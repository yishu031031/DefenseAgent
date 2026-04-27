"""Structured document extraction for DefenseAgent's RAG.

Lightweight extractors (PDF / HTML) that preserve images and tables as
referenced resources rather than dropping them. Each backend emits
`StructuredChunk` objects containing text with inline
`<resource_info>ID</resource_info>` markers, plus a list of persisted
`StructuredResource` records pointing to the saved image / table files.

Heavy backends (OCR, scanned-PDF parsing, docling, etc.) are deliberately
out of scope here: callers can plug in extra backends via the
`custom_backends` argument on `StructuredDocExtractor` without forcing
those dependencies on every install.
"""
from __future__ import annotations

import hashlib
import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

from DefenseAgent.config.profile import AgentProfile
from DefenseAgent.rag.base import RAGConfigError

logger = logging.getLogger(__name__)


# ---------- schema ----------


@dataclass(frozen=True, slots=True)
class StructuredResource:
    """A persisted non-text payload (image / table / etc.) attached to a chunk.

    `id` follows ms-agent's `<source>@<hash>@<ref>` pattern so the same input
    file always resolves to the same resource ids across runs.

    `kind` is an open string, not a closed enum: built-in extractors emit
    `"image"` and `"table"`, but SDK callers can register custom kinds
    (`"csv"`, `"audio"`, `"chart"`, ...) along with matching renderers via
    `LlamaIndexRAG.register_renderer()`.

    `extra` is an open dict for extractor-specific or downstream-renderer
    metadata that doesn't fit the canonical fields (e.g. an audio extractor
    might stash `{"duration_sec": 12.5}`, an OCR extractor might add
    `{"confidence": 0.89}`). The agent layer never reads it; it's a contract
    between the extractor that produced the resource and the renderer that
    consumes it.
    """

    id: str
    kind: str
    path: Path
    caption: str = ""
    mime_type: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StructuredChunk:
    """Text block with inline `<resource_info>ID</resource_info>` markers
    referencing items in `resources`."""

    text: str
    resources: list[StructuredResource] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------- backend protocol ----------


@runtime_checkable
class StructuredExtractor(Protocol):
    """Extractor backend contract.

    Backends own their persistence: when they return a `StructuredChunk`,
    every `StructuredResource.path` must point to an already-written file.
    The facade injects a `resources_dir` into the built-in backends; custom
    backends are free to manage persistence however they like.
    """

    def supports(self, source: str | Path) -> bool: ...

    def extract(self, source: str | Path) -> list[StructuredChunk]: ...


# ---------- helpers ----------


def _hash_file(path: Path) -> str:
    """Stable short hash of file contents; used to namespace resource ids and on-disk subdirs."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _table_to_markdown(rows: list[list[Any]]) -> str:
    """Convert a 2D list of cells (pdfplumber-style) to a GitHub-flavoured markdown table."""
    cleaned = [
        [("" if cell is None else str(cell)).strip().replace("|", "\\|") for cell in row]
        for row in rows
        if row
    ]
    if not cleaned:
        return ""
    width = max(len(r) for r in cleaned)
    cleaned = [r + [""] * (width - len(r)) for r in cleaned]
    header = "| " + " | ".join(cleaned[0]) + " |"
    sep = "| " + " | ".join(["---"] * width) + " |"
    body = "\n".join("| " + " | ".join(r) + " |" for r in cleaned[1:])
    return f"{header}\n{sep}\n{body}" if body else f"{header}\n{sep}"


# ---------- built-in backends ----------


class PyPdfExtractor:
    """PDF backend using pdfplumber for text, tables, and images.

    Produces one chunk per page; images are saved to
    `<resources_dir>/<source_hash>/p<page>_img<idx>.png` and referenced
    inline via `<resource_info>` markers.
    """

    def __init__(self, resources_dir: Path) -> None:
        self.resources_dir = resources_dir

    def supports(self, source: str | Path) -> bool:
        path = Path(source)
        return path.suffix.lower() == ".pdf" and path.is_file()

    def extract(self, source: str | Path) -> list[StructuredChunk]:
        try:
            import pdfplumber  # type: ignore
        except ImportError as e:
            raise RAGConfigError(
                "PDF extraction requires `pip install pdfplumber Pillow`"
            ) from e

        path = Path(source).resolve()
        source_hash = _hash_file(path)[:12]
        sub_dir = self.resources_dir / source_hash
        sub_dir.mkdir(parents=True, exist_ok=True)

        chunks: list[StructuredChunk] = []
        with pdfplumber.open(path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                chunk = self._extract_page(page, page_num, path, source_hash, sub_dir)
                if chunk is not None:
                    chunks.append(chunk)
        return chunks

    def _extract_page(
        self,
        page: Any,
        page_num: int,
        source_path: Path,
        source_hash: str,
        sub_dir: Path,
    ) -> StructuredChunk | None:
        text_parts: list[str] = []
        resources: list[StructuredResource] = []

        page_text = (page.extract_text() or "").strip()
        if page_text:
            text_parts.append(page_text)

        for table in page.extract_tables() or []:
            md = _table_to_markdown(table)
            if md:
                text_parts.append(md)

        for img_idx, img_info in enumerate(page.images or []):
            png_bytes = self._render_image(page, img_info)
            if png_bytes is None:
                continue
            rid = f"{source_path.name}@{source_hash}@p{page_num}_img{img_idx}"
            img_path = sub_dir / f"p{page_num}_img{img_idx}.png"
            img_path.write_bytes(png_bytes)
            resources.append(
                StructuredResource(
                    id=rid,
                    kind="image",
                    path=img_path,
                    caption=str(img_info.get("name", "") or ""),
                    mime_type="image/png",
                )
            )
            text_parts.append(f"<resource_info>{rid}</resource_info>")

        if not text_parts and not resources:
            return None
        return StructuredChunk(
            text="\n\n".join(text_parts),
            resources=resources,
            metadata={"source": str(source_path), "page": page_num},
        )

    @staticmethod
    def _render_image(page: Any, img_info: Mapping[str, Any]) -> bytes | None:
        """Crop the image bbox out of the rendered page and return PNG bytes; None on failure."""
        try:
            bbox = (
                img_info["x0"],
                img_info["top"],
                img_info["x1"],
                img_info["bottom"],
            )
            cropped = page.crop(bbox)
            pil_img = cropped.to_image(resolution=150).original
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            return buf.getvalue()
        except Exception as e:  # noqa: BLE001 - any failure → skip the image
            logger.debug("failed to render image bbox: %s", e)
            return None


class HtmlExtractor:
    """HTML backend using BeautifulSoup.

    Splits the document at h1/h2 boundaries; each section becomes one chunk.
    Inline `<img src="...">` referring to local files are copied into
    `<resources_dir>/<source_hash>/`. Remote URLs and `data:` URIs are
    skipped to keep the MVP free of network I/O.
    """

    def __init__(self, resources_dir: Path) -> None:
        self.resources_dir = resources_dir

    def supports(self, source: str | Path) -> bool:
        path = Path(source)
        return path.suffix.lower() in {".html", ".htm"} and path.is_file()

    def extract(self, source: str | Path) -> list[StructuredChunk]:
        try:
            from bs4 import BeautifulSoup  # type: ignore
        except ImportError as e:
            raise RAGConfigError(
                "HTML extraction requires `pip install beautifulsoup4`"
            ) from e

        path = Path(source).resolve()
        source_hash = _hash_file(path)[:12]
        sub_dir = self.resources_dir / source_hash
        sub_dir.mkdir(parents=True, exist_ok=True)

        html = path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(html, "html.parser")
        body = soup.body or soup

        sections = self._split_sections(body)
        chunks: list[StructuredChunk] = []
        for sec_idx, section in enumerate(sections):
            chunk = self._build_chunk(section, sec_idx, path, source_hash, sub_dir)
            if chunk is not None:
                chunks.append(chunk)
        return chunks

    @staticmethod
    def _split_sections(body: Any) -> list[list[Any]]:
        sections: list[list[Any]] = [[]]
        for elem in body.children:
            if getattr(elem, "name", None) in {"h1", "h2"}:
                sections.append([elem])
            else:
                sections[-1].append(elem)
        return [s for s in sections if s]

    def _build_chunk(
        self,
        section: list[Any],
        sec_idx: int,
        source_path: Path,
        source_hash: str,
        sub_dir: Path,
    ) -> StructuredChunk | None:
        text_parts: list[str] = []
        resources: list[StructuredResource] = []
        local_dir = source_path.parent

        def consume(elem: Any) -> None:
            name = getattr(elem, "name", None)
            if name is None:
                text = str(elem).strip()
                if text:
                    text_parts.append(text)
                return
            if name == "img":
                res = self._copy_image(
                    elem, sec_idx, len(resources), source_path, source_hash, sub_dir, local_dir
                )
                if res is not None:
                    resources.append(res)
                    text_parts.append(f"<resource_info>{res.id}</resource_info>")
                return
            if name == "table":
                rows = []
                for tr in elem.find_all("tr"):
                    cells = [c.get_text(strip=True) for c in tr.find_all(["th", "td"])]
                    if cells:
                        rows.append(cells)
                md = _table_to_markdown(rows)
                if md:
                    text_parts.append(md)
                return
            # Recursively descend into other tags so nested <img>/<table> are captured.
            children = list(getattr(elem, "children", []))
            if children:
                for child in children:
                    consume(child)
            else:
                text = elem.get_text(separator=" ", strip=True)
                if text:
                    text_parts.append(text)

        for elem in section:
            consume(elem)

        if not text_parts and not resources:
            return None
        return StructuredChunk(
            text="\n\n".join(text_parts),
            resources=resources,
            metadata={"source": str(source_path), "section": sec_idx},
        )

    @staticmethod
    def _copy_image(
        img_tag: Any,
        sec_idx: int,
        img_idx: int,
        source_path: Path,
        source_hash: str,
        sub_dir: Path,
        local_dir: Path,
    ) -> StructuredResource | None:
        src = img_tag.get("src", "")
        if not src or src.startswith(("http://", "https://", "data:")):
            return None
        src_path = (local_dir / src).resolve()
        if not src_path.is_file():
            return None
        ext = src_path.suffix.lstrip(".") or "png"
        rid = f"{source_path.name}@{source_hash}@s{sec_idx}_img{img_idx}"
        dst = sub_dir / f"s{sec_idx}_img{img_idx}.{ext}"
        dst.write_bytes(src_path.read_bytes())
        return StructuredResource(
            id=rid,
            kind="image",
            path=dst,
            caption=img_tag.get("alt", "") or "",
            mime_type=f"image/{ext}",
        )


# ---------- facade ----------


class StructuredDocExtractor:
    """Routes sources to the right backend and aggregates their chunks.

    Custom backends provided via `custom_backends` are tried before the
    built-in PDF / HTML extractors, so callers can override behaviour for
    specific source types (e.g. plug in a docling-based extractor for
    scanned PDFs) without modifying this module.
    """

    def __init__(
        self,
        profile: AgentProfile,
        *,
        resources_dir: str | Path | None = None,
        custom_backends: list[StructuredExtractor] | None = None,
    ) -> None:
        self.profile = profile
        self.resources_dir = self._resolve_resources_dir(resources_dir)
        self.resources_dir.mkdir(parents=True, exist_ok=True)
        backends: list[StructuredExtractor] = list(custom_backends or [])
        backends.extend(
            [
                PyPdfExtractor(self.resources_dir),
                HtmlExtractor(self.resources_dir),
            ]
        )
        self._backends: list[StructuredExtractor] = backends

    def register(self, backend: StructuredExtractor, *, prepend: bool = True) -> None:
        """Register a custom backend at runtime.

        `prepend=True` (default) puts the new backend ahead of the built-ins
        so it wins on overlapping `supports()` checks (the typical override
        case). Set `prepend=False` to register a fallback that only runs when
        no built-in claims the file.
        """
        if prepend:
            self._backends.insert(0, backend)
        else:
            self._backends.append(backend)

    def supports(self, source: str | Path) -> bool:
        """Return True iff at least one registered backend can handle `source`."""
        return self._pick_backend(source) is not None

    def extract(self, sources: "str | Path | list[str | Path]") -> list[StructuredChunk]:
        """Process one source or a list of sources; failures on individual sources are logged and skipped.

        Accepts either a single path (returns chunks for that one file) or a
        list of paths (returns chunks merged across all files), so callers can
        mirror either the per-file backend protocol or the batch facade in the
        same call.
        """
        if isinstance(sources, (str, Path)):
            sources_iter: list[str | Path] = [sources]
        else:
            sources_iter = list(sources)
        all_chunks: list[StructuredChunk] = []
        for source in sources_iter:
            backend = self._pick_backend(source)
            if backend is None:
                logger.warning("no backend supports source: %s", source)
                continue
            try:
                chunks = backend.extract(source)
            except Exception as e:  # noqa: BLE001 - one bad file shouldn't kill the batch
                logger.warning("extraction failed for %s: %s", source, e)
                continue
            all_chunks.extend(chunks)
        return all_chunks

    def _pick_backend(self, source: str | Path) -> StructuredExtractor | None:
        for backend in self._backends:
            try:
                if backend.supports(source):
                    return backend
            except Exception as e:  # noqa: BLE001 - backend probe shouldn't crash routing
                logger.debug("backend %s.supports() raised: %s", type(backend).__name__, e)
        return None

    def _resolve_resources_dir(self, override: str | Path | None) -> Path:
        if override is not None:
            return Path(override).resolve()
        if self.profile.rag.storage_dir:
            return (Path(self.profile.rag.storage_dir) / "resources").resolve()
        if self.profile.source_dir is None:
            raise RAGConfigError(
                "profile has no source_dir; pass resources_dir explicitly when "
                "loading an in-memory profile"
            )
        return (self.profile.source_dir / "rag" / "resources").resolve()
