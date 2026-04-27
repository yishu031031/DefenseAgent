from pathlib import Path
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv

from ms_agent.rag.llama_index_rag import LlamaIndexRAG as MsLlamaIndexRAG

from DefenseAgent.config.profile import AgentProfile
from DefenseAgent.rag._bridge import profile_to_rag_dictconfig
from DefenseAgent.rag.base import RAGConfigError
from DefenseAgent.rag.extraction import StructuredDocExtractor, StructuredResource
from DefenseAgent.rag.renderer import ResourceRenderer, default_renderers

if TYPE_CHECKING:
    from DefenseAgent.rag.extraction import StructuredChunk


_DEFAULT_DOC_GLOBS: tuple[str, ...] = (
    "*.md", "*.txt", "*.rst", "*.pdf", "*.html", "*.htm",
)


class LlamaIndexRAG(MsLlamaIndexRAG):
    """Inherits ms-agent's `LlamaIndexRAG`; adds profile-driven config, structured ingest, and a pluggable renderer registry.

    Adds:
      - `from_profile()` — auto-load persisted index or ingest documents on miss
      - `register_renderer()` — install a custom resource renderer at runtime
      - `render_resource(rid)` — fetch + format a resource for LLM consumption
      - `get_resource_kind() / get_resource_caption() / get_resource_mime()`
      - `extractor=` injection in the constructor for custom format support
    """

    def __init__(
        self,
        profile: AgentProfile,
        *,
        storage_path: str | Path | None = None,
        documents_path: str | Path | None = None,
        load_env: bool = True,
        dotenv_path: str | None = None,
        extractor: StructuredDocExtractor | None = None,
        renderers: dict[str, ResourceRenderer] | None = None,
    ) -> None:
        """Build the ms-agent DictConfig from `profile` + .env, ensure the storage dir exists, then defer to ms-agent's `__init__` (which loads the embedding model).

        `extractor` lets SDK callers plug in a `StructuredDocExtractor` configured
        with custom backends (e.g. a docx parser). When omitted, a default
        extractor with the built-in PDF + HTML backends is created lazily on
        the first `_auto_load()` call.

        `renderers` overrides the built-in renderer set at construction time.
        Most callers should leave this `None` and call `register_renderer()`
        to layer their own renderers on top of the defaults.
        """
        if load_env:
            load_dotenv(dotenv_path, override=False)
        config = profile_to_rag_dictconfig(
            profile,
            storage_path=storage_path,
            documents_path=documents_path,
        )
        Path(config.rag.storage_dir).mkdir(parents=True, exist_ok=True)
        super().__init__(config)
        self.profile = profile
        self._documents_dir: Path | None = (
            Path(config.documents_dir).resolve()
            if "documents_dir" in config
            else None
        )
        self._extractor: StructuredDocExtractor | None = extractor
        self._renderers: dict[str, ResourceRenderer] = (
            dict(renderers) if renderers is not None else default_renderers()
        )

    @classmethod
    async def from_profile(
        cls,
        profile: AgentProfile,
        *,
        storage_path: str | Path | None = None,
        documents_path: str | Path | None = None,
        load_env: bool = True,
        dotenv_path: str | None = None,
        auto_load: bool = True,
        extractor: StructuredDocExtractor | None = None,
        renderers: dict[str, ResourceRenderer] | None = None,
    ) -> "LlamaIndexRAG":
        """Convenience constructor that mirrors the rest of DefenseAgent. When `auto_load=True` (default), tries `load_index()` first and falls back to ingesting every file under the configured documents directory, then persists the index."""
        instance = cls(
            profile,
            storage_path=storage_path,
            documents_path=documents_path,
            load_env=load_env,
            dotenv_path=dotenv_path,
            extractor=extractor,
            renderers=renderers,
        )
        if auto_load:
            await instance._auto_load()
        return instance

    # ---- renderer registry ----

    def register_renderer(self, renderer: ResourceRenderer) -> None:
        """Register or override a `ResourceRenderer` keyed by `renderer.kind`.

        Built-in kinds (`"image"`, `"table"`) can be overridden by passing a
        renderer with the same `kind`. SDK callers add new kinds (`"csv"`,
        `"audio"`, ...) by registering a renderer that declares that kind.
        """
        self._renderers[renderer.kind] = renderer

    async def render_resource(self, resource_id: str) -> str:
        """Fetch a resource by id and render it via the registered renderer.

        Returns a diagnostic string when the id is unknown or no renderer
        matches — never raises, so the agent layer can hand the result
        straight to the LLM.
        """
        path = self.get_resource_path(resource_id)
        if path is None:
            return f"(no resource found with id={resource_id!r})"
        kind = self.get_resource_kind(resource_id) or "unknown"
        renderer = self._renderers.get(kind)
        if renderer is None:
            caption = self.get_resource_caption(resource_id) or ""
            cap = f' "{caption}"' if caption else ""
            return f"resource [{resource_id}]{cap} (kind={kind!r}) at {path}"
        resource = StructuredResource(
            id=resource_id,
            kind=kind,
            path=path,
            caption=self.get_resource_caption(resource_id) or "",
            mime_type=self.get_resource_mime(resource_id) or "",
        )
        try:
            return await renderer.render(resource)
        except Exception as e:  # noqa: BLE001 - diagnostic for the LLM
            return (
                f"(renderer for kind={kind!r} failed on {resource_id!r}: "
                f"{type(e).__name__}: {e})"
            )

    # ---- ingestion ----

    async def _auto_load(self) -> None:
        """Try `load_index()`; on miss, ingest the documents dir using the structured pipeline.

        Files claimed by `self.extractor.supports()` go through structured
        extraction (preserves images / tables); everything else falls back to
        the plain-text path. Both feed the same vector index.
        """
        try:
            await self.load_index()
            return
        except FileNotFoundError:
            pass
        if self._documents_dir is None or not self._documents_dir.is_dir():
            return
        files = _collect_document_files(self._documents_dir)
        if not files:
            return

        extractor = self._ensure_extractor()
        structured_files: list[Path] = []
        plain_files: list[Path] = []
        for f in files:
            try:
                if extractor.supports(f):
                    structured_files.append(f)
                else:
                    plain_files.append(f)
            except Exception:  # noqa: BLE001 - extractor probe shouldn't crash auto-load
                plain_files.append(f)

        if structured_files:
            chunks = extractor.extract(structured_files)
            if chunks:
                await self.add_structured_chunks(chunks)

        if plain_files:
            await self.add_documents_from_files([str(f) for f in plain_files])

        if structured_files or plain_files:
            await self.save_index()

    def _ensure_extractor(self) -> StructuredDocExtractor:
        """Build the default StructuredDocExtractor on demand if one wasn't injected."""
        if self._extractor is None:
            resources_dir = Path(self.storage_dir) / "resources"
            self._extractor = StructuredDocExtractor(
                self.profile, resources_dir=resources_dir,
            )
        return self._extractor

    async def add_structured_chunks(self, chunks: "list[StructuredChunk]") -> None:
        """Ingest pre-extracted structured chunks into the vector index.

        Each chunk's resource ids, paths, kinds, captions, and mime types are
        stored in node metadata so callers can map a retrieved chunk back to
        its associated images / tables / etc. via `get_resource_path()` and
        friends. Paths under `self.storage_dir` are persisted as POSIX-style
        relative strings so the index remains portable across machines.

        Idempotent on the index level: when called repeatedly, new chunks are
        appended to an existing index rather than replacing it.
        """
        if not chunks:
            return
        from llama_index.core import Document
        documents = [
            Document(
                text=c.text,
                metadata={
                    **c.metadata,
                    "resource_ids":      [r.id for r in c.resources],
                    "resource_paths":    [self._serialize_resource_path(r.path) for r in c.resources],
                    "resource_kinds":    [r.kind for r in c.resources],
                    "resource_captions": [r.caption for r in c.resources],
                    "resource_mimes":    [r.mime_type for r in c.resources],
                },
            )
            for c in chunks
        ]
        await self._add_documents_additive(documents)

    async def add_documents(self, documents: list[str]) -> None:
        """Override ms-agent's `add_documents` to be additive instead of replacing.

        ms-agent's base implementation does `self.index = VectorStoreIndex.from_documents(...)`,
        which silently wipes any previously-ingested chunks. We want repeated
        calls to accumulate so a single agent can mix structured chunks +
        plain-text documents in any order.
        """
        if not documents:
            raise ValueError("Document list cannot be empty")
        from llama_index.core import Document
        docs = [Document(text=d) for d in documents]
        await self._add_documents_additive(docs)

    async def add_documents_from_files(self, file_paths: list[str]) -> None:
        """Override ms-agent's `add_documents_from_files` to be additive (see add_documents)."""
        if not file_paths:
            raise ValueError("File path list cannot be empty")
        import os
        from llama_index.core.readers import SimpleDirectoryReader
        documents = []
        for file_path in file_paths:
            if not os.path.exists(file_path):
                raise ValueError(f"File {file_path} does not exist")
            if os.path.isfile(file_path):
                reader = SimpleDirectoryReader(input_files=[file_path])
            else:
                reader = SimpleDirectoryReader(input_dir=file_path)
            documents.extend(reader.load_data())
        await self._add_documents_additive(documents)

    async def _add_documents_additive(self, documents: list[Any]) -> None:
        """Append `documents` to `self.index`; create the index on the first call.

        Shared bottom-half for `add_documents`, `add_documents_from_files`, and
        `add_structured_chunks` — keeps the "build vs append" decision in one
        place so repeated ingest calls never silently clobber earlier chunks.
        """
        if not documents:
            return
        from llama_index.core import VectorStoreIndex
        if self.index is None:
            self.index = VectorStoreIndex.from_documents(documents)
        else:
            for doc in documents:
                self.index.insert(doc)
        if not self.retrieve_only:
            await self._setup_query_engine()

    # ---- resource lookup helpers ----

    def get_resource_path(self, resource_id: str) -> Path | None:
        """Look up a persisted resource path by id across every indexed document.

        Stored values can be either POSIX-relative-to-storage (the new default,
        portable) or absolute (legacy indexes / out-of-tree resources): both
        are resolved back to an absolute `Path` here so callers always get a
        usable filesystem path. Returns None when the index hasn't been built
        or the id wasn't found.
        """
        raw = self._lookup_resource_field(resource_id, "resource_paths")
        return self._deserialize_resource_path(raw) if raw is not None else None

    def get_resource_kind(self, resource_id: str) -> str | None:
        """Look up a resource's kind ('image' | 'table' | custom) by id."""
        return self._lookup_resource_field(resource_id, "resource_kinds")

    def get_resource_caption(self, resource_id: str) -> str | None:
        """Look up a resource's caption (may be empty for old indexes)."""
        return self._lookup_resource_field(resource_id, "resource_captions")

    def get_resource_mime(self, resource_id: str) -> str | None:
        """Look up a resource's mime type (may be empty for old indexes)."""
        return self._lookup_resource_field(resource_id, "resource_mimes")

    def _lookup_resource_field(self, resource_id: str, field: str) -> str | None:
        """Find `resource_id` in any indexed doc and return the matching entry from `field`.

        `field` is the metadata key holding the parallel array (e.g.
        `"resource_paths"`, `"resource_kinds"`). Older indexes built before
        the field existed simply return None — callers must tolerate that.
        """
        if self.index is None:
            return None
        for doc in self.index.docstore.docs.values():
            metadata = getattr(doc, "metadata", None) or {}
            ids = metadata.get("resource_ids", [])
            if resource_id not in ids:
                continue
            values = metadata.get(field, [])
            idx = ids.index(resource_id)
            if idx < len(values):
                return values[idx]
            return None
        return None

    def _serialize_resource_path(self, path: Path | str) -> str:
        """Serialize one resource path for index metadata.

        Returns a POSIX-style relative path when `path` lives under
        `self.storage_dir`, otherwise the absolute string form. Always uses
        forward slashes so a Windows-built index can be loaded on Linux.
        """
        abs_path = Path(path).resolve()
        storage_root = self._storage_root()
        if storage_root is not None:
            try:
                return abs_path.relative_to(storage_root).as_posix()
            except ValueError:
                pass  # resource lives outside storage_dir → keep absolute
        return abs_path.as_posix()

    def _deserialize_resource_path(self, raw: str) -> Path:
        """Inverse of `_serialize_resource_path`: relative entries are resolved
        against the current `storage_dir` so a moved index still finds its
        files; already-absolute entries are returned unchanged."""
        candidate = Path(raw)
        if candidate.is_absolute():
            return candidate
        storage_root = self._storage_root()
        if storage_root is None:
            return candidate.resolve()
        return (storage_root / candidate).resolve()

    def _storage_root(self) -> Path | None:
        """Return the resolved storage_dir as a Path, or None when unset."""
        if not getattr(self, "storage_dir", None):
            return None
        return Path(self.storage_dir).resolve()

    # ---- retrieval ----

    async def hybrid_retrieve(
        self,
        query: str,
        limit: int | None = None,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Vector + BM25 fused retrieval; degrades to vector-only when BM25 is unavailable.

        Defaults `limit` to `profile.rag.top_k` and `score_threshold` to
        `profile.rag.score_threshold`. Returns the same dict shape as
        `retrieve()`: `{text, score, metadata, node_id}`. Returns `[]` when no
        index has been loaded.
        """
        if self.index is None:
            return []
        top_k = limit if limit is not None else self.profile.rag.top_k
        threshold = (
            score_threshold
            if score_threshold is not None
            else self.profile.rag.score_threshold
        )
        raw = await super().hybrid_search(query, top_k=top_k)
        return [r for r in raw if r["score"] >= threshold]

    # ---- embedding setup (delegated from ms-agent) ----

    def _setup_embedding_model(self, config) -> None:
        """Dispatch on `config.rag.embedding_provider`: route 'openai' to our OpenAI-compatible installer (reuses mem0's EMBEDDING_* env, no torch/sentence-transformers needed); fall back to ms-agent's HuggingFace path otherwise."""
        provider = getattr(config.rag, "embedding_provider", "openai")
        if provider == "openai":
            self._install_openai_compat_embedding()
            return
        super()._setup_embedding_model(config)

    def _install_openai_compat_embedding(self) -> None:
        """Wire `Settings.embed_model` to llama-index's `OpenAILikeEmbedding`, reading the embedder fields already resolved (profile-first / env-fallback) on `self.config.rag` by `profile_to_rag_dictconfig`. Raises RAGConfigError when the resolved config is missing required fields or `llama-index-embeddings-openai-like` is not installed."""
        rag_cfg = self.config.rag
        model = _require(getattr(rag_cfg, "embedding", None), "EMBEDDING_MODEL", "embedding")
        api_key = _require(getattr(rag_cfg, "embedding_api_key", None), "EMBEDDING_API_KEY", "embedding_api_key")
        base_url = getattr(rag_cfg, "embedding_base_url", None) or None
        dims = getattr(rag_cfg, "embedding_dims", None)
        try:
            from llama_index.core import Settings
            from llama_index.embeddings.openai_like import OpenAILikeEmbedding
        except ImportError as e:
            raise RAGConfigError(
                "OpenAI-compatible RAG embedding requires "
                "`pip install llama-index-core llama-index-embeddings-openai-like`"
            ) from e
        kwargs: dict[str, Any] = {
            "model_name": model,
            "api_key": api_key,
            "embed_batch_size": 10,
        }
        if base_url:
            kwargs["api_base"] = base_url
        if dims is not None:
            kwargs["embed_dim"] = dims
        Settings.embed_model = OpenAILikeEmbedding(**kwargs)
        self.embedding_model = model


def _require(value: Any, env_var: str, profile_field: str) -> str:
    """Resolved-value getter for the OpenAI-compat embedder. The bridge already did profile-then-env resolution; if the field is still missing here, neither the profile nor .env supplied it — raise with both candidate sources named."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise RAGConfigError(
        f"OpenAI-compatible RAG embedding requires `{env_var}` (or "
        f"`profile.rag.{profile_field}`) to be set"
    )


def _collect_document_files(directory: Path) -> list[Path]:
    """Walk `directory` and return every file matching the default doc globs."""
    out: list[Path] = []
    for pattern in _DEFAULT_DOC_GLOBS:
        out.extend(p for p in directory.rglob(pattern) if p.is_file())
    return sorted(set(out))
