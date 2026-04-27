"""Tests for DefenseAgent.rag.extraction.

Built-in backends (PyPdfExtractor / HtmlExtractor) are exercised with
in-tmp_path fixtures: HTML uses a hand-built file, PDF uses a mocked
pdfplumber so we don't need reportlab or a real PDF binary in tests.

Schema, facade routing, custom-backend priority, and resource persistence
are all covered without network access or heavy parsers.
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from DefenseAgent.config import AgentProfile


def _make_profile(tmp_path: Path, **rag_kwargs):
    """In-tmp_path AgentProfile with sane RAG defaults; rag_kwargs override individual fields."""
    profile = AgentProfile(
        id="t",
        name="t",
        age=1,
        traits="t",
        backstory="b",
        initial_plan="p",
        rag={"enabled": True, **rag_kwargs},
    )
    (tmp_path / "profile.yaml").write_text("agent: {}", encoding="utf-8")
    profile._source_path = (tmp_path / "profile.yaml").resolve()
    return profile


# ---------- schema ----------


def test_structured_chunk_carries_text_metadata_and_resources():
    """StructuredChunk holds text, a resources list, and a metadata dict."""
    from DefenseAgent.rag.extraction import StructuredChunk, StructuredResource

    res = StructuredResource(
        id="x@h@r",
        kind="image",
        path=Path("/tmp/x.png"),
        caption="diagram",
        mime_type="image/png",
    )
    chunk = StructuredChunk(
        text="see <resource_info>x@h@r</resource_info>",
        resources=[res],
        metadata={"source": "x.pdf", "page": 1},
    )

    assert chunk.resources[0].id == "x@h@r"
    assert chunk.metadata["page"] == 1
    assert "<resource_info>x@h@r</resource_info>" in chunk.text


def test_structured_resource_is_frozen():
    """StructuredResource is immutable: post-extraction tampering raises."""
    from DefenseAgent.rag.extraction import StructuredResource

    res = StructuredResource(id="x", kind="image", path=Path("/tmp/x.png"))
    with pytest.raises(FrozenInstanceError):
        res.id = "y"  # type: ignore[misc]


# ---------- facade resources_dir resolution ----------


def test_facade_default_resources_dir(tmp_path: Path):
    """Default resources_dir is <profile.source_dir>/rag/resources/."""
    from DefenseAgent.rag.extraction import StructuredDocExtractor

    profile = _make_profile(tmp_path)
    extractor = StructuredDocExtractor(profile)
    expected = (tmp_path / "rag" / "resources").resolve()

    assert extractor.resources_dir == expected
    assert expected.is_dir()


def test_facade_storage_dir_overrides_default(tmp_path: Path):
    """When profile.rag.storage_dir is set, resources_dir lives under it."""
    from DefenseAgent.rag.extraction import StructuredDocExtractor

    storage = tmp_path / "store"
    profile = _make_profile(tmp_path, storage_dir=str(storage))
    extractor = StructuredDocExtractor(profile)

    assert extractor.resources_dir == (storage / "resources").resolve()


def test_facade_explicit_resources_dir_wins(tmp_path: Path):
    """Explicit `resources_dir` argument overrides everything else."""
    from DefenseAgent.rag.extraction import StructuredDocExtractor

    profile = _make_profile(tmp_path)
    custom = tmp_path / "elsewhere"
    extractor = StructuredDocExtractor(profile, resources_dir=custom)

    assert extractor.resources_dir == custom.resolve()


def test_facade_raises_when_in_memory_profile_has_no_anchor():
    """In-memory profile + no resources_dir + no storage_dir → RAGConfigError."""
    from DefenseAgent.rag.base import RAGConfigError
    from DefenseAgent.rag.extraction import StructuredDocExtractor

    profile = AgentProfile(
        id="a", name="a", age=1, traits="t", backstory="b", initial_plan="p",
        rag={"enabled": True},
    )
    with pytest.raises(RAGConfigError, match="source_dir"):
        StructuredDocExtractor(profile)


# ---------- backend routing ----------


def test_facade_picks_pypdf_for_pdf(tmp_path: Path):
    from DefenseAgent.rag.extraction import PyPdfExtractor, StructuredDocExtractor

    profile = _make_profile(tmp_path)
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    extractor = StructuredDocExtractor(profile)
    backend = extractor._pick_backend(pdf)
    assert isinstance(backend, PyPdfExtractor)


def test_facade_picks_html_for_html(tmp_path: Path):
    from DefenseAgent.rag.extraction import HtmlExtractor, StructuredDocExtractor

    profile = _make_profile(tmp_path)
    html = tmp_path / "doc.html"
    html.write_text("<html></html>", encoding="utf-8")

    extractor = StructuredDocExtractor(profile)
    backend = extractor._pick_backend(html)
    assert isinstance(backend, HtmlExtractor)


def test_facade_returns_none_for_unsupported_extension(tmp_path: Path):
    """A .txt file matches none of the built-in backends."""
    from DefenseAgent.rag.extraction import StructuredDocExtractor

    profile = _make_profile(tmp_path)
    txt = tmp_path / "doc.txt"
    txt.write_text("hi", encoding="utf-8")

    extractor = StructuredDocExtractor(profile)
    assert extractor._pick_backend(txt) is None


# ---------- custom backend hook ----------


def test_custom_backend_takes_priority_over_builtins(tmp_path: Path):
    """User-supplied backends are tried before PyPdfExtractor / HtmlExtractor."""
    from DefenseAgent.rag.extraction import StructuredChunk, StructuredDocExtractor

    profile = _make_profile(tmp_path)
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    captured: list[str] = []

    class CustomBackend:
        def supports(self, source):
            captured.append(f"supports:{Path(source).name}")
            return True

        def extract(self, source):
            captured.append(f"extract:{Path(source).name}")
            return [StructuredChunk(text="from custom", metadata={"backend": "custom"})]

    extractor = StructuredDocExtractor(profile, custom_backends=[CustomBackend()])
    chunks = extractor.extract([pdf])

    assert captured == ["supports:doc.pdf", "extract:doc.pdf"]
    assert len(chunks) == 1
    assert chunks[0].metadata["backend"] == "custom"


def test_extract_isolates_failure_to_single_source(tmp_path: Path):
    """One source raising mid-extraction shouldn't prevent the rest from being processed."""
    from DefenseAgent.rag.extraction import StructuredChunk, StructuredDocExtractor

    profile = _make_profile(tmp_path)
    good = tmp_path / "good.pdf"
    bad = tmp_path / "bad.pdf"
    good.write_bytes(b"%PDF-1.4\n")
    bad.write_bytes(b"%PDF-1.4\n")

    class FlakyBackend:
        def supports(self, source):
            return str(source).endswith(".pdf")

        def extract(self, source):
            if "bad" in str(source):
                raise RuntimeError("simulated parse failure")
            return [StructuredChunk(text=f"ok:{Path(source).name}")]

    extractor = StructuredDocExtractor(profile, custom_backends=[FlakyBackend()])
    chunks = extractor.extract([good, bad])

    assert len(chunks) == 1
    assert chunks[0].text == "ok:good.pdf"


# ---------- HtmlExtractor end-to-end ----------


def test_html_extractor_persists_local_image_and_emits_marker(tmp_path: Path):
    """Realistic HTML flow: <img src=local.png> gets copied + marker inserted in chunk text."""
    pytest.importorskip("bs4")
    from DefenseAgent.rag.extraction import HtmlExtractor

    img_src = tmp_path / "diagram.png"
    img_src.write_bytes(b"\x89PNG\r\n\x1a\nfake-png-bytes")
    html_path = tmp_path / "doc.html"
    html_path.write_text(
        '<html><body>'
        '<h1>Section A</h1>'
        '<p>hello world</p>'
        '<img src="diagram.png" alt="A diagram"/>'
        '</body></html>',
        encoding="utf-8",
    )

    resources_dir = tmp_path / "resources"
    extractor = HtmlExtractor(resources_dir)
    chunks = extractor.extract(html_path)

    assert chunks, "expected at least one chunk"
    chunk_with_image = next((c for c in chunks if c.resources), None)
    assert chunk_with_image is not None, "no chunk carried a resource"
    assert "<resource_info>" in chunk_with_image.text

    res = chunk_with_image.resources[0]
    assert res.kind == "image"
    assert res.caption == "A diagram"
    assert res.path.is_file()
    assert res.path.read_bytes() == img_src.read_bytes()
    assert res.path.parent.parent == resources_dir.resolve()


def test_html_extractor_skips_remote_and_data_uris(tmp_path: Path):
    """Remote URLs and data URIs are ignored to keep MVP free of network I/O."""
    pytest.importorskip("bs4")
    from DefenseAgent.rag.extraction import HtmlExtractor

    html_path = tmp_path / "doc.html"
    html_path.write_text(
        '<html><body>'
        '<h1>x</h1>'
        '<img src="https://example.com/a.png"/>'
        '<img src="data:image/png;base64,AAAA"/>'
        '</body></html>',
        encoding="utf-8",
    )

    extractor = HtmlExtractor(tmp_path / "resources")
    chunks = extractor.extract(html_path)

    assert all(not c.resources for c in chunks)


def test_html_extractor_renders_table_to_markdown(tmp_path: Path):
    pytest.importorskip("bs4")
    from DefenseAgent.rag.extraction import HtmlExtractor

    html_path = tmp_path / "doc.html"
    html_path.write_text(
        '<html><body>'
        '<h1>x</h1>'
        '<table>'
        '<tr><th>Header A</th><th>Header B</th></tr>'
        '<tr><td>a1</td><td>b1</td></tr>'
        '</table>'
        '</body></html>',
        encoding="utf-8",
    )

    extractor = HtmlExtractor(tmp_path / "resources")
    chunks = extractor.extract(html_path)

    text = "\n".join(c.text for c in chunks)
    assert "| Header A | Header B |" in text
    assert "| a1 | b1 |" in text


# ---------- PyPdfExtractor flow (mocked pdfplumber) ----------


def test_pypdf_extractor_emits_chunk_with_image_resource(tmp_path: Path):
    """Mock pdfplumber to verify the end-to-end PDF → chunk + resource flow.

    This test does not require a real PDF binary - it stubs pdfplumber.open
    to return a fake page that yields one image and one short text snippet.
    """
    from DefenseAgent.rag.extraction import PyPdfExtractor

    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF-1.4\nfake")  # supports() only checks suffix + is_file

    fake_page = MagicMock()
    fake_page.extract_text.return_value = "Page intro text"
    fake_page.extract_tables.return_value = []
    fake_page.images = [{"x0": 0, "top": 0, "x1": 10, "bottom": 10, "name": "fig1"}]

    fake_pdf = MagicMock()
    fake_pdf.pages = [fake_page]
    fake_pdf.__enter__.return_value = fake_pdf
    fake_pdf.__exit__.return_value = None

    fake_pdfplumber = MagicMock()
    fake_pdfplumber.open.return_value = fake_pdf

    with (
        patch.dict("sys.modules", {"pdfplumber": fake_pdfplumber}),
        patch.object(PyPdfExtractor, "_render_image", staticmethod(lambda page, info: b"PNGBYTES")),
    ):
        extractor = PyPdfExtractor(tmp_path / "resources")
        chunks = extractor.extract(pdf)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert "Page intro text" in chunk.text
    assert "<resource_info>" in chunk.text
    assert chunk.metadata["page"] == 1
    assert len(chunk.resources) == 1
    assert chunk.resources[0].path.read_bytes() == b"PNGBYTES"


def test_pypdf_extractor_returns_empty_when_page_is_empty(tmp_path: Path):
    """Pages with no text and no images produce no chunks."""
    from DefenseAgent.rag.extraction import PyPdfExtractor

    pdf = tmp_path / "blank.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    fake_page = MagicMock()
    fake_page.extract_text.return_value = ""
    fake_page.extract_tables.return_value = []
    fake_page.images = []

    fake_pdf = MagicMock()
    fake_pdf.pages = [fake_page]
    fake_pdf.__enter__.return_value = fake_pdf
    fake_pdf.__exit__.return_value = None

    fake_pdfplumber = MagicMock()
    fake_pdfplumber.open.return_value = fake_pdf

    with patch.dict("sys.modules", {"pdfplumber": fake_pdfplumber}):
        extractor = PyPdfExtractor(tmp_path / "resources")
        chunks = extractor.extract(pdf)

    assert chunks == []


def test_pypdf_extractor_raises_helpful_error_when_pdfplumber_missing(tmp_path: Path):
    """If pdfplumber isn't installed, surface a RAGConfigError with install hint."""
    from DefenseAgent.rag.base import RAGConfigError
    from DefenseAgent.rag.extraction import PyPdfExtractor

    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    with patch.dict("sys.modules", {"pdfplumber": None}):
        extractor = PyPdfExtractor(tmp_path / "resources")
        with pytest.raises(RAGConfigError, match="pdfplumber"):
            extractor.extract(pdf)


# ---------- Step 1.4 — register() / supports() ----------


def test_facade_register_prepends_by_default(tmp_path: Path):
    """A backend registered with prepend=True (default) wins over the built-ins for overlapping supports() — used to override behavior for a specific format."""
    from DefenseAgent.rag.extraction import StructuredDocExtractor

    profile = _profile_with_anchor(tmp_path)
    facade = StructuredDocExtractor(profile, resources_dir=tmp_path / "rs")

    custom = MagicMock(name="CustomPdfBackend")
    custom.supports.return_value = True
    custom.extract.return_value = []

    facade.register(custom)

    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 stub")
    picked = facade._pick_backend(pdf)
    assert picked is custom


def test_facade_register_append_runs_after_builtins(tmp_path: Path):
    """A backend registered with prepend=False is only consulted when no built-in claims the source — useful as a fallback."""
    from DefenseAgent.rag.extraction import StructuredDocExtractor

    profile = _profile_with_anchor(tmp_path)
    facade = StructuredDocExtractor(profile, resources_dir=tmp_path / "rs")

    fallback = MagicMock(name="FallbackBackend")
    fallback.supports.return_value = True   # would claim everything if checked first
    fallback.extract.return_value = []

    facade.register(fallback, prepend=False)

    # PDF should still go to PyPdfExtractor (built-in), not the fallback
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 stub")
    picked = facade._pick_backend(pdf)
    assert picked is not fallback   # built-in PyPdfExtractor took it

    # An unrecognized format falls through to the fallback
    weird = tmp_path / "doc.xyz"
    weird.write_text("data")
    picked2 = facade._pick_backend(weird)
    assert picked2 is fallback


def test_facade_supports_method_proxies_to_backends(tmp_path: Path):
    """`StructuredDocExtractor.supports(source)` returns True iff any registered backend can handle it."""
    from DefenseAgent.rag.extraction import StructuredDocExtractor

    profile = _profile_with_anchor(tmp_path)
    facade = StructuredDocExtractor(profile, resources_dir=tmp_path / "rs")

    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 stub")
    md = tmp_path / "doc.md"
    md.write_text("# hello")

    assert facade.supports(pdf) is True   # PyPdfExtractor claims it
    assert facade.supports(md) is False   # neither built-in claims .md


def test_facade_extract_accepts_single_source(tmp_path: Path):
    """`extract(source)` accepts a single Path/str (not just a list) — convenience for single-file callers."""
    from DefenseAgent.rag.extraction import StructuredDocExtractor

    profile = _profile_with_anchor(tmp_path)
    facade = StructuredDocExtractor(profile, resources_dir=tmp_path / "rs")

    custom = MagicMock(name="CustomBackend")
    custom.supports.return_value = True
    custom.extract.return_value = []
    facade.register(custom)

    facade.extract(tmp_path / "any.txt")  # single source
    custom.extract.assert_called_once()


def _profile_with_anchor(tmp_path: Path) -> AgentProfile:
    """Build a profile rooted at tmp_path so resources_dir resolution works."""
    yaml_path = tmp_path / "profile.yaml"
    yaml_path.write_text(
        'agent:\n'
        '  id: "t"\n  name: "T"\n  age: 30\n'
        '  traits: "x"\n  backstory: "x"\n  initial_plan: "x"\n',
        encoding="utf-8",
    )
    return AgentProfile.from_yaml(yaml_path)
