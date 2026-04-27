"""Tests for DefenseAgent.rag.LlamaIndexRAG (inherited from ms-agent's class).

llama-index isn't a hard dependency here, so we bypass ms-agent's __init__ (which
imports llama-index, downloads embedding models, and builds a SentenceSplitter).
A stub __init__ assigns just the attributes downstream code needs, letting us
exercise our profile→DictConfig translation, storage/document path resolution,
and the from_profile()/auto_load() flow offline.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from DefenseAgent.config import AgentProfile


def _set_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Populate env vars that `_flat_llm_config_from_env` reads (only needed when retrieve_only=False)."""
    monkeypatch.setenv("AGENT_LAB_LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.example.com")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-chat")


def _make_profile(
    tmp_path: Path,
    *,
    documents_dir: str | None = None,
    storage_dir: str | None = None,
    retrieve_only: bool = True,
    embedding_provider: str = "openai",
) -> AgentProfile:
    """Build an in-memory AgentProfile rooted at tmp_path with optional rag tweaks."""
    profile = AgentProfile(
        id="test_agent", name="Tester", age=25,
        traits="t", backstory="b", initial_plan="p",
        rag={
            "enabled": True,
            "documents_dir": documents_dir,
            "storage_dir": storage_dir,
            "retrieve_only": retrieve_only,
            "embedding_provider": embedding_provider,
        },
    )
    (tmp_path / "profile.yaml").write_text("agent: {}", encoding="utf-8")
    profile._source_path = (tmp_path / "profile.yaml").resolve()
    return profile


def _set_embedding_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Populate the EMBEDDING_* env vars our OpenAI-compat installer reads."""
    monkeypatch.setenv("EMBEDDING_API_KEY", "sk-emb")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://api.example.com")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("EMBEDDING_DIMS", "1536")


def _stub_parent_init(parent_self, config) -> None:
    """Stand-in for ms-agent's LlamaIndexRAG.__init__ that skips llama-index/HF imports."""
    parent_self.config = config
    parent_self.embedding_model = config.rag.embedding
    parent_self.chunk_size = config.rag.chunk_size
    parent_self.chunk_overlap = config.rag.chunk_overlap
    parent_self.retrieve_only = config.rag.retrieve_only
    parent_self.storage_dir = config.rag.storage_dir
    parent_self.index = None
    parent_self.query_engine = None


def _build_rag(
    profile: AgentProfile,
    **kwargs,
):
    """Construct a LlamaIndexRAG with the heavy parent __init__ stubbed out."""
    from ms_agent.rag.llama_index_rag import LlamaIndexRAG as MsLlamaIndexRAG
    from DefenseAgent.rag.llama_index_rag import LlamaIndexRAG

    with patch.object(MsLlamaIndexRAG, "__init__", _stub_parent_init):
        return LlamaIndexRAG(profile, load_env=False, **kwargs)


# ---------- inheritance + re-export contract ----------


def test_llama_index_rag_inherits_from_ms_agent():
    from ms_agent.rag.llama_index_rag import LlamaIndexRAG as MsLlamaIndexRAG
    from DefenseAgent.rag.llama_index_rag import LlamaIndexRAG

    assert issubclass(LlamaIndexRAG, MsLlamaIndexRAG)


def test_rag_mapping_overrides_with_our_subclass():
    from DefenseAgent.rag import LlamaIndexRAG, rag_mapping

    assert rag_mapping["LlamaIndexRAG"] is LlamaIndexRAG


def test_rag_base_abc_re_exported():
    from DefenseAgent.rag import RAG, RAGConfigError, RAGError, RAGProviderError

    assert issubclass(RAGConfigError, RAGError)
    assert issubclass(RAGProviderError, RAGError)
    assert RAG.__abstractmethods__ >= {"add_documents", "retrieve", "answer"}


# ---------- bridge: profile_to_rag_dictconfig ----------


def test_bridge_translates_rag_knobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The bridge resolves embedding fields per-profile-then-env. With env supplying the embedder, the resolved DictConfig should carry those env values verbatim."""
    from DefenseAgent.rag._bridge import profile_to_rag_dictconfig

    _set_embedding_env(monkeypatch)
    profile = _make_profile(tmp_path, retrieve_only=True)
    config = profile_to_rag_dictconfig(profile)

    assert config.rag.embedding == "text-embedding-3-small"
    assert config.rag.embedding_api_key == "sk-emb"
    assert config.rag.embedding_base_url == "https://api.example.com"
    assert config.rag.embedding_dims == 1536
    assert config.rag.chunk_size == 512
    assert config.rag.chunk_overlap == 50
    assert config.rag.retrieve_only is True
    assert Path(config.rag.storage_dir) == (tmp_path / "rag").resolve()
    assert config.use_huggingface is False
    assert "llm" not in config  # retrieve_only=True skips LLM block


def test_bridge_profile_embedding_overrides_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """When the profile populates embedding fields, those values win over .env per field."""
    from DefenseAgent.rag._bridge import profile_to_rag_dictconfig

    _set_embedding_env(monkeypatch)
    profile = AgentProfile(
        id="t", name="T", age=20, traits="t", backstory="b", initial_plan="p",
        rag={
            "enabled": True,
            "embedding": "BAAI/bge-large-en-v1.5",
            "embedding_api_key": "sk-from-profile",
            "embedding_base_url": "https://from-profile.example/v1",
            "embedding_dims": 768,
        },
    )
    (tmp_path / "profile.yaml").write_text("agent: {}", encoding="utf-8")
    profile._source_path = (tmp_path / "profile.yaml").resolve()

    config = profile_to_rag_dictconfig(profile)
    assert config.rag.embedding == "BAAI/bge-large-en-v1.5"
    assert config.rag.embedding_api_key == "sk-from-profile"
    assert config.rag.embedding_base_url == "https://from-profile.example/v1"
    assert config.rag.embedding_dims == 768


def test_bridge_partial_profile_falls_back_to_env_per_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Profile sets only `embedding` (model name); api_key/base_url/dims come from .env."""
    from DefenseAgent.rag._bridge import profile_to_rag_dictconfig

    _set_embedding_env(monkeypatch)
    profile = AgentProfile(
        id="t", name="T", age=20, traits="t", backstory="b", initial_plan="p",
        rag={"enabled": True, "embedding": "BAAI/bge-large-en-v1.5"},
    )
    (tmp_path / "profile.yaml").write_text("agent: {}", encoding="utf-8")
    profile._source_path = (tmp_path / "profile.yaml").resolve()

    config = profile_to_rag_dictconfig(profile)
    assert config.rag.embedding == "BAAI/bge-large-en-v1.5"      # from profile
    assert config.rag.embedding_api_key == "sk-emb"               # from env
    assert config.rag.embedding_base_url == "https://api.example.com"  # from env
    assert config.rag.embedding_dims == 1536                      # from env


def test_bridge_hf_provider_falls_back_to_default_embedding_when_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """HuggingFace path needs *some* embedding model name; if neither profile nor env supplies one, the bridge fills in the ms-agent default rather than letting the index build crash."""
    from DefenseAgent.rag._bridge import profile_to_rag_dictconfig

    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    profile = _make_profile(tmp_path, embedding_provider="huggingface")
    config = profile_to_rag_dictconfig(profile)
    assert config.rag.embedding == "Qwen/Qwen3-Embedding-0.6B"


def test_bridge_includes_llm_when_not_retrieve_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    _set_llm_env(monkeypatch)
    from DefenseAgent.rag._bridge import profile_to_rag_dictconfig

    profile = _make_profile(tmp_path, retrieve_only=False)
    config = profile_to_rag_dictconfig(profile)

    assert "llm" in config
    assert config.llm.model == "deepseek-chat"


def test_bridge_includes_documents_dir_when_configured(tmp_path: Path):
    from DefenseAgent.rag._bridge import profile_to_rag_dictconfig

    profile = _make_profile(tmp_path, documents_dir="docs")
    config = profile_to_rag_dictconfig(profile)

    assert "documents_dir" in config
    assert Path(config.documents_dir) == (tmp_path / "docs").resolve()


def test_bridge_explicit_paths_override_profile(tmp_path: Path):
    from DefenseAgent.rag._bridge import profile_to_rag_dictconfig

    profile = _make_profile(tmp_path, documents_dir="docs", storage_dir="rag-x")
    custom_storage = tmp_path / "elsewhere-storage"
    custom_docs = tmp_path / "elsewhere-docs"
    config = profile_to_rag_dictconfig(
        profile, storage_path=custom_storage, documents_path=custom_docs,
    )

    assert Path(config.rag.storage_dir) == custom_storage.resolve()
    assert Path(config.documents_dir) == custom_docs.resolve()


def test_bridge_raises_without_source_dir():
    """In-memory profile with no source_dir + no explicit storage_path → RAGConfigError."""
    from DefenseAgent.rag._bridge import profile_to_rag_dictconfig
    from DefenseAgent.rag.base import RAGConfigError

    profile = AgentProfile(
        id="a", name="A", age=1, traits="t", backstory="b", initial_plan="p",
        rag={"enabled": True},
    )
    with pytest.raises(RAGConfigError, match="source_dir"):
        profile_to_rag_dictconfig(profile)


# ---------- construction ----------


def test_construction_creates_storage_dir(tmp_path: Path):
    profile = _make_profile(tmp_path)
    rag = _build_rag(profile)

    expected = (tmp_path / "rag").resolve()
    assert Path(rag.storage_dir) == expected
    assert expected.exists()


def test_construction_resolves_documents_dir(tmp_path: Path):
    profile = _make_profile(tmp_path, documents_dir="docs")
    rag = _build_rag(profile)

    assert rag._documents_dir == (tmp_path / "docs").resolve()


def test_construction_without_documents_dir(tmp_path: Path):
    profile = _make_profile(tmp_path)
    rag = _build_rag(profile)

    assert rag._documents_dir is None


# ---------- from_profile + auto_load ----------


async def test_auto_load_returns_when_load_index_succeeds(tmp_path: Path):
    profile = _make_profile(tmp_path, documents_dir="docs")
    rag = _build_rag(profile)

    rag.load_index = AsyncMock(return_value=None)
    rag.add_documents_from_files = AsyncMock()
    rag.save_index = AsyncMock()

    await rag._auto_load()

    rag.load_index.assert_awaited_once()
    rag.add_documents_from_files.assert_not_awaited()
    rag.save_index.assert_not_awaited()


async def test_auto_load_falls_back_to_ingest_when_no_index(tmp_path: Path):
    profile = _make_profile(tmp_path, documents_dir="docs")
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "a.md").write_text("hello", encoding="utf-8")
    (docs_dir / "b.txt").write_text("world", encoding="utf-8")
    (docs_dir / "ignore.png").write_bytes(b"\x89PNG")  # not in default globs

    rag = _build_rag(profile)
    rag.load_index = AsyncMock(side_effect=FileNotFoundError)
    rag.add_documents_from_files = AsyncMock()
    rag.save_index = AsyncMock()

    await rag._auto_load()

    rag.add_documents_from_files.assert_awaited_once()
    files = rag.add_documents_from_files.await_args.args[0]
    assert sorted(Path(f).name for f in files) == ["a.md", "b.txt"]
    rag.save_index.assert_awaited_once()


async def test_auto_load_no_docs_dir_does_nothing(tmp_path: Path):
    profile = _make_profile(tmp_path)  # no documents_dir
    rag = _build_rag(profile)

    rag.load_index = AsyncMock(side_effect=FileNotFoundError)
    rag.add_documents_from_files = AsyncMock()
    rag.save_index = AsyncMock()

    await rag._auto_load()

    rag.add_documents_from_files.assert_not_awaited()
    rag.save_index.assert_not_awaited()


async def test_auto_load_empty_docs_dir_does_nothing(tmp_path: Path):
    profile = _make_profile(tmp_path, documents_dir="docs")
    (tmp_path / "docs").mkdir()  # exists but empty

    rag = _build_rag(profile)
    rag.load_index = AsyncMock(side_effect=FileNotFoundError)
    rag.add_documents_from_files = AsyncMock()
    rag.save_index = AsyncMock()

    await rag._auto_load()

    rag.add_documents_from_files.assert_not_awaited()
    rag.save_index.assert_not_awaited()


async def test_from_profile_passes_auto_load_flag(tmp_path: Path):
    """from_profile(auto_load=False) skips _auto_load entirely."""
    from ms_agent.rag.llama_index_rag import LlamaIndexRAG as MsLlamaIndexRAG
    from DefenseAgent.rag.llama_index_rag import LlamaIndexRAG

    profile = _make_profile(tmp_path, documents_dir="docs")
    with patch.object(MsLlamaIndexRAG, "__init__", _stub_parent_init):
        with patch.object(LlamaIndexRAG, "_auto_load", AsyncMock()) as mock_auto:
            await LlamaIndexRAG.from_profile(profile, load_env=False, auto_load=False)
            mock_auto.assert_not_awaited()


# ---------- embedding provider override (OpenAI-compat) ----------


def test_bridge_carries_embedding_provider(tmp_path: Path):
    from DefenseAgent.rag._bridge import profile_to_rag_dictconfig

    profile = _make_profile(tmp_path, embedding_provider="openai")
    config = profile_to_rag_dictconfig(profile)
    assert config.rag.embedding_provider == "openai"

    profile_hf = _make_profile(tmp_path, embedding_provider="huggingface")
    config_hf = profile_to_rag_dictconfig(profile_hf)
    assert config_hf.rag.embedding_provider == "huggingface"


def test_profile_or_env_str_prefers_profile(monkeypatch: pytest.MonkeyPatch):
    """`_profile_or_env_str` is the per-field resolver: profile-when-set wins; whitespace-only is treated as unset."""
    from DefenseAgent.rag._bridge import _profile_or_env_str

    monkeypatch.setenv("FOO", "from-env")
    assert _profile_or_env_str("from-profile", "FOO") == "from-profile"
    assert _profile_or_env_str("  from-profile  ", "FOO") == "from-profile"
    assert _profile_or_env_str(None, "FOO") == "from-env"
    assert _profile_or_env_str("", "FOO") == "from-env"
    assert _profile_or_env_str("   ", "FOO") == "from-env"


def test_profile_or_env_str_returns_none_when_neither_set(
    monkeypatch: pytest.MonkeyPatch,
):
    from DefenseAgent.rag._bridge import _profile_or_env_str

    monkeypatch.delenv("MISSING", raising=False)
    assert _profile_or_env_str(None, "MISSING") is None


def test_profile_or_env_int_handles_strings_and_invalid(
    monkeypatch: pytest.MonkeyPatch,
):
    from DefenseAgent.rag._bridge import _profile_or_env_int

    monkeypatch.setenv("DIMS", "1536")
    assert _profile_or_env_int(None, "DIMS") == 1536
    assert _profile_or_env_int(768, "DIMS") == 768  # profile wins
    monkeypatch.setenv("DIMS", "not-a-number")
    assert _profile_or_env_int(None, "DIMS") is None
    monkeypatch.delenv("DIMS", raising=False)
    assert _profile_or_env_int(None, "DIMS") is None


def test_install_openai_compat_embedding_raises_when_neither_profile_nor_env_supply_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """The runtime installer raises when both the profile and .env are silent — message names both candidate sources."""
    from DefenseAgent.rag.base import RAGConfigError

    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    profile = _make_profile(tmp_path, embedding_provider="openai")
    rag = _build_rag(profile)
    with pytest.raises(RAGConfigError, match="EMBEDDING_MODEL"):
        rag._install_openai_compat_embedding()


def test_install_openai_compat_embedding_wires_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """The OpenAI-compat installer reads env vars and assigns Settings.embed_model. We mock both llama-index modules in sys.modules so the test passes without llama-index installed."""
    _set_embedding_env(monkeypatch)
    profile = _make_profile(tmp_path, embedding_provider="openai")
    rag = _build_rag(profile)

    fake_settings = MagicMock(name="llama_index.core.Settings")
    fake_embedding_class = MagicMock(name="OpenAILikeEmbedding")
    fake_embedding_class.return_value = MagicMock(name="embedding_instance")

    fake_core = types.ModuleType("llama_index.core")
    fake_core.Settings = fake_settings
    fake_openai_like = types.ModuleType("llama_index.embeddings.openai_like")
    fake_openai_like.OpenAILikeEmbedding = fake_embedding_class

    with patch.dict(
        sys.modules,
        {
            "llama_index": types.ModuleType("llama_index"),
            "llama_index.core": fake_core,
            "llama_index.embeddings": types.ModuleType("llama_index.embeddings"),
            "llama_index.embeddings.openai_like": fake_openai_like,
        },
    ):
        rag._install_openai_compat_embedding()

    fake_embedding_class.assert_called_once()
    kwargs = fake_embedding_class.call_args.kwargs
    assert kwargs["model_name"] == "text-embedding-3-small"
    assert kwargs["api_key"] == "sk-emb"
    assert kwargs["api_base"] == "https://api.example.com"
    assert kwargs["embed_dim"] == 1536
    assert fake_settings.embed_model is fake_embedding_class.return_value
    assert rag.embedding_model == "text-embedding-3-small"


def test_install_openai_compat_embedding_raises_when_llama_index_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """No llama-index installed → wrap ImportError as RAGConfigError with an install hint."""
    _set_embedding_env(monkeypatch)
    profile = _make_profile(tmp_path, embedding_provider="openai")
    rag = _build_rag(profile)

    from DefenseAgent.rag.base import RAGConfigError

    # Force the import to fail by clearing any partial llama_index modules and
    # blocking the openai_like submodule.
    blockers = {
        "llama_index.embeddings.openai_like": None,
        "llama_index.core": None,
    }
    with patch.dict(sys.modules, blockers):
        with pytest.raises(RAGConfigError, match="llama-index-embeddings-openai-like"):
            rag._install_openai_compat_embedding()


def test_setup_embedding_model_dispatches_by_provider(tmp_path: Path):
    """provider='openai' → calls our installer; provider='huggingface' → defers to ms-agent's super()."""
    profile = _make_profile(tmp_path, embedding_provider="openai")
    rag = _build_rag(profile)

    with patch.object(
        type(rag), "_install_openai_compat_embedding", autospec=True,
    ) as mock_install:
        rag._setup_embedding_model(rag.config)
        mock_install.assert_called_once_with(rag)

    # Now switch to huggingface and verify super() is invoked instead.
    rag.config.rag.embedding_provider = "huggingface"
    from ms_agent.rag.llama_index_rag import LlamaIndexRAG as MsLlamaIndexRAG

    with patch.object(
        MsLlamaIndexRAG, "_setup_embedding_model", autospec=True,
    ) as mock_super:
        rag._setup_embedding_model(rag.config)
        mock_super.assert_called_once()


# ---------- hybrid_retrieve ----------


async def test_hybrid_retrieve_returns_empty_when_no_index(tmp_path: Path):
    """No index loaded → return [] without calling super()."""
    profile = _make_profile(tmp_path)
    rag = _build_rag(profile)
    rag.index = None

    result = await rag.hybrid_retrieve("anything")
    assert result == []


async def test_hybrid_retrieve_defaults_to_profile_top_k(tmp_path: Path):
    """When `limit` is None, top_k comes from profile.rag.top_k."""
    profile = _make_profile(tmp_path)
    profile.rag.top_k = 7
    rag = _build_rag(profile)
    rag.index = MagicMock()  # truthy → bypass empty-index guard

    fake_super = AsyncMock(return_value=[])
    with patch(
        "ms_agent.rag.llama_index_rag.LlamaIndexRAG.hybrid_search",
        fake_super,
    ):
        await rag.hybrid_retrieve("q")

    fake_super.assert_awaited_once_with("q", top_k=7)


async def test_hybrid_retrieve_explicit_limit_overrides_profile(tmp_path: Path):
    """An explicit `limit` arg overrides profile.rag.top_k."""
    profile = _make_profile(tmp_path)
    profile.rag.top_k = 7
    rag = _build_rag(profile)
    rag.index = MagicMock()

    fake_super = AsyncMock(return_value=[])
    with patch(
        "ms_agent.rag.llama_index_rag.LlamaIndexRAG.hybrid_search",
        fake_super,
    ):
        await rag.hybrid_retrieve("q", limit=2)

    fake_super.assert_awaited_once_with("q", top_k=2)


async def test_hybrid_retrieve_filters_below_score_threshold(tmp_path: Path):
    """Results with score < threshold are dropped."""
    profile = _make_profile(tmp_path)
    rag = _build_rag(profile)
    rag.index = MagicMock()

    payload = [
        {"text": "high", "score": 0.9, "metadata": {}, "node_id": "n1"},
        {"text": "mid", "score": 0.5, "metadata": {}, "node_id": "n2"},
        {"text": "low", "score": 0.1, "metadata": {}, "node_id": "n3"},
    ]
    fake_super = AsyncMock(return_value=payload)
    with patch(
        "ms_agent.rag.llama_index_rag.LlamaIndexRAG.hybrid_search",
        fake_super,
    ):
        result = await rag.hybrid_retrieve("q", score_threshold=0.4)

    assert [r["text"] for r in result] == ["high", "mid"]


async def test_hybrid_retrieve_defaults_to_profile_score_threshold(tmp_path: Path):
    """When `score_threshold` is None, profile.rag.score_threshold is used."""
    profile = _make_profile(tmp_path)
    profile.rag.score_threshold = 0.6
    rag = _build_rag(profile)
    rag.index = MagicMock()

    payload = [
        {"text": "keep", "score": 0.7, "metadata": {}, "node_id": "n1"},
        {"text": "drop", "score": 0.3, "metadata": {}, "node_id": "n2"},
    ]
    fake_super = AsyncMock(return_value=payload)
    with patch(
        "ms_agent.rag.llama_index_rag.LlamaIndexRAG.hybrid_search",
        fake_super,
    ):
        result = await rag.hybrid_retrieve("q")

    assert [r["text"] for r in result] == ["keep"]


# ---------- add_structured_chunks + get_resource_path ----------


async def test_add_structured_chunks_skips_when_empty(tmp_path: Path):
    """An empty list is a no-op; index stays None and no llama_index import is forced."""
    profile = _make_profile(tmp_path)
    rag = _build_rag(profile)
    rag.index = None

    await rag.add_structured_chunks([])

    assert rag.index is None


def _patch_llama_index_core(fake_doc_cls, fake_index_cls):
    """Build a context manager that drops fake llama_index.core into sys.modules."""
    fake_core = types.ModuleType("llama_index.core")
    fake_core.Document = fake_doc_cls
    fake_core.VectorStoreIndex = fake_index_cls
    return patch.dict(
        sys.modules,
        {
            "llama_index": types.ModuleType("llama_index"),
            "llama_index.core": fake_core,
        },
    )


async def test_add_structured_chunks_persists_paths_relative_to_storage(tmp_path: Path):
    """Resources living under storage_dir are serialized as POSIX-relative paths
    so the index stays portable across machines / OSes."""
    from DefenseAgent.rag.extraction import StructuredChunk, StructuredResource

    profile = _make_profile(tmp_path)
    rag = _build_rag(profile)
    rag.retrieve_only = True

    storage_root = Path(rag.storage_dir).resolve()
    img_path = storage_root / "resources" / "doc-hash" / "img0.png"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    img_path.write_bytes(b"PNG")

    chunks = [
        StructuredChunk(
            text="hello <resource_info>r1</resource_info>",
            resources=[
                StructuredResource(id="r1", kind="image", path=img_path, mime_type="image/png"),
            ],
            metadata={"source": "x.pdf", "page": 1},
        )
    ]

    fake_doc_cls = MagicMock(name="Document")
    fake_doc_cls.return_value = MagicMock(name="document_instance")
    fake_index_cls = MagicMock(name="VectorStoreIndex")
    fake_index_cls.from_documents.return_value = MagicMock(name="index_instance")

    with _patch_llama_index_core(fake_doc_cls, fake_index_cls):
        await rag.add_structured_chunks(chunks)

    fake_doc_cls.assert_called_once()
    kwargs = fake_doc_cls.call_args.kwargs
    assert kwargs["text"] == chunks[0].text
    assert kwargs["metadata"]["resource_ids"] == ["r1"]
    # Relative + POSIX (forward slashes) so a Windows-built index loads on Linux too.
    assert kwargs["metadata"]["resource_paths"] == ["resources/doc-hash/img0.png"]
    assert kwargs["metadata"]["resource_kinds"] == ["image"]
    assert kwargs["metadata"]["page"] == 1
    fake_index_cls.from_documents.assert_called_once()
    assert rag.index is fake_index_cls.from_documents.return_value


async def test_add_structured_chunks_keeps_absolute_for_out_of_tree_resource(tmp_path: Path):
    """Resources that live outside storage_dir fall back to absolute POSIX paths
    (we'd lose them on a move, but at least we don't lie about where they are)."""
    from DefenseAgent.rag.extraction import StructuredChunk, StructuredResource

    profile = _make_profile(tmp_path)
    rag = _build_rag(profile)
    rag.retrieve_only = True

    # Place the image OUTSIDE storage_dir (storage_dir is tmp_path / "rag").
    out_of_tree = tmp_path / "external" / "img.png"
    out_of_tree.parent.mkdir(parents=True, exist_ok=True)
    out_of_tree.write_bytes(b"PNG")

    chunks = [
        StructuredChunk(
            text="t",
            resources=[StructuredResource(id="r", kind="image", path=out_of_tree)],
        )
    ]

    fake_doc_cls = MagicMock(name="Document")
    fake_doc_cls.return_value = MagicMock(name="document_instance")
    fake_index_cls = MagicMock(name="VectorStoreIndex")
    fake_index_cls.from_documents.return_value = MagicMock(name="index_instance")

    with _patch_llama_index_core(fake_doc_cls, fake_index_cls):
        await rag.add_structured_chunks(chunks)

    paths = fake_doc_cls.call_args.kwargs["metadata"]["resource_paths"]
    assert paths == [out_of_tree.resolve().as_posix()]


def test_get_resource_path_returns_none_when_no_index(tmp_path: Path):
    profile = _make_profile(tmp_path)
    rag = _build_rag(profile)
    rag.index = None

    assert rag.get_resource_path("r1") is None


def test_get_resource_path_resolves_relative_against_storage(tmp_path: Path):
    """Stored relative paths are resolved against the **current** storage_dir,
    not whatever path was used at ingest time. This is the portability win."""
    profile = _make_profile(tmp_path)
    rag = _build_rag(profile)

    doc = MagicMock()
    doc.metadata = {
        "resource_ids": ["r0"],
        "resource_paths": ["resources/abc/img0.png"],
    }
    fake_index = MagicMock()
    fake_index.docstore.docs = {"d": doc}
    rag.index = fake_index

    expected = (Path(rag.storage_dir) / "resources" / "abc" / "img0.png").resolve()
    assert rag.get_resource_path("r0") == expected


def test_get_resource_path_simulates_cross_machine_move(tmp_path: Path):
    """Concrete portability check: ingest with storage_dir A, change storage_dir
    to B, lookup must still resolve (the relative path is stable, the anchor isn't)."""
    profile = _make_profile(tmp_path)
    rag = _build_rag(profile)

    doc = MagicMock()
    doc.metadata = {
        "resource_ids": ["r0"],
        "resource_paths": ["resources/abc/img0.png"],
    }
    fake_index = MagicMock()
    fake_index.docstore.docs = {"d": doc}
    rag.index = fake_index

    moved_storage = tmp_path / "moved-storage"
    moved_storage.mkdir()
    rag.storage_dir = str(moved_storage)

    expected = (moved_storage / "resources" / "abc" / "img0.png").resolve()
    assert rag.get_resource_path("r0") == expected


def test_get_resource_path_preserves_legacy_absolute_entries(tmp_path: Path):
    """Old indexes built before the relative-path change keep working: absolute
    entries are returned as-is."""
    profile = _make_profile(tmp_path)
    rag = _build_rag(profile)

    # Use a real OS-absolute path so is_absolute() works on both Windows and Linux.
    legacy = (tmp_path / "anywhere" / "img.png").resolve()

    doc = MagicMock()
    doc.metadata = {
        "resource_ids": ["legacy"],
        "resource_paths": [str(legacy)],
    }
    fake_index = MagicMock()
    fake_index.docstore.docs = {"d": doc}
    rag.index = fake_index

    assert rag.get_resource_path("legacy") == legacy
    assert rag.get_resource_path("missing") is None


# ---------- absolute-path documents_dir on in-memory profile ----------


def test_documents_dir_absolute_path_works_without_source_dir(tmp_path: Path):
    """Regression: an in-memory profile with an absolute `rag.documents_dir`
    must resolve cleanly without source_dir; only relative paths require an anchor."""
    from DefenseAgent.rag._bridge import _resolve_documents_path

    docs = tmp_path / "my_docs"
    docs.mkdir()

    # In-memory profile (no source_dir) + absolute documents_dir → should work
    profile = AgentProfile(
        id="t", name="t", age=1, traits="t",
        backstory="b", initial_plan="p",
        rag={"enabled": True, "documents_dir": str(docs.resolve())},
    )
    assert profile.source_dir is None
    resolved = _resolve_documents_path(profile, None)
    assert resolved == docs.resolve()


def test_documents_dir_relative_still_requires_source_dir():
    """Symmetry check: relative documents_dir on in-memory profile still raises."""
    from DefenseAgent.rag._bridge import _resolve_documents_path
    from DefenseAgent.rag.base import RAGConfigError

    profile = AgentProfile(
        id="t", name="t", age=1, traits="t",
        backstory="b", initial_plan="p",
        rag={"enabled": True, "documents_dir": "relative/docs"},
    )
    with pytest.raises(RAGConfigError) as ei:
        _resolve_documents_path(profile, None)
    assert "absolute path" in str(ei.value)
