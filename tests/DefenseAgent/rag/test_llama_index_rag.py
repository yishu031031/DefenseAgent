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
    assert RAG.__abstractmethods__ >= {"add_documents", "retrieve", "query"}


# ---------- bridge: profile_to_rag_dictconfig ----------


def test_bridge_translates_rag_knobs(tmp_path: Path):
    from DefenseAgent.rag._bridge import profile_to_rag_dictconfig

    profile = _make_profile(tmp_path, retrieve_only=True)
    config = profile_to_rag_dictconfig(profile)

    assert config.rag.embedding == "Qwen/Qwen3-Embedding-0.6B"
    assert config.rag.chunk_size == 512
    assert config.rag.chunk_overlap == 50
    assert config.rag.retrieve_only is True
    assert Path(config.rag.storage_dir) == (tmp_path / "rag").resolve()
    assert config.use_huggingface is False
    assert "llm" not in config  # retrieve_only=True skips LLM block


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


def test_read_embedding_env_happy_path(monkeypatch: pytest.MonkeyPatch):
    from DefenseAgent.rag.llama_index_rag import _read_embedding_env

    _set_embedding_env(monkeypatch)
    api_key, base_url, model, dims = _read_embedding_env()
    assert api_key == "sk-emb"
    assert base_url == "https://api.example.com"
    assert model == "text-embedding-3-small"
    assert dims == 1536


def test_read_embedding_env_missing_required_keys(monkeypatch: pytest.MonkeyPatch):
    from DefenseAgent.rag.base import RAGConfigError
    from DefenseAgent.rag.llama_index_rag import _read_embedding_env

    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    with pytest.raises(RAGConfigError, match="EMBEDDING_"):
        _read_embedding_env()


def test_read_embedding_env_dims_optional_and_invalid(
    monkeypatch: pytest.MonkeyPatch,
):
    from DefenseAgent.rag.llama_index_rag import _read_embedding_env

    monkeypatch.setenv("EMBEDDING_API_KEY", "k")
    monkeypatch.setenv("EMBEDDING_MODEL", "m")
    monkeypatch.delenv("EMBEDDING_BASE_URL", raising=False)

    monkeypatch.delenv("EMBEDDING_DIMS", raising=False)
    *_, dims = _read_embedding_env()
    assert dims is None

    monkeypatch.setenv("EMBEDDING_DIMS", "not-a-number")
    *_, dims = _read_embedding_env()
    assert dims is None


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
