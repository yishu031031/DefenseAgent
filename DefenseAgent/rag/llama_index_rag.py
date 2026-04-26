import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from ms_agent.rag.llama_index_rag import LlamaIndexRAG as MsLlamaIndexRAG

from DefenseAgent.config.profile import AgentProfile
from DefenseAgent.rag._bridge import profile_to_rag_dictconfig
from DefenseAgent.rag.base import RAGConfigError


_DEFAULT_DOC_GLOBS: tuple[str, ...] = ("*.md", "*.txt", "*.rst", "*.pdf")


class LlamaIndexRAG(MsLlamaIndexRAG):
    """Inherits ms-agent's `LlamaIndexRAG`; takes our AgentProfile and converts to DictConfig at construction.

    Adds `from_profile()` which auto-loads the persisted index from `profile.rag.storage_dir`
    if present, else builds a fresh index from `profile.rag.documents_dir` and persists it.
    """

    def __init__(
        self,
        profile: AgentProfile,
        *,
        storage_path: str | Path | None = None,
        documents_path: str | Path | None = None,
        load_env: bool = True,
        dotenv_path: str | None = None,
    ) -> None:
        """Build the ms-agent DictConfig from `profile` + .env, ensure the storage dir exists, then defer to ms-agent's `__init__` (which loads the embedding model)."""
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
    ) -> "LlamaIndexRAG":
        """Convenience constructor that mirrors the rest of DefenseAgent. When `auto_load=True` (default), tries `load_index()` first and falls back to ingesting every file under the configured documents directory, then persists the index."""
        instance = cls(
            profile,
            storage_path=storage_path,
            documents_path=documents_path,
            load_env=load_env,
            dotenv_path=dotenv_path,
        )
        if auto_load:
            await instance._auto_load()
        return instance

    async def _auto_load(self) -> None:
        """Try `load_index()`; on miss, ingest every document under `_documents_dir` and persist."""
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
        await self.add_documents_from_files([str(f) for f in files])
        await self.save_index()

    def _setup_embedding_model(self, config) -> None:
        """Dispatch on `config.rag.embedding_provider`: route 'openai' to our OpenAI-compatible installer (reuses mem0's EMBEDDING_* env, no torch/sentence-transformers needed); fall back to ms-agent's HuggingFace path otherwise."""
        provider = getattr(config.rag, "embedding_provider", "openai")
        if provider == "openai":
            self._install_openai_compat_embedding()
            return
        super()._setup_embedding_model(config)

    def _install_openai_compat_embedding(self) -> None:
        """Wire `Settings.embed_model` to llama-index's `OpenAILikeEmbedding`, reading the same EMBEDDING_API_KEY/BASE_URL/MODEL/DIMS env vars mem0 already uses; raises RAGConfigError when those are missing or `llama-index-embeddings-openai-like` is not installed."""
        api_key, base_url, model, dims = _read_embedding_env()
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


def _read_embedding_env() -> tuple[str, str, str, int | None]:
    """Pull EMBEDDING_API_KEY/BASE_URL/MODEL/DIMS from the process env (same shape mem0's bridge consumes); raises RAGConfigError when required fields are missing."""
    api_key = os.environ.get("EMBEDDING_API_KEY", "")
    base_url = os.environ.get("EMBEDDING_BASE_URL", "")
    model = os.environ.get("EMBEDDING_MODEL", "")
    if not api_key or not model:
        raise RAGConfigError(
            "EMBEDDING_API_KEY and EMBEDDING_MODEL must be set in .env "
            "for OpenAI-compatible RAG embedding"
        )
    raw_dims = os.environ.get("EMBEDDING_DIMS", "").strip()
    dims: int | None = None
    if raw_dims:
        try:
            dims = int(raw_dims)
        except ValueError:
            dims = None
    return api_key, base_url, model, dims


def _collect_document_files(directory: Path) -> list[Path]:
    """Walk `directory` and return every file matching the default doc globs."""
    out: list[Path] = []
    for pattern in _DEFAULT_DOC_GLOBS:
        out.extend(p for p in directory.rglob(pattern) if p.is_file())
    return sorted(set(out))
