import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv

from ms_agent.rag.llama_index_rag import LlamaIndexRAG as MsLlamaIndexRAG

from DefenseAgent.config.profile import AgentProfile
from DefenseAgent.rag._bridge import profile_to_rag_dictconfig
from DefenseAgent.rag.base import RAGConfigError

if TYPE_CHECKING:
    from DefenseAgent.rag.extraction import StructuredChunk


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

    async def add_structured_chunks(self, chunks: "list[StructuredChunk]") -> None:
        """Ingest pre-extracted structured chunks into the vector index.

        Each chunk's resource ids and on-disk paths are stored in node metadata
        so that callers can map a retrieved chunk back to its associated images
        / tables via `get_resource_path()`. Paths that fall under
        `self.storage_dir` are persisted as POSIX-style **relative** paths so the
        index remains portable across machines / OSes; paths outside that root
        are persisted as absolute strings as a safety fallback.
        """
        if not chunks:
            return
        from llama_index.core import Document, VectorStoreIndex
        documents = [
            Document(
                text=c.text,
                metadata={
                    **c.metadata,
                    "resource_ids": [r.id for r in c.resources],
                    "resource_paths": [self._serialize_resource_path(r.path) for r in c.resources],
                    "resource_kinds": [r.kind for r in c.resources],
                },
            )
            for c in chunks
        ]
        self.index = VectorStoreIndex.from_documents(documents)
        if not self.retrieve_only:
            await self._setup_query_engine()

    def get_resource_path(self, resource_id: str) -> Path | None:
        """Look up a persisted resource path by id across every indexed document.

        Stored values can be either POSIX-relative-to-storage (the new default,
        portable) or absolute (legacy indexes / out-of-tree resources): both are
        resolved back to an absolute `Path` here so callers always get a
        usable filesystem path. Returns None when the index hasn't been built
        or the id wasn't found.
        """
        if self.index is None:
            return None
        for doc in self.index.docstore.docs.values():
            metadata = getattr(doc, "metadata", None) or {}
            ids = metadata.get("resource_ids", [])
            paths = metadata.get("resource_paths", [])
            if resource_id in ids:
                return self._deserialize_resource_path(paths[ids.index(resource_id)])
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
