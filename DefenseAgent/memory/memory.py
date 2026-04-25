import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

from DefenseAgent.config.profile import AgentProfile
from DefenseAgent.memory.embedding import EmbeddingAdapter, EmbeddingConfigError
from DefenseAgent.memory.retriever import MemoryRetriever, ScoredMemory
from DefenseAgent.memory.stream import MemoryKind, MemoryRecord, MemoryStream


_EMBED_SUPPORTED_PROVIDERS = ("openai", "qwen", "vllm")
_EMBED_BASE_URL_REQUIRED = {"qwen", "vllm"}
_EMBED_API_KEY_OPTIONAL = {"vllm"}
_EMBED_VLLM_DEFAULT_KEY = "token-not-needed"


class Memory:
    """Module 4's unified facade; owns the embedding adapter, stream (RAM or SQLite-backed), and retriever."""

    def __init__(
        self,
        profile: AgentProfile,
        embedding_adapter: EmbeddingAdapter,
        *,
        stream: MemoryStream | None = None,
        retriever: MemoryRetriever | None = None,
        clock: Callable[[], datetime] | None = None,
        dedup_threshold: float | None = 0.95,
        recency_half_life_hours: float = 24.0,
        db_path: str | Path | None = None,
    ) -> None:
        """Build (or accept injected) MemoryStream + MemoryRetriever; if `db_path` is set, the stream persists to SQLite."""
        self.profile = profile
        self.embedding_adapter = embedding_adapter
        if stream is None:
            self.stream = MemoryStream(
                embedding_adapter,
                clock=clock,
                dedup_threshold=dedup_threshold,
                db_path=db_path,
            )
        else:
            self.stream = stream
        if retriever is None:
            self.retriever = MemoryRetriever(
                self.stream,
                embedding_adapter,
                profile.memory,
                clock=clock,
                recency_half_life_hours=recency_half_life_hours,
            )
        else:
            self.retriever = retriever

    @classmethod
    def from_env(
        cls,
        profile: AgentProfile,
        *,
        dotenv_path: str | None = None,
        load_env: bool = True,
        clock: Callable[[], datetime] | None = None,
        dedup_threshold: float | None = 0.95,
        recency_half_life_hours: float = 24.0,
        db_path: str | Path | None = None,
    ) -> "Memory":
        """Construct a Memory by reading the EMBEDDING_* env block; no persistence unless `db_path` is supplied."""
        adapter = _build_embedding_adapter_from_env(
            dotenv_path=dotenv_path, load_env=load_env,
        )
        return cls(
            profile=profile,
            embedding_adapter=adapter,
            clock=clock,
            dedup_threshold=dedup_threshold,
            recency_half_life_hours=recency_half_life_hours,
            db_path=db_path,
        )

    @classmethod
    def from_profile(
        cls,
        profile: AgentProfile,
        *,
        memory_dir: str | Path | None = None,
        persist: bool = True,
        dotenv_path: str | None = None,
        load_env: bool = True,
        clock: Callable[[], datetime] | None = None,
        dedup_threshold: float | None = 0.95,
        recency_half_life_hours: float = 24.0,
    ) -> "Memory":
        """Build a per-agent Memory; persist=True (default) backs it with SQLite at <memory_dir>/stream.db, persist=False keeps state purely in-process."""
        if persist:
            resolved_dir = _resolve_memory_dir(profile, memory_dir)
            resolved_dir.mkdir(parents=True, exist_ok=True)
            db_path: Path | None = resolved_dir / "stream.db"
        else:
            if memory_dir is not None:
                raise ValueError(
                    "persist=False is incompatible with memory_dir; pass one or the other"
                )
            db_path = None
        return cls.from_env(
            profile=profile,
            dotenv_path=dotenv_path,
            load_env=load_env,
            clock=clock,
            dedup_threshold=dedup_threshold,
            recency_half_life_hours=recency_half_life_hours,
            db_path=db_path,
        )

    async def remember(
        self,
        content: str,
        *,
        kind: MemoryKind = "observation",
        importance: float = 5.0,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        """Append a new record (embed + dedup + persist if configured); returns the stored or pre-existing record."""
        return await self.stream.add(
            content,
            kind=kind,
            importance=importance,
            metadata=metadata,
        )

    async def recall(
        self,
        query: str,
        *,
        top_k: int | None = None,
        kinds: tuple[MemoryKind, ...] | None = None,
    ) -> list[ScoredMemory]:
        """Run the hybrid retrieval pipeline for `query`, returning top-k ScoredMemory records."""
        return await self.retriever.retrieve(query, top_k=top_k, kinds=kinds)

    def close(self) -> None:
        """Close the underlying SQLite connection (no-op when the stream is RAM-only)."""
        self.stream.close()

    def __enter__(self) -> "Memory":
        """Support `with Memory.from_profile(...) as memory:` for deterministic close semantics."""
        return self

    def __exit__(self, *exc_info: Any) -> None:
        """Close the underlying SQLite connection on context exit."""
        self.close()

    def __len__(self) -> int:
        """Return the number of records in the underlying stream."""
        return len(self.stream)

    def __repr__(self) -> str:
        """Render a short diagnostic string including the agent name, record count, and db path."""
        db = self.stream.db_path
        suffix = f", db={db}" if db is not None else ""
        return f"Memory(profile={self.profile.name!r}, records={len(self)}{suffix})"


def _resolve_memory_dir(
    profile: AgentProfile,
    memory_dir: str | Path | None,
) -> Path:
    """Return an absolute path for the memory dir; defaults to `<profile.source_dir>/memory/`."""
    if memory_dir is not None:
        return Path(memory_dir).resolve()
    if profile.source_dir is None:
        raise EmbeddingConfigError(
            "profile has no source_dir; pass memory_dir explicitly when "
            "loading an in-memory profile"
        )
    return (profile.source_dir / "memory").resolve()


def _build_embedding_adapter_from_env(
    *,
    dotenv_path: str | None,
    load_env: bool,
) -> EmbeddingAdapter:
    """Read the EMBEDDING_* block, validate per-provider rules, and build the configured adapter."""
    if load_env:
        load_dotenv(dotenv_path, override=False)

    from DefenseAgent.memory.embedding import OpenAICompatibleEmbeddingAdapter

    provider = os.environ.get("EMBEDDING_PROVIDER", "").strip().lower()
    if not provider:
        raise EmbeddingConfigError(
            "EMBEDDING_PROVIDER is not set. "
            f"Supported values: {', '.join(_EMBED_SUPPORTED_PROVIDERS)}."
        )
    if provider not in _EMBED_SUPPORTED_PROVIDERS:
        raise EmbeddingConfigError(
            f"EMBEDDING_PROVIDER={provider!r} is not supported. "
            f"Supported values: {', '.join(_EMBED_SUPPORTED_PROVIDERS)}."
        )

    api_key = os.environ.get("EMBEDDING_API_KEY", "")
    base_url = os.environ.get("EMBEDDING_BASE_URL", "")
    model = os.environ.get("EMBEDDING_MODEL", "")

    if not model:
        raise EmbeddingConfigError(
            f"EMBEDDING_MODEL is not set for provider '{provider}'."
        )
    if provider in _EMBED_BASE_URL_REQUIRED and not base_url:
        raise EmbeddingConfigError(
            f"EMBEDDING_BASE_URL is required for provider '{provider}'."
        )
    if provider not in _EMBED_API_KEY_OPTIONAL and not api_key:
        raise EmbeddingConfigError(
            f"EMBEDDING_API_KEY is not set for provider '{provider}'."
        )

    if provider == "vllm" and not api_key:
        api_key = _EMBED_VLLM_DEFAULT_KEY

    return OpenAICompatibleEmbeddingAdapter(
        api_key=api_key, base_url=base_url, model=model,
    )
