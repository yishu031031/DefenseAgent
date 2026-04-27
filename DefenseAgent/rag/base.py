from abc import ABC, abstractmethod
from typing import Any

from DefenseAgent.config.profile import AgentProfile


class RAG(ABC):
    """RAG ABC matching ms-agent's scheme: ingest documents, retrieve passages, optionally synthesize an answer.

    DefenseAgent's RAG is the static-knowledge counterpart to mem0-based memory:
    pre-built reference corpus (textbooks, character lore, world docs) loaded once
    and queried during reasoning, distinct from per-turn experiential memory.
    """

    def __init__(self, profile: AgentProfile) -> None:
        """Bind the agent profile so subclasses can read storage_dir, embedding, etc. from it."""
        self.profile = profile

    @abstractmethod
    async def add_documents(self, documents: list[str]) -> None:
        """Ingest raw text documents into the index."""

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float = 0.0,
        **filters: Any,
    ) -> list[dict[str, Any]]:
        """Return top-`limit` passages matching `query` as dicts with at least `text` and `score`."""

    @abstractmethod
    async def answer(self, query: str) -> str:
        """Synthesize an answer from retrieved passages; only available when `retrieve_only=False`."""


class RAGError(Exception):
    """Base class for every error raised from the rag module."""


class RAGConfigError(RAGError):
    """Raised when RAG backend configuration (paths, embedding name, missing deps) is invalid."""


class RAGProviderError(RAGError):
    """Raised when the underlying llama-index / embedding / vector store call fails (original chained via __cause__)."""
