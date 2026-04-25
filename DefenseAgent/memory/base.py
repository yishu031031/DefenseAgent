from abc import ABC, abstractmethod

from DefenseAgent.config.profile import AgentProfile
from DefenseAgent.llm.types import Message


class Memory(ABC):
    """Memory ABC matching ms-agent's scheme: a `run(messages) -> messages` rewriter that ingests, retrieves, and injects in one call."""

    def __init__(self, profile: AgentProfile) -> None:
        """Bind the agent profile so subclasses can read storage_path, search_limit, etc. from it."""
        self.profile = profile

    @abstractmethod
    async def run(self, messages: list[Message]) -> list[Message]:
        """Refine `messages` and return the new list. Concrete subclasses ingest, search, compact, or rewrite in any combination."""


class MemoryError(Exception):
    """Base class for every error raised from the memory module."""


class MemoryConfigError(MemoryError):
    """Raised when memory backend configuration (path, providers, env vars) is missing or invalid."""


class MemoryProviderError(MemoryError):
    """Raised when the underlying mem0 / vector store / LLM call fails (original chained via __cause__)."""
