from abc import ABC, abstractmethod
from typing import Any

from openai import AsyncOpenAI


class MemoryError(Exception):
    """Base class for every error raised from the memory module."""


class MemoryNotFoundError(MemoryError):
    """Raised when a lookup by record id finds nothing."""


class EmbeddingConfigError(MemoryError):
    """Raised when the EMBEDDING_* env block is missing or invalid."""


class EmbeddingProviderError(MemoryError):
    """Raised when the embedding provider API returned an error (original chained via __cause__)."""


class EmbeddingAdapter(ABC):
    """Abstract base for every embedding provider; produces vectors from text."""

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Return the embedding vector for a single `text`."""

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return embedding vectors for a list of `texts` in input order."""


class OpenAICompatibleEmbeddingAdapter(EmbeddingAdapter):
    """EmbeddingAdapter for providers speaking OpenAI's /embeddings protocol (OpenAI, Qwen, vLLM)."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        client: Any | None = None,
    ) -> None:
        """Store the model name and construct (or accept) an AsyncOpenAI client pointed at `base_url`."""
        self.model = model
        if client is None:
            self._client = AsyncOpenAI(
                api_key=api_key or None, base_url=base_url or None,
            )
        else:
            self._client = client

    @property
    def _model(self) -> str:
        """Alias matching the LLM adapters' convention so demos/tests can read `.model` uniformly across modules."""
        return self.model

    async def embed(self, text: str) -> list[float]:
        """Return one embedding vector for `text`, wrapping provider errors in EmbeddingProviderError."""
        try:
            response = await self._client.embeddings.create(
                input=text, model=self.model,
            )
        except Exception as e:
            raise EmbeddingProviderError(str(e)) from e
        return list(response.data[0].embedding)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return embeddings for `texts` in input order; re-sorts provider output by the `index` field."""
        try:
            response = await self._client.embeddings.create(
                input=texts, model=self.model,
            )
        except Exception as e:
            raise EmbeddingProviderError(str(e)) from e
        items = sorted(response.data, key=lambda d: d.index)
        return [list(d.embedding) for d in items]
