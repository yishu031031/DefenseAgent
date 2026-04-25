from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from DefenseAgent.llm.types import (
    LLMResponse,
    Message,
    StreamChunk,
    StreamEnd,
    TextDelta,
)


def to_dict_safe(obj: Any) -> dict:
    """Best-effort dict conversion for the opaque SDK response object stored on .raw."""
    for attr in ("model_dump", "to_dict"):
        if hasattr(obj, attr):
            try:
                return getattr(obj, attr)()
            except Exception:
                pass
    return {"repr": repr(obj)}


class LLMAdapter(ABC):
    """Abstract base for every concrete LLM adapter; defines chat() and a default chat_stream()."""

    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        *,
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        system: str | None = None,
    ) -> LLMResponse:
        """Send `messages` to the provider and return a canonical LLMResponse."""

    async def chat_stream(
        self,
        messages: list[Message],
        *,
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        system: str | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Default streaming: await chat() and replay the whole response as one TextDelta + one StreamEnd."""
        response = await self.chat(
            messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            system=system,
        )
        if response.content:
            yield TextDelta(text=response.content)
        stop_reason = response.stop_reason
        if stop_reason is None:
            stop_reason = "end_turn"
        yield StreamEnd(
            stop_reason=stop_reason,
            usage=response.usage,
            raw=response.raw,
        )
