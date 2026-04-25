from dataclasses import dataclass, field
from typing import Any, Literal


Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ToolCall:
    """A tool invocation requested by the LLM, with parsed-dict arguments."""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Message:
    """One canonical conversation message in the harness's provider-agnostic format."""
    role: Role
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None


@dataclass
class TokenUsage:
    """Token accounting attached to every completed LLM call."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class LLMResponse:
    """Canonical result of a non-streaming LLM call."""
    content: str
    tool_calls: list[ToolCall]
    usage: TokenUsage
    stop_reason: str | None
    raw: dict[str, Any]


@dataclass
class TextDelta:
    """One incremental text chunk yielded during streaming."""
    text: str


@dataclass
class StreamEnd:
    """Terminal event of a streaming response; carries stop_reason and final usage."""
    stop_reason: str
    usage: TokenUsage
    raw: dict[str, Any]


StreamChunk = TextDelta | StreamEnd
