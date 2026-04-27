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
    """One canonical conversation message in the harness's provider-agnostic format. `content` is a plain string for text-only messages, or a list of OpenAI-style content blocks (`[{type:"text",text:...}, {type:"image_url",image_url:{url:...}}]`) for multimodal messages — only the OpenAI-compatible adapter currently consumes the list form; the Anthropic adapter raises a clear error if list content arrives."""
    role: Role
    content: str | list[dict[str, Any]]
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None


@dataclass
class TokenUsage:
    """Token accounting attached to every completed LLM call."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cache_read_tokens: int = 0       # tokens served from cache (billed cheap, ~0.1x)
    cache_creation_tokens: int = 0   # tokens written into cache (billed expensive, ~1.25x)


@dataclass
class LLMResponse:
    """Canonical result of a non-streaming LLM call."""
    content: str
    tool_calls: list[ToolCall]
    usage: TokenUsage
    stop_reason: str | None
    raw: dict[str, Any]
    reasoning_content: str = ""   # extended thinking / chain-of-thought (R1, o1, Claude thinking)
    id: str = ""                   # provider request id — useful for support tickets and log correlation


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
    reasoning_content: str = ""   # accumulated reasoning text from the whole stream
    id: str = ""


StreamChunk = TextDelta | StreamEnd
