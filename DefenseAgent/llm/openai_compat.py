import json
from typing import Any

from openai import AsyncOpenAI

from DefenseAgent.llm.base import LLMAdapter, to_dict_safe
from DefenseAgent.llm.errors import LLMAdapterError, LLMProviderError
from DefenseAgent.llm.types import (
    LLMResponse,
    Message,
    StreamEnd,
    TextDelta,
    TokenUsage,
    ToolCall,
)


_FINISH_REASON_MAP = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "length": "max_tokens",
}


class OpenAICompatibleAdapter(LLMAdapter):
    """Concrete LLMAdapter for providers speaking the OpenAI chat/completions protocol (OpenAI, Qwen, DeepSeek, vLLM, Google)."""

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
        self._client = client or AsyncOpenAI(
            api_key=api_key or None, base_url=base_url or None,
        )

    async def chat(
        self,
        messages: list[Message],
        *,
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        system: str | None = None,
    ) -> LLMResponse:
        """Send canonical messages to the OpenAI-compatible endpoint and return a parsed LLMResponse."""
        kwargs = self._build_request(
            messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            system=system,
        )
        try:
            response = await self._client.chat.completions.create(**kwargs)
        except Exception as e:
            raise LLMProviderError(
                provider="openai-compatible",
                status_code=getattr(e, "status_code", None),
                message=str(e),
            ) from e
        return _parse_response(response)

    async def chat_stream(
        self,
        messages: list[Message],
        *,
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        system: str | None = None,
    ):
        """Stream chat/completions chunks as TextDelta events followed by one StreamEnd."""
        kwargs = self._build_request(
            messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            system=system,
        )
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}

        try:
            stream = await self._client.chat.completions.create(**kwargs)
        except Exception as e:
            raise LLMProviderError(
                provider="openai-compatible",
                status_code=getattr(e, "status_code", None),
                message=str(e),
            ) from e

        stop_reason = "other"
        usage: TokenUsage | None = None
        reasoning_buf: list[str] = []
        response_id = ""

        async for chunk in stream:
            if chunk.choices:
                choice = chunk.choices[0]
                delta = getattr(choice, "delta", None)
                if delta is not None:
                    text = getattr(delta, "content", None)
                    if text:
                        yield TextDelta(text=text)
                    # DeepSeek-R1 / Qwen-QwQ / Kimi K1.5 emit incremental thinking
                    # under delta.reasoning_content; accumulate for StreamEnd.
                    r = getattr(delta, "reasoning_content", None)
                    if r:
                        reasoning_buf.append(r)
                finish = getattr(choice, "finish_reason", None)
                if finish:
                    stop_reason = _FINISH_REASON_MAP.get(finish, "other")
            # Capture id from the first chunk that carries one — every chunk in
            # a single stream shares the same id.
            chunk_id = getattr(chunk, "id", None)
            if chunk_id and not response_id:
                response_id = chunk_id
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage is not None:
                pt = getattr(chunk_usage, "prompt_tokens", 0) or 0
                ct = getattr(chunk_usage, "completion_tokens", 0) or 0
                total = getattr(chunk_usage, "total_tokens", None) or (pt + ct)
                cache_read, cache_creation = _extract_cache_tokens(chunk_usage)
                usage = TokenUsage(
                    prompt_tokens=pt,
                    completion_tokens=ct,
                    total_tokens=total,
                    cache_read_tokens=cache_read,
                    cache_creation_tokens=cache_creation,
                )

        yield StreamEnd(
            stop_reason=stop_reason,
            usage=usage or TokenUsage(0, 0, 0),
            raw={},
            reasoning_content="".join(reasoning_buf),
            id=response_id,
        )

    def _build_request(
        self,
        messages: list[Message],
        *,
        tools: list[dict] | None,
        temperature: float,
        max_tokens: int,
        system: str | None,
    ) -> dict[str, Any]:
        """Translate canonical args into the kwargs that openai.chat.completions.create expects."""
        has_system_in_messages = any(m.role == "system" for m in messages)
        if system is not None and has_system_in_messages:
            raise LLMAdapterError(
                "system kwarg and a role='system' message in `messages` are both set; "
                "pick one."
            )

        wire_messages: list[dict] = []
        if system is not None:
            wire_messages.append({"role": "system", "content": system})
        for m in messages:
            wire_messages.append(_message_to_wire(m))

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": wire_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = [_tool_schema_to_wire(t) for t in tools]
        return kwargs


def _message_to_wire(m: Message) -> dict:
    """Convert a canonical Message into OpenAI's wire dict (serializes tool_call arguments to JSON). `m.content` is passed through unchanged, so list-form content (multimodal: `[{type:"text",...},{type:"image_url",...}]`) flows directly to providers that speak the OpenAI content-block protocol — Qwen via DashScope, DeepSeek-VL, GLM, Kimi, etc. Mirrors ms-agent's `openai_llm.py` pattern of never restructuring already-list content."""
    if m.role == "assistant" and m.tool_calls:
        return {
            "role": "assistant",
            "content": m.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in m.tool_calls
            ],
        }
    if m.role == "tool":
        return {
            "role": "tool",
            "tool_call_id": m.tool_call_id,
            "name": m.name,
            "content": m.content,
        }
    return {"role": m.role, "content": m.content}


def _tool_schema_to_wire(schema: dict) -> dict:
    """Map a harness tool-schema dict into OpenAI's {type: function, function: {...}} shape."""
    return {
        "type": "function",
        "function": {
            "name": schema["name"],
            "description": schema.get("description", ""),
            "parameters": schema.get("input_schema") or {"type": "object", "properties": {}},
        },
    }


def _extract_cache_tokens(usage_raw: Any) -> tuple[int, int]:
    """Pull (cache_read_tokens, cache_creation_tokens) from an OpenAI-shaped usage object.

    Reads `usage.prompt_tokens_details.cached_tokens` (OpenAI native, DashScope) and
    `usage.prompt_tokens_details.cache_creation_input_tokens` (DashScope, Anthropic-via-OpenAI proxies).
    Most providers don't expose these — missing fields silently default to 0.
    """
    if usage_raw is None:
        return 0, 0
    details = getattr(usage_raw, "prompt_tokens_details", None)
    if details is None and isinstance(usage_raw, dict):
        details = usage_raw.get("prompt_tokens_details")
    if details is None:
        return 0, 0
    if isinstance(details, dict):
        read = int(details.get("cached_tokens", 0) or 0)
        created = int(details.get("cache_creation_input_tokens", 0) or 0)
    else:
        read = int(getattr(details, "cached_tokens", 0) or 0)
        created = int(getattr(details, "cache_creation_input_tokens", 0) or 0)
    return read, created


def _parse_response(response: Any) -> LLMResponse:
    """Turn an openai chat/completions response into a canonical LLMResponse."""
    choice = response.choices[0]
    msg = choice.message
    content = msg.content or ""

    # Reasoning models (DeepSeek-R1, Qwen-QwQ, Kimi K1.5, etc.) attach the
    # chain-of-thought to msg.reasoning_content; native OpenAI hides it.
    reasoning_content = getattr(msg, "reasoning_content", None) or ""

    tool_calls: list[ToolCall] = []
    raw_tool_calls = getattr(msg, "tool_calls", None) or []
    for tc in raw_tool_calls:
        raw_args = tc.function.arguments
        parsed = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=parsed))

    stop_reason = _FINISH_REASON_MAP.get(choice.finish_reason, "other")

    usage_raw = response.usage
    total = getattr(usage_raw, "total_tokens", None)
    if total is None:
        total = usage_raw.prompt_tokens + usage_raw.completion_tokens
    cache_read, cache_creation = _extract_cache_tokens(usage_raw)

    return LLMResponse(
        content=content,
        tool_calls=tool_calls,
        usage=TokenUsage(
            prompt_tokens=usage_raw.prompt_tokens,
            completion_tokens=usage_raw.completion_tokens,
            total_tokens=total,
            cache_read_tokens=cache_read,
            cache_creation_tokens=cache_creation,
        ),
        stop_reason=stop_reason,
        raw=to_dict_safe(response),
        reasoning_content=reasoning_content,
        id=getattr(response, "id", None) or "",
    )
