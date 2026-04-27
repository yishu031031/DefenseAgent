from typing import Any

from anthropic import AsyncAnthropic

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


_PASSTHROUGH_STOP_REASONS = {"end_turn", "tool_use", "max_tokens", "stop_sequence"}


class AnthropicAdapter(LLMAdapter):
    """Concrete LLMAdapter for Anthropic's Messages API, supporting chat and native streaming."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str | None = None,
        client: Any | None = None,
    ) -> None:
        """Store the model name and construct (or accept) an AsyncAnthropic client for API calls."""
        self.model = model
        if client is not None:
            self._client = client
            return
        kwargs: dict[str, Any] = {"api_key": api_key or None}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = AsyncAnthropic(**kwargs)

    async def chat(
        self,
        messages: list[Message],
        *,
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        system: str | None = None,
    ) -> LLMResponse:
        """Send canonical messages to Anthropic's /messages endpoint and return a parsed LLMResponse."""
        kwargs = self._build_request(
            messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            system=system,
        )
        try:
            response = await self._client.messages.create(**kwargs)
        except Exception as e:
            raise LLMProviderError(
                provider="claude",
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
        """Stream Anthropic deltas as TextDelta chunks followed by one StreamEnd."""
        kwargs = self._build_request(
            messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            system=system,
        )
        try:
            stream_cm = self._client.messages.stream(**kwargs)
        except Exception as e:
            raise LLMProviderError(
                provider="claude",
                status_code=getattr(e, "status_code", None),
                message=str(e),
            ) from e

        async with stream_cm as stream:
            async for delta_text in stream.text_stream:
                if delta_text:
                    yield TextDelta(text=delta_text)
            final = await stream.get_final_message()

        # Extract any thinking blocks from the final message — text_stream above
        # only yields text deltas, so reasoning content arrives as one payload at
        # the end rather than incrementally. Good enough for non-real-time UX.
        thinking_parts: list[str] = []
        for block in (getattr(final, "content", None) or []):
            if getattr(block, "type", None) == "thinking":
                thinking_parts.append(getattr(block, "thinking", "") or "")

        raw_stop = final.stop_reason
        stop_reason = raw_stop if raw_stop in _PASSTHROUGH_STOP_REASONS else "other"
        pt = getattr(final.usage, "input_tokens", 0) or 0
        ct = getattr(final.usage, "output_tokens", 0) or 0
        cache_read = getattr(final.usage, "cache_read_input_tokens", 0) or 0
        cache_creation = getattr(final.usage, "cache_creation_input_tokens", 0) or 0
        yield StreamEnd(
            stop_reason=stop_reason,
            usage=TokenUsage(
                prompt_tokens=pt,
                completion_tokens=ct,
                total_tokens=pt + ct,
                cache_read_tokens=cache_read,
                cache_creation_tokens=cache_creation,
            ),
            raw=to_dict_safe(final),
            reasoning_content="".join(thinking_parts),
            id=getattr(final, "id", None) or "",
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
        """Translate canonical args into the exact kwargs the Anthropic SDK's create/stream expects."""
        system_msgs = [m for m in messages if m.role == "system"]
        non_system = [m for m in messages if m.role != "system"]

        if system is not None and system_msgs:
            raise LLMAdapterError(
                "system kwarg and role='system' messages both supplied; pick one."
            )

        merged_system = system
        if system_msgs:
            merged_system = "\n".join(m.content for m in system_msgs)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [_message_to_wire(m) for m in non_system],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if merged_system is not None:
            kwargs["system"] = merged_system
        if tools:
            kwargs["tools"] = [_tool_schema_to_wire(t) for t in tools]
        return kwargs


def _message_to_wire(m: Message) -> dict:
    """Convert a canonical Message into Anthropic's wire dict (tool-result → user, tool_use → assistant blocks)."""
    if isinstance(m.content, list):
        raise LLMAdapterError(
            "AnthropicAdapter currently supports only text content; multimodal "
            "(list-shaped) content is not yet wired. Use OpenAICompatibleAdapter "
            "for image-bearing turns, or pass text-only content to this adapter."
        )
    if m.role == "tool":
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": m.tool_call_id,
                    "content": m.content,
                }
            ],
        }
    if m.role == "assistant" and m.tool_calls:
        blocks: list[dict] = []
        if m.content:
            blocks.append({"type": "text", "text": m.content})
        for tc in m.tool_calls:
            blocks.append(
                {
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.arguments,
                }
            )
        return {"role": "assistant", "content": blocks}
    return {"role": m.role, "content": m.content}


def _tool_schema_to_wire(schema: dict) -> dict:
    """Map a harness tool-schema dict into Anthropic's input_schema shape."""
    return {
        "name": schema["name"],
        "description": schema.get("description", ""),
        "input_schema": schema.get("input_schema") or {"type": "object", "properties": {}},
    }


def _parse_response(response: Any) -> LLMResponse:
    """Turn an Anthropic /messages response into a canonical LLMResponse."""
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for block in response.content:
        btype = getattr(block, "type", None)
        if btype == "text":
            text_parts.append(block.text)
        elif btype == "thinking":
            # Extended-thinking content lives alongside text/tool_use in the
            # content array; ms-agent's parser missed this case.
            thinking_parts.append(getattr(block, "thinking", "") or "")
        elif btype == "tool_use":
            tool_calls.append(
                ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
            )

    raw_stop = response.stop_reason
    stop_reason = raw_stop if raw_stop in _PASSTHROUGH_STOP_REASONS else "other"

    usage_raw = response.usage
    pt = usage_raw.input_tokens
    ct = usage_raw.output_tokens
    cache_read = getattr(usage_raw, "cache_read_input_tokens", 0) or 0
    cache_creation = getattr(usage_raw, "cache_creation_input_tokens", 0) or 0
    return LLMResponse(
        content="".join(text_parts),
        tool_calls=tool_calls,
        usage=TokenUsage(
            prompt_tokens=pt,
            completion_tokens=ct,
            total_tokens=pt + ct,
            cache_read_tokens=cache_read,
            cache_creation_tokens=cache_creation,
        ),
        stop_reason=stop_reason,
        raw=to_dict_safe(response),
        reasoning_content="".join(thinking_parts),
        id=getattr(response, "id", None) or "",
    )
