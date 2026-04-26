"""Tests for AnthropicAdapter (chat + chat_stream)."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from DefenseAgent.llm import (
    LLMProviderError,
    Message,
    StreamChunk,
    StreamEnd,
    TextDelta,
    ToolCall,
)
from DefenseAgent.llm.anthropic import AnthropicAdapter
from DefenseAgent.llm.errors import LLMAdapterError
from tests.DefenseAgent.llm.conftest import make_fake_anthropic_response


# ---------- happy path ----------


async def test_basic_chat_text_response(fake_anthropic_client):
    fake_anthropic_client.messages.create.return_value = make_fake_anthropic_response(
        text_blocks=["hello"],
        stop_reason="end_turn",
    )
    adapter = AnthropicAdapter(
        api_key="sk-ant",
        model="claude-sonnet-4-6",
        client=fake_anthropic_client,
    )

    resp = await adapter.chat([Message(role="user", content="hi")])

    kwargs = fake_anthropic_client.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-sonnet-4-6"
    assert kwargs["messages"] == [{"role": "user", "content": "hi"}]
    assert kwargs["max_tokens"] == 1024
    assert kwargs["temperature"] == 0.7
    assert "system" not in kwargs or kwargs["system"] is None

    assert resp.content == "hello"
    assert resp.tool_calls == []
    assert resp.stop_reason == "end_turn"
    assert resp.usage.prompt_tokens == 10
    assert resp.usage.completion_tokens == 3
    assert resp.usage.total_tokens == 13


# ---------- system handling ----------


async def test_system_kwarg_passed_as_system_param(fake_anthropic_client):
    fake_anthropic_client.messages.create.return_value = make_fake_anthropic_response(
        text_blocks=["ok"],
    )
    adapter = AnthropicAdapter(
        api_key="k", model="m", client=fake_anthropic_client
    )

    await adapter.chat(
        [Message(role="user", content="hi")],
        system="Be brief.",
    )

    kwargs = fake_anthropic_client.messages.create.call_args.kwargs
    assert kwargs["system"] == "Be brief."
    assert kwargs["messages"] == [{"role": "user", "content": "hi"}]


async def test_system_messages_in_list_merged_into_system_param(fake_anthropic_client):
    fake_anthropic_client.messages.create.return_value = make_fake_anthropic_response(
        text_blocks=["ok"],
    )
    adapter = AnthropicAdapter(
        api_key="k", model="m", client=fake_anthropic_client
    )

    await adapter.chat(
        [
            Message(role="system", content="line one"),
            Message(role="system", content="line two"),
            Message(role="user", content="hi"),
        ]
    )

    kwargs = fake_anthropic_client.messages.create.call_args.kwargs
    assert kwargs["system"] == "line one\nline two"
    assert kwargs["messages"] == [{"role": "user", "content": "hi"}]


async def test_system_kwarg_and_system_message_raises(fake_anthropic_client):
    adapter = AnthropicAdapter(
        api_key="k", model="m", client=fake_anthropic_client
    )

    with pytest.raises(LLMAdapterError):
        await adapter.chat(
            [
                Message(role="system", content="a"),
                Message(role="user", content="hi"),
            ],
            system="b",
        )


# ---------- tools (request) ----------


async def test_tools_schema_translated_to_anthropic_format(fake_anthropic_client):
    fake_anthropic_client.messages.create.return_value = make_fake_anthropic_response(
        text_blocks=["ok"],
    )
    adapter = AnthropicAdapter(
        api_key="k", model="m", client=fake_anthropic_client
    )
    tool_schema = [
        {
            "name": "get_weather",
            "description": "Get weather for a city.",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
    ]

    await adapter.chat(
        [Message(role="user", content="weather?")], tools=tool_schema
    )

    tools = fake_anthropic_client.messages.create.call_args.kwargs["tools"]
    assert tools == [
        {
            "name": "get_weather",
            "description": "Get weather for a city.",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
    ]


async def test_assistant_tool_calls_become_tool_use_content_blocks(
    fake_anthropic_client,
):
    fake_anthropic_client.messages.create.return_value = make_fake_anthropic_response(
        text_blocks=["ok"],
    )
    adapter = AnthropicAdapter(
        api_key="k", model="m", client=fake_anthropic_client
    )

    await adapter.chat(
        [
            Message(role="user", content="weather?"),
            Message(
                role="assistant",
                content="let me check",
                tool_calls=[
                    ToolCall(
                        id="toolu_1",
                        name="get_weather",
                        arguments={"city": "SF"},
                    )
                ],
            ),
            Message(
                role="tool",
                content='{"temp": 72}',
                tool_call_id="toolu_1",
                name="get_weather",
            ),
        ]
    )

    msgs = fake_anthropic_client.messages.create.call_args.kwargs["messages"]
    assert msgs[0] == {"role": "user", "content": "weather?"}
    assert msgs[1] == {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "let me check"},
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "get_weather",
                "input": {"city": "SF"},
            },
        ],
    }
    assert msgs[2] == {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_1",
                "content": '{"temp": 72}',
            }
        ],
    }


async def test_assistant_text_only_stays_as_plain_string(fake_anthropic_client):
    """An assistant message with only text (no tool_calls) uses plain content, not blocks."""
    fake_anthropic_client.messages.create.return_value = make_fake_anthropic_response(
        text_blocks=["ok"],
    )
    adapter = AnthropicAdapter(
        api_key="k", model="m", client=fake_anthropic_client
    )

    await adapter.chat(
        [
            Message(role="user", content="hi"),
            Message(role="assistant", content="hello there"),
            Message(role="user", content="what's up"),
        ]
    )

    msgs = fake_anthropic_client.messages.create.call_args.kwargs["messages"]
    assert msgs[1] == {"role": "assistant", "content": "hello there"}


# ---------- tools (response) ----------


async def test_response_tool_use_blocks_become_canonical_tool_calls(
    fake_anthropic_client,
):
    fake_anthropic_client.messages.create.return_value = make_fake_anthropic_response(
        text_blocks=["I'll check."],
        tool_use_blocks=[
            {"id": "toolu_abc", "name": "get_weather", "input": {"city": "SF"}}
        ],
        stop_reason="tool_use",
    )
    adapter = AnthropicAdapter(
        api_key="k", model="m", client=fake_anthropic_client
    )

    resp = await adapter.chat([Message(role="user", content="weather?")])

    assert resp.content == "I'll check."
    assert resp.stop_reason == "tool_use"
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].id == "toolu_abc"
    assert resp.tool_calls[0].name == "get_weather"
    assert resp.tool_calls[0].arguments == {"city": "SF"}


async def test_multiple_text_blocks_concatenated(fake_anthropic_client):
    fake_anthropic_client.messages.create.return_value = make_fake_anthropic_response(
        text_blocks=["hello ", "world"],
    )
    adapter = AnthropicAdapter(
        api_key="k", model="m", client=fake_anthropic_client
    )

    resp = await adapter.chat([Message(role="user", content="hi")])
    assert resp.content == "hello world"


# ---------- stop_reason mapping ----------


@pytest.mark.parametrize(
    "raw,normalized",
    [
        ("end_turn", "end_turn"),
        ("tool_use", "tool_use"),
        ("max_tokens", "max_tokens"),
        ("stop_sequence", "stop_sequence"),
        ("something_weird", "other"),
    ],
)
async def test_stop_reason_mapping(fake_anthropic_client, raw, normalized):
    fake_anthropic_client.messages.create.return_value = make_fake_anthropic_response(
        text_blocks=["x"],
        stop_reason=raw,
    )
    adapter = AnthropicAdapter(
        api_key="k", model="m", client=fake_anthropic_client
    )
    resp = await adapter.chat([Message(role="user", content="x")])
    assert resp.stop_reason == normalized


# ---------- error wrapping ----------


async def test_provider_exceptions_wrapped(fake_anthropic_client):
    fake_anthropic_client.messages.create.side_effect = RuntimeError("kaboom")
    adapter = AnthropicAdapter(
        api_key="k", model="m", client=fake_anthropic_client
    )

    with pytest.raises(LLMProviderError) as excinfo:
        await adapter.chat([Message(role="user", content="x")])

    assert excinfo.value.provider == "claude"
    assert "kaboom" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, RuntimeError)


# ============================================================
# Streaming (chat_stream)
# ============================================================


class _FakeTextStream:
    def __init__(self, texts):
        self._texts = texts

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for t in self._texts:
            yield t


class _FakeFinalMessage:
    def __init__(self, stop_reason: str, input_tokens: int, output_tokens: int):
        self.stop_reason = stop_reason
        self.usage = SimpleNamespace(
            input_tokens=input_tokens, output_tokens=output_tokens,
        )


class _FakeAnthropicStream:
    def __init__(self, text_deltas: list[str], final: _FakeFinalMessage):
        self.text_stream = _FakeTextStream(text_deltas)
        self._final = final

    async def get_final_message(self):
        return self._final


class _FakeAnthropicStreamCM:
    def __init__(self, stream):
        self._stream = stream

    async def __aenter__(self):
        return self._stream

    async def __aexit__(self, exc_type, exc, tb):
        return None


def _stream_client(text_deltas: list[str], *, stop_reason: str = "end_turn",
                   input_tokens: int = 5, output_tokens: int = 3):
    """Build a fake anthropic client whose messages.stream(...) returns a CM."""
    client = MagicMock()
    client.messages = MagicMock()
    client.messages.stream = MagicMock(
        return_value=_FakeAnthropicStreamCM(
            _FakeAnthropicStream(
                text_deltas=text_deltas,
                final=_FakeFinalMessage(stop_reason, input_tokens, output_tokens),
            ),
        ),
    )
    return client


async def test_anthropic_stream_yields_deltas_then_end():
    client = _stream_client(["Hel", "lo ", "world"])
    adapter = AnthropicAdapter(api_key="k", model="m", client=client)

    out: list[StreamChunk] = [
        c async for c in adapter.chat_stream([Message(role="user", content="hi")])
    ]

    assert [type(c).__name__ for c in out] == [
        "TextDelta", "TextDelta", "TextDelta", "StreamEnd",
    ]
    assert "".join(c.text for c in out[:3]) == "Hello world"
    end = out[-1]
    assert end.stop_reason == "end_turn"
    assert end.usage.prompt_tokens == 5
    assert end.usage.completion_tokens == 3
    assert end.usage.total_tokens == 8


async def test_anthropic_stream_maps_unknown_stop_reason_to_other():
    client = _stream_client(["x"], stop_reason="pause_turn")  # not in passthrough set
    adapter = AnthropicAdapter(api_key="k", model="m", client=client)
    chunks = [c async for c in adapter.chat_stream([Message(role="user", content="x")])]
    assert chunks[-1].stop_reason == "other"


async def test_anthropic_stream_passes_merged_system():
    client = _stream_client(["ok"])
    adapter = AnthropicAdapter(api_key="k", model="m", client=client)
    async for _ in adapter.chat_stream(
        [Message(role="user", content="hi")], system="Be brief.",
    ):
        pass
    assert client.messages.stream.call_args.kwargs["system"] == "Be brief."
