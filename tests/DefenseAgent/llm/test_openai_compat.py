"""Tests for OpenAICompatibleAdapter (chat + chat_stream)."""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from DefenseAgent.llm import (
    LLMProviderError,
    Message,
    StreamChunk,
    StreamEnd,
    TextDelta,
    ToolCall,
)
from DefenseAgent.llm.errors import LLMAdapterError
from DefenseAgent.llm.openai_compat import OpenAICompatibleAdapter
from tests.DefenseAgent.llm.conftest import make_fake_openai_response


# ---------- happy path: basic chat ----------


async def test_basic_chat_translates_user_message_and_parses_response(
    fake_openai_client,
):
    fake_openai_client.chat.completions.create.return_value = (
        make_fake_openai_response(content="hello", finish_reason="stop")
    )
    adapter = OpenAICompatibleAdapter(
        api_key="sk-test",
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
        client=fake_openai_client,
    )

    resp = await adapter.chat([Message(role="user", content="hi")])

    call = fake_openai_client.chat.completions.create.call_args
    assert call.kwargs["model"] == "gpt-4o-mini"
    assert call.kwargs["messages"] == [{"role": "user", "content": "hi"}]
    assert call.kwargs["temperature"] == 0.7
    assert call.kwargs["max_tokens"] == 1024
    assert call.kwargs.get("tools") is None

    assert resp.content == "hello"
    assert resp.tool_calls == []
    assert resp.stop_reason == "end_turn"
    assert resp.usage.prompt_tokens == 10
    assert resp.usage.completion_tokens == 3
    assert resp.usage.total_tokens == 13


async def test_chat_respects_temperature_and_max_tokens_overrides(fake_openai_client):
    fake_openai_client.chat.completions.create.return_value = (
        make_fake_openai_response()
    )
    adapter = OpenAICompatibleAdapter(
        api_key="k", base_url="u", model="m", client=fake_openai_client
    )

    await adapter.chat(
        [Message(role="user", content="x")],
        temperature=0.2,
        max_tokens=42,
    )

    kwargs = fake_openai_client.chat.completions.create.call_args.kwargs
    assert kwargs["temperature"] == 0.2
    assert kwargs["max_tokens"] == 42


# ---------- system handling ----------


async def test_system_kwarg_is_prepended_as_system_message(fake_openai_client):
    fake_openai_client.chat.completions.create.return_value = (
        make_fake_openai_response()
    )
    adapter = OpenAICompatibleAdapter(
        api_key="k", base_url="u", model="m", client=fake_openai_client
    )

    await adapter.chat(
        [Message(role="user", content="hi")],
        system="You are a helpful assistant.",
    )

    msgs = fake_openai_client.chat.completions.create.call_args.kwargs["messages"]
    assert msgs[0] == {"role": "system", "content": "You are a helpful assistant."}
    assert msgs[1] == {"role": "user", "content": "hi"}


async def test_system_message_in_list_is_passed_through(fake_openai_client):
    fake_openai_client.chat.completions.create.return_value = (
        make_fake_openai_response()
    )
    adapter = OpenAICompatibleAdapter(
        api_key="k", base_url="u", model="m", client=fake_openai_client
    )

    await adapter.chat(
        [
            Message(role="system", content="be brief"),
            Message(role="user", content="hi"),
        ]
    )

    msgs = fake_openai_client.chat.completions.create.call_args.kwargs["messages"]
    assert msgs[0] == {"role": "system", "content": "be brief"}
    assert msgs[1] == {"role": "user", "content": "hi"}


async def test_system_kwarg_and_system_in_messages_raises(fake_openai_client):
    adapter = OpenAICompatibleAdapter(
        api_key="k", base_url="u", model="m", client=fake_openai_client
    )

    with pytest.raises(LLMAdapterError):
        await adapter.chat(
            [
                Message(role="system", content="a"),
                Message(role="user", content="hi"),
            ],
            system="b",
        )


# ---------- tools (request side) ----------


async def test_tools_json_schema_translated_to_openai_function_format(
    fake_openai_client,
):
    fake_openai_client.chat.completions.create.return_value = (
        make_fake_openai_response()
    )
    adapter = OpenAICompatibleAdapter(
        api_key="k", base_url="u", model="m", client=fake_openai_client
    )
    tool_schema = [
        {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
    ]

    await adapter.chat([Message(role="user", content="weather in SF?")], tools=tool_schema)

    tools = fake_openai_client.chat.completions.create.call_args.kwargs["tools"]
    assert tools == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]


async def test_assistant_tool_calls_translated_to_openai_format(fake_openai_client):
    fake_openai_client.chat.completions.create.return_value = (
        make_fake_openai_response()
    )
    adapter = OpenAICompatibleAdapter(
        api_key="k", base_url="u", model="m", client=fake_openai_client
    )

    await adapter.chat(
        [
            Message(role="user", content="weather?"),
            Message(
                role="assistant",
                content="",
                tool_calls=[
                    ToolCall(id="call_1", name="get_weather", arguments={"city": "SF"})
                ],
            ),
            Message(
                role="tool",
                content='{"temp": 72}',
                tool_call_id="call_1",
                name="get_weather",
            ),
        ]
    )

    msgs = fake_openai_client.chat.completions.create.call_args.kwargs["messages"]
    assert msgs[1] == {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": json.dumps({"city": "SF"}),
                },
            }
        ],
    }
    assert msgs[2] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "name": "get_weather",
        "content": '{"temp": 72}',
    }


# ---------- tools (response side) ----------


async def test_response_tool_calls_parsed_into_canonical(fake_openai_client):
    fake_tool_call = SimpleNamespace(
        id="call_abc",
        type="function",
        function=SimpleNamespace(
            name="get_weather",
            arguments='{"city": "SF"}',
        ),
    )
    fake_openai_client.chat.completions.create.return_value = (
        make_fake_openai_response(
            content=None,
            tool_calls=[fake_tool_call],
            finish_reason="tool_calls",
        )
    )
    adapter = OpenAICompatibleAdapter(
        api_key="k", base_url="u", model="m", client=fake_openai_client
    )

    resp = await adapter.chat([Message(role="user", content="weather?")])

    assert resp.content == ""
    assert resp.stop_reason == "tool_use"
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].id == "call_abc"
    assert resp.tool_calls[0].name == "get_weather"
    assert resp.tool_calls[0].arguments == {"city": "SF"}  # parsed dict, not string


# ---------- finish_reason mapping ----------


@pytest.mark.parametrize(
    "raw,normalized",
    [
        ("stop", "end_turn"),
        ("tool_calls", "tool_use"),
        ("length", "max_tokens"),
        ("content_filter", "other"),
        ("function_call", "other"),
    ],
)
async def test_finish_reason_mapping(fake_openai_client, raw, normalized):
    fake_openai_client.chat.completions.create.return_value = (
        make_fake_openai_response(finish_reason=raw)
    )
    adapter = OpenAICompatibleAdapter(
        api_key="k", base_url="u", model="m", client=fake_openai_client
    )

    resp = await adapter.chat([Message(role="user", content="x")])
    assert resp.stop_reason == normalized


# ---------- error wrapping ----------


async def test_provider_exceptions_wrapped_in_llm_provider_error(fake_openai_client):
    fake_openai_client.chat.completions.create.side_effect = RuntimeError("boom")
    adapter = OpenAICompatibleAdapter(
        api_key="k", base_url="u", model="m", client=fake_openai_client
    )

    with pytest.raises(LLMProviderError) as excinfo:
        await adapter.chat([Message(role="user", content="x")])

    assert excinfo.value.provider == "openai-compatible"
    assert "boom" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, RuntimeError)


# ============================================================
# Streaming (chat_stream)
# ============================================================


def _openai_delta_chunk(content: str | None = None, finish_reason: str | None = None):
    """Shape of one chunk from the openai SDK's streaming iterator."""
    delta = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=None)


def _openai_usage_chunk(prompt_tokens: int, completion_tokens: int):
    """A usage-only chunk (sent at the end when stream_options.include_usage=True)."""
    return SimpleNamespace(
        choices=[],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


async def _fake_openai_stream(chunks):
    for c in chunks:
        yield c


async def test_openai_stream_yields_text_deltas_then_end(fake_openai_client):
    fake_openai_client.chat.completions.create.return_value = _fake_openai_stream([
        _openai_delta_chunk(content="Hel"),
        _openai_delta_chunk(content="lo "),
        _openai_delta_chunk(content="world"),
        _openai_delta_chunk(finish_reason="stop"),
        _openai_usage_chunk(prompt_tokens=4, completion_tokens=3),
    ])

    adapter = OpenAICompatibleAdapter(
        api_key="k", base_url="u", model="m", client=fake_openai_client,
    )

    out: list[StreamChunk] = [
        c async for c in adapter.chat_stream([Message(role="user", content="hi")])
    ]

    assert [type(c).__name__ for c in out] == [
        "TextDelta", "TextDelta", "TextDelta", "StreamEnd",
    ]
    assert "".join(c.text for c in out[:3]) == "Hello world"
    end = out[-1]
    assert end.stop_reason == "end_turn"
    assert end.usage.prompt_tokens == 4
    assert end.usage.completion_tokens == 3
    assert end.usage.total_tokens == 7


async def test_openai_stream_finish_reason_maps_tool_calls(fake_openai_client):
    fake_openai_client.chat.completions.create.return_value = _fake_openai_stream([
        _openai_delta_chunk(content="ok"),
        _openai_delta_chunk(finish_reason="tool_calls"),
        _openai_usage_chunk(1, 1),
    ])
    adapter = OpenAICompatibleAdapter(
        api_key="k", base_url="u", model="m", client=fake_openai_client,
    )
    chunks = [c async for c in adapter.chat_stream([Message(role="user", content="x")])]
    end = chunks[-1]
    assert isinstance(end, StreamEnd)
    assert end.stop_reason == "tool_use"


async def test_openai_stream_without_usage_chunk_defaults_to_zero(fake_openai_client):
    """Some providers may not honor stream_options.include_usage — must not crash."""
    fake_openai_client.chat.completions.create.return_value = _fake_openai_stream([
        _openai_delta_chunk(content="hi"),
        _openai_delta_chunk(finish_reason="stop"),
    ])
    adapter = OpenAICompatibleAdapter(
        api_key="k", base_url="u", model="m", client=fake_openai_client,
    )
    chunks = [c async for c in adapter.chat_stream([Message(role="user", content="x")])]
    end = chunks[-1]
    assert isinstance(end, StreamEnd)
    assert end.usage.total_tokens == 0


async def test_openai_stream_passes_stream_true_and_include_usage(fake_openai_client):
    fake_openai_client.chat.completions.create.return_value = _fake_openai_stream([
        _openai_delta_chunk(finish_reason="stop"),
        _openai_usage_chunk(1, 1),
    ])
    adapter = OpenAICompatibleAdapter(
        api_key="k", base_url="u", model="m", client=fake_openai_client,
    )
    async for _ in adapter.chat_stream([Message(role="user", content="hi")]):
        pass

    kwargs = fake_openai_client.chat.completions.create.call_args.kwargs
    assert kwargs["stream"] is True
    assert kwargs["stream_options"] == {"include_usage": True}
    assert kwargs["model"] == "m"
