"""Shared fixtures and helpers for LLM-adapter tests.

Provides fake client + response builders for both Anthropic and OpenAI-compatible
adapters so tests stay fully offline.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


# ---------- OpenAI-compatible SDK fakes ----------


def make_fake_openai_response(
    *,
    content: str | None = "hello",
    tool_calls: list | None = None,
    finish_reason: str = "stop",
    prompt_tokens: int = 10,
    completion_tokens: int = 3,
    total_tokens: int | None = None,
):
    """Build a minimal fake OpenAI chat.completions response object."""
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens if total_tokens is not None else prompt_tokens + completion_tokens,
    )
    return SimpleNamespace(choices=[choice], usage=usage, model="test-model", id="resp_1")


@pytest.fixture
def fake_openai_client():
    """Fake AsyncOpenAI-shaped client whose create() is an AsyncMock."""
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock()
    return client


# ---------- Anthropic SDK fakes ----------


def make_fake_anthropic_response(
    *,
    text_blocks: list[str] | None = None,
    tool_use_blocks: list[dict] | None = None,
    stop_reason: str = "end_turn",
    input_tokens: int = 10,
    output_tokens: int = 3,
):
    """Build a minimal fake Anthropic messages.create response object."""
    content = []
    for t in text_blocks or []:
        content.append(SimpleNamespace(type="text", text=t))
    for tu in tool_use_blocks or []:
        content.append(
            SimpleNamespace(
                type="tool_use",
                id=tu["id"],
                name=tu["name"],
                input=tu["input"],
            )
        )
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
        id="msg_1",
        model="test-model",
    )


@pytest.fixture
def fake_anthropic_client():
    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = AsyncMock()
    return client
