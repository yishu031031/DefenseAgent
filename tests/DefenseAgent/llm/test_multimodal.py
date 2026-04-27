"""Tests for the multimodal content path: list-shaped Message content + OpenAI adapter pass-through + Anthropic friendly error.

Approach mirrors ms-agent's `openai_llm.py:565-609` pattern: when `Message.content` is already a list (`[{type: text, text: ...}, {type: image_url, image_url: {url: ...}}, ...]`), the adapter must hand it to the provider unchanged. Anthropic adapter doesn't (yet) speak that shape, so it raises an `LLMAdapterError` with a friendly message instead of producing a confusing wire-format error downstream.
"""
from unittest.mock import AsyncMock

import pytest

from DefenseAgent.llm.types import Message
from DefenseAgent.llm.errors import LLMAdapterError
from DefenseAgent.llm.anthropic import AnthropicAdapter
from DefenseAgent.llm.openai_compat import (
    OpenAICompatibleAdapter,
    _message_to_wire,
)
from tests.DefenseAgent.llm.conftest import make_fake_openai_response


# ---------- Message accepts list content ----------


def test_message_accepts_list_content_for_multimodal():
    """The dataclass annotation now permits a list-of-content-blocks for `content`. No runtime validation — OpenAI's adapter is the one that consumes the structured form."""
    blocks = [
        {"type": "text", "text": "what is in this image?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
    ]
    msg = Message(role="user", content=blocks)
    assert msg.content is blocks
    assert msg.role == "user"


def test_message_still_accepts_plain_string_content():
    """Backwards compatibility: text-only callers don't need to change."""
    msg = Message(role="user", content="hi")
    assert msg.content == "hi"


# ---------- OpenAI adapter: list content passes through unchanged ----------


def test_openai_message_to_wire_passes_list_content_through():
    """When `m.content` is a list, `_message_to_wire` must hand it to the provider as-is — mirrors ms-agent's openai_llm pattern of never restructuring already-list content."""
    blocks = [
        {"type": "text", "text": "describe this"},
        {"type": "image_url", "image_url": {"url": "https://x.example/a.jpg"}},
    ]
    wire = _message_to_wire(Message(role="user", content=blocks))
    assert wire == {"role": "user", "content": blocks}


async def test_openai_chat_sends_multimodal_message_through_to_client(fake_openai_client):
    """End-to-end through `chat()`: a multimodal user message reaches the OpenAI client's `messages` argument with content still in list form."""
    fake_openai_client.chat.completions.create.return_value = make_fake_openai_response(
        content="ok"
    )
    adapter = OpenAICompatibleAdapter(
        api_key="k", base_url="u", model="m", client=fake_openai_client
    )

    blocks = [
        {"type": "text", "text": "what is this?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR"}},
    ]
    await adapter.chat([Message(role="user", content=blocks)])

    sent_messages = fake_openai_client.chat.completions.create.call_args.kwargs["messages"]
    assert sent_messages == [{"role": "user", "content": blocks}]


# ---------- Anthropic adapter: friendly error on list content ----------


async def test_anthropic_chat_raises_friendly_error_on_list_content(fake_anthropic_client):
    """The Anthropic adapter doesn't translate OpenAI-style content-block lists; it must surface an explicit error instead of producing a confusing wire-format message."""
    adapter = AnthropicAdapter(
        api_key="k", model="m", client=fake_anthropic_client
    )

    blocks = [
        {"type": "text", "text": "describe this"},
        {"type": "image_url", "image_url": {"url": "https://x/a.jpg"}},
    ]
    with pytest.raises(LLMAdapterError) as e:
        await adapter.chat([Message(role="user", content=blocks)])
    msg = str(e.value)
    assert "multimodal" in msg.lower() or "list" in msg.lower()
    assert "AnthropicAdapter" in msg
