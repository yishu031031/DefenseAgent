"""Tests for `agent.run(task, images=...)` ergonomics.

`BaseAgent._build_user_message(task, images)` is the single shared entry point. With `images=None` it returns a plain text Message (backwards compatible). With a list, it builds an OpenAI-style multimodal Message (`[{type: text, text}, {type: image_url, image_url: {url}}, ...]`). Each image accepts a local file path (becomes a `data:` URL with inferred MIME), an `http(s)://` URL (passes through), or a pre-built `data:` URL (passes through).
"""
import base64
from pathlib import Path

import pytest

from DefenseAgent.agent.base import _resolve_image_url
from DefenseAgent.agent import ReActAgent, SimpleAgent
from DefenseAgent.llm.types import Message
from DefenseAgent.tools import ToolRegistry

from tests.DefenseAgent.agent._support import (
    ScriptedLLM,
    fake_memory,
    make_profile,
    make_test_config,
    resp,
)


# ---------- _resolve_image_url ----------


def test_resolve_image_url_passes_https_through():
    url = "https://example.com/img.png"
    assert _resolve_image_url(url) == url


def test_resolve_image_url_passes_http_through():
    url = "http://example.com/img.png"
    assert _resolve_image_url(url) == url


def test_resolve_image_url_passes_existing_data_url_through():
    data_url = "data:image/jpeg;base64,/9j/AAAA"
    assert _resolve_image_url(data_url) == data_url


def test_resolve_image_url_local_file_becomes_data_url(tmp_path: Path):
    """Local file → `data:<mime>;base64,...` with MIME inferred from extension."""
    img = tmp_path / "tiny.png"
    raw = b"\x89PNG\r\n\x1a\nfake-png-bytes"
    img.write_bytes(raw)

    url = _resolve_image_url(img)
    assert url.startswith("data:image/png;base64,")
    payload = url.split(",", 1)[1]
    assert base64.b64decode(payload) == raw


def test_resolve_image_url_local_file_accepts_str_path(tmp_path: Path):
    img = tmp_path / "x.jpg"
    img.write_bytes(b"jpeg-bytes")
    url = _resolve_image_url(str(img))
    assert url.startswith("data:image/jpeg;base64,")


def test_resolve_image_url_unknown_extension_falls_back_to_png(tmp_path: Path):
    """When `mimetypes.guess_type` can't determine the type, the helper defaults to image/png so OpenAI providers still accept it."""
    img = tmp_path / "blob.unknownext"
    img.write_bytes(b"some-bytes")
    url = _resolve_image_url(img)
    assert url.startswith("data:image/png;base64,")


def test_resolve_image_url_raises_on_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        _resolve_image_url(tmp_path / "nope.png")


# ---------- BaseAgent._build_user_message ----------


def _agent() -> ReActAgent:
    """Spin up a ReActAgent (concrete BaseAgent subclass) just to access the inherited `_build_user_message` helper."""
    profile = make_profile()
    config = make_test_config(
        profile=profile,
        llm=ScriptedLLM([]),
        memory=fake_memory(profile),
        tools=ToolRegistry(),
    )
    return ReActAgent(config)


def test_build_user_message_text_only_when_images_is_none():
    msg = _agent()._build_user_message("hi", None)
    assert msg.role == "user"
    assert msg.content == "hi"


def test_build_user_message_text_only_when_images_is_empty_list():
    msg = _agent()._build_user_message("hi", [])
    assert msg.content == "hi"


def test_build_user_message_with_one_image_url():
    url = "https://example.com/a.jpg"
    msg = _agent()._build_user_message("describe", [url])
    assert msg.role == "user"
    assert msg.content == [
        {"type": "text", "text": "describe"},
        {"type": "image_url", "image_url": {"url": url}},
    ]


def test_build_user_message_with_local_file(tmp_path: Path):
    img = tmp_path / "x.png"
    img.write_bytes(b"png-bytes")
    msg = _agent()._build_user_message("what is this?", [img])
    assert isinstance(msg.content, list)
    assert msg.content[0] == {"type": "text", "text": "what is this?"}
    block = msg.content[1]
    assert block["type"] == "image_url"
    assert block["image_url"]["url"].startswith("data:image/png;base64,")


def test_build_user_message_with_mixed_inputs(tmp_path: Path):
    """A run can mix local files and remote URLs in one call."""
    img_local = tmp_path / "a.jpg"
    img_local.write_bytes(b"jpeg-bytes")
    msg = _agent()._build_user_message(
        "compare",
        [img_local, "https://example.com/b.png"],
    )
    assert isinstance(msg.content, list)
    assert msg.content[0]["type"] == "text"
    assert msg.content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert msg.content[2]["image_url"]["url"] == "https://example.com/b.png"


# ---------- end-to-end: agent.run(images=...) reaches the LLM ----------


async def test_simple_agent_run_with_images_threads_through_to_llm():
    """SimpleAgent.run(task, images=[...]) sends a list-shaped user message to the LLM. Verifies the helper is wired into the actual `run()` call site, not just available on BaseAgent."""
    llm = ScriptedLLM([resp(content="ok")])
    profile = make_profile()
    config = make_test_config(
        profile=profile,
        llm=llm,
        memory=fake_memory(profile),
        tools=ToolRegistry(),
        save_outcome=False,
        reflect_after_run=False,
    )
    agent = SimpleAgent(config)
    await agent.run("what is in this image?", images=["https://example.com/img.png"])

    sent_messages = llm.calls[0]["messages"]
    user_msg = sent_messages[-1]
    assert isinstance(user_msg.content, list)
    assert user_msg.content[0] == {"type": "text", "text": "what is in this image?"}
    assert user_msg.content[1]["type"] == "image_url"
    assert user_msg.content[1]["image_url"]["url"] == "https://example.com/img.png"


async def test_react_agent_run_with_images_threads_through_to_llm():
    """ReActAgent.run(task, images=[...]) — same end-to-end path, exercised against the loop-strategy agent."""
    llm = ScriptedLLM([resp(content="answer")])
    profile = make_profile()
    config = make_test_config(
        profile=profile,
        llm=llm,
        memory=fake_memory(profile),
        tools=ToolRegistry(),
        save_outcome=False,
        reflect_after_run=False,
    )
    agent = ReActAgent(config)
    await agent.run("describe", images=["https://example.com/x.jpg"])

    sent_messages = llm.calls[0]["messages"]
    user_msg = sent_messages[-1]
    assert isinstance(user_msg.content, list)
    assert user_msg.content[0] == {"type": "text", "text": "describe"}
    assert user_msg.content[1]["image_url"]["url"] == "https://example.com/x.jpg"


async def test_agent_run_without_images_keeps_text_content():
    """Backwards compatibility — pre-existing callers that don't pass `images` still send a plain text Message."""
    llm = ScriptedLLM([resp(content="ok")])
    profile = make_profile()
    config = make_test_config(
        profile=profile,
        llm=llm,
        memory=fake_memory(profile),
        tools=ToolRegistry(),
        save_outcome=False,
        reflect_after_run=False,
    )
    agent = SimpleAgent(config)
    await agent.run("plain text task")

    sent_messages = llm.calls[0]["messages"]
    user_msg = sent_messages[-1]
    assert user_msg.content == "plain text task"
