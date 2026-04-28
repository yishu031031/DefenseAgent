"""Tests for DefenseAgent.memory.Mem0Memory (mem0-backed adapter inherited from ms-agent's `DefaultMemory`).

The full mem0 stack pulls qdrant + an LLM + an embedder, none of which we want to spin up in unit tests.
We mock mem0.Memory.from_config so the adapter's wiring (config translation + Message-boundary conversion + memory_type filtering)
can be exercised offline.
"""
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from DefenseAgent.config import AgentProfile
from DefenseAgent.llm.types import Message, ToolCall


from DefenseAgent.examples import EXAMPLE_PROFILE_PATH as _EXAMPLE_PROFILE


def _set_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Populate the env vars Mem0Memory's config translator reads."""
    monkeypatch.setenv("AGENT_LAB_LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.example.com")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-chat")
    monkeypatch.setenv("EMBEDDING_API_KEY", "sk-test-emb")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://api.example.com")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("EMBEDDING_DIMS", "1536")


def _build_memory(profile: AgentProfile) -> Any:
    """Construct a Mem0Memory whose internal mem0 client is a MagicMock — patched at ms-agent's _init_memory_obj seam."""
    from DefenseAgent.memory.mem0_memory import Mem0Memory

    fake_mem0 = MagicMock(name="mem0.Memory")
    fake_mem0.add.return_value = None
    fake_mem0.search.return_value = {"results": []}
    fake_mem0.get_all.return_value = {"results": []}

    with patch.object(
        Mem0Memory,
        "_init_memory_obj",
        return_value=fake_mem0,
    ):
        memory = Mem0Memory(profile, load_env=False)
    memory._fake_mem0 = fake_mem0
    return memory


def _make_profile(tmp_path: Path) -> AgentProfile:
    """Build an in-memory AgentProfile whose source_dir points at tmp_path."""
    profile = AgentProfile(
        id="test_agent", name="Tester", age=25,
        traits="t", backstory="b", initial_plan="p",
    )
    (tmp_path / "profile.yaml").write_text("agent: {}", encoding="utf-8")
    profile._source_path = (tmp_path / "profile.yaml").resolve()
    return profile


# ---------- inheritance contract ----------


def test_mem0_memory_inherits_from_ms_agent_default_memory():
    from ms_agent.memory.default_memory import DefaultMemory as MsDefaultMemory
    from DefenseAgent.memory.mem0_memory import Mem0Memory

    assert issubclass(Mem0Memory, MsDefaultMemory)


def test_context_compressor_inherits_from_ms_agent():
    from ms_agent.memory.condenser.context_compressor import (
        ContextCompressor as MsCC,
    )
    from DefenseAgent.memory.context_compressor import ContextCompressor

    assert issubclass(ContextCompressor, MsCC)


def test_shared_memory_manager_inherits_from_ms_agent():
    from ms_agent.memory.memory_manager import SharedMemoryManager as MsSMM
    from DefenseAgent.memory.shared import SharedMemoryManager

    assert issubclass(SharedMemoryManager, MsSMM)


def test_memory_mapping_is_re_exported():
    from DefenseAgent.memory import memory_mapping

    assert "default_memory" in memory_mapping
    assert "context_compressor" in memory_mapping


# ---------- construction + config translation ----------


def test_mem0_memory_construction_resolves_storage_under_profile_source_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    _set_env(monkeypatch)
    profile = _make_profile(tmp_path)

    memory = _build_memory(profile)

    expected = (tmp_path / "memory").resolve()
    assert Path(memory.output_dir) == expected
    assert (tmp_path / "memory").exists()


def test_mem0_memory_explicit_storage_path_overrides_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    _set_env(monkeypatch)
    profile = _make_profile(tmp_path)
    custom = tmp_path / "elsewhere"

    from DefenseAgent.memory.mem0_memory import Mem0Memory

    fake_mem0 = MagicMock()
    fake_mem0.add.return_value = None
    fake_mem0.search.return_value = {"results": []}
    fake_mem0.get_all.return_value = {"results": []}
    with patch("mem0.Memory.from_config", return_value=fake_mem0):
        memory = Mem0Memory(profile, storage_path=custom, load_env=False)

    assert Path(memory.output_dir) == custom.resolve()


def test_construction_raises_without_llm_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """No AGENT_LAB_LLM_PROVIDER → bridge refuses to build the mem0 config."""
    monkeypatch.delenv("AGENT_LAB_LLM_PROVIDER", raising=False)
    profile = _make_profile(tmp_path)

    from DefenseAgent.memory.mem0_memory import Mem0Memory

    with pytest.raises(ValueError, match="AGENT_LAB_LLM_PROVIDER"):
        Mem0Memory(profile, load_env=False)


def test_construction_raises_without_embedding_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("AGENT_LAB_LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-chat")
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    profile = _make_profile(tmp_path)

    from DefenseAgent.memory.mem0_memory import Mem0Memory

    with pytest.raises(ValueError, match="EMBEDDING_"):
        Mem0Memory(profile, load_env=False)


# ---------- public API methods ----------


async def test_run_no_user_message_returns_input_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    _set_env(monkeypatch)
    profile = _make_profile(tmp_path)
    memory = _build_memory(profile)

    messages = [Message(role="system", content="you are a test")]
    result = await memory.run(messages)

    assert len(result) == 1
    assert result[0].role == "system"
    memory._fake_mem0.search.assert_not_called()


async def test_run_search_no_hits_returns_input_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    _set_env(monkeypatch)
    profile = _make_profile(tmp_path)
    memory = _build_memory(profile)
    memory._fake_mem0.search.return_value = {"results": []}

    messages = [Message(role="user", content="anything")]
    result = await memory.run(messages)

    assert len(result) == 1
    assert result[0].content == "anything"


async def test_run_is_a_passthrough_in_defenseagent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """DefenseAgent.Mem0Memory.run() returns messages unchanged — the LLM accesses memory through the built-in `memory_recall` tool, not via passive system-prompt injection (which would collide with BaseAgent's `system=` kwarg)."""
    _set_env(monkeypatch)
    profile = _make_profile(tmp_path)
    memory = _build_memory(profile)
    memory._fake_mem0.search.return_value = {
        "results": [{"id": "m1", "memory": "Maya prefers the library."}]
    }

    messages = [
        Message(role="system", content="You are Maya."),
        Message(role="user", content="where do I study?"),
    ]
    result = await memory.run(messages)

    assert result == messages
    memory._fake_mem0.search.assert_not_called()


async def test_add_filters_ignore_roles_then_forwards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    _set_env(monkeypatch)
    profile = _make_profile(tmp_path)
    memory = _build_memory(profile)
    # Default ignore_roles == ["tool", "system"]; user messages should pass through.
    messages = [
        Message(role="system", content="ignored"),
        Message(role="user", content="kept"),
        Message(role="tool", content="ignored too", tool_call_id="x"),
    ]
    await memory.add(messages, memory_type="trajectory")

    memory._fake_mem0.add.assert_called_once()
    args, kwargs = memory._fake_mem0.add.call_args
    forwarded = args[0]
    assert isinstance(forwarded, list)
    assert all(m["role"] == "user" for m in forwarded)
    assert kwargs["metadata"] == {"memory_type": "trajectory"}


async def test_add_filters_via_inherited_parse_messages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """ignore_roles filtering is enforced inside ms-agent's add() via parse_messages — the adapter doesn't pre-filter."""
    _set_env(monkeypatch)
    profile = _make_profile(tmp_path)
    memory = _build_memory(profile)

    # All-system messages: ms-agent's parse_messages drops them, so any
    # mem0.add call sees an empty payload (or skips entirely).
    await memory.add([Message(role="system", content="x")])

    if memory._fake_mem0.add.called:
        forwarded = memory._fake_mem0.add.call_args.args[0]
        assert forwarded == []


def test_search_records_filters_by_memory_type(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    _set_env(monkeypatch)
    profile = _make_profile(tmp_path)
    memory = _build_memory(profile)
    memory._fake_mem0.search.return_value = {
        "results": [
            {"memory": "trajectory record", "memory_type": "trajectory"},
            {"memory": "outcome record",    "memory_type": "outcome"},
        ]
    }

    only_traj = memory.search_records("anything", memory_type="trajectory")
    assert [r["memory"] for r in only_traj] == ["trajectory record"]

    all_hits = memory.search_records("anything")
    assert len(all_hits) == 2


def test_get_all_filters_results_by_memory_type(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    _set_env(monkeypatch)
    profile = _make_profile(tmp_path)
    memory = _build_memory(profile)
    memory._fake_mem0.get_all.return_value = {
        "results": [
            {"memory": "x", "memory_type": "reflection"},
            {"memory": "y"},
        ]
    }

    reflections = memory.get_all(memory_type="reflection")
    assert [r["memory"] for r in reflections] == ["x"]


def test_mem0_search_failures_propagate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """search_texts() inherits ms-agent's wrapper which doesn't catch mem0 errors — the agent layer's _handle_memory_recall does that instead."""
    _set_env(monkeypatch)
    profile = _make_profile(tmp_path)
    memory = _build_memory(profile)
    memory._fake_mem0.search.side_effect = RuntimeError("network down")

    with pytest.raises(RuntimeError):
        memory.search_texts("anything")
