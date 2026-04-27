"""Smoke tests for the unified `AgentConfig` constructor path.

Asserts that `ReActAgent(config)` / `SimpleAgent(config)` / `PlanAndSolveAgent(config)`
build a working agent without manual component wiring, and that the toggles
(`use_memory`, `use_reflection`, `use_compressor`, `use_logger`, `use_tools`)
flip the right modules on/off.
"""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from DefenseAgent import AgentConfig, PlanAndSolveAgent, ReActAgent, SimpleAgent
from DefenseAgent.agent.base import RAG_SEARCH_TOOL_NAME, MEMORY_RECALL_TOOL_NAME

from tests.DefenseAgent.agent._support import make_profile


def _set_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Populate the env vars LLM.from_env reads so AgentConfig sync construction succeeds."""
    monkeypatch.setenv("AGENT_LAB_LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.example.com")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-chat")
    monkeypatch.setenv("EMBEDDING_API_KEY", "sk-test-emb")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://api.example.com")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("EMBEDDING_DIMS", "1536")


def _build_with_mocked_mem0(cls, config: AgentConfig):
    """Construct an agent with mem0 patched out so no real DB is touched."""
    from DefenseAgent.memory.mem0_memory import Mem0Memory
    with patch.object(
        Mem0Memory, "_init_memory_obj", return_value=MagicMock(name="mem0"),
    ):
        return cls(config)


def test_react_agent_from_config_wires_every_component(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _set_env(monkeypatch)
    config = AgentConfig(profile=make_profile(), load_env=False, storage_path=tmp_path)
    agent = _build_with_mocked_mem0(ReActAgent, config)

    assert agent.profile is config.profile
    assert agent.llm is not None
    assert agent.memory is not None
    assert agent.tools is not None
    assert agent.reflector is not None
    assert agent.compressor is not None
    assert agent._config is config


def test_simple_and_plan_agents_accept_the_same_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _set_env(monkeypatch)
    config = AgentConfig(profile=make_profile(), load_env=False, use_logger=False, storage_path=tmp_path)

    simple = _build_with_mocked_mem0(SimpleAgent, config)
    plan = _build_with_mocked_mem0(PlanAndSolveAgent, config)

    assert isinstance(simple, SimpleAgent)
    assert isinstance(plan, PlanAndSolveAgent)
    assert simple.profile is plan.profile is config.profile


def test_use_memory_false_disables_memory_subsystem(monkeypatch: pytest.MonkeyPatch):
    _set_env(monkeypatch)
    config = AgentConfig(
        profile=make_profile(),
        load_env=False,
        use_memory=False,
        use_reflection=False,
        use_compressor=False,
        use_logger=False,
    )
    # No mem0 patch needed — Mem0Memory should not even be constructed.
    agent = ReActAgent(config)

    assert agent.memory is None
    assert agent.reflector is None
    assert MEMORY_RECALL_TOOL_NAME not in agent._agent_tools
    assert agent.save_outcome is False
    assert agent.save_trajectory is False
    assert agent.reflect_after_run is False


def test_use_compressor_and_logger_toggles(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _set_env(monkeypatch)
    config = AgentConfig(
        profile=make_profile(),
        load_env=False,
        use_compressor=False,
        use_logger=False,
        storage_path=tmp_path,
    )
    agent = _build_with_mocked_mem0(ReActAgent, config)

    assert agent.compressor is None
    assert agent.logger is None


def test_use_rag_none_follows_profile_disabled_by_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _set_env(monkeypatch)
    config = AgentConfig(profile=make_profile(), load_env=False, use_logger=False, storage_path=tmp_path)
    agent = _build_with_mocked_mem0(ReActAgent, config)

    # make_profile() has rag.enabled=False, so rag stays unwired.
    assert agent.rag is None
    assert RAG_SEARCH_TOOL_NAME not in agent._agent_tools


def test_extra_tools_are_registered(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _set_env(monkeypatch)

    def calculator(expression: str) -> str:
        """Compute a math expression."""
        return str(eval(expression))  # noqa: S307 — test fixture

    config = AgentConfig(
        profile=make_profile(),
        load_env=False,
        use_logger=False,
        tools=[calculator],
        storage_path=tmp_path,
    )
    agent = _build_with_mocked_mem0(ReActAgent, config)

    assert "calculator" in agent.tools.names()


def test_max_steps_default_is_taken_from_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _set_env(monkeypatch)
    config = AgentConfig(
        profile=make_profile(max_steps=99),  # profile cap = 99
        load_env=False,
        use_logger=False,
        max_steps=4,                          # config cap = 4
        storage_path=tmp_path,
    )
    agent = _build_with_mocked_mem0(ReActAgent, config)

    # Config wins over profile when no per-call override is given.
    assert agent._resolve_max_steps(None) == 4
    # Per-call override always wins.
    assert agent._resolve_max_steps(7) == 7
