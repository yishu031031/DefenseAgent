"""Tests for `DefenseAgent.create_agent` — the one-shot SDK factory.

Verifies the three input shapes (AgentConfig / dict / path), the strategy
kwarg routing, and error paths.
"""
from pathlib import Path

import pytest

from DefenseAgent import (
    AgentConfig,
    PlanAndSolveAgent,
    ReActAgent,
    SimpleAgent,
    create_agent,
)

from tests.DefenseAgent.agent._support import (
    ScriptedLLM,
    fake_memory,
    make_profile,
    make_test_config,
)


def _config() -> AgentConfig:
    """Build a fully-injected AgentConfig that bypasses .env entirely."""
    profile = make_profile()
    return make_test_config(
        profile=profile,
        llm=ScriptedLLM([]),
        memory=fake_memory(profile),
    )


def test_create_agent_with_agent_config_returns_react_by_default():
    cfg = _config()
    agent = create_agent(cfg)
    assert isinstance(agent, ReActAgent)


def test_create_agent_simple_strategy():
    cfg = _config()
    agent = create_agent(cfg, strategy="simple")
    assert isinstance(agent, SimpleAgent)


def test_create_agent_plan_and_solve_strategy():
    cfg = _config()
    agent = create_agent(cfg, strategy="plan_and_solve")
    assert isinstance(agent, PlanAndSolveAgent)


def test_create_agent_from_dict_routes_through_agent_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """A dict input is forwarded as **kwargs to AgentConfig(...)."""
    profile = make_profile()
    cfg_dict = {
        "profile": profile,
        "load_env": False,
        "use_tools": False,
        "use_memory": False,
        "use_reflection": False,
        "use_compressor": False,
        "use_logger": False,
        "llm": ScriptedLLM([]),
    }
    agent = create_agent(cfg_dict, strategy="simple")
    assert isinstance(agent, SimpleAgent)


def test_create_agent_from_path_loads_yaml(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """A str/Path input is treated as a profile YAML path."""
    monkeypatch.setenv("AGENT_LAB_LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.example.com")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-chat")
    monkeypatch.setenv("EMBEDDING_API_KEY", "sk-test")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://api.example.com")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("EMBEDDING_DIMS", "1536")

    from DefenseAgent.examples import EXAMPLE_PROFILE_PATH as profile_path
    assert profile_path.is_file()

    # Patch out the mem0 stack so we don't touch a real DB.
    from unittest.mock import MagicMock, patch
    from DefenseAgent.memory.mem0_memory import Mem0Memory

    with patch.object(
        Mem0Memory, "_init_memory_obj", return_value=MagicMock(name="mem0"),
    ):
        # Pass as str
        agent_str = create_agent(str(profile_path))
        # And as Path
        agent_path = create_agent(profile_path)

    assert isinstance(agent_str, ReActAgent)
    assert isinstance(agent_path, ReActAgent)


def test_create_agent_unknown_strategy_raises():
    cfg = _config()
    with pytest.raises(ValueError, match="unknown strategy"):
        create_agent(cfg, strategy="bogus")  # type: ignore[arg-type]


def test_create_agent_unsupported_input_type_raises():
    with pytest.raises(TypeError, match="unsupported config type"):
        create_agent(42)  # type: ignore[arg-type]
