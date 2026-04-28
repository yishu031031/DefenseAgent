"""Tests for the BaseAgent abstract contract — instantiation guard, max_steps resolution, close lifecycle, from_profile wiring."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from DefenseAgent.agent import BaseAgent, PlanAndSolveAgent, ReActAgent
from DefenseAgent.config import AgentProfile
from DefenseAgent.tools import ToolRegistry

from tests.DefenseAgent.agent._support import (
    ScriptedLLM,
    fake_memory,
    make_profile,
    resp,
    make_test_config,
)


from DefenseAgent.examples import EXAMPLE_PROFILE_PATH as _EXAMPLE_PROFILE


def _set_env_for_real_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    """Populate the env vars Agent.from_profile reads (LLM provider + embedding) so validation passes."""
    monkeypatch.setenv("AGENT_LAB_LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.example.com")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-chat")
    monkeypatch.setenv("EMBEDDING_API_KEY", "sk-test-emb")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://api.example.com")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("EMBEDDING_DIMS", "1536")


# ---------- abstract contract ----------


def test_agent_base_class_cannot_be_instantiated():
    profile = make_profile()
    memory = fake_memory(profile)
    with pytest.raises(TypeError):
        BaseAgent(  # type: ignore[abstract]
            profile, llm=ScriptedLLM([]), memory=memory, tools=ToolRegistry(),
        )


# ---------- max_steps resolution ----------


def test_resolve_max_steps_uses_explicit_override():
    profile = make_profile(max_steps=10)
    agent = ReActAgent(make_test_config(
            profile=profile,
            llm=ScriptedLLM([]),
            memory=fake_memory(profile),
            tools=ToolRegistry(),
        ))
    assert agent._resolve_max_steps(3) == 3
    assert agent._resolve_max_steps(None) == 10


def test_resolve_max_steps_reads_from_profile_when_no_override():
    profile = make_profile(max_steps=7)
    agent = ReActAgent(make_test_config(
            profile=profile,
            llm=ScriptedLLM([]),
            memory=fake_memory(profile),
            tools=ToolRegistry(),
        ))
    assert agent._resolve_max_steps(None) == 7


# ---------- close + context manager ----------


async def test_close_is_idempotent():
    profile = make_profile()
    agent = ReActAgent(make_test_config(
            profile=profile,
            llm=ScriptedLLM([]),
            memory=fake_memory(profile),
            tools=ToolRegistry(),
        ))
    await agent.close()
    await agent.close()  # no error on second call


async def test_async_context_manager_closes_on_exit():
    profile = make_profile()
    agent = ReActAgent(make_test_config(
            profile=profile,
            llm=ScriptedLLM([resp(content="x")]),
            memory=fake_memory(profile),
            tools=ToolRegistry(),
            memory_recall_top_k=0,
            save_outcome=False,
            reflect_after_run=False,
        ))
    async with agent as managed:
        result = await managed.run("task", max_steps=2)
        assert result.final_answer == "x"
    await agent.close()


# ---------- from_profile (mem0 construction patched) ----------


async def _patched_from_profile(cls, profile: AgentProfile):
    """Run cls.from_profile with Mem0Memory's mem0 init patched out."""
    from DefenseAgent.memory.mem0_memory import Mem0Memory

    with patch.object(
        Mem0Memory,
        "_init_memory_obj",
        return_value=MagicMock(name="mem0"),
    ):
        return await cls.from_profile(profile, load_env=False)


async def test_from_profile_wires_every_component(monkeypatch: pytest.MonkeyPatch):
    """Agent.from_profile must construct every composed module against Maya's real bundle."""
    _set_env_for_real_construction(monkeypatch)
    profile = AgentProfile.from_yaml(_EXAMPLE_PROFILE)

    agent = await _patched_from_profile(ReActAgent, profile)
    try:
        assert agent.profile is profile
        assert agent.llm is not None
        assert agent.memory is not None
        assert agent.tools is not None
        assert agent.reflector is not None
        assert "tabular-report" in agent.tools
    finally:
        await agent.close()


async def test_from_profile_works_for_plan_and_solve(monkeypatch: pytest.MonkeyPatch):
    """from_profile must work on PlanAndSolveAgent too (same mechanism via classmethod)."""
    _set_env_for_real_construction(monkeypatch)
    profile = AgentProfile.from_yaml(_EXAMPLE_PROFILE)

    agent = await _patched_from_profile(PlanAndSolveAgent, profile)
    try:
        assert isinstance(agent, PlanAndSolveAgent)
        assert isinstance(agent, BaseAgent)
    finally:
        await agent.close()


# ---------- RAG wiring through from_profile ----------


async def test_from_profile_skips_rag_when_disabled(monkeypatch: pytest.MonkeyPatch):
    """profile.rag.enabled=False (Maya's default) → agent.rag is None and rag_search is not registered."""
    _set_env_for_real_construction(monkeypatch)
    profile = AgentProfile.from_yaml(_EXAMPLE_PROFILE)
    assert profile.rag.enabled is False

    agent = await _patched_from_profile(ReActAgent, profile)
    try:
        assert agent.rag is None
        from DefenseAgent.agent.base import RAG_SEARCH_TOOL_NAME
        assert RAG_SEARCH_TOOL_NAME not in agent._agent_tools
    finally:
        await agent.close()


async def test_from_profile_builds_rag_when_enabled(monkeypatch: pytest.MonkeyPatch):
    """profile.rag.enabled=True → LlamaIndexRAG.from_profile is invoked during async setup and the result wired onto the agent + rag_search registered."""
    from unittest.mock import AsyncMock

    _set_env_for_real_construction(monkeypatch)
    profile = AgentProfile.from_yaml(_EXAMPLE_PROFILE)
    profile.rag.enabled = True

    fake_rag = MagicMock(name="LlamaIndexRAG")
    fake_rag.retrieve = AsyncMock(return_value=[])

    from DefenseAgent.memory.mem0_memory import Mem0Memory
    from DefenseAgent.rag.llama_index_rag import LlamaIndexRAG
    from DefenseAgent.agent.base import RAG_SEARCH_TOOL_NAME

    with patch.object(
        Mem0Memory, "_init_memory_obj", return_value=MagicMock(name="mem0"),
    ):
        with patch.object(
            LlamaIndexRAG, "from_profile", AsyncMock(return_value=fake_rag),
        ):
            agent = await ReActAgent.from_profile(profile, load_env=False)

    try:
        assert agent.rag is fake_rag
        assert RAG_SEARCH_TOOL_NAME in agent._agent_tools
    finally:
        await agent.close()
