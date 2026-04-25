"""Tests for the Agent base class — from_profile wiring, max_steps resolution, close lifecycle."""
from pathlib import Path

import pytest

from DefenseAgent.agent import Agent, PlanAndSolveAgent, ReActAgent
from DefenseAgent.config import AgentProfile
from DefenseAgent.memory import Memory
from DefenseAgent.tools import ToolRegistry

from tests.DefenseAgent.agent._support import ScriptedLLM, ZeroEmbedder, make_profile, resp


_MAYA_PROFILE = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "agents" / "maya_rodriguez" / "profile.yaml"
)


# ---------- abstract contract ----------


def test_agent_base_class_cannot_be_instantiated():
    profile = make_profile()
    memory = Memory(profile=profile, embedding_adapter=ZeroEmbedder())
    with pytest.raises(TypeError):
        Agent(  # type: ignore[abstract]
            profile, llm=ScriptedLLM([]), memory=memory, tools=ToolRegistry(),
        )


# ---------- max_steps resolution ----------


def test_resolve_max_steps_uses_explicit_override():
    profile = make_profile(max_steps=10)
    agent = ReActAgent(
        profile,
        llm=ScriptedLLM([]),  # type: ignore[arg-type]
        memory=Memory(profile=profile, embedding_adapter=ZeroEmbedder()),
        tools=ToolRegistry(),
    )
    assert agent._resolve_max_steps(3) == 3
    assert agent._resolve_max_steps(None) == 10


def test_resolve_max_steps_reads_from_profile_when_no_override():
    profile = make_profile(max_steps=7)
    agent = ReActAgent(
        profile,
        llm=ScriptedLLM([]),  # type: ignore[arg-type]
        memory=Memory(profile=profile, embedding_adapter=ZeroEmbedder()),
        tools=ToolRegistry(),
    )
    assert agent._resolve_max_steps(None) == 7


# ---------- close + context manager ----------


async def test_close_is_idempotent():
    profile = make_profile()
    agent = ReActAgent(
        profile,
        llm=ScriptedLLM([]),  # type: ignore[arg-type]
        memory=Memory(profile=profile, embedding_adapter=ZeroEmbedder()),
        tools=ToolRegistry(),
    )
    await agent.close()
    await agent.close()  # no error on second call


async def test_async_context_manager_closes_on_exit():
    profile = make_profile()
    agent = ReActAgent(
        profile,
        llm=ScriptedLLM([resp(content="x")]),  # type: ignore[arg-type]
        memory=Memory(profile=profile, embedding_adapter=ZeroEmbedder()),
        tools=ToolRegistry(),
        memory_recall_top_k=0,
        persist_outcome=False,
        reflect_after_run=False,
    )
    async with agent as managed:
        result = await managed.run("task", max_steps=2)
        assert result.final_answer == "x"
    # After exit: close has been called. Calling it again should still succeed.
    await agent.close()


# ---------- from_profile (real Maya bundle) ----------


async def test_from_profile_wires_every_component(monkeypatch: pytest.MonkeyPatch):
    """Agent.from_profile must construct every composed module against Maya's real bundle."""
    # Make env valid so LLM.from_env + Memory.from_env pass validation; we don't
    # actually call the network (no agent.run() in this test).
    monkeypatch.setenv("AGENT_LAB_LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.example.com")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-chat")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("EMBEDDING_API_KEY", "sk-test")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")

    profile = AgentProfile.from_yaml(_MAYA_PROFILE)
    agent = await ReActAgent.from_profile(profile, persist_memory=False, load_env=False)
    try:
        assert agent.profile is profile
        assert agent.llm is not None
        assert agent.memory is not None
        assert agent.tools is not None
        assert agent.reflector is not None
        # Maya's profile declares one skill (tabular-report).
        assert "tabular-report" in agent.tools
    finally:
        await agent.close()


async def test_from_profile_works_for_plan_and_solve(monkeypatch: pytest.MonkeyPatch):
    """from_profile must work on PlanAndSolveAgent too (same mechanism via classmethod)."""
    monkeypatch.setenv("AGENT_LAB_LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.example.com")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-chat")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("EMBEDDING_API_KEY", "sk-test")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")

    profile = AgentProfile.from_yaml(_MAYA_PROFILE)
    agent = await PlanAndSolveAgent.from_profile(profile, persist_memory=False, load_env=False)
    try:
        assert isinstance(agent, PlanAndSolveAgent)
        assert isinstance(agent, Agent)
    finally:
        await agent.close()
