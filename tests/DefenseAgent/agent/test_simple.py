"""Tests for SimpleAgent — single-turn LLM strategy.

Covers:
  • One LLM call per `run()` (no loop)
  • Identity prompt is supplied as `system`
  • Memory condensation runs before the call
  • Outcome persistence with success and failure tagging
  • Reflection fires on every exit path (success / failure)
  • `extra_instructions` appends below the identity
"""
from unittest.mock import AsyncMock, MagicMock

import pytest

from DefenseAgent.agent import BaseAgent, SimpleAgent
from DefenseAgent.tools import ToolRegistry

from tests.DefenseAgent.agent._support import (
    ScriptedLLM,
    fake_memory,
    make_profile,
    make_test_config,
    resp,
)


def _make_agent(
    *,
    llm=None,
    memory=None,
    save_outcome: bool = True,
    reflect_after_run: bool = True,
    extra_instructions: str | None = None,
    reflector=None,
) -> SimpleAgent:
    """Build a SimpleAgent with stubbed deps for offline tests."""
    profile = make_profile()
    config = make_test_config(
        profile=profile,
        llm=llm or ScriptedLLM([resp(content="hi")]),
        memory=memory or fake_memory(profile),
        tools=ToolRegistry(),
        reflector=reflector,
        save_outcome=save_outcome,
        reflect_after_run=reflect_after_run,
        extra_instructions=extra_instructions,
    )
    return SimpleAgent(config)


def test_simple_agent_inherits_base_agent():
    agent = _make_agent()
    assert isinstance(agent, BaseAgent)


async def test_run_makes_exactly_one_llm_call():
    llm = ScriptedLLM([resp(content="answer")])
    agent = _make_agent(
        llm=llm, save_outcome=False, reflect_after_run=False,
    )
    result = await agent.run("hello")
    assert result.final_answer == "answer"
    assert len(llm.calls) == 1
    assert len(result.steps) == 1
    assert result.steps[0].kind == "answer"


async def test_run_passes_identity_prompt_as_system():
    llm = ScriptedLLM([resp(content="ok")])
    agent = _make_agent(
        llm=llm, save_outcome=False, reflect_after_run=False,
    )
    await agent.run("ping")
    system = llm.calls[0]["system"]
    assert system is not None
    assert "Tester" in system  # from make_profile() default name


async def test_extra_instructions_appended_to_system_prompt():
    llm = ScriptedLLM([resp(content="ok")])
    agent = _make_agent(
        llm=llm,
        extra_instructions="ALWAYS BE TERSE.",
        save_outcome=False,
        reflect_after_run=False,
    )
    await agent.run("ping")
    assert llm.calls[0]["system"].rstrip().endswith("ALWAYS BE TERSE.")


async def test_run_invokes_memory_condensation_before_chat():
    profile = make_profile()
    memory = fake_memory(profile)
    config = make_test_config(
        profile=profile,
        llm=ScriptedLLM([resp(content="ok")]),
        memory=memory,
        tools=ToolRegistry(),
        save_outcome=False,
        reflect_after_run=False,
    )
    agent = SimpleAgent(config)
    await agent.run("hi")
    assert memory.run.await_count == 1


async def test_save_outcome_writes_success_record():
    memory = fake_memory()
    agent = _make_agent(memory=memory, reflect_after_run=False)
    await agent.run("question")
    assert memory.add.await_count == 1
    call = memory.add.await_args
    assert call.kwargs["memory_type"] == "outcome"


async def test_save_outcome_writes_failure_record_when_llm_raises():
    memory = fake_memory()
    failing_llm = MagicMock()
    failing_llm.chat = AsyncMock(side_effect=RuntimeError("boom"))
    agent = _make_agent(
        llm=failing_llm, memory=memory, reflect_after_run=False,
    )
    with pytest.raises(RuntimeError):
        await agent.run("question")
    assert memory.add.await_count == 1
    assert memory.add.await_args.kwargs["memory_type"] == "failure"


async def test_save_outcome_disabled_skips_writes():
    memory = fake_memory()
    agent = _make_agent(
        memory=memory, save_outcome=False, reflect_after_run=False,
    )
    await agent.run("question")
    assert memory.add.await_count == 0


async def test_reflect_after_run_runs_on_success():
    reflector = MagicMock()
    reflector.maybe_reflect = AsyncMock(return_value=[])
    agent = _make_agent(
        reflector=reflector, save_outcome=False,
    )
    await agent.run("question")
    assert reflector.maybe_reflect.await_count == 1


async def test_reflect_after_run_runs_on_failure_too():
    """Reflection must fire even when the LLM call raises — failure traces still feed insight."""
    reflector = MagicMock()
    reflector.maybe_reflect = AsyncMock(return_value=[])
    failing_llm = MagicMock()
    failing_llm.chat = AsyncMock(side_effect=RuntimeError("boom"))
    agent = _make_agent(
        llm=failing_llm, reflector=reflector, save_outcome=False,
    )
    with pytest.raises(RuntimeError):
        await agent.run("question")
    assert reflector.maybe_reflect.await_count == 1


async def test_max_steps_argument_is_accepted_but_ignored():
    """The interface accepts max_steps for uniformity with ReAct/PlanAndSolve, but SimpleAgent has no loop to cap."""
    llm = ScriptedLLM([resp(content="ok")])
    agent = _make_agent(
        llm=llm, save_outcome=False, reflect_after_run=False,
    )
    result = await agent.run("ping", max_steps=99)
    assert result.final_answer == "ok"
    assert len(llm.calls) == 1
