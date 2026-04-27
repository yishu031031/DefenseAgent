"""Tests for DefenseAgent.agent.plan_and_solve.PlanAndSolveAgent."""
import pytest

from DefenseAgent.agent import AgentError, PlanAndSolveAgent
from DefenseAgent.agent.plan_and_solve import _parse_plan
from DefenseAgent.llm.types import ToolCall

from DefenseAgent.tools import ToolRegistry

from tests.DefenseAgent.agent._support import (
    ScriptedLLM,
    fake_memory,
    make_profile,
    make_test_config,
    resp,
)


def _bare_agent(llm, *, tools=None, max_substeps_per_step=3) -> PlanAndSolveAgent:
    """Build a PlanAndSolveAgent with recall/persist/reflection disabled for tight tests."""
    profile = make_profile()
    tools = tools or ToolRegistry()
    memory = fake_memory(profile)
    config = make_test_config(
        profile=profile,
        llm=llm,
        memory=memory,
        tools=tools,
        memory_recall_top_k=0,
        max_substeps_per_step=max_substeps_per_step,
        save_outcome=False,
        reflect_after_run=False,
    )
    return PlanAndSolveAgent(config)


# ---------- plan parser ----------


def test_parse_plan_handles_dot_and_paren_styles():
    text = "1. First\n2) Second\n  3.   Third with whitespace\n"
    assert _parse_plan(text) == ["First", "Second", "Third with whitespace"]


def test_parse_plan_ignores_non_numbered_lines():
    text = "Preamble\n1. Real step\nAside\n2. Another step\nEpilogue"
    assert _parse_plan(text) == ["Real step", "Another step"]


def test_parse_plan_empty_returns_empty():
    assert _parse_plan("") == []
    assert _parse_plan("no steps here") == []


# ---------- full flow ----------


async def test_plan_execute_synthesize_end_to_end():
    llm = ScriptedLLM(
        [
            resp(content="1. Compute x squared\n2. Format the answer"),
            # Step 1: model calls a tool
            resp(
                content="I'll call square.",
                tool_calls=[ToolCall(id="t1", name="square", arguments={"x": 3})],
            ),
            resp(content="Step 1 result: 9"),
            # Step 2: model answers directly
            resp(content="Step 2 result: formatted as '9'"),
            # Synthesis
            resp(content="The result is 9."),
        ]
    )
    registry = ToolRegistry()

    @registry.tool
    def square(x: int) -> int:
        """Squared."""
        return x * x

    agent = _bare_agent(llm, tools=registry)
    result = await agent.run("Compute 3 squared and format it.", max_steps=5)

    assert result.final_answer == "The result is 9."

    kinds = [s.kind for s in result.steps]
    assert kinds[0] == "plan"
    assert "tool_call" in kinds
    assert "tool_result" in kinds
    assert kinds[-1] == "answer"

    # Aggregate usage covers all 5 LLM calls.
    assert result.usage.total_tokens == 15 * 5


async def test_plan_step_count_capped_by_max_steps():
    # Plan has 5 steps but max_steps=2 → only 2 executed + 1 synthesis call.
    llm = ScriptedLLM(
        [
            resp(content="1. a\n2. b\n3. c\n4. d\n5. e"),
            resp(content="did a"),
            resp(content="did b"),
            resp(content="synthesis"),
        ]
    )
    agent = _bare_agent(llm)
    result = await agent.run("multi-step task", max_steps=2)

    assert result.final_answer == "synthesis"
    plan_step = next(s for s in result.steps if s.kind == "plan")
    assert plan_step.content.splitlines() == ["a", "b"]


# ---------- error paths ----------


async def test_empty_plan_raises_agent_error():
    llm = ScriptedLLM([resp(content="I don't know how to plan this.")])
    agent = _bare_agent(llm)
    with pytest.raises(AgentError):
        await agent.run("unparseable")


async def test_bad_plan_persists_failure_outcome():
    """Empty-plan failure records a FAILED outcome at importance 6.0 so reflection can see the failure."""
    profile = make_profile()
    memory = fake_memory(profile)
    llm = ScriptedLLM([resp(content="not a plan")])
    config = make_test_config(
        profile=profile,
        llm=llm,
        memory=memory,
        tools=ToolRegistry(),
        memory_recall_top_k=0,
        max_substeps_per_step=2,
        save_outcome=True,
        reflect_after_run=False,
    )
    agent = PlanAndSolveAgent(config)
    with pytest.raises(AgentError):
        await agent.run("unparseable")

    from tests.DefenseAgent.agent._support import added_calls
    failures = [c for c in added_calls(memory) if c["memory_type"] == "failure"]
    assert len(failures) == 1
    failure_msg = failures[0]["messages"][0]
    assert "Q: unparseable" in failure_msg.content
    assert "FAILED" in failure_msg.content


async def test_bad_plan_failure_skipped_when_save_outcome_false():
    """save_outcome=False disables the failure outcome write just like the success outcome."""
    profile = make_profile()
    memory = fake_memory(profile)
    llm = ScriptedLLM([resp(content="garbage")])
    config = make_test_config(
        profile=profile,
        llm=llm,
        memory=memory,
        tools=ToolRegistry(),
        memory_recall_top_k=0,
        max_substeps_per_step=2,
        save_outcome=False,
        reflect_after_run=False,
    )
    agent = PlanAndSolveAgent(config)
    with pytest.raises(AgentError):
        await agent.run("q")
    assert len(memory) == 0


async def test_substep_cap_returns_incomplete_marker_and_continues():
    # Step execution keeps emitting tool calls; substep cap kicks in.
    llm = ScriptedLLM(
        [
            resp(content="1. Do the thing"),
            resp(
                content="",
                tool_calls=[ToolCall(id="t", name="noop", arguments={})],
            ),
            resp(
                content="",
                tool_calls=[ToolCall(id="t", name="noop", arguments={})],
            ),
            # Synthesis LLM call
            resp(content="final answer"),
        ]
    )
    registry = ToolRegistry()

    @registry.tool
    def noop() -> str:
        """no-op"""
        return "x"

    agent = _bare_agent(llm, tools=registry, max_substeps_per_step=2)
    result = await agent.run("loopy task")

    assert result.final_answer == "final answer"
    # Synthesis prompt received an "incomplete" marker for step 1.
    # We verify by inspecting the last recorded call's user message.
    last_call = llm.calls[-1]
    user_msg = last_call["messages"][0].content
    assert "step 1 incomplete" in user_msg


# ---------- wiring ----------


async def test_plan_system_prompt_contains_identity_but_not_exec_instructions():
    llm = ScriptedLLM(
        [
            resp(content="1. single step"),
            resp(content="done"),
            resp(content="synthesis"),
        ]
    )
    agent = _bare_agent(llm)
    await agent.run("q")

    # The planning call is the first; its system prompt must NOT contain the
    # per-step execution instruction (which only appears for phase 2 calls).
    planning_system = llm.calls[0]["system"]
    assert "You are Tester" in planning_system
    assert "executing ONE step" not in planning_system

    # The execution call (second) MUST have the exec instruction.
    exec_system = llm.calls[1]["system"]
    assert "executing ONE step" in exec_system


async def test_save_outcome_stores_final_answer():
    profile = make_profile()
    memory = fake_memory(profile)
    llm = ScriptedLLM(
        [
            resp(content="1. do it"),
            resp(content="did it"),
            resp(content="final"),
        ]
    )
    config = make_test_config(
        profile=profile,
        llm=llm,
        memory=memory,
        tools=ToolRegistry(),
        memory_recall_top_k=0,
        max_substeps_per_step=2,
        save_outcome=True,
        reflect_after_run=False,
    )
    agent = PlanAndSolveAgent(config)
    await agent.run("task A")

    from tests.DefenseAgent.agent._support import added_calls
    outcomes = [c for c in added_calls(memory) if c["memory_type"] == "outcome"]
    assert len(outcomes) == 1
    msg = outcomes[0]["messages"][0]
    assert "Q: task A" in msg.content
    assert "A: final" in msg.content
