"""Tests for DefenseAgent.agent.react.ReActAgent."""
import pytest

from DefenseAgent.agent import AgentStepLimitError, ReActAgent
from DefenseAgent.llm.types import ToolCall

from DefenseAgent.tools import ToolRegistry

from tests.DefenseAgent.agent._support import (
    ScriptedLLM,
    fake_memory,
    make_profile,
    make_test_config,
    resp,
)


def _bare_agent(llm, *, profile=None, tools=None, memory=None) -> ReActAgent:
    """Build a ReActAgent with recall/persist/reflection disabled — minimal wiring for loop tests."""
    profile = profile or make_profile()
    tools = tools or ToolRegistry()
    memory = memory or fake_memory(profile)
    config = make_test_config(
        profile=profile,
        llm=llm,
        memory=memory,
        tools=tools,
        memory_recall_top_k=0,
        save_outcome=False,
        reflect_after_run=False,
    )
    return ReActAgent(config)


# ---------- happy paths ----------


async def test_direct_answer_when_llm_emits_no_tool_calls():
    llm = ScriptedLLM([resp(content="The answer is 42.")])
    agent = _bare_agent(llm)

    result = await agent.run("What's the answer?", max_steps=5)

    assert result.final_answer == "The answer is 42."
    assert len(result.steps) == 1
    assert result.steps[0].kind == "answer"
    assert result.stopped_reason == "answered"
    assert result.usage.total_tokens == 15


async def test_executes_tool_call_then_answers():
    llm = ScriptedLLM(
        [
            resp(
                content="Let me compute.",
                tool_calls=[ToolCall(id="tc1", name="square", arguments={"x": 5})],
            ),
            resp(content="The answer is 25."),
        ]
    )
    registry = ToolRegistry()

    @registry.tool
    def square(x: int) -> int:
        """Squared."""
        return x * x

    agent = _bare_agent(llm, tools=registry)
    result = await agent.run("Square 5.", max_steps=5)

    assert result.final_answer == "The answer is 25."
    assert [s.kind for s in result.steps] == ["tool_call", "tool_result", "answer"]
    assert result.steps[0].tool_calls[0].name == "square"
    assert result.steps[1].tool_results[0].content == "25"
    # Token usage accumulates across both LLM calls.
    assert result.usage.total_tokens == 30


async def test_multiple_tool_calls_in_one_response_all_execute():
    llm = ScriptedLLM(
        [
            resp(
                content="",
                tool_calls=[
                    ToolCall(id="a", name="echo", arguments={"text": "one"}),
                    ToolCall(id="b", name="echo", arguments={"text": "two"}),
                ],
            ),
            resp(content="done"),
        ]
    )
    registry = ToolRegistry()

    @registry.tool
    def echo(text: str) -> str:
        """Echo the text."""
        return text

    agent = _bare_agent(llm, tools=registry)
    result = await agent.run("echo both", max_steps=5)

    assert result.final_answer == "done"
    tool_result_step = next(s for s in result.steps if s.kind == "tool_result")
    contents = [m.content for m in tool_result_step.tool_results]
    assert contents == ["one", "two"]


# ---------- max_steps / failure ----------


async def test_max_steps_exhausted_raises():
    def never_ending():
        return resp(
            content="",
            tool_calls=[ToolCall(id="t", name="square", arguments={"x": 1})],
        )

    llm = ScriptedLLM([never_ending() for _ in range(3)])
    registry = ToolRegistry()

    @registry.tool
    def square(x: int) -> int:
        """Squared."""
        return x * x

    agent = _bare_agent(llm, tools=registry)
    with pytest.raises(AgentStepLimitError):
        await agent.run("loop forever", max_steps=3)


# ---------- memory + reflection wiring ----------


async def test_save_outcome_writes_observation_to_memory():
    profile = make_profile()
    memory = fake_memory(profile)
    llm = ScriptedLLM([resp(content="final answer")])
    config = make_test_config(
        profile=profile,
        llm=llm,
        memory=memory,
        tools=ToolRegistry(),
        memory_recall_top_k=0,
        save_outcome=True,
        reflect_after_run=False,
    )
    agent = ReActAgent(config)

    await agent.run("describe cats", max_steps=2)

    from tests.DefenseAgent.agent._support import added_calls
    calls = added_calls(memory)
    outcome = next(c for c in calls if c["memory_type"] == "outcome")
    written = outcome["messages"][0]
    assert "Q: describe cats" in written.content
    assert "A: final answer" in written.content


async def test_condense_memory_chain_runs_memory_then_compressor():
    """Memory tools are pipelined in registration order; each one's output feeds the next."""
    from unittest.mock import AsyncMock, MagicMock

    profile = make_profile()
    memory = fake_memory(profile)

    fake_compressor = MagicMock(name="ContextCompressor")
    fake_compressor.run = AsyncMock(side_effect=lambda msgs, **kw: msgs)

    llm = ScriptedLLM([resp(content="ok")])
    config = make_test_config(
        profile=profile,
        llm=llm,
        memory=memory,
        tools=ToolRegistry(),
        compressor=fake_compressor,
        memory_recall_top_k=0,
        save_outcome=False,
        save_trajectory=False,
        reflect_after_run=False,
    )
    agent = ReActAgent(config)
    # P2: builder wraps the bare memory in a MemoryOrchestrator before placing
    # it in the chain — the orchestrator's `.run()` delegates to the wrapped
    # persistent backend (the mock), so awaits propagate as expected.
    assert agent.memory_tools[0] is agent.memory
    assert agent.memory_tools[1] is fake_compressor
    assert agent.memory.persistent is memory  # type: ignore[union-attr]

    await agent.run("anything", max_steps=2)

    # Both tools were invoked at least once before the LLM call.
    assert memory.run.await_count >= 1
    assert fake_compressor.run.await_count >= 1


async def test_condense_memory_swallows_tool_errors_and_continues():
    """A misbehaving memory tool must not crash the agent loop — the chain logs and skips it."""
    from unittest.mock import AsyncMock

    profile = make_profile()
    memory = fake_memory(profile)
    memory.run = AsyncMock(side_effect=RuntimeError("memory blew up"))

    llm = ScriptedLLM([resp(content="answer")])
    config = make_test_config(
        profile=profile,
        llm=llm,
        memory=memory,
        tools=ToolRegistry(),
        memory_recall_top_k=0,
        save_outcome=False,
        save_trajectory=False,
        reflect_after_run=False,
    )
    agent = ReActAgent(config)
    result = await agent.run("anything", max_steps=2)

    # Run still completed despite the broken memory tool.
    assert result.final_answer == "answer"


async def test_memory_run_is_invoked_on_every_loop_turn():
    """The condense_memory chain calls memory.run(messages) before each LLM call — that's where injection happens now."""
    profile = make_profile()
    memory = fake_memory(profile)

    llm = ScriptedLLM([resp(content="ok")])
    config = make_test_config(
        profile=profile,
        llm=llm,
        memory=memory,
        tools=ToolRegistry(),
        memory_recall_top_k=3,
        save_outcome=False,
        reflect_after_run=False,
    )
    agent = ReActAgent(config)
    await agent.run("anything", max_steps=2)

    # The chain must have asked memory.run() at least once before the LLM call.
    assert memory.run.await_count >= 1
    # Identity/instructions are still in the static system prompt.
    system_prompt = llm.calls[0]["system"]
    assert "You are Tester" in system_prompt
    assert "memory_recall" in system_prompt.lower()


# ---------- system-prompt shape ----------


async def test_system_prompt_contains_identity_and_instructions():
    llm = ScriptedLLM([resp(content="done")])
    agent = _bare_agent(llm)
    await agent.run("task", max_steps=2)

    prompt = llm.calls[0]["system"]
    assert "You are Tester" in prompt
    assert "25-year-old" in prompt
    # ReAct instruction block — any of these anchors is enough.
    assert "memory_recall" in prompt.lower() or "call tools" in prompt.lower()


async def test_memory_recall_is_always_in_forwarded_tool_specs():
    """Agent-owned memory_recall is always present, even with an empty user registry."""
    llm_empty = ScriptedLLM([resp(content="done")])
    empty_agent = _bare_agent(llm_empty)
    await empty_agent.run("task", max_steps=2)
    specs = llm_empty.calls[0]["tools"]
    assert specs is not None
    assert [s["name"] for s in specs] == ["memory_recall"]

    registry = ToolRegistry()

    @registry.tool
    def noop() -> str:
        """no-op"""
        return "ok"

    llm_full = ScriptedLLM([resp(content="done")])
    full_agent = _bare_agent(llm_full, tools=registry)
    await full_agent.run("task", max_steps=2)
    specs_full = llm_full.calls[0]["tools"]
    # User tools appear first, then agent-owned built-ins.
    assert [s["name"] for s in specs_full] == ["noop", "memory_recall"]


# ---------- context manager ----------


async def test_context_manager_closes_memory_and_tools():
    llm = ScriptedLLM([resp(content="done")])
    agent = _bare_agent(llm)

    async with agent as managed:
        assert managed is agent
        await agent.run("q", max_steps=2)

    # After exit, memory's SQLite (if any) is closed; no explicit assertion
    # needed — Memory.close() is a no-op when db_path is None, and this test
    # uses the in-memory Memory.
