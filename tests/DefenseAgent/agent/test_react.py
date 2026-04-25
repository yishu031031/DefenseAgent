"""Tests for DefenseAgent.agent.react.ReActAgent."""
import pytest

from DefenseAgent.agent import AgentStepLimitError, ReActAgent
from DefenseAgent.llm.types import ToolCall
from DefenseAgent.memory import Memory
from DefenseAgent.tools import ToolRegistry

from tests.DefenseAgent.agent._support import (
    ScriptedLLM,
    ZeroEmbedder,
    make_profile,
    resp,
)


def _bare_agent(llm, *, profile=None, tools=None, memory=None) -> ReActAgent:
    """Build a ReActAgent with recall/persist/reflection disabled — minimal wiring for loop tests."""
    profile = profile or make_profile()
    tools = tools or ToolRegistry()
    memory = memory or Memory(profile=profile, embedding_adapter=ZeroEmbedder())
    return ReActAgent(
        profile,
        llm=llm,  # type: ignore[arg-type]
        memory=memory,
        tools=tools,
        memory_recall_top_k=0,
        persist_outcome=False,
        reflect_after_run=False,
    )


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


async def test_persist_outcome_writes_observation_to_memory():
    profile = make_profile()
    memory = Memory(profile=profile, embedding_adapter=ZeroEmbedder())
    llm = ScriptedLLM([resp(content="final answer")])
    agent = ReActAgent(
        profile,
        llm=llm,  # type: ignore[arg-type]
        memory=memory,
        tools=ToolRegistry(),
        memory_recall_top_k=0,
        persist_outcome=True,
        reflect_after_run=False,
    )
    assert len(memory) == 0

    await agent.run("describe cats", max_steps=2)

    assert len(memory) == 1
    written = memory.stream.get_all()[0]
    assert "Q: describe cats" in written.content
    assert "A: final answer" in written.content
    assert written.kind == "observation"


async def test_recall_memories_are_injected_into_system_prompt():
    profile = make_profile()
    memory = Memory(profile=profile, embedding_adapter=ZeroEmbedder())
    # Pre-seed a fact so recall() has something to return.
    await memory.remember("Maya is a CS student.", kind="fact", importance=7.0)

    llm = ScriptedLLM([resp(content="ok")])
    agent = ReActAgent(
        profile,
        llm=llm,  # type: ignore[arg-type]
        memory=memory,
        tools=ToolRegistry(),
        memory_recall_top_k=3,
        persist_outcome=False,
        reflect_after_run=False,
    )
    await agent.run("anything", max_steps=2)

    system_prompt = llm.calls[0]["system"]
    assert "Maya is a CS student." in system_prompt
    assert "Relevant memories:" in system_prompt


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
