"""Tests for the Agent-owned memory_recall tool — the live memory access the LLM can call mid-loop."""
import pytest

from DefenseAgent.agent import ReActAgent
from DefenseAgent.agent.base import MEMORY_RECALL_TOOL_NAME
from DefenseAgent.llm.types import ToolCall

from DefenseAgent.tools import ToolRegistry

from tests.DefenseAgent.agent._support import (
    ScriptedLLM,
    fake_memory,
    make_profile,
    make_test_config,
    resp,
)


# ---------- spec exposure ----------


def test_memory_recall_appears_in_combined_tool_specs():
    profile = make_profile()
    memory = fake_memory(profile)
    agent = ReActAgent(make_test_config(
        profile=profile, llm=ScriptedLLM([]), memory=memory, tools=ToolRegistry(),
    ))
    specs = agent._combined_tool_specs()
    assert specs is not None
    names = [s["name"] for s in specs]
    assert MEMORY_RECALL_TOOL_NAME in names
    # Schema has the expected shape.
    recall_spec = next(s for s in specs if s["name"] == MEMORY_RECALL_TOOL_NAME)
    assert recall_spec["input_schema"]["required"] == ["query"]
    assert "query" in recall_spec["input_schema"]["properties"]
    assert "top_k" in recall_spec["input_schema"]["properties"]


def test_user_tools_appear_before_agent_builtins_in_spec():
    profile = make_profile()
    memory = fake_memory(profile)
    registry = ToolRegistry()

    @registry.tool
    def first() -> str:
        """first"""
        return "1"

    @registry.tool
    def second() -> str:
        """second"""
        return "2"

    agent = ReActAgent(make_test_config(
        profile=profile, llm=ScriptedLLM([]), memory=memory, tools=registry,
    ))
    specs = agent._combined_tool_specs()
    assert specs is not None
    assert [s["name"] for s in specs] == ["first", "second", "memory_recall"]


# ---------- dispatch ----------


async def test_llm_can_invoke_memory_recall_and_receive_hits():
    profile = make_profile()
    memory = fake_memory(profile)
    memory.search_records.return_value = [
        {"memory": "Maya prefers studying in the library.", "memory_type": "preference"},
    ]

    llm = ScriptedLLM(
        [
            resp(
                content="Let me check memory.",
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name=MEMORY_RECALL_TOOL_NAME,
                        arguments={"query": "where does Maya study?"},
                    )
                ],
            ),
            resp(content="She prefers the library."),
        ]
    )
    agent = ReActAgent(make_test_config(
        profile=profile, llm=llm, memory=memory, tools=ToolRegistry(),
        memory_recall_top_k=0, save_outcome=False,
        save_trajectory=False, reflect_after_run=False,
    ))

    result = await agent.run("study location?", max_steps=5)
    assert result.final_answer == "She prefers the library."

    tool_result_step = next(s for s in result.steps if s.kind == "tool_result")
    # The dispatch fed the tool result back as a role="tool" message with name=memory_recall.
    assert tool_result_step.tool_results[0].name == MEMORY_RECALL_TOOL_NAME
    assert "prefers studying in the library" in tool_result_step.tool_results[0].content


async def test_memory_recall_empty_returns_diagnostic_not_crash():
    profile = make_profile()
    memory = fake_memory(profile)
    llm = ScriptedLLM(
        [
            resp(
                content="",
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name=MEMORY_RECALL_TOOL_NAME,
                        arguments={"query": "anything"},
                    )
                ],
            ),
            resp(content="Nothing found, answering from priors."),
        ]
    )
    agent = ReActAgent(make_test_config(
        profile=profile, llm=llm, memory=memory, tools=ToolRegistry(),
        memory_recall_top_k=0, save_outcome=False,
        save_trajectory=False, reflect_after_run=False,
    ))

    result = await agent.run("q", max_steps=3)
    tool_result_step = next(s for s in result.steps if s.kind == "tool_result")
    assert "no memories matched" in tool_result_step.tool_results[0].content


async def test_memory_recall_empty_query_is_handled_gracefully():
    profile = make_profile()
    memory = fake_memory(profile)
    llm = ScriptedLLM(
        [
            resp(
                content="",
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name=MEMORY_RECALL_TOOL_NAME,
                        arguments={"query": ""},
                    )
                ],
            ),
            resp(content="done"),
        ]
    )
    agent = ReActAgent(make_test_config(
        profile=profile, llm=llm, memory=memory, tools=ToolRegistry(),
        memory_recall_top_k=0, save_outcome=False,
        save_trajectory=False, reflect_after_run=False,
    ))
    result = await agent.run("q", max_steps=3)
    tool_result_step = next(s for s in result.steps if s.kind == "tool_result")
    assert "empty query" in tool_result_step.tool_results[0].content


async def test_memory_recall_top_k_is_clamped():
    """top_k is coerced to int and clamped to [1, 20]; extreme values must not crash."""
    profile = make_profile()
    memory = fake_memory(profile)
    agent = ReActAgent(make_test_config(
        profile=profile, llm=ScriptedLLM([]), memory=memory, tools=ToolRegistry(),
    ))

    # top_k=999 → clamped to 20 internally; handler returns the empty-match diagnostic.
    out = await agent._handle_memory_recall({"query": "anything", "top_k": 999})
    assert "no memories matched" in out

    # top_k=0 → clamped up to 1; handler still safe.
    out_zero = await agent._handle_memory_recall({"query": "anything", "top_k": 0})
    assert "no memories matched" in out_zero

    # Non-integer top_k → falls back to default 5, no crash.
    out_bad = await agent._handle_memory_recall({"query": "anything", "top_k": "abc"})
    assert "no memories matched" in out_bad


# ---------- dispatch preserves ordering across mixed tool calls ----------


async def test_dispatch_preserves_order_across_user_and_agent_tools():
    profile = make_profile()
    memory = fake_memory(profile)
    memory.search_records.return_value = [
        {"memory": "cached fact", "memory_type": "fact"},
    ]

    registry = ToolRegistry()

    @registry.tool
    def echo(text: str) -> str:
        """echo"""
        return f"echoed: {text}"

    llm = ScriptedLLM(
        [
            resp(
                content="",
                tool_calls=[
                    # Order: user tool, agent tool, user tool
                    ToolCall(id="c1", name="echo", arguments={"text": "a"}),
                    ToolCall(
                        id="c2",
                        name=MEMORY_RECALL_TOOL_NAME,
                        arguments={"query": "cached"},
                    ),
                    ToolCall(id="c3", name="echo", arguments={"text": "b"}),
                ],
            ),
            resp(content="done"),
        ]
    )
    agent = ReActAgent(make_test_config(
        profile=profile, llm=llm, memory=memory, tools=registry,
        memory_recall_top_k=0, save_outcome=False,
        save_trajectory=False, reflect_after_run=False,
    ))
    result = await agent.run("multi", max_steps=3)
    tool_result_step = next(s for s in result.steps if s.kind == "tool_result")
    names = [m.name for m in tool_result_step.tool_results]
    assert names == ["echo", MEMORY_RECALL_TOOL_NAME, "echo"]
    assert tool_result_step.tool_results[0].content == "echoed: a"
    assert "cached fact" in tool_result_step.tool_results[1].content
    assert tool_result_step.tool_results[2].content == "echoed: b"
