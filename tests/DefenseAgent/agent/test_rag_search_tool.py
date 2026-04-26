"""Tests for the Agent-owned `rag_search` tool — the live knowledge-base access exposed to the LLM mid-loop.

Mirrors `test_memory_recall_tool.py` but stubs the RAG backend with a MagicMock instead of touching llama-index.
"""
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from DefenseAgent.agent import (
    MEMORY_RECALL_TOOL_NAME,
    RAG_SEARCH_TOOL_NAME,
    ReActAgent,
    SimpleAgent,
)
from DefenseAgent.llm.types import ToolCall
from DefenseAgent.tools import ToolRegistry

from tests.DefenseAgent.agent._support import (
    ScriptedLLM,
    fake_memory,
    make_profile,
    resp,
    make_test_config,
)


def _fake_rag(hits: list[dict[str, Any]] | None = None) -> Any:
    """Build a MagicMock standing in for LlamaIndexRAG with an awaitable retrieve()."""
    rag = MagicMock(name="LlamaIndexRAG")
    rag.retrieve = AsyncMock(return_value=list(hits or []))
    return rag


# ---------- registration / spec exposure ----------


def test_rag_search_absent_when_no_rag_backend():
    profile = make_profile()
    agent = ReActAgent(make_test_config(
            profile=profile,
            llm=ScriptedLLM([]),
            memory=fake_memory(profile),
            tools=ToolRegistry(),
        ))
    specs = agent._combined_tool_specs() or []
    assert RAG_SEARCH_TOOL_NAME not in [s["name"] for s in specs]
    assert RAG_SEARCH_TOOL_NAME not in agent._agent_tools


def test_rag_search_appears_after_memory_recall_when_rag_wired():
    profile = make_profile()
    agent = ReActAgent(make_test_config(
            profile=profile,
            llm=ScriptedLLM([]),
            memory=fake_memory(profile),
            tools=ToolRegistry(),
            rag=_fake_rag(),
        ))
    specs = agent._combined_tool_specs()
    assert specs is not None
    names = [s["name"] for s in specs]
    # Order: user tools (none here) → memory_recall → rag_search.
    assert names == [MEMORY_RECALL_TOOL_NAME, RAG_SEARCH_TOOL_NAME]
    rag_spec = next(s for s in specs if s["name"] == RAG_SEARCH_TOOL_NAME)
    assert rag_spec["input_schema"]["required"] == ["query"]
    assert "top_k" in rag_spec["input_schema"]["properties"]


def test_rag_search_registered_on_simple_agent_too():
    profile = make_profile()
    agent = SimpleAgent(make_test_config(
            profile=profile,
            llm=ScriptedLLM([]),
            memory=fake_memory(profile),
            tools=ToolRegistry(),
            rag=_fake_rag(),
        ))
    assert RAG_SEARCH_TOOL_NAME in agent._agent_tools


def test_react_prompt_mentions_rag_search_only_when_rag_present():
    profile = make_profile()
    no_rag = ReActAgent(make_test_config(
            profile=profile,
            llm=ScriptedLLM([]),
            memory=fake_memory(profile),
            tools=ToolRegistry(),
        ))
    assert "rag_search" not in no_rag._build_system_prompt()

    with_rag = ReActAgent(make_test_config(
            profile=profile,
            llm=ScriptedLLM([]),
            memory=fake_memory(profile),
            tools=ToolRegistry(),
            rag=_fake_rag(),
        ))
    assert "rag_search" in with_rag._build_system_prompt()


# ---------- handler dispatch ----------


async def test_handle_rag_search_returns_formatted_hits():
    profile = make_profile()
    rag = _fake_rag(
        hits=[
            {"text": "Trees are O(log n) on average.", "score": 0.91},
            {"text": "B+ trees keep all data in leaves.", "score": 0.78},
        ]
    )
    agent = ReActAgent(make_test_config(
            profile=profile,
            llm=ScriptedLLM([]),
            memory=fake_memory(profile),
            tools=ToolRegistry(),
            rag=rag,
        ))
    out = await agent._handle_rag_search({"query": "B+ tree complexity", "top_k": 3})

    assert "Trees are O(log n) on average." in out
    assert "score=0.91" in out
    rag.retrieve.assert_awaited_once()
    args, kwargs = rag.retrieve.call_args
    assert args == ("B+ tree complexity",)
    assert kwargs["limit"] == 3
    assert kwargs["score_threshold"] == profile.rag.score_threshold


async def test_handle_rag_search_empty_query_diagnostic():
    profile = make_profile()
    rag = _fake_rag()
    agent = ReActAgent(make_test_config(
            profile=profile,
            llm=ScriptedLLM([]),
            memory=fake_memory(profile),
            tools=ToolRegistry(),
            rag=rag,
        ))
    out = await agent._handle_rag_search({"query": "   "})
    assert "empty query" in out
    rag.retrieve.assert_not_awaited()


async def test_handle_rag_search_no_hits_diagnostic():
    profile = make_profile()
    rag = _fake_rag(hits=[])
    agent = ReActAgent(make_test_config(
            profile=profile,
            llm=ScriptedLLM([]),
            memory=fake_memory(profile),
            tools=ToolRegistry(),
            rag=rag,
        ))
    out = await agent._handle_rag_search({"query": "obscure topic"})
    assert "no documents matched" in out


async def test_handle_rag_search_swallows_backend_error():
    profile = make_profile()
    rag = _fake_rag()
    rag.retrieve.side_effect = RuntimeError("vector store offline")
    agent = ReActAgent(make_test_config(
            profile=profile,
            llm=ScriptedLLM([]),
            memory=fake_memory(profile),
            tools=ToolRegistry(),
            rag=rag,
        ))
    out = await agent._handle_rag_search({"query": "anything"})
    assert "rag_search failed" in out
    assert "vector store offline" in out


async def test_handle_rag_search_top_k_defaults_and_clamps():
    """top_k missing → falls back to profile.rag.top_k; out-of-range clamps to [1, 20]."""
    profile = make_profile()
    rag = _fake_rag()
    agent = ReActAgent(make_test_config(
            profile=profile,
            llm=ScriptedLLM([]),
            memory=fake_memory(profile),
            tools=ToolRegistry(),
            rag=rag,
        ))

    await agent._handle_rag_search({"query": "x"})
    assert rag.retrieve.await_args.kwargs["limit"] == profile.rag.top_k

    await agent._handle_rag_search({"query": "x", "top_k": 999})
    assert rag.retrieve.await_args.kwargs["limit"] == 20

    await agent._handle_rag_search({"query": "x", "top_k": 0})
    assert rag.retrieve.await_args.kwargs["limit"] == 1


async def test_llm_can_invoke_rag_search_through_dispatch():
    """End-to-end: LLM emits a ToolCall for rag_search, _dispatch_tool_calls routes it to our handler, and the result is a tool-role message."""
    profile = make_profile()
    rag = _fake_rag(
        hits=[{"text": "A heap supports O(log n) push/pop.", "score": 0.88}]
    )
    agent = ReActAgent(make_test_config(
            profile=profile,
            llm=ScriptedLLM([]),
            memory=fake_memory(profile),
            tools=ToolRegistry(),
            rag=rag,
        ))
    tc = ToolCall(id="t1", name=RAG_SEARCH_TOOL_NAME, arguments={"query": "heap"})

    [result] = await agent._dispatch_tool_calls([tc])
    assert result.role == "tool"
    assert result.tool_call_id == "t1"
    assert "heap" in result.content.lower()
    rag.retrieve.assert_awaited_once()
