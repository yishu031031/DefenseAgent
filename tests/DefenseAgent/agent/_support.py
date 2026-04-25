"""Shared stubs for the agent test suite: a scripted LLM, a fake DefaultMemory, profile factories."""
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from DefenseAgent.config import AgentProfile
from DefenseAgent.llm.types import LLMResponse, Message, TokenUsage, ToolCall


class ScriptedLLM:
    """LLM stub that plays back a pre-built list of LLMResponse objects; records every chat() call for assertions."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        """Store the scripted responses (copied) and the empty call log."""
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def chat(
        self,
        messages,
        *,
        system=None,
        tools=None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Record the call, pop and return the next scripted response; raises when the script is exhausted."""
        self.calls.append(
            {
                "messages": list(messages),
                "system": system,
                "tools": tools,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        if not self._responses:
            raise AssertionError(
                "ScriptedLLM ran out of responses — check the test's expected call count."
            )
        return self._responses.pop(0)


def make_profile(max_steps: int = 10) -> AgentProfile:
    """Build a minimal AgentProfile suitable for agent-loop tests."""
    return AgentProfile(
        id="test_agent",
        name="Tester",
        age=25,
        traits="focused, terse",
        backstory="A test fixture.",
        initial_plan="Run tests.",
        cognitive={"max_steps_per_cycle": max_steps},  # type: ignore[arg-type]
    )


def resp(content: str = "", tool_calls: list[ToolCall] | None = None) -> LLMResponse:
    """Build a ready-to-enqueue LLMResponse with realistic-looking TokenUsage."""
    calls = list(tool_calls) if tool_calls else []
    return LLMResponse(
        content=content,
        tool_calls=calls,
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        stop_reason="tool_use" if calls else "end_turn",
        raw={},
    )


def fake_memory(profile: AgentProfile | None = None) -> Any:
    """Build a MagicMock standing in for DefaultMemory: profile + AsyncMock add() + sync search_records/get_all returning []."""
    profile = profile or make_profile()
    mem = MagicMock(name="DefaultMemory")
    mem.profile = profile
    mem.add = AsyncMock(return_value=None)
    mem.search_records = MagicMock(return_value=[])
    mem.get_all = MagicMock(return_value=[])
    mem.run = AsyncMock(side_effect=lambda msgs, **kw: msgs)
    return mem


def fake_memory_with_records(
    profile: AgentProfile | None = None,
    *,
    search_results: list[dict[str, Any]] | None = None,
) -> Any:
    """Same as `fake_memory` but search_records returns the given list, useful for memory_recall tool tests."""
    mem = fake_memory(profile)
    mem.search_records = MagicMock(return_value=list(search_results or []))
    return mem


def added_calls(memory: Any) -> list[dict[str, Any]]:
    """Flatten memory.add.await_args_list into [{messages, memory_type}, ...] for assertions."""
    out: list[dict[str, Any]] = []
    for call in memory.add.await_args_list:
        args = call.args
        kwargs = call.kwargs
        messages = args[0] if args else kwargs.get("messages", [])
        out.append({
            "messages": list(messages),
            "memory_type": kwargs.get("memory_type"),
        })
    return out
