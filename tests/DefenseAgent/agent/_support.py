"""Shared stubs for the agent test suite: a scripted LLM, a fake Mem0Memory, profile factories."""
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from DefenseAgent.agent import AgentConfig
from DefenseAgent.config import AgentProfile
from DefenseAgent.llm.types import LLMResponse, Message, TokenUsage, ToolCall
from DefenseAgent.tools import ToolRegistry


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
    """Build a MagicMock standing in for Mem0Memory: profile + AsyncMock add() + sync search_records/get_all returning []."""
    profile = profile or make_profile()
    mem = MagicMock(name="Mem0Memory")
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


def make_test_config(
    *,
    profile: AgentProfile | None = None,
    llm: Any = None,
    memory: Any = None,
    tools: ToolRegistry | None = None,
    reflector: Any = None,
    compressor: Any = None,
    rag: Any = None,
    save_outcome: bool = False,
    save_trajectory: bool = False,
    reflect_after_run: bool = False,
    memory_recall_top_k: int = 0,
    extra_instructions: str | None = None,
    max_substeps_per_step: int = 3,
    max_steps: int | None = None,
) -> AgentConfig:
    """Build an AgentConfig that injects pre-built test stubs and disables every
    auto-built side-channel (compressor, logger, env-driven build paths) so unit
    tests stay hermetic and offline.

    The default behavior knobs (`save_outcome=False`, `reflect_after_run=False`,
    `memory_recall_top_k=0`) keep simple test agents minimal — flip them per-test
    when exercising those code paths.
    """
    return AgentConfig(
        profile=profile or make_profile(),
        load_env=False,                       # never touch .env in tests
        # subsystem toggles — skip all auto-builds; injection takes precedence
        use_tools=tools is None,              # only auto-register skills if no registry injected
        use_memory=memory is not None,        # only build mem0 if no memory injected (and we never do)
        use_reflection=reflector is not None, # only build Reflector if no reflector injected
        use_compressor=compressor is not None,  # only auto-build compressor when injected one wins via injection field
        use_logger=False,                     # tests don't write logs
        use_rag=rag is not None,              # only build RAG if injected
        # injections
        llm=llm,
        memory=memory,
        tool_registry=tools,
        reflector=reflector,
        compressor=compressor,
        rag=rag,
        # behavior knobs
        memory_recall_top_k=memory_recall_top_k,
        save_outcome=save_outcome,
        save_trajectory=save_trajectory,
        reflect_after_run=reflect_after_run,
        extra_instructions=extra_instructions,
        max_substeps_per_step=max_substeps_per_step,
        max_steps=max_steps,
    )
