"""Tests for the two ReAct behavioral upgrades: trajectory persistence + reflection on every exit path."""
import pytest

from DefenseAgent.agent import AgentStepLimitError, ReActAgent
from DefenseAgent.llm.types import ToolCall
from DefenseAgent.memory import Memory
from DefenseAgent.reflection import Reflector
from DefenseAgent.tools import ToolRegistry

from tests.DefenseAgent.agent._support import (
    ScriptedLLM,
    ZeroEmbedder,
    make_profile,
    resp,
)


class _FakeReflector:
    """Reflector stand-in that records how many times check_and_reflect() was awaited and can be made to raise."""

    def __init__(self, *, raise_on_reflect: bool = False):
        """Configure whether reflection should raise; start with zero call count."""
        self.call_count = 0
        self._raise = raise_on_reflect

    async def check_and_reflect(self):
        """Count each call; raise RuntimeError if configured to do so."""
        self.call_count += 1
        if self._raise:
            raise RuntimeError("reflection boom")
        return []


# ---------- trajectory persistence ----------


async def test_trajectory_writes_one_observation_per_step():
    """Each agent step with tool calls produces exactly ONE trajectory record (not one per call)."""
    profile = make_profile()
    memory = Memory(profile=profile, embedding_adapter=ZeroEmbedder())
    registry = ToolRegistry()

    @registry.tool
    def square(x: int) -> int:
        """squared"""
        return x * x

    llm = ScriptedLLM(
        [
            resp(
                content="",
                tool_calls=[ToolCall(id="c1", name="square", arguments={"x": 3})],
            ),
            resp(
                content="",
                tool_calls=[ToolCall(id="c2", name="square", arguments={"x": 4})],
            ),
            resp(content="final answer"),
        ]
    )
    agent = ReActAgent(
        profile,
        llm=llm,  # type: ignore[arg-type]
        memory=memory,
        tools=registry,
        memory_recall_top_k=0,
        persist_outcome=True,
        persist_trajectory=True,
        reflect_after_run=False,
    )

    assert len(memory) == 0
    await agent.run("task", max_steps=5)
    # 2 trajectory steps + 1 outcome = 3 records.
    assert len(memory) == 3

    records = memory.stream.get_all()
    trajectory_records = [r for r in records if r.metadata.get("trajectory")]
    assert len(trajectory_records) == 2
    assert trajectory_records[0].metadata["tool_names"] == ["square"]
    assert trajectory_records[0].metadata["step"] == 0
    assert trajectory_records[1].metadata["step"] == 1
    # Content carries the call + result preview.
    assert "square(" in trajectory_records[0].content
    assert "→" in trajectory_records[0].content


async def test_trajectory_consolidates_multiple_tool_calls_into_one_record():
    """A single LLM turn with N concurrent tool calls must produce ONE trajectory record, not N."""
    profile = make_profile()
    memory = Memory(profile=profile, embedding_adapter=ZeroEmbedder())
    registry = ToolRegistry()

    @registry.tool
    def echo(text: str) -> str:
        """echo"""
        return f"echoed {text}"

    llm = ScriptedLLM(
        [
            resp(
                content="",
                tool_calls=[
                    ToolCall(id="a", name="echo", arguments={"text": "first"}),
                    ToolCall(id="b", name="echo", arguments={"text": "second"}),
                    ToolCall(id="c", name="echo", arguments={"text": "third"}),
                ],
            ),
            resp(content="done"),
        ]
    )
    agent = ReActAgent(
        profile,
        llm=llm,  # type: ignore[arg-type]
        memory=memory,
        tools=registry,
        memory_recall_top_k=0,
        persist_outcome=False,
        persist_trajectory=True,
        reflect_after_run=False,
    )
    await agent.run("task", max_steps=5)

    trajectory_records = [
        r for r in memory.stream.get_all() if r.metadata.get("trajectory")
    ]
    assert len(trajectory_records) == 1
    record = trajectory_records[0]
    # Metadata lists every tool name called in that step.
    assert record.metadata["tool_names"] == ["echo", "echo", "echo"]
    # Content summarizes all three calls with `; ` between them.
    assert record.content.count("echo(") == 3
    assert record.content.count(";") >= 2


async def test_trajectory_importance_defaults_to_five():
    """New default is 5.0 — equal footing with organic observations so memory_recall surfaces past attempts."""
    profile = make_profile()
    memory = Memory(profile=profile, embedding_adapter=ZeroEmbedder())
    registry = ToolRegistry()

    @registry.tool
    def noop() -> str:
        """no-op"""
        return "ok"

    llm = ScriptedLLM(
        [
            resp(
                content="",
                tool_calls=[ToolCall(id="c1", name="noop", arguments={})],
            ),
            resp(content="done"),
        ]
    )
    agent = ReActAgent(
        profile,
        llm=llm,  # type: ignore[arg-type]
        memory=memory,
        tools=registry,
        memory_recall_top_k=0,
        persist_outcome=True,
        persist_trajectory=True,
        reflect_after_run=False,
    )
    await agent.run("task", max_steps=5)

    records = memory.stream.get_all()
    trajectory = next(r for r in records if r.metadata.get("trajectory"))
    outcome = next(r for r in records if not r.metadata.get("trajectory"))
    assert trajectory.importance == 5.0
    assert outcome.importance == 5.0  # success outcomes use the same default


async def test_persist_trajectory_false_writes_no_trajectory_records():
    profile = make_profile()
    memory = Memory(profile=profile, embedding_adapter=ZeroEmbedder())
    registry = ToolRegistry()

    @registry.tool
    def noop() -> str:
        """no-op"""
        return "ok"

    llm = ScriptedLLM(
        [
            resp(
                content="",
                tool_calls=[ToolCall(id="c1", name="noop", arguments={})],
            ),
            resp(content="done"),
        ]
    )
    agent = ReActAgent(
        profile,
        llm=llm,  # type: ignore[arg-type]
        memory=memory,
        tools=registry,
        memory_recall_top_k=0,
        persist_outcome=True,
        persist_trajectory=False,  # ← key knob
        reflect_after_run=False,
    )
    await agent.run("task", max_steps=5)

    records = memory.stream.get_all()
    assert all(not r.metadata.get("trajectory") for r in records)


async def test_trajectory_previews_truncate_long_results():
    profile = make_profile()
    memory = Memory(profile=profile, embedding_adapter=ZeroEmbedder())
    registry = ToolRegistry()

    @registry.tool
    def verbose() -> str:
        """returns a long string"""
        return "x" * 500

    llm = ScriptedLLM(
        [
            resp(
                content="",
                tool_calls=[ToolCall(id="c1", name="verbose", arguments={})],
            ),
            resp(content="done"),
        ]
    )
    agent = ReActAgent(
        profile,
        llm=llm,  # type: ignore[arg-type]
        memory=memory,
        tools=registry,
        memory_recall_top_k=0,
        persist_outcome=False,
        persist_trajectory=True,
        reflect_after_run=False,
    )
    await agent.run("t", max_steps=3)

    trajectory = next(r for r in memory.stream.get_all() if r.metadata.get("trajectory"))
    # 500-char result should have been cut down with "..." before being stored.
    assert "..." in trajectory.content
    assert len(trajectory.content) < 400


# ---------- reflection on every exit path ----------


async def test_reflection_fires_on_success():
    profile = make_profile()
    memory = Memory(profile=profile, embedding_adapter=ZeroEmbedder())
    reflector = _FakeReflector()
    agent = ReActAgent(
        profile,
        llm=ScriptedLLM([resp(content="final")]),  # type: ignore[arg-type]
        memory=memory,
        tools=ToolRegistry(),
        reflector=reflector,  # type: ignore[arg-type]
        memory_recall_top_k=0,
        persist_outcome=False,
        persist_trajectory=False,
        reflect_after_run=True,
    )
    await agent.run("q", max_steps=2)
    assert reflector.call_count == 1


async def test_reflection_fires_on_max_steps_exhaustion():
    """The whole point of the fix — reflection must run when a run FAILS too."""
    profile = make_profile()
    memory = Memory(profile=profile, embedding_adapter=ZeroEmbedder())
    registry = ToolRegistry()

    @registry.tool
    def noop() -> str:
        """no-op"""
        return "ok"

    llm = ScriptedLLM(
        [
            resp(
                content="",
                tool_calls=[ToolCall(id=f"c{i}", name="noop", arguments={})],
            )
            for i in range(3)
        ]
    )
    reflector = _FakeReflector()
    agent = ReActAgent(
        profile,
        llm=llm,  # type: ignore[arg-type]
        memory=memory,
        tools=registry,
        reflector=reflector,  # type: ignore[arg-type]
        memory_recall_top_k=0,
        persist_outcome=False,
        persist_trajectory=False,
        reflect_after_run=True,
    )
    with pytest.raises(AgentStepLimitError):
        await agent.run("loop", max_steps=3)
    # Reflection still fired despite the failure.
    assert reflector.call_count == 1


async def test_reflection_failure_does_not_mask_success():
    profile = make_profile()
    memory = Memory(profile=profile, embedding_adapter=ZeroEmbedder())
    reflector = _FakeReflector(raise_on_reflect=True)
    agent = ReActAgent(
        profile,
        llm=ScriptedLLM([resp(content="final")]),  # type: ignore[arg-type]
        memory=memory,
        tools=ToolRegistry(),
        reflector=reflector,  # type: ignore[arg-type]
        memory_recall_top_k=0,
        persist_outcome=False,
        persist_trajectory=False,
        reflect_after_run=True,
    )
    # The run must return normally even though reflection raised.
    result = await agent.run("q", max_steps=2)
    assert result.final_answer == "final"
    assert reflector.call_count == 1


async def test_reflection_failure_does_not_mask_step_limit_error():
    profile = make_profile()
    memory = Memory(profile=profile, embedding_adapter=ZeroEmbedder())
    registry = ToolRegistry()

    @registry.tool
    def noop() -> str:
        """no-op"""
        return "ok"

    llm = ScriptedLLM(
        [
            resp(
                content="",
                tool_calls=[ToolCall(id=f"c{i}", name="noop", arguments={})],
            )
            for i in range(2)
        ]
    )
    reflector = _FakeReflector(raise_on_reflect=True)
    agent = ReActAgent(
        profile,
        llm=llm,  # type: ignore[arg-type]
        memory=memory,
        tools=registry,
        reflector=reflector,  # type: ignore[arg-type]
        memory_recall_top_k=0,
        persist_outcome=False,
        persist_trajectory=False,
        reflect_after_run=True,
    )
    # Original exception (AgentStepLimitError) must propagate, not the reflection error.
    with pytest.raises(AgentStepLimitError):
        await agent.run("loop", max_steps=2)
    assert reflector.call_count == 1


async def test_failure_path_persists_outcome_with_failed_prefix():
    """When max_steps is exhausted the failure is recorded as an outcome at importance 6.0."""
    profile = make_profile()
    memory = Memory(profile=profile, embedding_adapter=ZeroEmbedder())
    registry = ToolRegistry()

    @registry.tool
    def noop() -> str:
        """no-op"""
        return "ok"

    llm = ScriptedLLM(
        [
            resp(
                content="",
                tool_calls=[ToolCall(id=f"c{i}", name="noop", arguments={})],
            )
            for i in range(3)
        ]
    )
    agent = ReActAgent(
        profile,
        llm=llm,  # type: ignore[arg-type]
        memory=memory,
        tools=registry,
        memory_recall_top_k=0,
        persist_outcome=True,
        persist_trajectory=False,
        reflect_after_run=False,
    )
    with pytest.raises(AgentStepLimitError):
        await agent.run("hard task", max_steps=3)

    outcome_records = [
        r for r in memory.stream.get_all() if not r.metadata.get("trajectory")
    ]
    assert len(outcome_records) == 1
    failure = outcome_records[0]
    assert failure.content.startswith("Q: hard task")
    assert "FAILED" in failure.content
    assert "max_steps=3" in failure.content
    assert failure.importance == 6.0


async def test_failure_outcome_skipped_when_persist_outcome_false():
    """persist_outcome=False disables outcome writes on both success and failure paths."""
    profile = make_profile()
    memory = Memory(profile=profile, embedding_adapter=ZeroEmbedder())
    registry = ToolRegistry()

    @registry.tool
    def noop() -> str:
        """no-op"""
        return "ok"

    llm = ScriptedLLM(
        [
            resp(
                content="",
                tool_calls=[ToolCall(id=f"c{i}", name="noop", arguments={})],
            )
            for i in range(2)
        ]
    )
    agent = ReActAgent(
        profile,
        llm=llm,  # type: ignore[arg-type]
        memory=memory,
        tools=registry,
        memory_recall_top_k=0,
        persist_outcome=False,
        persist_trajectory=False,
        reflect_after_run=False,
    )
    with pytest.raises(AgentStepLimitError):
        await agent.run("task", max_steps=2)
    assert len(memory) == 0


async def test_reflect_after_run_false_skips_reflection_on_both_paths():
    profile = make_profile()
    memory = Memory(profile=profile, embedding_adapter=ZeroEmbedder())
    reflector = _FakeReflector()
    agent = ReActAgent(
        profile,
        llm=ScriptedLLM([resp(content="done")]),  # type: ignore[arg-type]
        memory=memory,
        tools=ToolRegistry(),
        reflector=reflector,  # type: ignore[arg-type]
        memory_recall_top_k=0,
        persist_outcome=False,
        persist_trajectory=False,
        reflect_after_run=False,
    )
    await agent.run("q", max_steps=2)
    assert reflector.call_count == 0
