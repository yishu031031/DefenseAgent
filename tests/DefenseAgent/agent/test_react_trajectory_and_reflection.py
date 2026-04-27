"""Tests for the two ReAct behavioral upgrades: trajectory persistence + reflection on every exit path."""
import pytest

from DefenseAgent.agent import AgentStepLimitError, ReActAgent
from DefenseAgent.llm.types import ToolCall

from DefenseAgent.reflection import Reflector
from DefenseAgent.tools import ToolRegistry

from tests.DefenseAgent.agent._support import (
    ScriptedLLM,
    fake_memory,
    make_profile,
    resp,
    make_test_config,
)


class _FakeReflector:
    """Reflector stand-in that records how many times maybe_reflect() was awaited and can be made to raise."""

    def __init__(self, *, raise_on_reflect: bool = False):
        """Configure whether reflection should raise; start with zero call count."""
        self.call_count = 0
        self._raise = raise_on_reflect

    async def maybe_reflect(self):
        """Count each call; raise RuntimeError if configured to do so."""
        self.call_count += 1
        if self._raise:
            raise RuntimeError("reflection boom")
        return []


# ---------- trajectory persistence ----------


async def test_trajectory_writes_one_observation_per_step():
    """Each agent step with tool calls produces exactly ONE trajectory record (not one per call)."""
    profile = make_profile()
    memory = fake_memory(profile)
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
    agent = ReActAgent(make_test_config(
            profile=profile,
            llm=llm,
            memory=memory,
            tools=registry,
            memory_recall_top_k=0,
            save_outcome=True,
            save_trajectory=True,
            reflect_after_run=False,
        ))

    await agent.run("task", max_steps=5)

    from tests.DefenseAgent.agent._support import added_calls
    calls = added_calls(memory)
    trajectory_calls = [c for c in calls if c["memory_type"] == "trajectory"]
    outcome_calls = [c for c in calls if c["memory_type"] == "outcome"]
    assert len(trajectory_calls) == 2
    assert len(outcome_calls) == 1

    first_traj = trajectory_calls[0]["messages"][0].content
    second_traj = trajectory_calls[1]["messages"][0].content
    assert "Trajectory step 0" in first_traj
    assert "Trajectory step 1" in second_traj
    assert "square(" in first_traj
    assert "→" in first_traj


async def test_trajectory_consolidates_multiple_tool_calls_into_one_record():
    """A single LLM turn with N concurrent tool calls must produce ONE trajectory record, not N."""
    profile = make_profile()
    memory = fake_memory(profile)
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
    agent = ReActAgent(make_test_config(
            profile=profile,
            llm=llm,
            memory=memory,
            tools=registry,
            memory_recall_top_k=0,
            save_outcome=False,
            save_trajectory=True,
            reflect_after_run=False,
        ))
    await agent.run("task", max_steps=5)

    from tests.DefenseAgent.agent._support import added_calls
    trajectory_calls = [
        c for c in added_calls(memory) if c["memory_type"] == "trajectory"
    ]
    assert len(trajectory_calls) == 1
    content = trajectory_calls[0]["messages"][0].content
    # Content summarizes all three calls with `; ` between them.
    assert content.count("echo(") == 3
    assert content.count(";") >= 2


async def test_save_trajectory_false_writes_no_trajectory_records():
    profile = make_profile()
    memory = fake_memory(profile)
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
    agent = ReActAgent(make_test_config(
            profile=profile,
            llm=llm,
            memory=memory,
            tools=registry,
            memory_recall_top_k=0,
            save_outcome=True,
            save_trajectory=False,  # ← key knob
            reflect_after_run=False,
        ))
    await agent.run("task", max_steps=5)

    from tests.DefenseAgent.agent._support import added_calls
    calls = added_calls(memory)
    assert all(c["memory_type"] != "trajectory" for c in calls)


async def test_trajectory_previews_truncate_long_results():
    profile = make_profile()
    memory = fake_memory(profile)
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
    agent = ReActAgent(make_test_config(
            profile=profile,
            llm=llm,
            memory=memory,
            tools=registry,
            memory_recall_top_k=0,
            save_outcome=False,
            save_trajectory=True,
            reflect_after_run=False,
        ))
    await agent.run("t", max_steps=3)

    from tests.DefenseAgent.agent._support import added_calls
    trajectory = next(c for c in added_calls(memory) if c["memory_type"] == "trajectory")
    content = trajectory["messages"][0].content
    # 500-char result should have been cut down with "..." before being stored.
    assert "..." in content
    assert len(content) < 400


# ---------- reflection on every exit path ----------


async def test_reflection_fires_on_success():
    profile = make_profile()
    memory = fake_memory(profile)
    reflector = _FakeReflector()
    agent = ReActAgent(make_test_config(
            profile=profile,
            llm=ScriptedLLM([resp(content="final")]),
            memory=memory,
            tools=ToolRegistry(),
            reflector=reflector,
            memory_recall_top_k=0,
            save_outcome=False,
            save_trajectory=False,
            reflect_after_run=True,
        ))
    await agent.run("q", max_steps=2)
    assert reflector.call_count == 1


async def test_reflection_fires_on_max_steps_exhaustion():
    """The whole point of the fix — reflection must run when a run FAILS too."""
    profile = make_profile()
    memory = fake_memory(profile)
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
    agent = ReActAgent(make_test_config(
            profile=profile,
            llm=llm,
            memory=memory,
            tools=registry,
            reflector=reflector,
            memory_recall_top_k=0,
            save_outcome=False,
            save_trajectory=False,
            reflect_after_run=True,
        ))
    with pytest.raises(AgentStepLimitError):
        await agent.run("loop", max_steps=3)
    # Reflection still fired despite the failure.
    assert reflector.call_count == 1


async def test_reflection_failure_does_not_mask_success():
    profile = make_profile()
    memory = fake_memory(profile)
    reflector = _FakeReflector(raise_on_reflect=True)
    agent = ReActAgent(make_test_config(
            profile=profile,
            llm=ScriptedLLM([resp(content="final")]),
            memory=memory,
            tools=ToolRegistry(),
            reflector=reflector,
            memory_recall_top_k=0,
            save_outcome=False,
            save_trajectory=False,
            reflect_after_run=True,
        ))
    # The run must return normally even though reflection raised.
    result = await agent.run("q", max_steps=2)
    assert result.final_answer == "final"
    assert reflector.call_count == 1


async def test_reflection_failure_does_not_mask_step_limit_error():
    profile = make_profile()
    memory = fake_memory(profile)
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
    agent = ReActAgent(make_test_config(
            profile=profile,
            llm=llm,
            memory=memory,
            tools=registry,
            reflector=reflector,
            memory_recall_top_k=0,
            save_outcome=False,
            save_trajectory=False,
            reflect_after_run=True,
        ))
    # Original exception (AgentStepLimitError) must propagate, not the reflection error.
    with pytest.raises(AgentStepLimitError):
        await agent.run("loop", max_steps=2)
    assert reflector.call_count == 1


async def test_failure_path_persists_outcome_with_failed_prefix():
    """When max_steps is exhausted the failure is recorded as an outcome at importance 6.0."""
    profile = make_profile()
    memory = fake_memory(profile)
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
    agent = ReActAgent(make_test_config(
            profile=profile,
            llm=llm,
            memory=memory,
            tools=registry,
            memory_recall_top_k=0,
            save_outcome=True,
            save_trajectory=False,
            reflect_after_run=False,
        ))
    with pytest.raises(AgentStepLimitError):
        await agent.run("hard task", max_steps=3)

    from tests.DefenseAgent.agent._support import added_calls
    failures = [c for c in added_calls(memory) if c["memory_type"] == "failure"]
    assert len(failures) == 1
    failure_msg = failures[0]["messages"][0]
    assert "Q: hard task" in failure_msg.content
    assert "FAILED" in failure_msg.content
    assert "max_steps=3" in failure_msg.content


async def test_failure_outcome_skipped_when_save_outcome_false():
    """save_outcome=False disables outcome writes on both success and failure paths."""
    profile = make_profile()
    memory = fake_memory(profile)
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
    agent = ReActAgent(make_test_config(
            profile=profile,
            llm=llm,
            memory=memory,
            tools=registry,
            memory_recall_top_k=0,
            save_outcome=False,
            save_trajectory=False,
            reflect_after_run=False,
        ))
    with pytest.raises(AgentStepLimitError):
        await agent.run("task", max_steps=2)
    assert len(memory) == 0


async def test_reflect_after_run_false_skips_reflection_on_both_paths():
    profile = make_profile()
    memory = fake_memory(profile)
    reflector = _FakeReflector()
    agent = ReActAgent(make_test_config(
            profile=profile,
            llm=ScriptedLLM([resp(content="done")]),
            memory=memory,
            tools=ToolRegistry(),
            reflector=reflector,
            memory_recall_top_k=0,
            save_outcome=False,
            save_trajectory=False,
            reflect_after_run=False,
        ))
    await agent.run("q", max_steps=2)
    assert reflector.call_count == 0
