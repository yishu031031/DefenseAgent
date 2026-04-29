"""Tests for DefenseAgent.reflection — Reflector + parsers, all offline via mocked Mem0Memory."""
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from DefenseAgent.config.profile import AgentProfile
from DefenseAgent.llm import LLM, LLMResponse, Message, TokenUsage
from DefenseAgent.llm.base import LLMAdapter
from DefenseAgent.reflection import ImportanceScorer, InsightSynthesizer, Reflector
from DefenseAgent.reflection.scorer import parse_importance_response
from DefenseAgent.reflection.synthesizer import (
    format_memories_for_prompt,
    parse_reflection_response,
)


_NOW = datetime(2026, 4, 22, 18, 0, tzinfo=timezone.utc)


def _fixed_clock() -> datetime:
    """Deterministic clock so reflect_now's _last_reflection_time is predictable."""
    return _NOW


# ---- stubs ----


class _StubLLMAdapter(LLMAdapter):
    """LLMAdapter that pops scripted responses from a queue (FIFO)."""

    def __init__(self, responses: list[str] | None = None):
        """Hold the canned response strings + an empty call log."""
        self._responses = list(responses or [])
        self.calls: list[dict] = []

    async def chat(self, messages, *, tools=None, temperature=0.7,
                   max_tokens=1024, system=None):
        """Pop the next canned string and return it as an LLMResponse."""
        self.calls.append({"messages": messages, "temperature": temperature})
        content = self._responses.pop(0) if self._responses else ""
        return LLMResponse(
            content=content, tool_calls=[],
            usage=TokenUsage(10, 5, 15), stop_reason="end_turn", raw={},
        )


def _profile(reflection_threshold: int = 5) -> AgentProfile:
    """Minimal AgentProfile with a tunable cognitive.reflection_threshold."""
    return AgentProfile(
        id="agent_test", name="Tester", age=20,
        traits="t", backstory="b", initial_plan="p",
        cognitive={"reflection_threshold": reflection_threshold},  # type: ignore[arg-type]
    )


def _stub_llm(responses: list[str] | None = None) -> LLM:
    """Wrap _StubLLMAdapter in our LLM facade."""
    return LLM(_StubLLMAdapter(responses))


def _stub_memory(
    profile: AgentProfile,
    *,
    records: list[dict[str, Any]] | None = None,
) -> Any:
    """Build a MagicMock standing in for Mem0Memory: profile + get_all returning canned records + AsyncMock add()."""
    mem = MagicMock()
    mem.profile = profile
    mem.get_all = MagicMock(return_value=list(records or []))
    mem.add = AsyncMock(return_value=None)
    return mem


# ============================================================
# Parsers
# ============================================================


def test_parse_importance_extracts_first_integer():
    assert parse_importance_response("8") == 8.0
    assert parse_importance_response("I'd rate this an 8.") == 8.0


def test_parse_importance_clips_to_one_through_ten():
    assert parse_importance_response("0") == 1.0
    assert parse_importance_response("42") == 10.0
    assert parse_importance_response("11") == 10.0


def test_parse_importance_falls_back_on_unparseable():
    assert parse_importance_response("no number here") == 5.0
    assert parse_importance_response("") == 5.0
    assert parse_importance_response(None) == 5.0  # type: ignore[arg-type]


def test_parse_reflection_strips_bullets_and_numbers():
    text = "1. First insight\n- second insight\n• third insight\n4) fourth"
    assert parse_reflection_response(text, n=4) == [
        "First insight", "second insight", "third insight", "fourth",
    ]


def test_parse_reflection_caps_at_n():
    text = "alpha\nbeta\ngamma\ndelta"
    assert parse_reflection_response(text, n=2) == ["alpha", "beta"]


def test_parse_reflection_drops_empty_lines():
    text = "first\n\n\nsecond\n"
    assert parse_reflection_response(text, n=5) == ["first", "second"]


def test_parse_reflection_empty_returns_empty():
    assert parse_reflection_response("", n=3) == []
    assert parse_reflection_response(None, n=3) == []  # type: ignore[arg-type]


def test_format_memories_for_prompt_renders_mem0_records():
    records = [
        {"memory": "Maya prefers the library.", "memory_type": "preference"},
        {"memory": "Got stuck on problem 3.", "memory_type": "observation"},
        {"memory": "Plain memory with no type."},
    ]
    rendered = format_memories_for_prompt(records)
    assert "[preference] Maya prefers the library." in rendered
    assert "[observation] Got stuck on problem 3." in rendered
    assert "[observation] Plain memory with no type." in rendered


# ============================================================
# ImportanceScorer
# ============================================================


async def test_importance_scorer_returns_llm_score():
    llm = _stub_llm(["8"])
    scorer = ImportanceScorer(llm)
    score = await scorer.score("I won the science fair.")
    assert score == 8.0


async def test_importance_scorer_passes_content_to_prompt():
    llm = _stub_llm(["6"])
    scorer = ImportanceScorer(llm)
    await scorer.score("specific test content")
    body = llm.adapter.calls[0]["messages"][0].content
    assert "specific test content" in body


# ============================================================
# InsightSynthesizer
# ============================================================


async def test_synthesizer_returns_n_insights_from_records():
    llm = _stub_llm(["insight A\ninsight B\ninsight C"])
    synth = InsightSynthesizer(llm, num_insights=3)

    records = [{"memory": "x", "memory_type": "observation"}]
    insights = await synth.synthesize(records)
    assert insights == ["insight A", "insight B", "insight C"]


async def test_synthesizer_empty_records_short_circuits():
    llm = _stub_llm([])
    synth = InsightSynthesizer(llm, num_insights=3)
    assert await synth.synthesize([]) == []
    assert llm.adapter.calls == []


async def test_synthesizer_caps_output_at_num_insights():
    llm = _stub_llm(["one\ntwo\nthree\nfour\nfive"])
    synth = InsightSynthesizer(llm, num_insights=2)
    records = [{"memory": "x"}]
    assert await synth.synthesize(records) == ["one", "two"]


# ============================================================
# Reflector
# ============================================================


async def test_reflector_score_importance_delegates_to_scorer():
    profile = _profile()
    memory = _stub_memory(profile)
    llm = _stub_llm(["7"])
    reflector = Reflector(memory, llm, clock=_fixed_clock)
    assert await reflector.score_importance("anything") == 7.0


async def test_reflector_unreflected_count_excludes_reflections():
    profile = _profile()
    records = [
        {"memory": "obs A", "memory_type": "observation"},
        {"memory": "obs B", "memory_type": "observation"},
        {"memory": "old reflection", "memory_type": "reflection"},
    ]
    memory = _stub_memory(profile, records=records)
    reflector = Reflector(memory, _stub_llm(), clock=_fixed_clock)
    assert reflector.unreflected_count == 2


async def test_reflector_maybe_reflect_below_threshold_is_noop():
    profile = _profile(reflection_threshold=5)
    records = [{"memory": f"obs {i}", "memory_type": "observation"} for i in range(3)]
    memory = _stub_memory(profile, records=records)
    reflector = Reflector(memory, _stub_llm(), clock=_fixed_clock)

    result = await reflector.maybe_reflect()
    assert result == []
    memory.add.assert_not_called()


async def test_reflector_maybe_reflect_above_threshold_writes_back():
    profile = _profile(reflection_threshold=2)
    records = [{"memory": f"obs {i}", "memory_type": "observation"} for i in range(3)]
    memory = _stub_memory(profile, records=records)
    llm = _stub_llm(["I learn best when I struggle first.\nI prefer quiet study spaces."])
    reflector = Reflector(memory, llm, num_insights=2, clock=_fixed_clock)

    stored = await reflector.maybe_reflect()
    assert len(stored) == 2
    assert stored[0]["memory_type"] == "reflection"
    assert "struggle first" in stored[0]["memory"]
    # Each insight became an add() call.
    assert memory.add.await_count == 2


async def test_reflector_reflect_now_advances_cutoff():
    profile = _profile()
    records = [{"memory": "obs", "memory_type": "observation"}]
    memory = _stub_memory(profile, records=records)
    llm = _stub_llm(["a single insight"])
    reflector = Reflector(memory, llm, num_insights=1, clock=_fixed_clock)

    assert reflector._last_reflection_time is None
    await reflector.reflect_now()
    assert reflector._last_reflection_time == _NOW


async def test_reflector_reflect_now_with_no_records_is_noop():
    profile = _profile()
    memory = _stub_memory(profile, records=[])
    reflector = Reflector(memory, _stub_llm(), clock=_fixed_clock)
    assert await reflector.reflect_now() == []
    memory.add.assert_not_called()


async def test_reflector_writes_with_memory_type_reflection():
    profile = _profile()
    records = [{"memory": "obs A", "memory_type": "observation"}]
    memory = _stub_memory(profile, records=records)
    llm = _stub_llm(["new insight"])
    reflector = Reflector(memory, llm, num_insights=1, clock=_fixed_clock)

    await reflector.reflect_now()
    args, kwargs = memory.add.call_args
    assert kwargs.get("memory_type") == "reflection"
    forwarded_messages = args[0]
    assert forwarded_messages[0].content == "new insight"


async def test_reflector_writes_into_semantic_tier():
    """P2: reflections live in the SEMANTIC tier (Hello-Agents lifecycle for
    distilled facts/lessons), not Episodic. The Reflector should pass tier
    explicitly so the persistent backend tags metadata correctly."""
    from DefenseAgent.memory import MemoryTier

    profile = _profile()
    records = [{"memory": "obs A", "memory_type": "observation"}]
    memory = _stub_memory(profile, records=records)
    llm = _stub_llm(["distilled lesson"])
    reflector = Reflector(memory, llm, num_insights=1, clock=_fixed_clock)

    await reflector.reflect_now()
    _, kwargs = memory.add.call_args
    assert kwargs.get("tier") == MemoryTier.SEMANTIC


async def test_reflector_normalizes_importance_to_unit_range():
    """The legacy 1-10 reflection_importance scale must be normalized to
    [0, 1] before storage so it matches MemoryItem's invariant. 8 → 0.8."""
    profile = _profile()
    records = [{"memory": "obs", "memory_type": "observation"}]
    memory = _stub_memory(profile, records=records)
    llm = _stub_llm(["insight"])
    reflector = Reflector(
        memory, llm,
        num_insights=1,
        reflection_importance=8.0,
        clock=_fixed_clock,
    )

    await reflector.reflect_now()
    _, kwargs = memory.add.call_args
    assert kwargs.get("importance") == 0.8


async def test_reflector_clamps_misconfigured_importance():
    """A reflection_importance set above 10 (misconfig) must not propagate an
    out-of-range value into MemoryItem and trigger the validator."""
    profile = _profile()
    records = [{"memory": "obs", "memory_type": "observation"}]
    memory = _stub_memory(profile, records=records)
    llm = _stub_llm(["insight"])
    reflector = Reflector(
        memory, llm,
        num_insights=1,
        reflection_importance=15.0,
        clock=_fixed_clock,
    )

    await reflector.reflect_now()
    _, kwargs = memory.add.call_args
    assert kwargs.get("importance") == 1.0
