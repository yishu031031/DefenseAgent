"""Tests for DefenseAgent.reflection — all behaviors offline.

Stubs:
  - _StubLLM returns a canned response from a queue; records calls.
  - _StubEmbedder assigns vectors from a lookup table.
Both support the full LLM/EmbeddingAdapter contracts our code uses.
"""
from datetime import datetime, timedelta, timezone

import pytest

from DefenseAgent.config.profile import AgentProfile
from DefenseAgent.llm import LLM, LLMAdapter, LLMResponse, Message, TokenUsage
from DefenseAgent.llm import LLMProviderError
from DefenseAgent.memory import EmbeddingAdapter, Memory, MemoryRecord
from DefenseAgent.reflection import ImportanceScorer, InsightSynthesizer, Reflector
from DefenseAgent.reflection.scorer import (
    parse_importance_response as _parse_importance_response,
)
from DefenseAgent.reflection.synthesizer import (
    parse_reflection_response as _parse_reflection_response,
)


_NOW = datetime(2026, 4, 22, 18, 0, tzinfo=timezone.utc)


def _fixed_clock() -> datetime:
    return _NOW


# ---- stubs ----------------------------------------------------------------


class _StubLLMAdapter(LLMAdapter):
    """Returns canned responses from a queue. Records every call."""

    def __init__(self, responses: list[str] | None = None):
        self._responses = list(responses or [])
        self.calls: list[dict] = []

    async def chat(self, messages, *, tools=None, temperature=0.7,
                   max_tokens=1024, system=None):
        self.calls.append({"messages": messages, "temperature": temperature,
                           "max_tokens": max_tokens, "system": system})
        if self._responses:
            content = self._responses.pop(0)
        else:
            content = ""
        return LLMResponse(
            content=content, tool_calls=[],
            usage=TokenUsage(10, 5, 15),
            stop_reason="end_turn", raw={},
        )


class _RaisingLLMAdapter(LLMAdapter):
    async def chat(self, messages, **kwargs):
        raise LLMProviderError(provider="stub", status_code=500, message="boom")


class _StubEmbedder(EmbeddingAdapter):
    def __init__(self, table: dict[str, list[float]] | None = None):
        self._table = table or {}

    async def embed(self, text: str) -> list[float]:
        if text in self._table:
            return list(self._table[text])
        # A degenerate but distinct vector per unknown text, keyed by hash
        # so dedup across different contents doesn't accidentally collapse.
        h = abs(hash(text)) % 1000
        return [float(h) / 1000, 0.0, 0.0, 0.1]

    async def embed_batch(self, texts):
        return [await self.embed(t) for t in texts]


def _profile(reflection_threshold: int = 5) -> AgentProfile:
    return AgentProfile(
        id="agent_test",
        name="Test",
        age=25,
        traits="reflective",
        backstory="A test agent.",
        initial_plan="Think.",
        cognitive={"reflection_threshold": reflection_threshold},
    )


def _memory(reflection_threshold: int = 5) -> Memory:
    return Memory(
        _profile(reflection_threshold),
        _StubEmbedder(),
        clock=_fixed_clock,
        dedup_threshold=None,   # off for tests — want deterministic counts
    )


# ============================================================
# Parsing: _parse_importance_response
# ============================================================


def test_importance_parse_plain_integer():
    assert _parse_importance_response("7") == 7.0


def test_importance_parse_integer_in_sentence():
    assert _parse_importance_response("I'd rate this about 8, personally.") == 8.0


def test_importance_parse_clips_to_upper_bound():
    assert _parse_importance_response("42") == 10.0


def test_importance_parse_clips_to_lower_bound():
    # Regex only matches non-negative ints — "0" is below the 1–10 range.
    assert _parse_importance_response("0") == 1.0


def test_importance_parse_returns_five_for_unparseable():
    assert _parse_importance_response("hmm no idea") == 5.0


def test_importance_parse_returns_five_for_empty():
    assert _parse_importance_response("") == 5.0


def test_importance_parse_returns_float_not_int():
    result = _parse_importance_response("7")
    assert isinstance(result, float)


# ============================================================
# Parsing: _parse_reflection_response
# ============================================================


def test_reflection_parse_clean_lines():
    resp = "She learns faster when stuck.\nShe loves the library.\nShe studies with peers."
    out = _parse_reflection_response(resp, n=3)
    assert out == [
        "She learns faster when stuck.",
        "She loves the library.",
        "She studies with peers.",
    ]


def test_reflection_parse_strips_numbered_prefixes():
    resp = "1. First insight.\n2. Second insight.\n3. Third insight."
    out = _parse_reflection_response(resp, n=3)
    assert out == ["First insight.", "Second insight.", "Third insight."]


def test_reflection_parse_strips_bullet_prefixes():
    resp = "- alpha\n* beta\n• gamma"
    out = _parse_reflection_response(resp, n=3)
    assert out == ["alpha", "beta", "gamma"]


def test_reflection_parse_drops_empty_lines():
    resp = "one\n\n\ntwo\n"
    assert _parse_reflection_response(resp, n=5) == ["one", "two"]


def test_reflection_parse_takes_at_most_n():
    resp = "a\nb\nc\nd\ne"
    assert _parse_reflection_response(resp, n=2) == ["a", "b"]


def test_reflection_parse_empty_response():
    assert _parse_reflection_response("", n=3) == []


def test_reflection_parse_whitespace_only():
    assert _parse_reflection_response("   \n\t\n  ", n=3) == []


# ============================================================
# score_importance (integration with LLM)
# ============================================================


async def test_score_importance_parses_llm_response():
    adapter = _StubLLMAdapter(["8"])
    llm = LLM(adapter)
    reflector = Reflector(_memory(), llm, clock=_fixed_clock)
    assert await reflector.score_importance("won the competition") == 8.0


async def test_score_importance_defaults_on_garbage():
    adapter = _StubLLMAdapter(["not a number"])
    reflector = Reflector(_memory(), LLM(adapter), clock=_fixed_clock)
    assert await reflector.score_importance("something") == 5.0


async def test_score_importance_propagates_provider_errors():
    reflector = Reflector(_memory(), LLM(_RaisingLLMAdapter()), clock=_fixed_clock)
    with pytest.raises(LLMProviderError):
        await reflector.score_importance("anything")


async def test_score_importance_uses_low_temperature():
    adapter = _StubLLMAdapter(["5"])
    reflector = Reflector(_memory(), LLM(adapter), clock=_fixed_clock)
    await reflector.score_importance("x")
    assert adapter.calls[0]["temperature"] == 0.0


# ============================================================
# unreflected_count
# ============================================================


async def test_unreflected_count_empty_stream_is_zero():
    reflector = Reflector(_memory(), LLM(_StubLLMAdapter()), clock=_fixed_clock)
    assert reflector.unreflected_count == 0


async def test_unreflected_count_ignores_reflections():
    mem = _memory()
    reflector = Reflector(mem, LLM(_StubLLMAdapter()), clock=_fixed_clock)

    await mem.remember("obs 1", importance=5)
    await mem.remember("obs 2", importance=6)
    # Add a reflection manually (bypassing reflector) — must not count.
    await mem.remember("reflection 1", kind="reflection", importance=8)

    assert reflector.unreflected_count == 2


async def test_unreflected_count_resets_after_reflection():
    mem = _memory()
    adapter = _StubLLMAdapter(["alpha\nbeta\ngamma"])
    reflector = Reflector(mem, LLM(adapter), clock=_fixed_clock)

    for i in range(6):
        await mem.remember(f"obs {i}", importance=5)
    assert reflector.unreflected_count == 6

    await reflector.reflect_now()
    # After reflection, everything added so far is "pre-reflection"
    # (timestamps are equal to _NOW; the cutoff is also _NOW, so
    # `r.timestamp > cutoff` is False for all existing records).
    assert reflector.unreflected_count == 0


async def test_unreflected_count_respects_cutoff():
    mem = _memory()
    reflector = Reflector(mem, LLM(_StubLLMAdapter(["a\nb\nc"])), clock=_fixed_clock)

    # Back-date an old observation to before NOW.
    mem.stream._clock = lambda: _NOW - timedelta(hours=2)
    await mem.remember("old obs", importance=5)
    mem.stream._clock = _fixed_clock

    # Force a reflection cutoff (advance _last_reflection_time to NOW).
    reflector._last_reflection_time = _NOW

    # A record older than the cutoff should NOT be counted.
    assert reflector.unreflected_count == 0


# ============================================================
# reflect_now
# ============================================================


async def test_reflect_now_parses_and_stores_insights():
    mem = _memory()
    adapter = _StubLLMAdapter([
        "Maya learns best when stuck.\n"
        "She loves the library environment.\n"
        "Study groups help her retain material."
    ])
    reflector = Reflector(
        mem, LLM(adapter), num_insights=3, reflection_importance=8.0,
        clock=_fixed_clock,
    )
    for i in range(5):
        await mem.remember(f"obs {i}", importance=6)

    stored = await reflector.reflect_now()

    assert len(stored) == 3
    assert all(r.kind == "reflection" for r in stored)
    assert all(r.importance == 8.0 for r in stored)
    assert stored[0].content == "Maya learns best when stuck."


async def test_reflect_now_tolerates_bullets_and_numbering():
    mem = _memory()
    adapter = _StubLLMAdapter([
        "1. First insight.\n- Second insight.\n* Third insight."
    ])
    reflector = Reflector(mem, LLM(adapter), clock=_fixed_clock)
    await mem.remember("obs", importance=5)

    stored = await reflector.reflect_now()
    assert [r.content for r in stored] == [
        "First insight.", "Second insight.", "Third insight.",
    ]


async def test_reflect_now_returns_empty_when_no_recent_memories():
    mem = _memory()
    adapter = _StubLLMAdapter(["a\nb\nc"])
    reflector = Reflector(mem, LLM(adapter), clock=_fixed_clock)
    # Stream is empty — should return [] without calling the LLM.
    stored = await reflector.reflect_now()
    assert stored == []
    assert adapter.calls == []


async def test_reflect_now_advances_cutoff_even_on_empty_response():
    mem = _memory()
    adapter = _StubLLMAdapter([""])
    reflector = Reflector(mem, LLM(adapter), clock=_fixed_clock)
    await mem.remember("obs", importance=5)

    stored = await reflector.reflect_now()
    assert stored == []
    # Second call with no new records → still no-op, no second LLM call.
    stored2 = await reflector.reflect_now()
    assert stored2 == []
    assert len(adapter.calls) == 1   # only the first reflect_now called LLM


async def test_reflect_now_does_not_double_count_its_own_reflections():
    mem = _memory()
    adapter = _StubLLMAdapter([
        "First round insight 1.\nFirst round insight 2.\nFirst round insight 3.",
    ])
    reflector = Reflector(mem, LLM(adapter), clock=_fixed_clock)
    for i in range(5):
        await mem.remember(f"obs {i}", importance=5)

    await reflector.reflect_now()
    # The three newly-stored reflections must NOT be treated as unreflected
    # observations for the next round.
    assert reflector.unreflected_count == 0


async def test_reflect_now_stores_reflections_with_configured_importance():
    mem = _memory()
    adapter = _StubLLMAdapter(["a\nb\nc"])
    reflector = Reflector(
        mem, LLM(adapter), reflection_importance=9.5, clock=_fixed_clock,
    )
    await mem.remember("obs", importance=5)

    stored = await reflector.reflect_now()
    assert all(r.importance == 9.5 for r in stored)


# ============================================================
# check_and_reflect — trigger logic
# ============================================================


async def test_check_and_reflect_below_threshold_is_noop():
    mem = _memory(reflection_threshold=5)
    adapter = _StubLLMAdapter(["a\nb\nc"])
    reflector = Reflector(mem, LLM(adapter), clock=_fixed_clock)
    for i in range(3):   # 3 < threshold(5)
        await mem.remember(f"obs {i}", importance=5)

    stored = await reflector.check_and_reflect()
    assert stored == []
    assert adapter.calls == []   # LLM was not called


async def test_check_and_reflect_at_threshold_triggers():
    mem = _memory(reflection_threshold=5)
    adapter = _StubLLMAdapter(["alpha\nbeta\ngamma"])
    reflector = Reflector(mem, LLM(adapter), clock=_fixed_clock)
    for i in range(5):   # exactly at threshold
        await mem.remember(f"obs {i}", importance=5)

    stored = await reflector.check_and_reflect()
    assert len(stored) == 3
    assert len(adapter.calls) == 1


async def test_check_and_reflect_above_threshold_triggers():
    mem = _memory(reflection_threshold=5)
    adapter = _StubLLMAdapter(["one\ntwo\nthree"])
    reflector = Reflector(mem, LLM(adapter), clock=_fixed_clock)
    for i in range(10):
        await mem.remember(f"obs {i}", importance=5)
    stored = await reflector.check_and_reflect()
    assert len(stored) == 3


async def test_check_and_reflect_noop_until_fresh_observations():
    mem = _memory(reflection_threshold=3)
    adapter = _StubLLMAdapter(["r1a\nr1b\nr1c", "r2a\nr2b\nr2c"])
    reflector = Reflector(mem, LLM(adapter), clock=_fixed_clock)

    for i in range(3):
        await mem.remember(f"obs {i}", importance=5)
    first_stored = await reflector.check_and_reflect()
    assert len(first_stored) == 3

    # Without new observations, a second check_and_reflect must no-op
    # even though the stream has 6 records total (3 obs + 3 reflections).
    second_stored = await reflector.check_and_reflect()
    assert second_stored == []
    assert len(adapter.calls) == 1


# ============================================================
# Component classes: ImportanceScorer + InsightSynthesizer
# ============================================================


async def test_importance_scorer_uses_configured_default_on_parse_failure():
    """ImportanceScorer's default_score is returned when LLM output is unparseable."""
    scorer = ImportanceScorer(LLM(_StubLLMAdapter(["garbage"])), default_score=3.0)
    assert await scorer.score("anything") == 3.0


async def test_importance_scorer_honors_temperature_and_max_tokens():
    """Constructor-configured temperature and max_tokens reach the LLM call."""
    adapter = _StubLLMAdapter(["5"])
    scorer = ImportanceScorer(LLM(adapter), temperature=0.1, max_tokens=24)
    await scorer.score("x")
    call = adapter.calls[0]
    assert call["temperature"] == 0.1
    assert call["max_tokens"] == 24


async def test_insight_synthesizer_returns_empty_for_empty_records():
    """No records → no LLM call, empty list."""
    adapter = _StubLLMAdapter([])
    synth = InsightSynthesizer(LLM(adapter))
    assert await synth.synthesize([]) == []
    assert adapter.calls == []


async def test_insight_synthesizer_respects_num_insights_cap():
    """The cap comes from the constructor; excess lines from the LLM are dropped."""
    adapter = _StubLLMAdapter(["a\nb\nc\nd\ne"])
    synth = InsightSynthesizer(LLM(adapter), num_insights=2)
    records = [
        MemoryRecord(id="r1", content="c1", timestamp=_NOW, kind="observation",
                     importance=5.0, embedding=[], metadata={}),
    ]
    insights = await synth.synthesize(records)
    assert insights == ["a", "b"]


async def test_reflector_accepts_injected_scorer_and_synthesizer():
    """The facade uses caller-supplied components verbatim, not its defaults."""
    scorer_adapter = _StubLLMAdapter(["9"])
    synth_adapter = _StubLLMAdapter(["custom insight 1\ncustom insight 2\ncustom insight 3"])
    custom_scorer = ImportanceScorer(LLM(scorer_adapter))
    custom_synth = InsightSynthesizer(LLM(synth_adapter), num_insights=3)

    mem = _memory()
    reflector = Reflector(
        mem, LLM(_StubLLMAdapter([])),      # facade's own LLM is unused
        scorer=custom_scorer,
        synthesizer=custom_synth,
        clock=_fixed_clock,
    )

    score = await reflector.score_importance("sample")
    assert score == 9.0
    assert len(scorer_adapter.calls) == 1   # custom scorer's adapter was used

    await mem.remember("obs", importance=5)
    stored = await reflector.reflect_now()
    assert len(stored) == 3
    assert stored[0].content == "custom insight 1"
    assert len(synth_adapter.calls) == 1    # custom synthesizer's adapter was used


# ============================================================
# End-to-end integration
# ============================================================


async def test_reflections_are_retrievable_via_memory():
    """After reflecting, memory.recall() surfaces reflections alongside observations."""
    # Use a table so the "pattern" query and the reflections share an axis.
    table = {
        "what patterns are emerging?": [0.0, 0.0, 1.0, 0.0],
        "Maya learns faster under pressure.": [0.0, 0.0, 1.0, 0.0],
    }
    profile = _profile()
    mem = Memory(profile, _StubEmbedder(table), clock=_fixed_clock, dedup_threshold=None)
    adapter = _StubLLMAdapter(["Maya learns faster under pressure."])
    reflector = Reflector(mem, LLM(adapter), num_insights=1, clock=_fixed_clock)

    for i in range(3):
        await mem.remember(f"obs {i}", importance=6)

    await reflector.reflect_now()

    results = await mem.recall("what patterns are emerging?", top_k=3)
    reflection_hits = [s for s in results if s.record.kind == "reflection"]
    assert len(reflection_hits) == 1
    assert reflection_hits[0].record.content == "Maya learns faster under pressure."
