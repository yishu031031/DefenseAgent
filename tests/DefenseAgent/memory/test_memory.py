"""Tests for DefenseAgent.memory.memory — everything the module owns in one file.

Covers (in order, by section):
  • Error hierarchy
  • Canonical types (MemoryRecord, ScoredMemory)
  • cosine similarity helper
  • MemoryStream (add, dedup, get_*, BM25 maintenance, add_record)
  • MemoryRetriever (hybrid retrieval + per-kind rules + scoring axes)
  • Memory facade (construction, from_env, remember, recall)
"""
import math
from datetime import datetime, timedelta, timezone

import pytest

from DefenseAgent.config.profile import AgentProfile, MemoryConfig
from DefenseAgent.memory import (
    EmbeddingAdapter,
    EmbeddingConfigError,
    EmbeddingProviderError,
    Memory,
    MemoryError,
    MemoryNotFoundError,
    MemoryRecord,
    ScoredMemory,
)
from DefenseAgent.memory.embedding import OpenAICompatibleEmbeddingAdapter
from DefenseAgent.memory.retriever import MemoryRetriever, RRF_K
from DefenseAgent.memory.stream import MemoryStream, cosine


_NOW = datetime(2026, 4, 22, 18, 0, tzinfo=timezone.utc)


def _fixed_clock():
    return _NOW


# ============================================================
# Shared helpers
# ============================================================


class _StubEmbedder(EmbeddingAdapter):
    """Returns vectors from a lookup table. Unknown text → zero-like vector."""

    def __init__(self, table: dict[str, list[float]] | None = None, dim: int = 3):
        self._table = table or {}
        self._dim = dim
        self.embed_calls: list[str] = []

    async def embed(self, text: str) -> list[float]:
        self.embed_calls.append(text)
        if text in self._table:
            return list(self._table[text])
        any_item = next(iter(self._table.values()), [0.0] * self._dim)
        return [0.0] * len(any_item)

    async def embed_batch(self, texts):
        return [await self.embed(t) for t in texts]


def _record(id_: str, *, content: str, kind: str, importance: float,
            embedding: list[float], age_hours: float = 0.0,
            metadata: dict | None = None) -> MemoryRecord:
    return MemoryRecord(
        id=id_,
        content=content,
        timestamp=_NOW - timedelta(hours=age_hours),
        kind=kind,
        importance=importance,
        embedding=list(embedding),
        metadata=metadata or {},
    )


def _profile(id_: str = "agent_1", **overrides) -> AgentProfile:
    base = dict(
        id=id_, name="Test", age=20,
        traits="curious", backstory="A test agent.",
        initial_plan="Do things.",
    )
    base.update(overrides)
    return AgentProfile(**base)


def _stream_with(records: list[MemoryRecord], adapter: EmbeddingAdapter) -> MemoryStream:
    stream = MemoryStream(adapter, clock=_fixed_clock, dedup_threshold=None)
    for r in records:
        stream.add_record(r)
    return stream


# ============================================================
# Error hierarchy
# ============================================================


def test_all_subclasses_inherit_from_memory_error():
    assert issubclass(EmbeddingConfigError, MemoryError)
    assert issubclass(EmbeddingProviderError, MemoryError)
    assert issubclass(MemoryNotFoundError, MemoryError)


def test_errors_carry_messages():
    assert "blank" in str(EmbeddingConfigError("EMBEDDING_PROVIDER is blank"))
    assert "429" in str(EmbeddingProviderError("provider returned 429"))
    assert "missing" in str(MemoryNotFoundError("id not found: missing"))


def test_provider_error_preserves_cause():
    original = RuntimeError("network down")
    with pytest.raises(EmbeddingProviderError) as excinfo:
        try:
            raise original
        except RuntimeError as e:
            raise EmbeddingProviderError("wrapped") from e
    assert excinfo.value.__cause__ is original


# ============================================================
# Canonical types
# ============================================================


def _plain_record(**overrides):
    base = dict(
        id="a1", content="hi", timestamp=_NOW, kind="observation",
        importance=5.0, embedding=[0.1, 0.2, 0.3], metadata={},
    )
    base.update(overrides)
    return MemoryRecord(**base)


def test_memory_record_fields_and_defaults():
    r = _plain_record()
    assert r.id == "a1"
    assert r.kind == "observation"
    assert r.importance == 5.0
    assert r.embedding == [0.1, 0.2, 0.3]
    assert r.metadata == {}


@pytest.mark.parametrize("kind", [
    "observation", "fact", "preference", "plan", "reflection",
])
def test_memory_record_accepts_all_kinds(kind):
    assert _plain_record(kind=kind).kind == kind


def test_memory_record_preserves_metadata():
    r = _plain_record(metadata={"status": "active", "source": "class"})
    assert r.metadata == {"status": "active", "source": "class"}


def test_scored_memory_construction():
    r = _plain_record()
    s = ScoredMemory(
        record=r, score=0.87,
        recency_score=0.9, importance_score=0.5, relevance_score=0.8,
        dense_rank=1, sparse_rank=3,
    )
    assert s.record is r
    assert s.score == 0.87
    assert s.dense_rank == 1
    assert s.sparse_rank == 3


def test_scored_memory_ranks_may_be_none():
    s = ScoredMemory(
        record=_plain_record(), score=0.0,
        recency_score=0.0, importance_score=0.0, relevance_score=0.0,
        dense_rank=None, sparse_rank=None,
    )
    assert s.dense_rank is None
    assert s.sparse_rank is None


# ============================================================
# cosine similarity
# ============================================================


def test_cosine_identical_vectors_score_one():
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


def test_cosine_orthogonal_vectors_score_zero():
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_opposite_vectors_score_negative_one():
    assert cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_magnitude_independent():
    assert cosine([3.0, 4.0], [30.0, 40.0]) == pytest.approx(1.0)


def test_cosine_example_known_value():
    # Angle between (1,1) and (1,0) is 45°; cos(45°) = sqrt(2)/2
    assert cosine([1.0, 1.0], [1.0, 0.0]) == pytest.approx(math.sqrt(2) / 2)


def test_cosine_dimension_mismatch_raises_memory_error():
    with pytest.raises(MemoryError):
        cosine([1.0, 2.0], [1.0, 2.0, 3.0])


def test_cosine_zero_vector_returns_zero_not_nan():
    result = cosine([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    assert not math.isnan(result)


# ============================================================
# MemoryStream
# ============================================================


def test_empty_stream_has_zero_length():
    stream = MemoryStream(_StubEmbedder(), clock=_fixed_clock)
    assert len(stream) == 0
    assert stream.get_all() == []


async def test_add_generates_embedding_and_appends():
    adapter = _StubEmbedder({"hello world": [1.0, 0.0, 0.0]})
    stream = MemoryStream(adapter, clock=_fixed_clock, dedup_threshold=None)

    record = await stream.add("hello world")

    assert record.content == "hello world"
    assert record.embedding == [1.0, 0.0, 0.0]
    assert record.kind == "observation"
    assert record.importance == 5.0
    assert record.timestamp == _NOW
    assert len(stream) == 1
    assert adapter.embed_calls == ["hello world"]


async def test_add_assigns_unique_ids():
    adapter = _StubEmbedder({"a": [1.0, 0.0], "b": [0.0, 1.0]})
    stream = MemoryStream(adapter, clock=_fixed_clock, dedup_threshold=None)
    r1, r2 = await stream.add("a"), await stream.add("b")
    assert r1.id != r2.id and r1.id and r2.id


async def test_add_accepts_importance_kind_metadata():
    adapter = _StubEmbedder({"fact text": [1.0, 0.0]})
    stream = MemoryStream(adapter, clock=_fixed_clock, dedup_threshold=None)
    r = await stream.add(
        "fact text", kind="fact", importance=9.0,
        metadata={"source": "classroom"},
    )
    assert r.kind == "fact"
    assert r.importance == 9.0
    assert r.metadata == {"source": "classroom"}


async def test_get_recent_returns_most_recent_first():
    adapter = _StubEmbedder({"a": [1, 0], "b": [0, 1], "c": [1, 1]})
    stream = MemoryStream(adapter, clock=_fixed_clock, dedup_threshold=None)
    r1 = await stream.add("a")
    r2 = await stream.add("b")
    r3 = await stream.add("c")
    assert [r.id for r in stream.get_recent(2)] == [r3.id, r2.id]


async def test_get_recent_with_n_larger_than_stream_returns_all():
    adapter = _StubEmbedder({"x": [1, 0]})
    stream = MemoryStream(adapter, clock=_fixed_clock, dedup_threshold=None)
    r = await stream.add("x")
    assert stream.get_recent(100) == [r]


async def test_get_all_preserves_insertion_order():
    adapter = _StubEmbedder({"a": [1, 0], "b": [0, 1], "c": [1, 1]})
    stream = MemoryStream(adapter, clock=_fixed_clock, dedup_threshold=None)
    r1 = await stream.add("a")
    r2 = await stream.add("b")
    r3 = await stream.add("c")
    assert [r.id for r in stream.get_all()] == [r1.id, r2.id, r3.id]


async def test_get_by_id_hit_and_miss():
    adapter = _StubEmbedder({"x": [1, 0]})
    stream = MemoryStream(adapter, clock=_fixed_clock, dedup_threshold=None)
    r = await stream.add("x")
    assert stream.get_by_id(r.id) is r
    with pytest.raises(MemoryNotFoundError):
        stream.get_by_id("no-such-id")


def test_add_record_appends_without_embedding_call():
    adapter = _StubEmbedder({})
    stream = MemoryStream(adapter, clock=_fixed_clock, dedup_threshold=None)
    record = MemoryRecord(
        id="manual-1", content="manually built", timestamp=_NOW,
        kind="fact", importance=7.0, embedding=[0.5, 0.5], metadata={},
    )
    stream.add_record(record)
    assert len(stream) == 1
    assert stream.get_by_id("manual-1") is record
    assert adapter.embed_calls == []


def test_add_record_updates_bm25_index():
    adapter = _StubEmbedder({})
    stream = MemoryStream(adapter, clock=_fixed_clock, dedup_threshold=None)
    stream.add_record(MemoryRecord(
        id="manual-1", content="python data structures", timestamp=_NOW,
        kind="observation", importance=5.0, embedding=[0.5, 0.5], metadata={},
    ))
    assert stream.bm25.score("python")["manual-1"] > 0


async def test_add_updates_bm25_index():
    adapter = _StubEmbedder({
        "python data structures class": [1, 0, 0],
        "spanish homework tonight": [0, 1, 0],
    })
    stream = MemoryStream(adapter, clock=_fixed_clock, dedup_threshold=None)
    r1 = await stream.add("python data structures class")
    r2 = await stream.add("spanish homework tonight")
    scores = stream.bm25.score("python")
    assert scores[r1.id] > scores[r2.id]


async def test_dedup_near_duplicate_same_kind_returns_existing():
    adapter = _StubEmbedder({
        "I went to class": [1.0, 0.0, 0.0],
        "I went to class today": [0.9999, 0.01, 0.0],
    })
    stream = MemoryStream(adapter, clock=_fixed_clock, dedup_threshold=0.95)
    first = await stream.add("I went to class")
    second = await stream.add("I went to class today")
    assert second is first
    assert len(stream) == 1


async def test_dedup_does_not_block_across_kinds():
    adapter = _StubEmbedder({"Maya loves algorithms": [1.0, 0.0, 0.0]})
    stream = MemoryStream(adapter, clock=_fixed_clock, dedup_threshold=0.95)
    obs = await stream.add("Maya loves algorithms", kind="observation")
    fact = await stream.add("Maya loves algorithms", kind="fact")
    assert obs.id != fact.id
    assert len(stream) == 2


async def test_dedup_threshold_none_disables():
    adapter = _StubEmbedder({"hi": [1.0, 0.0]})
    stream = MemoryStream(adapter, clock=_fixed_clock, dedup_threshold=None)
    r1 = await stream.add("hi")
    r2 = await stream.add("hi")
    assert r1.id != r2.id
    assert len(stream) == 2


async def test_dedup_below_threshold_accepts_new_record():
    adapter = _StubEmbedder({
        "python class": [1.0, 0.0, 0.0],
        "spanish class": [0.0, 1.0, 0.0],
    })
    stream = MemoryStream(adapter, clock=_fixed_clock, dedup_threshold=0.5)
    r1 = await stream.add("python class")
    r2 = await stream.add("spanish class")
    assert r1.id != r2.id
    assert len(stream) == 2


# ============================================================
# MemoryRetriever
# ============================================================


def _mem_config(**overrides) -> MemoryConfig:
    return MemoryConfig(**overrides)


async def test_retriever_empty_stream_returns_empty_list():
    adapter = _StubEmbedder({"q": [1, 0, 0]})
    stream = MemoryStream(adapter, clock=_fixed_clock, dedup_threshold=None)
    retriever = MemoryRetriever(stream, adapter, _mem_config(), clock=_fixed_clock)
    assert await retriever.retrieve("q") == []


async def test_retriever_top_k_from_memory_config_by_default():
    records = [_record(f"r{i}", content=f"doc {i}", kind="observation",
                       importance=5, embedding=[1, 0, 0]) for i in range(5)]
    adapter = _StubEmbedder({"q": [1, 0, 0], **{r.content: [1, 0, 0] for r in records}})
    stream = _stream_with(records, adapter)
    retriever = MemoryRetriever(
        stream, adapter, _mem_config(retrieval_top_k=3), clock=_fixed_clock,
    )
    assert len(await retriever.retrieve("q")) == 3


async def test_retriever_respects_explicit_top_k_override():
    records = [_record(f"r{i}", content=f"doc {i}", kind="observation",
                       importance=5, embedding=[1, 0, 0]) for i in range(5)]
    adapter = _StubEmbedder({"q": [1, 0, 0], **{r.content: [1, 0, 0] for r in records}})
    stream = _stream_with(records, adapter)
    retriever = MemoryRetriever(
        stream, adapter, _mem_config(retrieval_top_k=10), clock=_fixed_clock,
    )
    assert len(await retriever.retrieve("q", top_k=2)) == 2


async def test_retriever_query_embedded_exactly_once():
    records = [_record(f"r{i}", content=f"r{i}", kind="observation",
                       importance=5, embedding=[1, 0]) for i in range(3)]
    adapter = _StubEmbedder({"python class": [1, 0], **{f"r{i}": [1, 0] for i in range(3)}})
    stream = _stream_with(records, adapter)
    retriever = MemoryRetriever(stream, adapter, _mem_config(), clock=_fixed_clock)
    adapter.embed_calls = []
    await retriever.retrieve("python class")
    assert adapter.embed_calls == ["python class"]


async def test_retriever_scored_memory_carries_all_components():
    adapter = _StubEmbedder({"python": [1, 0, 0], "r1": [1, 0, 0]})
    r = _record("r1", content="python class", kind="observation",
                importance=6, embedding=[1, 0, 0])
    stream = _stream_with([r], adapter)
    retriever = MemoryRetriever(stream, adapter, _mem_config(), clock=_fixed_clock)
    s = (await retriever.retrieve("python"))[0]
    assert 0 <= s.recency_score <= 1
    assert 0 <= s.importance_score <= 1
    assert 0 <= s.relevance_score <= 1
    assert s.score > 0
    assert s.dense_rank == 1
    assert s.sparse_rank == 1


# ---- scoring axes (one at a time) ----


async def test_recency_dominates_when_only_recency_weight_nonzero():
    older = _record("old", content="generic content", kind="observation",
                    importance=10, embedding=[1, 0], age_hours=48.0)
    newer = _record("new", content="generic content", kind="observation",
                    importance=1, embedding=[1, 0], age_hours=0.0)
    adapter = _StubEmbedder({"query": [1, 0]})
    stream = _stream_with([older, newer], adapter)
    config = _mem_config(recency_weight=1.0, importance_weight=0.0, relevance_weight=0.0)
    retriever = MemoryRetriever(stream, adapter, config, clock=_fixed_clock)
    results = await retriever.retrieve("query")
    assert [s.record.id for s in results[:2]] == ["new", "old"]


async def test_importance_dominates_when_only_importance_weight_nonzero():
    recent_low = _record("low", content="generic content", kind="observation",
                         importance=1, embedding=[1, 0], age_hours=0.0)
    old_high = _record("high", content="generic content", kind="observation",
                       importance=10, embedding=[1, 0], age_hours=48.0)
    adapter = _StubEmbedder({"query": [1, 0]})
    stream = _stream_with([recent_low, old_high], adapter)
    config = _mem_config(recency_weight=0.0, importance_weight=1.0, relevance_weight=0.0)
    retriever = MemoryRetriever(stream, adapter, config, clock=_fixed_clock)
    results = await retriever.retrieve("query")
    assert [s.record.id for s in results[:2]] == ["high", "low"]


async def test_relevance_dominates_when_only_relevance_weight_nonzero():
    irrelevant = _record("ir", content="spanish homework", kind="observation",
                         importance=10, embedding=[0.0, 1.0], age_hours=0.0)
    relevant = _record("re", content="python classroom", kind="observation",
                       importance=1, embedding=[1.0, 0.0], age_hours=200.0)
    adapter = _StubEmbedder({"python": [1.0, 0.0]})
    stream = _stream_with([irrelevant, relevant], adapter)
    config = _mem_config(recency_weight=0.0, importance_weight=0.0, relevance_weight=1.0)
    retriever = MemoryRetriever(stream, adapter, config, clock=_fixed_clock)
    results = await retriever.retrieve("python")
    assert results[0].record.id == "re"


# ---- per-kind rules ----


async def test_fact_bypasses_recency_decay():
    old_fact = _record("f", content="I'm allergic to peanuts", kind="fact",
                       importance=8, embedding=[0, 0, 1], age_hours=500.0)
    adapter = _StubEmbedder({"q": [0, 0, 1]})
    stream = _stream_with([old_fact], adapter)
    retriever = MemoryRetriever(stream, adapter, _mem_config(), clock=_fixed_clock)
    results = await retriever.retrieve("q")
    assert results[0].recency_score == 1.0


async def test_preference_bypasses_recency_decay():
    old_pref = _record("p", content="I hate morning classes", kind="preference",
                       importance=7, embedding=[0, 0, 1], age_hours=500.0)
    adapter = _StubEmbedder({"q": [0, 0, 1]})
    stream = _stream_with([old_pref], adapter)
    retriever = MemoryRetriever(stream, adapter, _mem_config(), clock=_fixed_clock)
    results = await retriever.retrieve("q")
    assert results[0].recency_score == 1.0


async def test_observation_decays_with_age():
    new_obs = _record("new", content="a", kind="observation",
                      importance=5, embedding=[1, 0], age_hours=0.0)
    old_obs = _record("old", content="a", kind="observation",
                      importance=5, embedding=[1, 0], age_hours=24.0)
    adapter = _StubEmbedder({"q": [1, 0]})
    stream = _stream_with([new_obs, old_obs], adapter)
    retriever = MemoryRetriever(stream, adapter, _mem_config(), clock=_fixed_clock)
    results = await retriever.retrieve("q")
    new_result = next(s for s in results if s.record.id == "new")
    old_result = next(s for s in results if s.record.id == "old")
    assert new_result.recency_score == pytest.approx(1.0, abs=0.01)
    assert old_result.recency_score == pytest.approx(0.5, abs=0.01)   # 1 half-life


async def test_plan_with_status_done_is_excluded():
    active = _record("a", content="active plan", kind="plan", importance=7,
                     embedding=[1, 0], metadata={"status": "active"})
    done = _record("d", content="done plan", kind="plan", importance=7,
                   embedding=[1, 0], metadata={"status": "done"})
    adapter = _StubEmbedder({"q": [1, 0]})
    stream = _stream_with([active, done], adapter)
    retriever = MemoryRetriever(stream, adapter, _mem_config(), clock=_fixed_clock)
    results = await retriever.retrieve("q")
    assert [s.record.id for s in results] == ["a"]


async def test_plan_without_status_is_kept():
    rec = _record("a", content="plan without status", kind="plan",
                  importance=7, embedding=[1, 0])
    adapter = _StubEmbedder({"q": [1, 0]})
    stream = _stream_with([rec], adapter)
    retriever = MemoryRetriever(stream, adapter, _mem_config(), clock=_fixed_clock)
    assert [s.record.id for s in await retriever.retrieve("q")] == ["a"]


# ---- kinds filter ----


async def test_kinds_filter_restricts_results():
    obs = _record("o", content="obs", kind="observation", importance=5, embedding=[1, 0])
    fact = _record("f", content="fact", kind="fact", importance=5, embedding=[1, 0])
    adapter = _StubEmbedder({"q": [1, 0]})
    stream = _stream_with([obs, fact], adapter)
    retriever = MemoryRetriever(stream, adapter, _mem_config(), clock=_fixed_clock)
    results = await retriever.retrieve("q", kinds=("fact",))
    assert [s.record.id for s in results] == ["f"]


async def test_kinds_filter_accepts_multiple():
    obs = _record("o", content="obs", kind="observation", importance=5, embedding=[1, 0])
    fact = _record("f", content="fact", kind="fact", importance=5, embedding=[1, 0])
    pref = _record("p", content="pref", kind="preference", importance=5, embedding=[1, 0])
    adapter = _StubEmbedder({"q": [1, 0]})
    stream = _stream_with([obs, fact, pref], adapter)
    retriever = MemoryRetriever(stream, adapter, _mem_config(), clock=_fixed_clock)
    results = await retriever.retrieve("q", kinds=("fact", "preference"))
    assert {s.record.id for s in results} == {"f", "p"}


# ---- hybrid retrieval: dense vs sparse ----


async def test_dense_win_when_bm25_ties():
    """Identical tokens → identical BM25; differing embeddings → dense decides."""
    r_close = _record("close", content="python python", kind="observation",
                      importance=5, embedding=[1.0, 0.0])
    r_far = _record("far", content="python python", kind="observation",
                    importance=5, embedding=[0.0, 1.0])
    adapter = _StubEmbedder({"python": [1.0, 0.0]})
    stream = _stream_with([r_close, r_far], adapter)
    retriever = MemoryRetriever(stream, adapter, _mem_config(), clock=_fixed_clock)
    results = await retriever.retrieve("python")
    assert results[0].record.id == "close"


async def test_sparse_win_when_embeddings_tie():
    """Identical embeddings → dense ties; different tokens → BM25 decides."""
    r_kw = _record("kw", content="python data structures", kind="observation",
                   importance=5, embedding=[1.0, 0.0])
    r_nokw = _record("nokw", content="spanish history homework", kind="observation",
                     importance=5, embedding=[1.0, 0.0])
    adapter = _StubEmbedder({"python": [1.0, 0.0]})
    stream = _stream_with([r_kw, r_nokw], adapter)
    retriever = MemoryRetriever(stream, adapter, _mem_config(), clock=_fixed_clock)
    results = await retriever.retrieve("python")
    assert results[0].record.id == "kw"


def test_rrf_constant_is_the_standard_value():
    """Guard against accidental mis-tuning."""
    assert RRF_K == 60


# ---- token budget (max_working_memory_tokens) ----


async def test_retrieval_respects_tight_token_budget():
    """Budget binds before top_k — fewer records returned than top_k allows."""
    # Each content is 40 chars → ~10 tokens. 5 records × 10 = 50 tokens.
    contents = ["word " * 8 for _ in range(5)]   # "word word word word word word word word "
    records = [
        _record(f"r{i}", content=contents[i], kind="observation",
                importance=5, embedding=[1, 0])
        for i in range(5)
    ]
    adapter = _StubEmbedder({"q": [1, 0], **{c: [1, 0] for c in contents}})
    stream = _stream_with(records, adapter)
    # top_k=5 would return all; budget=25 tokens fits ~2 records (top-1 unconditional + 1 more).
    config = _mem_config(retrieval_top_k=5, max_working_memory_tokens=25)
    retriever = MemoryRetriever(stream, adapter, config, clock=_fixed_clock)

    results = await retriever.retrieve("q")

    assert 1 <= len(results) < 5   # budget binds, not top_k
    total_tokens = sum(max(1, len(s.record.content) // 4) for s in results)
    # The budget check stops BEFORE adding a record that would overflow; so
    # with the top-1 always-in rule, total_tokens may slightly exceed the
    # budget only when top-1 alone exceeds it — verify sanely:
    assert total_tokens <= 25 + (len(results[0].record.content) // 4)


async def test_retrieval_always_includes_top_1_even_if_over_budget():
    """A tight budget should never yield an empty result."""
    big_content = "x" * 10000    # ~2500 tokens alone
    record = _record("big", content=big_content, kind="observation",
                     importance=5, embedding=[1, 0])
    adapter = _StubEmbedder({"q": [1, 0], big_content: [1, 0]})
    stream = _stream_with([record], adapter)
    config = _mem_config(retrieval_top_k=5, max_working_memory_tokens=100)
    retriever = MemoryRetriever(stream, adapter, config, clock=_fixed_clock)

    results = await retriever.retrieve("q")
    assert len(results) == 1
    assert results[0].record.id == "big"


async def test_retrieval_top_k_binds_when_budget_is_roomy():
    """Generous budget → top_k is the binding constraint."""
    records = [
        _record(f"r{i}", content="short", kind="observation",
                importance=5, embedding=[1, 0])
        for i in range(10)
    ]
    adapter = _StubEmbedder({"q": [1, 0], "short": [1, 0]})
    stream = _stream_with(records, adapter)
    config = _mem_config(retrieval_top_k=3, max_working_memory_tokens=4000)
    retriever = MemoryRetriever(stream, adapter, config, clock=_fixed_clock)

    results = await retriever.retrieve("q")
    assert len(results) == 3   # top_k binds (budget is ~4000, content total is tiny)


async def test_retrieval_budget_caps_regardless_of_explicit_top_k():
    """Caller-supplied top_k override is still subject to the token budget."""
    contents = ["word " * 8 for _ in range(6)]
    records = [
        _record(f"r{i}", content=contents[i], kind="observation",
                importance=5, embedding=[1, 0])
        for i in range(6)
    ]
    adapter = _StubEmbedder({"q": [1, 0], **{c: [1, 0] for c in contents}})
    stream = _stream_with(records, adapter)
    config = _mem_config(retrieval_top_k=1, max_working_memory_tokens=25)
    retriever = MemoryRetriever(stream, adapter, config, clock=_fixed_clock)

    # Caller asks for 10; budget-aware loop still caps based on tokens.
    results = await retriever.retrieve("q", top_k=10)
    assert len(results) >= 1
    assert len(results) < 6   # at least one dropped due to budget


# ============================================================
# Memory facade
# ============================================================


def test_memory_constructed_from_profile_and_adapter():
    embedder = _StubEmbedder()
    profile = _profile()
    memory = Memory(profile, embedder, clock=_fixed_clock)
    assert memory.profile is profile
    assert memory.embedding_adapter is embedder


def test_memory_exposes_stream_and_retriever_properties():
    """Advanced callers should still reach the underlying pieces."""
    memory = Memory(_profile(), _StubEmbedder(), clock=_fixed_clock)
    assert isinstance(memory.stream, MemoryStream)
    assert isinstance(memory.retriever, MemoryRetriever)


def test_memory_len_reflects_stream_size():
    memory = Memory(_profile(), _StubEmbedder(), clock=_fixed_clock)
    assert len(memory) == 0


def test_memory_repr_is_informative():
    memory = Memory(_profile(id_="student_1"), _StubEmbedder(), clock=_fixed_clock)
    assert "Memory" in repr(memory)


# ---- Memory.from_env ----

_ENV_VARS = ["EMBEDDING_PROVIDER", "EMBEDDING_API_KEY", "EMBEDDING_BASE_URL", "EMBEDDING_MODEL"]


@pytest.fixture
def clear_embedding_env(monkeypatch):
    for v in _ENV_VARS:
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(
        "DefenseAgent.memory.memory.load_dotenv", lambda *a, **kw: None,
    )
    yield


def _set(monkeypatch, **kv):
    for k, v in kv.items():
        monkeypatch.setenv(k, v)


def test_from_env_qwen(monkeypatch, clear_embedding_env):
    _set(monkeypatch,
         EMBEDDING_PROVIDER="qwen",
         EMBEDDING_API_KEY="sk-q",
         EMBEDDING_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1",
         EMBEDDING_MODEL="text-embedding-v3")
    memory = Memory.from_env(_profile(), load_env=False)
    assert isinstance(memory.embedding_adapter, OpenAICompatibleEmbeddingAdapter)
    assert memory.embedding_adapter._model == "text-embedding-v3"


def test_from_env_openai_allows_empty_base_url(monkeypatch, clear_embedding_env):
    _set(monkeypatch,
         EMBEDDING_PROVIDER="openai",
         EMBEDDING_API_KEY="sk-oai",
         EMBEDDING_MODEL="text-embedding-3-small")
    memory = Memory.from_env(_profile(), load_env=False)
    assert isinstance(memory.embedding_adapter, OpenAICompatibleEmbeddingAdapter)


def test_from_env_vllm_allows_empty_api_key(monkeypatch, clear_embedding_env):
    _set(monkeypatch,
         EMBEDDING_PROVIDER="vllm",
         EMBEDDING_BASE_URL="http://localhost:8000/v1",
         EMBEDDING_MODEL="BAAI/bge-large-en")
    memory = Memory.from_env(_profile(), load_env=False)
    assert isinstance(memory.embedding_adapter, OpenAICompatibleEmbeddingAdapter)


def test_from_env_provider_case_insensitive(monkeypatch, clear_embedding_env):
    _set(monkeypatch,
         EMBEDDING_PROVIDER="  Qwen  ",
         EMBEDDING_API_KEY="k", EMBEDDING_BASE_URL="u", EMBEDDING_MODEL="m")
    memory = Memory.from_env(_profile(), load_env=False)
    assert isinstance(memory.embedding_adapter, OpenAICompatibleEmbeddingAdapter)


def test_from_env_missing_provider_raises(clear_embedding_env):
    with pytest.raises(EmbeddingConfigError) as e:
        Memory.from_env(_profile(), load_env=False)
    assert "EMBEDDING_PROVIDER" in str(e.value)


def test_from_env_unknown_provider_raises_and_lists_supported(monkeypatch, clear_embedding_env):
    _set(monkeypatch, EMBEDDING_PROVIDER="deepseek")
    with pytest.raises(EmbeddingConfigError) as e:
        Memory.from_env(_profile(), load_env=False)
    msg = str(e.value)
    for p in ("openai", "qwen", "vllm"):
        assert p in msg


def test_from_env_missing_model_raises(monkeypatch, clear_embedding_env):
    _set(monkeypatch, EMBEDDING_PROVIDER="openai", EMBEDDING_API_KEY="k")
    with pytest.raises(EmbeddingConfigError) as e:
        Memory.from_env(_profile(), load_env=False)
    assert "MODEL" in str(e.value).upper()


def test_from_env_missing_api_key_for_qwen(monkeypatch, clear_embedding_env):
    _set(monkeypatch, EMBEDDING_PROVIDER="qwen",
         EMBEDDING_BASE_URL="u", EMBEDDING_MODEL="m")
    with pytest.raises(EmbeddingConfigError) as e:
        Memory.from_env(_profile(), load_env=False)
    assert "API_KEY" in str(e.value).upper()


def test_from_env_missing_base_url_for_qwen(monkeypatch, clear_embedding_env):
    _set(monkeypatch, EMBEDDING_PROVIDER="qwen",
         EMBEDDING_API_KEY="k", EMBEDDING_MODEL="m")
    with pytest.raises(EmbeddingConfigError) as e:
        Memory.from_env(_profile(), load_env=False)
    assert "BASE_URL" in str(e.value).upper()


def test_from_env_missing_base_url_for_vllm(monkeypatch, clear_embedding_env):
    _set(monkeypatch, EMBEDDING_PROVIDER="vllm", EMBEDDING_MODEL="m")
    with pytest.raises(EmbeddingConfigError) as e:
        Memory.from_env(_profile(), load_env=False)
    assert "BASE_URL" in str(e.value).upper()


# ---- remember() / recall() ----


async def test_remember_delegates_to_stream():
    embedder = _StubEmbedder({"I learned recursion today": [1, 0, 0, 0]})
    memory = Memory(_profile(), embedder, clock=_fixed_clock)
    record = await memory.remember("I learned recursion today", importance=7.0)
    assert record.content == "I learned recursion today"
    assert record.importance == 7.0
    assert len(memory) == 1


async def test_remember_accepts_all_kinds():
    memory = Memory(_profile(), _StubEmbedder(), clock=_fixed_clock)
    kinds = ("observation", "fact", "preference", "plan", "reflection")
    records = [await memory.remember(k, kind=k) for k in kinds]
    assert [r.kind for r in records] == list(kinds)


async def test_remember_forwards_metadata():
    memory = Memory(_profile(), _StubEmbedder(), clock=_fixed_clock)
    r = await memory.remember("plan thing", kind="plan", metadata={"status": "active"})
    assert r.metadata == {"status": "active"}


async def test_recall_delegates_to_retriever():
    embedder = _StubEmbedder({
        "query": [1, 0, 0, 0],
        "relevant content": [1, 0, 0, 0],
        "unrelated": [0, 1, 0, 0],
    })
    memory = Memory(_profile(), embedder, clock=_fixed_clock)
    await memory.remember("relevant content", importance=6)
    await memory.remember("unrelated", importance=6)
    results = await memory.recall("query", top_k=1)
    assert len(results) == 1
    assert results[0].record.content == "relevant content"


async def test_recall_top_k_defaults_to_profile_retrieval_top_k():
    """The facade threads MemoryConfig.retrieval_top_k through to the retriever."""
    profile = _profile(memory={"retrieval_top_k": 3})
    table = {"q": [1, 0, 0, 0]}
    for i in range(7):
        v = [0.0, 0.0, 0.0, 0.0]
        v[i % 4] = 1.0 + i * 0.01
        table[f"obs {i}"] = v
    memory = Memory(profile, _StubEmbedder(table), clock=_fixed_clock)
    for i in range(7):
        await memory.remember(f"obs {i}", importance=5)
    assert len(await memory.recall("q")) == 3
