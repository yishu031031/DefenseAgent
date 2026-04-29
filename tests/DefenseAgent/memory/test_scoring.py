"""Tests for DefenseAgent.memory.scoring — pure functions, no mem0 involved.

Each component (recency / frequency / hybrid) is tested in isolation, plus
end-to-end ordering behavior of `rank_items`.
"""
from datetime import datetime, timedelta, timezone

import math
import pytest

from DefenseAgent.config.profile import ScoringWeights
from DefenseAgent.memory.scoring import (
    frequency_score,
    hybrid_score,
    rank_items,
    recency_score,
)
from DefenseAgent.memory.types import MemoryItem, MemoryTier


# ---------- recency_score ----------


def test_recency_score_at_half_life_is_half():
    """The decay constant is calibrated so age = half_life ⇒ score = 0.5 exactly."""
    now = datetime(2026, 4, 1, tzinfo=timezone.utc)
    seven_days_ago = now - timedelta(days=7)
    score = recency_score(seven_days_ago, half_life_days=7.0, now=now)
    assert math.isclose(score, 0.5, rel_tol=1e-6)


def test_recency_score_two_half_lives_is_quarter():
    now = datetime(2026, 4, 1, tzinfo=timezone.utc)
    two_half_lives_ago = now - timedelta(days=14)
    score = recency_score(two_half_lives_ago, half_life_days=7.0, now=now)
    assert math.isclose(score, 0.25, rel_tol=1e-6)


def test_recency_score_freshly_written_is_one():
    now = datetime.now(timezone.utc)
    assert recency_score(now, half_life_days=7.0, now=now) == 1.0


def test_recency_score_none_created_at_returns_one():
    """Newly-written records may not have a propagated timestamp; treat as fresh
    rather than penalize for missing data."""
    assert recency_score(None, half_life_days=7.0) == 1.0


def test_recency_score_handles_naive_datetime():
    """Some mem0 backends emit naive datetimes; recency must not crash on them."""
    now = datetime(2026, 4, 1, tzinfo=timezone.utc)
    naive = datetime(2026, 3, 25)  # naive — should be coerced to UTC
    score = recency_score(naive, half_life_days=7.0, now=now)
    assert math.isclose(score, 0.5, rel_tol=1e-6)


def test_recency_score_negative_age_clamps_to_one():
    """If created_at is in the future (clock skew across nodes), don't return
    a >1.0 score that would dominate ranking."""
    now = datetime(2026, 4, 1, tzinfo=timezone.utc)
    future = now + timedelta(days=3)
    score = recency_score(future, half_life_days=7.0, now=now)
    assert score == 1.0


def test_recency_score_zero_half_life_returns_one():
    """Defensive: a misconfigured zero half-life shouldn't produce inf/nan."""
    now = datetime.now(timezone.utc)
    score = recency_score(now - timedelta(days=10), half_life_days=0.0, now=now)
    assert score == 1.0


# ---------- frequency_score ----------


def test_frequency_score_zero_count_is_zero():
    assert frequency_score(0) == 0.0


def test_frequency_score_one_access_is_half():
    assert math.isclose(frequency_score(1), 0.5)


def test_frequency_score_saturates_below_one():
    assert frequency_score(10) < 1.0
    assert frequency_score(1000) < 1.0
    assert frequency_score(1000) > frequency_score(10)


def test_frequency_score_negative_count_treated_as_zero():
    """Defensive against bad metadata; never produce negative scores."""
    assert frequency_score(-5) == 0.0


# ---------- hybrid_score ----------


def test_hybrid_score_combines_all_components():
    weights = ScoringWeights(
        similarity=1.0, recency=1.0, importance=1.0, frequency=1.0,
        recency_half_life_days=7.0,
    )
    now = datetime(2026, 4, 1, tzinfo=timezone.utc)
    item = MemoryItem(
        content="x", importance=0.8, access_count=1, created_at=now,
    )
    score = hybrid_score(item, similarity=0.6, weights=weights, now=now)
    # 0.6 (sim) + 1.0 (rec, fresh) + 0.8 (imp) + 0.5 (freq, count=1)
    assert math.isclose(score, 0.6 + 1.0 + 0.8 + 0.5)


def test_hybrid_score_clamps_out_of_range_similarity():
    """Defensive against backends that occasionally emit similarity > 1
    (negative cosine raised to power, normalization bug, etc.)."""
    weights = ScoringWeights(
        similarity=1.0, recency=0.0, importance=0.0, frequency=0.0,
        recency_half_life_days=7.0,
    )
    item = MemoryItem(content="x")
    too_high = hybrid_score(item, similarity=1.5, weights=weights)
    assert too_high == 1.0
    negative = hybrid_score(item, similarity=-0.3, weights=weights)
    assert negative == 0.0


# ---------- rank_items end-to-end ----------


def test_rank_items_sorts_descending_by_hybrid_score():
    weights = ScoringWeights(
        similarity=1.0, recency=0.0, importance=0.0, frequency=0.0,
        recency_half_life_days=7.0,
    )
    a = MemoryItem(content="a")
    b = MemoryItem(content="b")
    c = MemoryItem(content="c")
    ranked = rank_items(
        [(a, 0.5), (b, 0.9), (c, 0.1)],
        weights=weights,
    )
    assert [item.content for item, _ in ranked] == ["b", "a", "c"]


def test_rank_items_respects_limit():
    weights = ScoringWeights(
        similarity=1.0, recency=0.0, importance=0.0, frequency=0.0,
        recency_half_life_days=7.0,
    )
    items = [(MemoryItem(content=str(i)), float(i) / 10) for i in range(10)]
    top3 = rank_items(items, weights=weights, limit=3)
    assert len(top3) == 3
    assert [item.content for item, _ in top3] == ["9", "8", "7"]


def test_rank_items_high_importance_beats_high_similarity_when_weighted():
    """The whole point of hybrid scoring: a slightly-less-relevant but very
    important record can outrank a slightly-more-relevant trivial one."""
    weights = ScoringWeights(
        similarity=0.2, recency=0.0, importance=0.8, frequency=0.0,
        recency_half_life_days=7.0,
    )
    important = MemoryItem(content="important", importance=1.0)
    relevant = MemoryItem(content="relevant", importance=0.1)
    ranked = rank_items(
        [(important, 0.6), (relevant, 0.9)],
        weights=weights,
    )
    assert [item.content for item, _ in ranked] == ["important", "relevant"]


def test_rank_items_recency_breaks_ties_among_equal_similarity():
    """Two equally-similar records — the fresher one should win."""
    weights = ScoringWeights(
        similarity=0.5, recency=0.5, importance=0.0, frequency=0.0,
        recency_half_life_days=7.0,
    )
    now = datetime(2026, 4, 1, tzinfo=timezone.utc)
    fresh = MemoryItem(content="fresh", created_at=now)
    stale = MemoryItem(content="stale", created_at=now - timedelta(days=14))
    ranked = rank_items(
        [(fresh, 0.5), (stale, 0.5)],
        weights=weights,
        now=now,
    )
    assert [item.content for item, _ in ranked] == ["fresh", "stale"]
