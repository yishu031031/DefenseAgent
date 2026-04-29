"""Hybrid retrieval scoring — combine vector similarity with recency,
importance, and access-frequency signals into a single rank.

Borrowed from Hello-Agents' "memory recall" formulation. Each component is
normalized into [0, 1] before being weighted by the values in
`AgentProfile.memory.scoring`. Weights need not sum to 1 — overweighting is
intentional when offline evaluation says so.

Components:
- similarity: passed in (cosine from the vector backend, expected [0, 1])
- recency: exponential decay over the record's age, half-life from config
- importance: the record's stored importance, already [0, 1]
- frequency: a saturating function of access_count, asymptotic to 1
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Iterable

from DefenseAgent.config.profile import ScoringWeights
from DefenseAgent.memory.types import MemoryItem


def recency_score(
    created_at: datetime | None,
    *,
    half_life_days: float,
    now: datetime | None = None,
) -> float:
    """Exponential decay: a record half this many days old scores 0.5; one
    twice that old scores 0.25; freshly written or undated records score 1.0.
    Returns a value in (0, 1]."""
    if created_at is None:
        # Newly written records may not have a propagated timestamp yet — treat
        # them as fresh rather than penalizing for missing data.
        return 1.0
    if half_life_days <= 0:
        return 1.0
    ts_now = now if now is not None else datetime.now(timezone.utc)
    if ts_now.tzinfo is None:
        ts_now = ts_now.replace(tzinfo=timezone.utc)
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    age_seconds = max(0.0, (ts_now - created_at).total_seconds())
    age_days = age_seconds / 86_400.0
    decay = math.exp(-math.log(2) * age_days / half_life_days)
    return decay


def frequency_score(access_count: int) -> float:
    """Saturating frequency signal: 0 at count=0, ~0.5 at count=1, ~0.91 at
    count=10, approaches 1 as count grows. No tuning constant needed because
    the asymptote is fixed, but heavy hitters never quite saturate so further
    accesses still nudge the rank."""
    if access_count <= 0:
        return 0.0
    return 1.0 - 1.0 / (1.0 + access_count)


def hybrid_score(
    item: MemoryItem,
    *,
    similarity: float,
    weights: ScoringWeights,
    now: datetime | None = None,
) -> float:
    """Combined relevance: weighted sum of similarity, recency, importance,
    and frequency. The result is NOT normalized to [0, 1] — its absolute scale
    depends on the weights — but it is monotonic for ranking and stable
    across tiers (so cross-tier sorting is well-defined)."""
    sim = max(0.0, min(1.0, similarity))
    rec = recency_score(
        item.created_at, half_life_days=weights.recency_half_life_days, now=now,
    )
    imp = max(0.0, min(1.0, item.importance))
    freq = frequency_score(item.access_count)
    return (
        sim * weights.similarity
        + rec * weights.recency
        + imp * weights.importance
        + freq * weights.frequency
    )


def rank_items(
    scored: Iterable[tuple[MemoryItem, float]],
    *,
    weights: ScoringWeights,
    now: datetime | None = None,
    limit: int | None = None,
) -> list[tuple[MemoryItem, float]]:
    """Re-rank a stream of (item, raw_similarity) pairs using `hybrid_score`.
    Returns a descending-sorted list; clipping to `limit` happens after sort
    so a low-similarity but high-importance/recency item can still beat a
    high-similarity but stale one. The raw similarity is replaced by the
    hybrid score in the returned tuples."""
    out = [
        (item, hybrid_score(item, similarity=sim, weights=weights, now=now))
        for item, sim in scored
    ]
    out.sort(key=lambda pair: pair[1], reverse=True)
    if limit is not None:
        out = out[:limit]
    return out
