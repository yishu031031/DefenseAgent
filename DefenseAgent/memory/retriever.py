from dataclasses import dataclass
from datetime import datetime
from typing import Callable

from DefenseAgent.config.profile import MemoryConfig
from DefenseAgent.memory.embedding import EmbeddingAdapter
from DefenseAgent.memory.stream import (
    MemoryKind,
    MemoryRecord,
    MemoryStream,
    _default_clock,
    cosine,
)


RRF_K = 60
_NO_DECAY_KINDS = frozenset({"fact", "preference"})


@dataclass
class ScoredMemory:
    """A record plus every score component that produced its rank during retrieval."""
    record: MemoryRecord
    score: float
    recency_score: float
    importance_score: float
    relevance_score: float
    dense_rank: int | None
    sparse_rank: int | None


def estimate_tokens(text: str) -> int:
    """Approximate token count at ~4 characters per token; minimum 1."""
    return max(1, len(text) // 4)


def ranks_descending(scores: dict[str, float]) -> dict[str, int]:
    """Return a {id: 1-indexed rank} dict sorted by value descending (ties break by insertion order)."""
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    ranks: dict[str, int] = {}
    for rank, (rid, _) in enumerate(ordered, start=1):
        ranks[rid] = rank
    return ranks


class MemoryRetriever:
    """Hybrid retrieval: dense cosine + sparse BM25 fused with RRF, weighted per MemoryConfig."""

    def __init__(
        self,
        stream: MemoryStream,
        embedding_adapter: EmbeddingAdapter,
        memory_config: MemoryConfig,
        *,
        clock: Callable[[], datetime] | None = None,
        recency_half_life_hours: float = 24.0,
    ) -> None:
        """Bind the stream, embedding adapter, scoring weights, clock, and recency decay half-life."""
        self.stream = stream
        self.embedding_adapter = embedding_adapter
        self.memory_config = memory_config
        self.recency_half_life_hours = recency_half_life_hours
        if clock is None:
            self._clock = _default_clock
        else:
            self._clock = clock

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        kinds: tuple[MemoryKind, ...] | None = None,
    ) -> list[ScoredMemory]:
        """Run the hybrid pipeline and return top-k scored memories, respecting the token budget."""
        candidates = self._select_candidates(kinds)
        if not candidates:
            return []

        if top_k is None:
            k = self.memory_config.retrieval_top_k
        else:
            k = top_k
        if k <= 0:
            return []

        query_embedding = await self.embedding_adapter.embed(query)

        dense_scores: dict[str, float] = {}
        for r in candidates:
            similarity = cosine(query_embedding, r.embedding)
            if similarity < 0.0:
                similarity = 0.0
            dense_scores[r.id] = similarity
        dense_rank = ranks_descending(dense_scores)

        sparse_raw = self.stream.bm25.score(query)
        sparse_scores: dict[str, float] = {}
        for r in candidates:
            if r.id in sparse_raw:
                sparse_scores[r.id] = sparse_raw[r.id]
            else:
                sparse_scores[r.id] = 0.0
        sparse_rank = ranks_descending(sparse_scores)

        raw_rrf: dict[str, float] = {}
        for r in candidates:
            raw_rrf[r.id] = (
                1.0 / (RRF_K + dense_rank[r.id])
                + 1.0 / (RRF_K + sparse_rank[r.id])
            )
        rrf_max = max(raw_rrf.values())
        hybrid_relevance: dict[str, float] = {}
        for rid, raw in raw_rrf.items():
            hybrid_relevance[rid] = raw / rrf_max

        now = self._clock()
        scored: list[ScoredMemory] = []
        for r in candidates:
            recency_score = self._compute_recency_score(r, now)
            importance_score = r.importance / 10.0
            relevance_score = hybrid_relevance[r.id]
            composite = (
                self.memory_config.recency_weight * recency_score
                + self.memory_config.importance_weight * importance_score
                + self.memory_config.relevance_weight * relevance_score
            )
            scored.append(
                ScoredMemory(
                    record=r,
                    score=composite,
                    recency_score=recency_score,
                    importance_score=importance_score,
                    relevance_score=relevance_score,
                    dense_rank=dense_rank[r.id],
                    sparse_rank=sparse_rank[r.id],
                )
            )

        scored.sort(key=lambda s: s.score, reverse=True)
        return self._apply_token_budget(scored, k)

    def _select_candidates(
        self,
        kinds: tuple[MemoryKind, ...] | None,
    ) -> list[MemoryRecord]:
        """Return the records eligible for ranking after filtering by `kinds` and excluding done plans."""
        candidates: list[MemoryRecord] = []
        for r in self.stream.get_all():
            if kinds is not None and r.kind not in kinds:
                continue
            if r.kind == "plan" and r.metadata.get("status") == "done":
                continue
            candidates.append(r)
        return candidates

    def _compute_recency_score(self, record: MemoryRecord, now: datetime) -> float:
        """Return 1.0 for kinds that bypass decay, else 2**(-age_hours/half_life)."""
        if record.kind in _NO_DECAY_KINDS:
            return 1.0
        age_seconds = (now - record.timestamp).total_seconds()
        age_hours = age_seconds / 3600.0
        if age_hours < 0.0:
            age_hours = 0.0
        return 2.0 ** (-age_hours / self.recency_half_life_hours)

    def _apply_token_budget(
        self,
        scored: list[ScoredMemory],
        k: int,
    ) -> list[ScoredMemory]:
        """Walk `scored` in order, stopping at the token budget; always include top-1, always cap at k."""
        budget = self.memory_config.max_working_memory_tokens
        picked: list[ScoredMemory] = []
        running_tokens = 0
        for s in scored:
            cost = estimate_tokens(s.record.content)
            if picked and running_tokens + cost > budget:
                break
            picked.append(s)
            running_tokens += cost
            if len(picked) >= k:
                break
        return picked
