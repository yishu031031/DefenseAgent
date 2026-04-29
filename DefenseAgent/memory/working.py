"""WorkingMemory — in-memory, session-scoped short-term storage.

Plays the role of Hello-Agents' working/scratch tier: cheap, immediate,
TTL-bounded, capacity-bounded. No vector embeddings — search is naive
substring matching against item content. For semantic recall the orchestrator
routes to the persistent tiers (Episodic / Semantic / Procedural) via mem0.

Eviction policy is intentionally simple:
- On every add/search, drop items older than `ttl_seconds`.
- When at capacity, drop the oldest item (FIFO). Importance is *not* consulted
  here — short-term memory is meant to roll over; the consolidation job (P5)
  is what promotes high-importance items into longer tiers before they expire.
"""

from __future__ import annotations

from collections import deque
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Callable

from DefenseAgent.config.profile import AgentProfile
from DefenseAgent.memory.types import MemoryItem, MemoryTier


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class WorkingMemory:
    """In-memory short-term memory satisfying `WorkingMemoryProtocol`.

    Thread/coroutine safety: not safe under concurrent mutation. The agent
    loop is sequential per agent instance, so this matches the typical usage;
    callers spawning concurrent coroutines that all write to the same
    WorkingMemory must add their own lock.
    """

    def __init__(
        self,
        *,
        capacity: int = 50,
        ttl_seconds: int = 3600,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1; got {capacity}")
        if ttl_seconds < 1:
            raise ValueError(f"ttl_seconds must be >= 1; got {ttl_seconds}")
        self._capacity = capacity
        self._ttl = timedelta(seconds=ttl_seconds)
        self._items: deque[MemoryItem] = deque()
        self._clock = clock or _utcnow

    @classmethod
    def from_profile(
        cls,
        profile: AgentProfile,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> "WorkingMemory":
        """Build from `profile.memory.tier_limits.working_capacity` and
        `working_ttl_seconds`. The orchestrator uses this so a user who never
        thought about Working still gets sensible defaults."""
        limits = profile.memory.tier_limits
        return cls(
            capacity=limits.working_capacity,
            ttl_seconds=limits.working_ttl_seconds,
            clock=clock,
        )

    # -- WorkingMemoryProtocol surface ----------------------------------

    def add(self, item: MemoryItem) -> None:
        """Append `item` to the working buffer. The orchestrator passes items
        already tagged tier=WORKING; we coerce defensively in case a caller
        builds the MemoryItem directly with the wrong tier."""
        if item.tier is not MemoryTier.WORKING:
            item = replace(item, tier=MemoryTier.WORKING)
        if item.created_at is None:
            item = replace(item, created_at=self._clock())
        self._prune_expired()
        while len(self._items) >= self._capacity:
            self._items.popleft()  # oldest-out FIFO
        self._items.append(item)

    def search(self, query: str, *, limit: int) -> list[MemoryItem]:
        """Substring-match `query` (case-insensitive) against item content,
        return up to `limit` most-recently-added hits. Empty query → []."""
        self._prune_expired()
        if not query:
            return []
        q = query.lower()
        hits = [i for i in self._items if q in i.content.lower()]
        hits.reverse()  # most recent first
        return hits[:limit] if limit else hits

    def clear(self) -> None:
        """Drop everything. Typically called between sessions, or after a
        consolidation pass has promoted the high-importance items elsewhere."""
        self._items.clear()

    # -- introspection (not part of the protocol) -----------------------

    def __len__(self) -> int:
        return len(self._items)

    def snapshot(self) -> list[MemoryItem]:
        """Return a copy of current items in insertion order. Useful for tests
        and the consolidation job (P5) that needs to inspect what's eligible
        for promotion before the TTL takes them."""
        self._prune_expired()
        return list(self._items)

    # -- internals ------------------------------------------------------

    def _prune_expired(self) -> None:
        """Drop items older than `ttl_seconds`. Items are appended in time
        order, so the deque is sorted by `created_at` and we can stop at the
        first non-expired item (cheap O(k) where k = # expired this call)."""
        cutoff = self._clock() - self._ttl
        while self._items:
            head = self._items[0]
            if head.created_at is None or head.created_at >= cutoff:
                break
            self._items.popleft()
