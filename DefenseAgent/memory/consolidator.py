"""MemoryConsolidator — lifecycle job for the four-tier architecture.

Promotes high-importance items along the lifecycle pipeline:

    WORKING ─(importance ≥ promote_to_episodic_threshold)─→ EPISODIC
    EPISODIC ─(≥ promote_to_semantic_threshold)──────────→ SEMANTIC
    SEMANTIC ─(≥ promote_to_procedural_threshold)────────→ PROCEDURAL

On promotion the source's importance is multiplied by
`importance_boost_on_promotion` (capped at 1.0) — the Hello-Agents convention
for "this item earned its place in a longer-lived tier, weight it accordingly."

Disabled by default. Drive it as a one-shot via `await run_once()` or as a
background asyncio task via `start()` / `stop()`. The background loop is
best-effort: any exception in a single pass is swallowed so the loop survives
to the next interval.

Limitation (v1): the "already-promoted" check is in-memory only — restart
may re-promote items that already moved up. For production durability, wrap
this consolidator and persist the `_promoted_ids` set externally.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from DefenseAgent.config.profile import ConsolidationConfig
from DefenseAgent.llm.types import Message
from DefenseAgent.memory.orchestrator import MemoryOrchestrator
from DefenseAgent.memory.types import MemoryItem, MemoryTier


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# Tier promotion edges — each maps source → (target, threshold attribute name).
_PROMOTION_EDGES: list[tuple[MemoryTier, MemoryTier, str]] = [
    (MemoryTier.WORKING, MemoryTier.EPISODIC, "promote_to_episodic_threshold"),
    (MemoryTier.EPISODIC, MemoryTier.SEMANTIC, "promote_to_semantic_threshold"),
    (MemoryTier.SEMANTIC, MemoryTier.PROCEDURAL, "promote_to_procedural_threshold"),
]


@dataclass
class ConsolidationStats:
    """Per-pass counters returned from `run_once()` for observability /
    metrics. Each field is incremented exactly once per item moved."""

    promoted_to_episodic: int = 0
    promoted_to_semantic: int = 0
    promoted_to_procedural: int = 0
    skipped_already_promoted: int = 0
    skipped_below_threshold: int = 0
    errors: int = 0
    promoted_ids: list[str] = field(default_factory=list)


class MemoryConsolidator:
    """Lifecycle promotion job over a `MemoryOrchestrator`."""

    def __init__(
        self,
        orchestrator: MemoryOrchestrator,
        config: ConsolidationConfig | None = None,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.orchestrator = orchestrator
        self.config = config or orchestrator.profile.memory.consolidation
        self._clock = clock or _utcnow
        self._task: asyncio.Task[None] | None = None
        self._stop_event: asyncio.Event | None = None
        # Track records we've already promoted to avoid re-doing work each
        # pass. Keyed by `record_id` for persistent records; for working
        # records (no stable id), keyed by content+timestamp digest.
        self._promoted_ids: set[str] = set()

    # ------------------------------------------------------------------ public

    async def run_once(self) -> ConsolidationStats:
        """Execute one promotion pass across all three lifecycle edges."""
        stats = ConsolidationStats()
        await self._promote_working(stats)
        for source, target, threshold_attr in _PROMOTION_EDGES[1:]:
            await self._promote_persistent(source, target, threshold_attr, stats)
        return stats

    async def start(self) -> None:
        """Spawn a background asyncio task that calls `run_once()` every
        `interval_seconds`. Idempotent — calling start() while already running
        is a no-op."""
        if self._task is not None and not self._task.done():
            return
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(self._run_loop(), name="memory-consolidator")

    async def stop(self) -> None:
        """Signal the background task to exit and await its completion. Safe
        to call when not running."""
        if self._task is None or self._stop_event is None:
            return
        self._stop_event.set()
        try:
            await self._task
        finally:
            self._task = None
            self._stop_event = None

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    # --------------------------------------------------------------- internals

    async def _run_loop(self) -> None:
        """Background loop: run_once → wait interval → repeat. Errors in one
        pass don't kill the loop (consolidation is best-effort)."""
        assert self._stop_event is not None
        interval = self.config.interval_seconds
        while not self._stop_event.is_set():
            try:
                await self.run_once()
            except Exception:  # noqa: BLE001 — log-and-continue is intentional
                # No logger dependency at this layer; production wiring should
                # subclass and override this if metrics/alerting is desired.
                pass
            # Sleep until interval expires OR stop is signaled.
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=interval)
            except asyncio.TimeoutError:
                continue

    async def _promote_working(self, stats: ConsolidationStats) -> None:
        """Move qualifying WORKING items into EPISODIC. Working has no stable
        record id, so we synthesize one from content + timestamp for the
        already-promoted check; collisions only matter if two items happen to
        share both, which is exceedingly rare for free-text content."""
        working = self.orchestrator.working
        if working is None:
            return
        threshold = self.config.promote_to_episodic_threshold
        for item in working.snapshot():
            key = self._working_item_key(item)
            if key in self._promoted_ids:
                stats.skipped_already_promoted += 1
                continue
            if item.importance < threshold:
                stats.skipped_below_threshold += 1
                continue
            try:
                await self._promote(item, MemoryTier.EPISODIC)
                stats.promoted_to_episodic += 1
                stats.promoted_ids.append(key)
                self._promoted_ids.add(key)
            except Exception:  # noqa: BLE001
                stats.errors += 1

    async def _promote_persistent(
        self,
        source_tier: MemoryTier,
        target_tier: MemoryTier,
        threshold_attr: str,
        stats: ConsolidationStats,
    ) -> None:
        """Move qualifying items from one persistent tier to the next."""
        threshold: float = getattr(self.config, threshold_attr)
        items = self.orchestrator.get_items(tier=source_tier)
        for item in items:
            key = item.record_id or self._working_item_key(item)
            if key in self._promoted_ids:
                stats.skipped_already_promoted += 1
                continue
            if item.importance < threshold:
                stats.skipped_below_threshold += 1
                continue
            try:
                await self._promote(item, target_tier)
                self._increment_counter(stats, target_tier)
                stats.promoted_ids.append(key)
                self._promoted_ids.add(key)
            except Exception:  # noqa: BLE001
                stats.errors += 1

    async def _promote(self, item: MemoryItem, target_tier: MemoryTier) -> None:
        """Write a copy of `item` into `target_tier` with importance boosted
        and a `consolidated_from` link back to the source. We don't delete the
        source — the persistent tier keeps it as historical evidence; future
        passes skip it via the `_promoted_ids` set."""
        boosted = min(1.0, item.importance * self.config.importance_boost_on_promotion)
        consolidated_from: list[str] = list(item.consolidated_from)
        if item.record_id:
            consolidated_from.append(item.record_id)
        extra = dict(item.extra or {})
        extra["consolidated_from_tier"] = item.tier.value
        await self.orchestrator.add(
            [Message(role="user", content=item.content)],
            tier=target_tier,
            memory_type=item.memory_type,
            importance=boosted,
            source_run_id=item.source_run_id,
            extra=extra,
        )

    @staticmethod
    def _increment_counter(stats: ConsolidationStats, target_tier: MemoryTier) -> None:
        if target_tier is MemoryTier.EPISODIC:
            stats.promoted_to_episodic += 1
        elif target_tier is MemoryTier.SEMANTIC:
            stats.promoted_to_semantic += 1
        elif target_tier is MemoryTier.PROCEDURAL:
            stats.promoted_to_procedural += 1

    @staticmethod
    def _working_item_key(item: MemoryItem) -> str:
        """Stable-enough id for an item lacking a backend record_id. Used both
        for working items (which have no id) and as a fallback for persistent
        items whose backend didn't return one."""
        ts = item.created_at.isoformat() if item.created_at else ""
        return f"{ts}::{hash(item.content)}"
