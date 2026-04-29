"""Memory data model — tier-aware, importance-scored, lifecycle-tracked.

Inspired by Hello-Agents' four-layer memory architecture (Working / Episodic /
Semantic / Perceptual). The DefenseAgent adaptation drops Perceptual (no
multimodal need) and adds Procedural — attack patterns, SOPs, and workflows
are first-class for a defense agent.

The on-disk source of truth is still the mem0 record dict; this module is a
typed view over it. `to_metadata()` / `from_record()` are the only conversion
boundaries — anywhere else in the codebase a MemoryItem can be passed around
without reaching for raw dict keys.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class MemoryTier(str, Enum):
    """Lifecycle tier of a memory record. Drives retention, consolidation, and
    retrieval scoring decisions made by the orchestrator and scoring layer."""

    WORKING = "working"        # session-scoped scratchpad, in-memory only
    EPISODIC = "episodic"      # raw events / trajectories, capacity-bounded
    SEMANTIC = "semantic"      # distilled facts / reflections / lessons
    PROCEDURAL = "procedural"  # attack patterns / SOPs / workflows


# Neutral importance — neither protected from eviction nor early-evicted.
DEFAULT_IMPORTANCE = 0.5

# Metadata keys this module owns. Anything else stored under `extra` is
# round-tripped untouched so callers can stash custom fields.
_TIER_KEY = "tier"
_IMPORTANCE_KEY = "importance"
_ACCESS_COUNT_KEY = "access_count"
_LAST_ACCESSED_KEY = "last_accessed_at"
_SOURCE_RUN_KEY = "source_run_id"
_CONSOLIDATED_FROM_KEY = "consolidated_from"
_MEMORY_TYPE_KEY = "memory_type"  # legacy — pre-tier label kept for back-compat


@dataclass(frozen=True)
class MemoryItem:
    """A normalized memory record. `content` is the text mem0 indexes; every
    other field is metadata used by scoring or lifecycle. Frozen so callers
    can't accidentally mutate a record they got back from a search; use
    `with_access()` / `dataclasses.replace()` to derive variants."""

    content: str
    tier: MemoryTier = MemoryTier.EPISODIC
    memory_type: str | None = None
    importance: float = DEFAULT_IMPORTANCE
    access_count: int = 0
    created_at: datetime | None = None
    last_accessed_at: datetime | None = None
    source_run_id: str | None = None
    consolidated_from: tuple[str, ...] = field(default_factory=tuple)
    record_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.importance <= 1.0:
            raise ValueError(
                f"importance must be in [0, 1]; got {self.importance!r}"
            )

    def to_metadata(self) -> dict[str, Any]:
        """Serialize non-content fields into a flat dict suitable for mem0's
        `metadata=` kwarg. None-valued and zero-default fields are still
        emitted so records written by this version are recognizable later."""
        payload: dict[str, Any] = {
            _TIER_KEY: self.tier.value,
            _IMPORTANCE_KEY: self.importance,
            _ACCESS_COUNT_KEY: self.access_count,
        }
        if self.memory_type is not None:
            payload[_MEMORY_TYPE_KEY] = self.memory_type
        if self.last_accessed_at is not None:
            payload[_LAST_ACCESSED_KEY] = self.last_accessed_at.isoformat()
        if self.source_run_id is not None:
            payload[_SOURCE_RUN_KEY] = self.source_run_id
        if self.consolidated_from:
            payload[_CONSOLIDATED_FROM_KEY] = list(self.consolidated_from)
        for k, v in self.extra.items():
            payload.setdefault(k, v)
        return payload

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> MemoryItem:
        """Build a MemoryItem from a mem0 record dict. Tolerates legacy records
        (pre-tier/importance) by defaulting tier=EPISODIC, importance=0.5."""
        metadata = dict(record.get("metadata") or {})
        memory_type = record.get("memory_type") or metadata.pop(_MEMORY_TYPE_KEY, None)

        tier_raw = metadata.pop(_TIER_KEY, MemoryTier.EPISODIC.value)
        try:
            tier = MemoryTier(tier_raw)
        except ValueError:
            tier = MemoryTier.EPISODIC

        try:
            importance = float(metadata.pop(_IMPORTANCE_KEY, DEFAULT_IMPORTANCE))
        except (TypeError, ValueError):
            importance = DEFAULT_IMPORTANCE
        importance = max(0.0, min(1.0, importance))

        try:
            access_count = int(metadata.pop(_ACCESS_COUNT_KEY, 0))
        except (TypeError, ValueError):
            access_count = 0

        last_accessed_at = _parse_iso(metadata.pop(_LAST_ACCESSED_KEY, None))
        created_at = _parse_iso(record.get("created_at"))

        source_run_id = metadata.pop(_SOURCE_RUN_KEY, None)
        consolidated_from_raw = metadata.pop(_CONSOLIDATED_FROM_KEY, None) or []
        consolidated_from = tuple(consolidated_from_raw)

        return cls(
            content=record.get("memory") or record.get("content") or "",
            tier=tier,
            memory_type=memory_type,
            importance=importance,
            access_count=access_count,
            created_at=created_at,
            last_accessed_at=last_accessed_at,
            source_run_id=source_run_id,
            consolidated_from=consolidated_from,
            record_id=record.get("id"),
            extra=metadata,
        )

    def with_access(self, *, now: datetime | None = None) -> MemoryItem:
        """Return a new item with access_count + 1 and last_accessed_at bumped.
        The orchestrator calls this after a successful retrieval and writes the
        result back via mem0's update API so frequency/recency scoring is fed
        accurate data."""
        ts = now if now is not None else datetime.now(timezone.utc)
        return replace(
            self,
            access_count=self.access_count + 1,
            last_accessed_at=ts,
        )

    def with_importance(self, importance: float) -> MemoryItem:
        """Return a new item with `importance` overridden — used by
        consolidation when promoting a record into a higher tier."""
        return replace(self, importance=max(0.0, min(1.0, importance)))


def _parse_iso(value: Any) -> datetime | None:
    """Parse an ISO-8601 string (or pass-through datetime) into a tz-aware
    UTC datetime. mem0 emits timestamps in either form depending on backend
    and version; tolerate both. Falsy / unparseable input → None."""
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    return None
