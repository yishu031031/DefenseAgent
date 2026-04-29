"""Tests for DefenseAgent.memory.types — the MemoryItem / MemoryTier data model.

Pure-Python tests, no mem0 involved: round-trip serialization, default
handling for legacy (pre-tier) records, importance validation, and the
`with_access` / `with_importance` derive-new-instance helpers.
"""
from datetime import datetime, timedelta, timezone

import pytest

from DefenseAgent.memory.types import (
    DEFAULT_IMPORTANCE,
    MemoryItem,
    MemoryTier,
)


# ---------- construction & validation ----------


def test_memory_item_defaults_are_safe():
    """An item built with only `content=` must be a valid EPISODIC record at
    neutral importance — this is the path legacy callers take when they upgrade
    without setting tier-aware fields."""
    item = MemoryItem(content="hello")
    assert item.tier == MemoryTier.EPISODIC
    assert item.importance == DEFAULT_IMPORTANCE
    assert item.access_count == 0
    assert item.memory_type is None
    assert item.consolidated_from == ()


@pytest.mark.parametrize("bad", [-0.01, 1.01, 2.0, -10.0])
def test_memory_item_rejects_out_of_range_importance(bad: float):
    with pytest.raises(ValueError, match="importance"):
        MemoryItem(content="x", importance=bad)


def test_memory_item_accepts_boundary_importance():
    """0.0 and 1.0 are valid (the consolidation policy uses both ends)."""
    assert MemoryItem(content="x", importance=0.0).importance == 0.0
    assert MemoryItem(content="x", importance=1.0).importance == 1.0


def test_memory_item_is_frozen():
    item = MemoryItem(content="x")
    with pytest.raises(AttributeError):
        item.importance = 0.9  # type: ignore[misc]


# ---------- to_metadata / from_record round-trip ----------


def test_to_metadata_emits_tier_importance_access_count():
    """These three fields are always present so records written by this version
    are recognizable as tier-aware on read-back."""
    item = MemoryItem(
        content="x",
        tier=MemoryTier.SEMANTIC,
        importance=0.8,
        access_count=3,
    )
    meta = item.to_metadata()
    assert meta["tier"] == "semantic"
    assert meta["importance"] == 0.8
    assert meta["access_count"] == 3


def test_to_metadata_omits_unset_optional_fields():
    """memory_type / last_accessed_at / source_run_id / consolidated_from are
    None-by-default; they should not pollute the metadata dict when unset."""
    item = MemoryItem(content="x")
    meta = item.to_metadata()
    assert "memory_type" not in meta
    assert "last_accessed_at" not in meta
    assert "source_run_id" not in meta
    assert "consolidated_from" not in meta


def test_to_metadata_preserves_extra_fields():
    item = MemoryItem(content="x", extra={"custom_tag": "foo"})
    meta = item.to_metadata()
    assert meta["custom_tag"] == "foo"


def test_to_metadata_owned_keys_beat_extra():
    """If the user puts `tier` in `extra`, the canonical field wins — `extra` is
    a passthrough channel for unknown keys, not a back door."""
    item = MemoryItem(
        content="x", tier=MemoryTier.SEMANTIC, extra={"tier": "should_lose"},
    )
    assert item.to_metadata()["tier"] == "semantic"


def test_round_trip_preserves_all_fields():
    now = datetime.now(timezone.utc)
    item = MemoryItem(
        content="round-trip me",
        tier=MemoryTier.PROCEDURAL,
        memory_type="reflection",
        importance=0.92,
        access_count=5,
        last_accessed_at=now,
        source_run_id="run-42",
        consolidated_from=("a", "b"),
        extra={"foo": "bar"},
    )
    record = {"memory": item.content, "metadata": item.to_metadata()}
    back = MemoryItem.from_record(record)

    assert back.content == "round-trip me"
    assert back.tier == MemoryTier.PROCEDURAL
    assert back.memory_type == "reflection"
    assert back.importance == 0.92
    assert back.access_count == 5
    assert back.last_accessed_at is not None
    assert back.source_run_id == "run-42"
    assert back.consolidated_from == ("a", "b")
    assert back.extra == {"foo": "bar"}


# ---------- from_record tolerance for legacy & malformed input ----------


def test_from_record_legacy_record_defaults_to_episodic():
    """A record written by the pre-tier code only has 'memory' (and maybe
    'memory_type') — must decode cleanly with safe defaults."""
    legacy = {"memory": "old", "metadata": {"memory_type": "trajectory"}}
    item = MemoryItem.from_record(legacy)
    assert item.tier == MemoryTier.EPISODIC
    assert item.importance == DEFAULT_IMPORTANCE
    assert item.access_count == 0
    assert item.memory_type == "trajectory"


def test_from_record_unknown_tier_falls_back_to_episodic():
    """A future-tier value (e.g. 'perceptual' from another fork) shouldn't crash
    the read path — degrade to EPISODIC and preserve the rest of the data."""
    item = MemoryItem.from_record(
        {"memory": "x", "metadata": {"tier": "perceptual"}}
    )
    assert item.tier == MemoryTier.EPISODIC


def test_from_record_clamps_out_of_range_importance():
    item = MemoryItem.from_record(
        {"memory": "x", "metadata": {"importance": 1.5}}
    )
    assert item.importance == 1.0
    item2 = MemoryItem.from_record(
        {"memory": "x", "metadata": {"importance": -0.5}}
    )
    assert item2.importance == 0.0


def test_from_record_top_level_memory_type_wins_over_metadata():
    item = MemoryItem.from_record({
        "memory": "x",
        "memory_type": "outcome",
        "metadata": {"memory_type": "trajectory"},
    })
    assert item.memory_type == "outcome"


def test_from_record_captures_record_id():
    item = MemoryItem.from_record({"id": "uuid-1", "memory": "x"})
    assert item.record_id == "uuid-1"


# ---------- with_access / with_importance ----------


def test_with_access_bumps_count_and_timestamp():
    item = MemoryItem(content="x", access_count=2)
    bumped = item.with_access()
    assert bumped.access_count == 3
    assert bumped.last_accessed_at is not None
    # original is frozen — unchanged.
    assert item.access_count == 2
    assert item.last_accessed_at is None


def test_with_access_explicit_now():
    fixed = datetime(2026, 1, 1, tzinfo=timezone.utc)
    bumped = MemoryItem(content="x").with_access(now=fixed)
    assert bumped.last_accessed_at == fixed


def test_with_importance_clamps_into_range():
    item = MemoryItem(content="x", importance=0.5)
    boosted = item.with_importance(1.5)
    assert boosted.importance == 1.0
    assert item.importance == 0.5  # original untouched


# ---------- timestamp parsing tolerance ----------


def test_from_record_parses_iso_string_timestamp():
    item = MemoryItem.from_record({
        "memory": "x",
        "created_at": "2026-04-01T12:00:00Z",
    })
    assert item.created_at is not None
    assert item.created_at.tzinfo is not None


def test_from_record_tolerates_malformed_timestamp():
    item = MemoryItem.from_record({
        "memory": "x",
        "metadata": {"last_accessed_at": "not-a-date"},
    })
    assert item.last_accessed_at is None


def test_from_record_tolerates_naive_datetime():
    """mem0 backends sometimes emit naive datetimes; treat them as UTC rather
    than crashing the comparison ops downstream scoring will run."""
    naive = datetime(2026, 4, 1, 12, 0, 0)
    item = MemoryItem.from_record({"memory": "x", "created_at": naive})
    assert item.created_at is not None
    assert item.created_at.tzinfo is not None
