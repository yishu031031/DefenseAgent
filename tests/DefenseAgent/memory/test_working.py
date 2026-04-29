"""Tests for DefenseAgent.memory.working.WorkingMemory (P4).

Pure-Python: no mem0, no LLM. Uses an injected clock to make TTL behavior
deterministic.
"""
from datetime import datetime, timedelta, timezone

import pytest

from DefenseAgent.memory.types import MemoryItem, MemoryTier
from DefenseAgent.memory.working import WorkingMemory


class _ClockStub:
    """Hand-cranked clock for deterministic TTL tests."""

    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, delta: timedelta) -> None:
        self.now += delta


def _item(content: str, **overrides) -> MemoryItem:
    return MemoryItem(content=content, **overrides)


# ---------- construction validation ----------


@pytest.mark.parametrize("bad_capacity", [0, -1, -100])
def test_capacity_must_be_positive(bad_capacity: int):
    with pytest.raises(ValueError, match="capacity"):
        WorkingMemory(capacity=bad_capacity, ttl_seconds=60)


@pytest.mark.parametrize("bad_ttl", [0, -1])
def test_ttl_must_be_positive(bad_ttl: int):
    with pytest.raises(ValueError, match="ttl_seconds"):
        WorkingMemory(capacity=10, ttl_seconds=bad_ttl)


# ---------- add / search basics ----------


def test_add_then_search_finds_substring():
    wm = WorkingMemory(capacity=10, ttl_seconds=60)
    wm.add(_item("Maya prefers the library"))
    wm.add(_item("today is sunny"))

    hits = wm.search("library", limit=5)
    assert len(hits) == 1
    assert hits[0].content == "Maya prefers the library"


def test_search_is_case_insensitive():
    wm = WorkingMemory(capacity=10, ttl_seconds=60)
    wm.add(_item("Library Hours: 9am-5pm"))

    assert wm.search("library", limit=5)[0].content.startswith("Library")


def test_search_returns_most_recent_first():
    """Reverse-chronological order matches the LLM's expectation that
    "more recent context wins" — same convention chat models use."""
    wm = WorkingMemory(capacity=10, ttl_seconds=60)
    wm.add(_item("matching first"))
    wm.add(_item("matching second"))
    wm.add(_item("matching third"))

    hits = wm.search("matching", limit=10)
    assert [h.content for h in hits] == ["matching third", "matching second", "matching first"]


def test_search_empty_query_returns_empty():
    wm = WorkingMemory(capacity=10, ttl_seconds=60)
    wm.add(_item("anything"))
    assert wm.search("", limit=5) == []


def test_search_respects_limit():
    wm = WorkingMemory(capacity=10, ttl_seconds=60)
    for i in range(5):
        wm.add(_item(f"matching {i}"))

    assert len(wm.search("matching", limit=2)) == 2


def test_add_coerces_tier_to_working():
    """The orchestrator's `add()` builds items already tagged WORKING, but a
    direct caller might pass an EPISODIC item by accident — coerce defensively."""
    wm = WorkingMemory(capacity=10, ttl_seconds=60)
    wm.add(_item("x", tier=MemoryTier.EPISODIC))

    snap = wm.snapshot()
    assert snap[0].tier == MemoryTier.WORKING


def test_add_stamps_created_at_when_missing():
    clock = _ClockStub(datetime(2026, 4, 1, tzinfo=timezone.utc))
    wm = WorkingMemory(capacity=10, ttl_seconds=60, clock=clock)
    wm.add(_item("no timestamp"))

    snap = wm.snapshot()
    assert snap[0].created_at == clock.now


def test_add_preserves_existing_created_at():
    """A caller-supplied `created_at` is kept verbatim — but it must be within
    the TTL window or the next prune drops the item before snapshot sees it."""
    clock = _ClockStub(datetime(2026, 4, 1, tzinfo=timezone.utc))
    wm = WorkingMemory(capacity=10, ttl_seconds=120, clock=clock)
    fresh = clock.now - timedelta(seconds=30)
    wm.add(_item("preserved", created_at=fresh))

    assert wm.snapshot()[0].created_at == fresh


# ---------- capacity (FIFO) ----------


def test_capacity_evicts_oldest_first():
    wm = WorkingMemory(capacity=3, ttl_seconds=3600)
    for i in range(5):
        wm.add(_item(f"item {i}"))

    contents = [i.content for i in wm.snapshot()]
    assert contents == ["item 2", "item 3", "item 4"]


def test_len_tracks_size():
    wm = WorkingMemory(capacity=10, ttl_seconds=60)
    assert len(wm) == 0
    wm.add(_item("one"))
    assert len(wm) == 1
    wm.add(_item("two"))
    assert len(wm) == 2


# ---------- TTL eviction ----------


def test_ttl_drops_expired_items_on_search():
    clock = _ClockStub(datetime(2026, 4, 1, tzinfo=timezone.utc))
    wm = WorkingMemory(capacity=10, ttl_seconds=60, clock=clock)
    wm.add(_item("expires soon"))

    clock.advance(timedelta(seconds=120))
    assert wm.search("expires", limit=5) == []
    assert len(wm) == 0


def test_ttl_drops_expired_items_on_add():
    """Adds also trigger expiration so a long-idle WorkingMemory doesn't grow
    unbounded between searches."""
    clock = _ClockStub(datetime(2026, 4, 1, tzinfo=timezone.utc))
    wm = WorkingMemory(capacity=10, ttl_seconds=60, clock=clock)
    wm.add(_item("old"))

    clock.advance(timedelta(seconds=120))
    wm.add(_item("new"))

    contents = [i.content for i in wm.snapshot()]
    assert contents == ["new"]


def test_ttl_keeps_fresh_items():
    clock = _ClockStub(datetime(2026, 4, 1, tzinfo=timezone.utc))
    wm = WorkingMemory(capacity=10, ttl_seconds=60, clock=clock)
    wm.add(_item("recent"))

    clock.advance(timedelta(seconds=30))
    assert len(wm.search("recent", limit=5)) == 1


def test_ttl_partial_eviction_keeps_younger_items():
    """The deque is sorted by created_at, so the loop should stop at the first
    not-yet-expired item — this catches the case where only a prefix is gone."""
    clock = _ClockStub(datetime(2026, 4, 1, tzinfo=timezone.utc))
    wm = WorkingMemory(capacity=10, ttl_seconds=60, clock=clock)
    wm.add(_item("oldest"))
    clock.advance(timedelta(seconds=40))
    wm.add(_item("middle"))
    clock.advance(timedelta(seconds=40))  # oldest now 80s old, middle 40s
    wm.add(_item("newest"))

    # oldest should drop, middle and newest survive (TTL=60s, oldest is 80s).
    contents = [i.content for i in wm.snapshot()]
    assert contents == ["middle", "newest"]


# ---------- clear ----------


def test_clear_drops_everything():
    wm = WorkingMemory(capacity=10, ttl_seconds=60)
    for i in range(5):
        wm.add(_item(f"item {i}"))

    wm.clear()
    assert len(wm) == 0
    assert wm.search("item", limit=5) == []


# ---------- from_profile ----------


def test_from_profile_reads_tier_limits():
    from DefenseAgent.config.profile import AgentProfile, MemoryConfig, TierLimits

    profile = AgentProfile(
        id="t", name="t",
        memory=MemoryConfig(
            tier_limits=TierLimits(working_capacity=7, working_ttl_seconds=120),
        ),
    )
    wm = WorkingMemory.from_profile(profile)

    assert wm._capacity == 7
    assert wm._ttl == timedelta(seconds=120)
