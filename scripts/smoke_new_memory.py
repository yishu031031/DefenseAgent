"""End-to-end smoke test for the new tier-aware memory module.

Exercises the full public API surface (MemoryItem, MemoryTier, WorkingMemory,
MemoryOrchestrator, MemoryConsolidator) using a mocked mem0 backend so no
external services or .env credentials are required. Prints a pass/fail line
per scenario and exits non-zero if any assertion fails.

Usage:
    python scripts/smoke_new_memory.py
"""
from __future__ import annotations

import asyncio
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch


PASS = "  [PASS]"
FAIL = "  [FAIL]"


class TestRunner:
    def __init__(self) -> None:
        self.failed = 0
        self.passed = 0

    async def run(self, name: str, coro):
        try:
            await coro
            print(f"{PASS} {name}")
            self.passed += 1
        except Exception as e:  # noqa: BLE001
            self.failed += 1
            print(f"{FAIL} {name}")
            print(f"         {type(e).__name__}: {e}")
            for line in traceback.format_exc().splitlines()[-5:]:
                print(f"         {line}")

    def sync(self, name: str, fn):
        try:
            fn()
            print(f"{PASS} {name}")
            self.passed += 1
        except Exception as e:  # noqa: BLE001
            self.failed += 1
            print(f"{FAIL} {name}")
            print(f"         {type(e).__name__}: {e}")
            for line in traceback.format_exc().splitlines()[-5:]:
                print(f"         {line}")


# ----------------------------------------------------------------- helpers


def _set_env(monkeypatch_dict: dict[str, str]) -> None:
    import os
    for k, v in monkeypatch_dict.items():
        os.environ[k] = v


_FAKE_ENV = {
    "AGENT_LAB_LLM_PROVIDER": "deepseek",
    "DEEPSEEK_API_KEY": "sk-test",
    "DEEPSEEK_BASE_URL": "https://api.example.com",
    "DEEPSEEK_MODEL": "deepseek-chat",
    "EMBEDDING_API_KEY": "sk-test-emb",
    "EMBEDDING_BASE_URL": "https://api.example.com",
    "EMBEDDING_MODEL": "text-embedding-3-small",
    "EMBEDDING_DIMS": "1536",
}


def _make_profile(tmp_path: Path):
    from DefenseAgent.config import AgentProfile
    from DefenseAgent.config.profile import (
        ConsolidationConfig,
        MemoryConfig,
        ScoringWeights,
        TierLimits,
    )

    profile = AgentProfile(
        id="smoke_agent", name="SmokeTester", age=25,
        traits="t", backstory="b", initial_plan="p",
        memory=MemoryConfig(
            scoring=ScoringWeights(
                similarity=0.4, importance=0.4, recency=0.1, frequency=0.1,
            ),
            tier_limits=TierLimits(
                working_capacity=10, working_ttl_seconds=3600,
            ),
            consolidation=ConsolidationConfig(
                enabled=True,
                promote_to_episodic_threshold=0.7,
                promote_to_semantic_threshold=0.8,
                importance_boost_on_promotion=1.2,
            ),
        ),
    )
    (tmp_path / "profile.yaml").write_text("agent: {}", encoding="utf-8")
    profile._source_path = (tmp_path / "profile.yaml").resolve()
    return profile


def _build_orchestrator(profile, *, with_working: bool = True):
    """Build a MemoryOrchestrator whose persistent layer is a mocked mem0
    client. Returns (orchestrator, fake_mem0_client) so tests can inspect
    or seed the underlying mock."""
    from DefenseAgent.memory import MemoryOrchestrator
    from DefenseAgent.memory.mem0_memory import Mem0Memory
    from DefenseAgent.memory.working import WorkingMemory

    fake_mem0 = MagicMock(name="mem0.Memory")
    fake_mem0.add.return_value = None
    fake_mem0.search.return_value = {"results": []}
    fake_mem0.get_all.return_value = {"results": []}

    with patch.object(Mem0Memory, "_init_memory_obj", return_value=fake_mem0):
        persistent = Mem0Memory(profile, load_env=False)

    working = WorkingMemory.from_profile(profile) if with_working else None
    orch = MemoryOrchestrator(profile, persistent, working=working)
    orch._fake_mem0 = fake_mem0  # type: ignore[attr-defined]
    return orch, fake_mem0


# ------------------------------------------------------------- test cases


def t_imports_resolve():
    from DefenseAgent.memory import (
        ConsolidationStats,
        DEFAULT_IMPORTANCE,
        Mem0Memory,
        Memory,
        MemoryConsolidator,
        MemoryItem,
        MemoryOrchestrator,
        MemoryTier,
        SharedMemoryManager,
        WorkingMemory,
        WorkingMemoryProtocol,
    )
    # Touch each so unused-import warnings aren't a false positive.
    assert all(x is not None for x in (
        ConsolidationStats, DEFAULT_IMPORTANCE, Mem0Memory, Memory,
        MemoryConsolidator, MemoryItem, MemoryOrchestrator, MemoryTier,
        SharedMemoryManager, WorkingMemory, WorkingMemoryProtocol,
    ))


def t_memory_item_round_trip():
    from DefenseAgent.memory import MemoryItem, MemoryTier
    item = MemoryItem(
        content="round trip me",
        tier=MemoryTier.SEMANTIC,
        memory_type="reflection",
        importance=0.85,
        access_count=4,
        source_run_id="run-1",
    )
    record = {"memory": item.content, "metadata": item.to_metadata()}
    back = MemoryItem.from_record(record)
    assert back.content == "round trip me"
    assert back.tier == MemoryTier.SEMANTIC
    assert back.memory_type == "reflection"
    assert back.importance == 0.85
    assert back.access_count == 4
    assert back.source_run_id == "run-1"


def t_memory_item_legacy_record_decodes_with_defaults():
    from DefenseAgent.memory import MemoryItem, MemoryTier
    legacy = {"memory": "old", "metadata": {"memory_type": "trajectory"}}
    item = MemoryItem.from_record(legacy)
    assert item.tier == MemoryTier.EPISODIC
    assert item.importance == 0.5
    assert item.memory_type == "trajectory"


def t_memory_item_validates_importance():
    from DefenseAgent.memory import MemoryItem
    try:
        MemoryItem(content="x", importance=2.0)
    except ValueError:
        return
    raise AssertionError("expected ValueError for importance=2.0")


def t_working_memory_basic_ops():
    from DefenseAgent.memory import MemoryItem, MemoryTier, WorkingMemory
    wm = WorkingMemory(capacity=3, ttl_seconds=60)
    for i in range(3):
        wm.add(MemoryItem(content=f"item {i}", tier=MemoryTier.WORKING))
    assert len(wm) == 3
    hits = wm.search("item", limit=10)
    assert len(hits) == 3
    assert hits[0].content == "item 2"  # most recent first


def t_working_memory_capacity_eviction():
    from DefenseAgent.memory import MemoryItem, MemoryTier, WorkingMemory
    wm = WorkingMemory(capacity=2, ttl_seconds=60)
    for i in range(5):
        wm.add(MemoryItem(content=f"item {i}", tier=MemoryTier.WORKING))
    assert len(wm) == 2
    contents = [i.content for i in wm.snapshot()]
    assert contents == ["item 3", "item 4"]


def t_scoring_components():
    from datetime import timedelta
    import math
    from DefenseAgent.memory.scoring import (
        frequency_score,
        recency_score,
    )

    now = datetime(2026, 4, 1, tzinfo=timezone.utc)
    seven_ago = now - timedelta(days=7)
    assert math.isclose(
        recency_score(seven_ago, half_life_days=7.0, now=now), 0.5,
        rel_tol=1e-6,
    )
    assert frequency_score(0) == 0.0
    assert math.isclose(frequency_score(1), 0.5)
    assert frequency_score(100) > 0.99


def t_scoring_hybrid_rerank_prefers_importance():
    from DefenseAgent.config.profile import ScoringWeights
    from DefenseAgent.memory import MemoryItem
    from DefenseAgent.memory.scoring import rank_items

    weights = ScoringWeights(
        similarity=0.2, importance=0.8, recency=0.0, frequency=0.0,
        recency_half_life_days=7.0,
    )
    important = MemoryItem(content="important", importance=1.0)
    relevant = MemoryItem(content="relevant", importance=0.1)
    ranked = rank_items(
        [(important, 0.6), (relevant, 0.9)],
        weights=weights,
    )
    assert ranked[0][0].content == "important"


# -- async tests -----------------------------------------------------------


async def t_orchestrator_routes_episodic(tmp_path: Path):
    from DefenseAgent.llm.types import Message
    profile = _make_profile(tmp_path)
    orch, fake_mem0 = _build_orchestrator(profile)

    await orch.add_episodic(
        [Message(role="user", content="something happened")],
        memory_type="trajectory",
        importance=0.6,
    )
    metadata = fake_mem0.add.call_args.kwargs["metadata"]
    assert metadata["tier"] == "episodic"
    assert metadata["memory_type"] == "trajectory"
    assert metadata["importance"] == 0.6


async def t_orchestrator_routes_semantic(tmp_path: Path):
    from DefenseAgent.llm.types import Message
    profile = _make_profile(tmp_path)
    orch, fake_mem0 = _build_orchestrator(profile)

    await orch.add_semantic(
        [Message(role="user", content="lesson learned")],
        memory_type="reflection",
        importance=0.9,
    )
    metadata = fake_mem0.add.call_args.kwargs["metadata"]
    assert metadata["tier"] == "semantic"


async def t_orchestrator_routes_procedural(tmp_path: Path):
    from DefenseAgent.llm.types import Message
    profile = _make_profile(tmp_path)
    orch, fake_mem0 = _build_orchestrator(profile)

    await orch.add_procedural(
        [Message(role="user", content="isolate-host playbook")],
        memory_type="sop",
        importance=0.95,
    )
    metadata = fake_mem0.add.call_args.kwargs["metadata"]
    assert metadata["tier"] == "procedural"


async def t_orchestrator_routes_working(tmp_path: Path):
    from DefenseAgent.llm.types import Message
    profile = _make_profile(tmp_path)
    orch, fake_mem0 = _build_orchestrator(profile)

    from DefenseAgent.memory import MemoryTier
    await orch.add(
        [Message(role="user", content="scratch note")],
        tier=MemoryTier.WORKING,
        importance=0.4,
    )
    # Working writes should NOT touch mem0.
    fake_mem0.add.assert_not_called()
    # And they should land in the working layer.
    assert orch.working is not None
    assert len(orch.working) == 1
    snap = orch.working.snapshot()
    assert "scratch note" in snap[0].content


async def t_orchestrator_recall_hybrid_default(tmp_path: Path):
    """The orchestrator's recall() defaults to hybrid scoring. With weights
    favoring importance, the high-importance hit should win."""
    from DefenseAgent.memory import MemoryTier
    profile = _make_profile(tmp_path)
    profile.memory.scoring.similarity = 0.2
    profile.memory.scoring.importance = 0.8
    profile.memory.scoring.recency = 0.0
    profile.memory.scoring.frequency = 0.0
    orch, fake_mem0 = _build_orchestrator(profile, with_working=False)
    fake_mem0.search.return_value = {
        "results": [
            {"memory": "trivial", "score": 0.9,
             "metadata": {"tier": "episodic", "importance": 0.1}},
            {"memory": "important", "score": 0.6,
             "metadata": {"tier": "episodic", "importance": 1.0}},
        ]
    }

    items = orch.recall("anything", limit=2)
    assert items[0].content == "important", \
        f"expected importance to win, got {[i.content for i in items]}"


async def t_orchestrator_recall_working_prepended(tmp_path: Path):
    from DefenseAgent.llm.types import Message
    from DefenseAgent.memory import MemoryTier
    profile = _make_profile(tmp_path)
    orch, fake_mem0 = _build_orchestrator(profile)
    fake_mem0.search.return_value = {
        "results": [
            {"memory": "old persistent fresh result",
             "score": 0.95, "metadata": {"tier": "episodic"}},
        ]
    }
    await orch.add(
        [Message(role="user", content="fresh in working")],
        tier=MemoryTier.WORKING,
    )

    items = orch.recall("fresh", limit=5)
    # Working content gets a `[role] ` prefix when ingested via the
    # orchestrator's `add(messages=...)` path — that's intentional so
    # who-said-what survives substring search.
    assert "fresh in working" in items[0].content, \
        f"working should win first slot, got {[i.content for i in items]}"


async def t_consolidator_promotes_high_importance_working(tmp_path: Path):
    from DefenseAgent.memory import (
        MemoryConsolidator,
        MemoryItem,
        MemoryTier,
    )
    profile = _make_profile(tmp_path)
    orch, fake_mem0 = _build_orchestrator(profile)
    orch.working.add(MemoryItem(
        content="critical observation",
        tier=MemoryTier.WORKING,
        importance=0.9,
    ))
    orch.working.add(MemoryItem(
        content="trivia",
        tier=MemoryTier.WORKING,
        importance=0.2,
    ))

    consolidator = MemoryConsolidator(orch)
    stats = await consolidator.run_once()
    assert stats.promoted_to_episodic == 1
    assert stats.skipped_below_threshold == 1
    metadata = fake_mem0.add.call_args.kwargs["metadata"]
    assert metadata["tier"] == "episodic"


async def t_consolidator_idempotent_across_runs(tmp_path: Path):
    from DefenseAgent.memory import (
        MemoryConsolidator,
        MemoryItem,
        MemoryTier,
    )
    profile = _make_profile(tmp_path)
    orch, fake_mem0 = _build_orchestrator(profile)
    orch.working.add(MemoryItem(
        content="critical",
        tier=MemoryTier.WORKING,
        importance=0.9,
    ))

    consolidator = MemoryConsolidator(orch)
    s1 = await consolidator.run_once()
    s2 = await consolidator.run_once()
    assert s1.promoted_to_episodic == 1
    assert s2.promoted_to_episodic == 0


async def t_consolidator_background_loop(tmp_path: Path):
    from DefenseAgent.config.profile import ConsolidationConfig
    from DefenseAgent.memory import (
        MemoryConsolidator,
        MemoryItem,
        MemoryTier,
    )
    profile = _make_profile(tmp_path)
    profile.memory.consolidation = ConsolidationConfig(
        enabled=True,
        interval_seconds=1,
        promote_to_episodic_threshold=0.5,
    )
    orch, fake_mem0 = _build_orchestrator(profile)
    orch.working.add(MemoryItem(
        content="x", tier=MemoryTier.WORKING, importance=0.9,
    ))

    consolidator = MemoryConsolidator(orch)
    await consolidator.start()
    await asyncio.sleep(0.05)
    await consolidator.stop()
    assert not consolidator.is_running
    assert fake_mem0.add.call_count >= 1


async def t_orchestrator_search_records_back_compat(tmp_path: Path):
    """The legacy search_records() entry point still works for callers that
    haven't migrated to recall()."""
    profile = _make_profile(tmp_path)
    orch, fake_mem0 = _build_orchestrator(profile)
    fake_mem0.search.return_value = {
        "results": [
            {"memory": "a hit", "score": 0.7, "metadata": {"tier": "episodic"}},
        ]
    }

    hits = orch.search_records("anything", limit=5)
    assert len(hits) == 1
    assert hits[0]["memory"] == "a hit"
    assert "_hybrid_score" in hits[0]


# ----------------------------------------------------------------- main


async def main() -> int:
    import tempfile

    runner = TestRunner()
    print("\n=== Pure-Python tests (no mem0 needed) ===\n")
    runner.sync("imports resolve",                    t_imports_resolve)
    runner.sync("MemoryItem round-trip",              t_memory_item_round_trip)
    runner.sync("MemoryItem legacy record decode",    t_memory_item_legacy_record_decodes_with_defaults)
    runner.sync("MemoryItem importance validation",   t_memory_item_validates_importance)
    runner.sync("WorkingMemory basic ops",            t_working_memory_basic_ops)
    runner.sync("WorkingMemory capacity eviction",    t_working_memory_capacity_eviction)
    runner.sync("scoring components",                 t_scoring_components)
    runner.sync("hybrid rerank prefers importance",   t_scoring_hybrid_rerank_prefers_importance)

    _set_env(_FAKE_ENV)
    print("\n=== Orchestrator tests (mocked mem0 backend) ===\n")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        await runner.run("orchestrator routes EPISODIC",     t_orchestrator_routes_episodic(tmp_path))
        await runner.run("orchestrator routes SEMANTIC",     t_orchestrator_routes_semantic(tmp_path))
        await runner.run("orchestrator routes PROCEDURAL",   t_orchestrator_routes_procedural(tmp_path))
        await runner.run("orchestrator routes WORKING",      t_orchestrator_routes_working(tmp_path))
        await runner.run("recall hybrid scoring default",    t_orchestrator_recall_hybrid_default(tmp_path))
        await runner.run("recall prepends WORKING hits",     t_orchestrator_recall_working_prepended(tmp_path))
        await runner.run("search_records back-compat shim",  t_orchestrator_search_records_back_compat(tmp_path))

    print("\n=== Consolidator tests (mocked mem0 backend) ===\n")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        await runner.run("consolidator promotes high-importance WORKING",
                         t_consolidator_promotes_high_importance_working(tmp_path))
        await runner.run("consolidator idempotent across runs",
                         t_consolidator_idempotent_across_runs(tmp_path))
        await runner.run("consolidator background loop start/stop",
                         t_consolidator_background_loop(tmp_path))

    print()
    print(f"=== Summary: {runner.passed} passed, {runner.failed} failed ===")
    return 0 if runner.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
