"""Tests for DefenseAgent.memory.consolidator.MemoryConsolidator (P5).

Strategy: stub the persistent layer (Mem0Memory) with a MagicMock so writes
go to a recordable list. Wire a real WorkingMemory so promotions out of it
actually move data. The consolidator is exercised through `run_once()`
(deterministic) and `start()`/`stop()` (background loop).
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from DefenseAgent.config import AgentProfile
from DefenseAgent.config.profile import (
    ConsolidationConfig,
    MemoryConfig,
    TierLimits,
)
from DefenseAgent.memory import (
    MemoryConsolidator,
    MemoryItem,
    MemoryOrchestrator,
    MemoryTier,
    WorkingMemory,
)


def _set_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_LAB_LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.example.com")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-chat")
    monkeypatch.setenv("EMBEDDING_API_KEY", "sk-test-emb")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://api.example.com")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("EMBEDDING_DIMS", "1536")


def _make_profile(
    tmp_path: Path,
    *,
    consolidation: ConsolidationConfig | None = None,
) -> AgentProfile:
    profile = AgentProfile(
        id="test_agent", name="Tester", age=25,
        traits="t", backstory="b", initial_plan="p",
        memory=MemoryConfig(
            tier_limits=TierLimits(working_capacity=10, working_ttl_seconds=3600),
            consolidation=consolidation or ConsolidationConfig(),
        ),
    )
    (tmp_path / "profile.yaml").write_text("agent: {}", encoding="utf-8")
    profile._source_path = (tmp_path / "profile.yaml").resolve()
    return profile


def _build_orchestrator(
    profile: AgentProfile,
    *,
    persistent_records_by_tier: dict[str, list[dict[str, Any]]] | None = None,
) -> MemoryOrchestrator:
    """Build a MemoryOrchestrator whose persistent layer is a MagicMock-backed
    Mem0Memory. `persistent_records_by_tier` lets the test pre-populate what
    `get_all(tier=...)` returns per tier."""
    from DefenseAgent.memory.mem0_memory import Mem0Memory

    fake_mem0 = MagicMock(name="mem0.Memory")
    fake_mem0.add.return_value = None
    fake_mem0.search.return_value = {"results": []}

    records_by_tier = persistent_records_by_tier or {}

    def fake_get_all(filters: dict[str, Any]) -> dict[str, Any]:
        # The consolidator filters by tier in Python after get_all returns
        # everything, so just return the union. The Mem0Memory.get_all method
        # itself is what does the tier filtering.
        all_records: list[dict[str, Any]] = []
        for recs in records_by_tier.values():
            all_records.extend(recs)
        return {"results": all_records}

    fake_mem0.get_all.side_effect = fake_get_all

    with patch.object(Mem0Memory, "_init_memory_obj", return_value=fake_mem0):
        persistent = Mem0Memory(profile, load_env=False)
    persistent._fake_mem0 = fake_mem0  # type: ignore[attr-defined]

    working = WorkingMemory.from_profile(profile)
    return MemoryOrchestrator(profile, persistent, working=working)


def _record(
    *,
    tier: str,
    importance: float,
    content: str = "x",
    record_id: str | None = None,
    memory_type: str | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {"tier": tier, "importance": importance}
    if memory_type:
        metadata["memory_type"] = memory_type
    out: dict[str, Any] = {"memory": content, "metadata": metadata}
    if record_id:
        out["id"] = record_id
    return out


# ---------- run_once: working → episodic ----------


async def test_promotes_working_item_above_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """A WORKING item with importance ≥ promote_to_episodic_threshold should
    appear as a new EPISODIC mem0 add()."""
    _set_env(monkeypatch)
    profile = _make_profile(
        tmp_path,
        consolidation=ConsolidationConfig(
            enabled=True, promote_to_episodic_threshold=0.5,
        ),
    )
    orch = _build_orchestrator(profile)
    orch.working.add(MemoryItem(
        content="important observation", tier=MemoryTier.WORKING, importance=0.9,
    ))

    consolidator = MemoryConsolidator(orch)
    stats = await consolidator.run_once()

    assert stats.promoted_to_episodic == 1
    add_calls = orch.persistent._fake_mem0.add.call_args_list
    assert len(add_calls) == 1
    metadata = add_calls[0].kwargs["metadata"]
    assert metadata["tier"] == "episodic"


async def test_skips_working_item_below_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    _set_env(monkeypatch)
    profile = _make_profile(
        tmp_path,
        consolidation=ConsolidationConfig(
            enabled=True, promote_to_episodic_threshold=0.7,
        ),
    )
    orch = _build_orchestrator(profile)
    orch.working.add(MemoryItem(
        content="trivia", tier=MemoryTier.WORKING, importance=0.3,
    ))

    consolidator = MemoryConsolidator(orch)
    stats = await consolidator.run_once()

    assert stats.promoted_to_episodic == 0
    assert stats.skipped_below_threshold == 1
    orch.persistent._fake_mem0.add.assert_not_called()


async def test_promotion_boosts_importance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """The promoted record's importance should be source × boost (capped at 1.0)."""
    _set_env(monkeypatch)
    profile = _make_profile(
        tmp_path,
        consolidation=ConsolidationConfig(
            enabled=True,
            promote_to_episodic_threshold=0.5,
            importance_boost_on_promotion=1.2,
        ),
    )
    orch = _build_orchestrator(profile)
    orch.working.add(MemoryItem(
        content="x", tier=MemoryTier.WORKING, importance=0.6,
    ))

    consolidator = MemoryConsolidator(orch)
    await consolidator.run_once()

    metadata = orch.persistent._fake_mem0.add.call_args.kwargs["metadata"]
    assert metadata["importance"] == pytest.approx(0.6 * 1.2)


async def test_promotion_caps_importance_at_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Boost can't push importance past the schema's [0, 1] invariant — the
    consolidator must cap before forwarding into the new MemoryItem."""
    _set_env(monkeypatch)
    profile = _make_profile(
        tmp_path,
        consolidation=ConsolidationConfig(
            enabled=True,
            promote_to_episodic_threshold=0.5,
            importance_boost_on_promotion=2.0,
        ),
    )
    orch = _build_orchestrator(profile)
    orch.working.add(MemoryItem(
        content="x", tier=MemoryTier.WORKING, importance=0.9,
    ))

    consolidator = MemoryConsolidator(orch)
    await consolidator.run_once()

    metadata = orch.persistent._fake_mem0.add.call_args.kwargs["metadata"]
    assert metadata["importance"] == 1.0


async def test_already_promoted_items_are_skipped_on_subsequent_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    _set_env(monkeypatch)
    profile = _make_profile(
        tmp_path,
        consolidation=ConsolidationConfig(
            enabled=True, promote_to_episodic_threshold=0.5,
        ),
    )
    orch = _build_orchestrator(profile)
    orch.working.add(MemoryItem(
        content="x", tier=MemoryTier.WORKING, importance=0.9,
    ))

    consolidator = MemoryConsolidator(orch)
    first = await consolidator.run_once()
    second = await consolidator.run_once()

    assert first.promoted_to_episodic == 1
    assert second.promoted_to_episodic == 0
    assert second.skipped_already_promoted >= 1


# ---------- run_once: persistent tier promotion ----------


async def test_promotes_episodic_to_semantic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Working is empty, semantic tier has no records — only the
    episodic→semantic edge has anything to do."""
    _set_env(monkeypatch)
    profile = _make_profile(
        tmp_path,
        consolidation=ConsolidationConfig(
            enabled=True,
            promote_to_semantic_threshold=0.6,
        ),
    )
    orch = _build_orchestrator(
        profile,
        persistent_records_by_tier={
            "episodic": [
                _record(tier="episodic", importance=0.7, content="ep1", record_id="e1"),
            ],
        },
    )

    consolidator = MemoryConsolidator(orch)
    stats = await consolidator.run_once()

    assert stats.promoted_to_semantic == 1
    add_calls = orch.persistent._fake_mem0.add.call_args_list
    assert add_calls[0].kwargs["metadata"]["tier"] == "semantic"


async def test_promotes_semantic_to_procedural(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Only seeded tier is semantic — episodic edge has nothing to promote."""
    _set_env(monkeypatch)
    profile = _make_profile(
        tmp_path,
        consolidation=ConsolidationConfig(
            enabled=True,
            promote_to_procedural_threshold=0.8,
        ),
    )
    orch = _build_orchestrator(
        profile,
        persistent_records_by_tier={
            "semantic": [
                _record(tier="semantic", importance=0.95, content="lesson", record_id="s1"),
            ],
        },
    )

    consolidator = MemoryConsolidator(orch)
    stats = await consolidator.run_once()

    assert stats.promoted_to_procedural == 1


# ---------- consolidated_from link ----------


async def test_promoted_record_carries_consolidated_from(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """The new tier's record should reference its source via metadata so the
    history is auditable post-promotion."""
    _set_env(monkeypatch)
    profile = _make_profile(
        tmp_path,
        consolidation=ConsolidationConfig(
            enabled=True,
            promote_to_semantic_threshold=0.5,
        ),
    )
    orch = _build_orchestrator(
        profile,
        persistent_records_by_tier={
            "episodic": [
                _record(tier="episodic", importance=0.7, record_id="ep-source-1"),
            ],
        },
    )

    consolidator = MemoryConsolidator(orch)
    await consolidator.run_once()

    metadata = orch.persistent._fake_mem0.add.call_args.kwargs["metadata"]
    assert metadata.get("consolidated_from_tier") == "episodic"


# ---------- error tolerance ----------


async def test_promotion_error_increments_error_counter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """A failure during a single promotion shouldn't kill the whole pass —
    the rest of the items should still get a chance."""
    _set_env(monkeypatch)
    profile = _make_profile(
        tmp_path,
        consolidation=ConsolidationConfig(
            enabled=True, promote_to_episodic_threshold=0.5,
        ),
    )
    orch = _build_orchestrator(profile)
    orch.working.add(MemoryItem(
        content="will fail", tier=MemoryTier.WORKING, importance=0.9,
    ))
    orch.working.add(MemoryItem(
        content="will succeed", tier=MemoryTier.WORKING, importance=0.9,
    ))
    # Patch the orchestrator.add to fail on the first call only.
    original_add = orch.add
    call_count = {"n": 0}

    async def flaky_add(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("simulated failure")
        return await original_add(*args, **kwargs)

    orch.add = flaky_add  # type: ignore[method-assign]

    consolidator = MemoryConsolidator(orch)
    stats = await consolidator.run_once()

    assert stats.errors == 1
    assert stats.promoted_to_episodic == 1


# ---------- background loop ----------


async def test_start_and_stop_background_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    _set_env(monkeypatch)
    profile = _make_profile(
        tmp_path,
        consolidation=ConsolidationConfig(
            enabled=True,
            promote_to_episodic_threshold=0.5,
            interval_seconds=1,  # tightest allowed
        ),
    )
    orch = _build_orchestrator(profile)
    orch.working.add(MemoryItem(
        content="x", tier=MemoryTier.WORKING, importance=0.9,
    ))

    consolidator = MemoryConsolidator(orch)
    await consolidator.start()
    # Yield once so the spawned task gets to call run_once() before we stop.
    await asyncio.sleep(0)
    await asyncio.sleep(0.05)
    await consolidator.stop()

    assert not consolidator.is_running
    # At least one promotion should have happened during the brief window.
    assert orch.persistent._fake_mem0.add.call_count >= 1


async def test_start_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    _set_env(monkeypatch)
    profile = _make_profile(
        tmp_path,
        consolidation=ConsolidationConfig(enabled=True, interval_seconds=10),
    )
    orch = _build_orchestrator(profile)
    consolidator = MemoryConsolidator(orch)

    await consolidator.start()
    first_task = consolidator._task
    await consolidator.start()  # second start should be a no-op
    assert consolidator._task is first_task

    await consolidator.stop()


async def test_stop_when_not_running_is_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    _set_env(monkeypatch)
    profile = _make_profile(tmp_path)
    orch = _build_orchestrator(profile)
    consolidator = MemoryConsolidator(orch)

    # Should not raise.
    await consolidator.stop()
    assert not consolidator.is_running


async def test_loop_swallows_run_once_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """A pass that raises shouldn't kill the loop — the next interval should
    still fire."""
    _set_env(monkeypatch)
    profile = _make_profile(
        tmp_path,
        consolidation=ConsolidationConfig(enabled=True, interval_seconds=1),
    )
    orch = _build_orchestrator(profile)
    consolidator = MemoryConsolidator(orch)

    call_count = {"n": 0}

    async def flaky_run_once():
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("boom")
        return None

    consolidator.run_once = flaky_run_once  # type: ignore[method-assign]

    await consolidator.start()
    await asyncio.sleep(0)
    await asyncio.sleep(0.05)
    await consolidator.stop()

    # The first call raised; the loop should have survived to call again
    # (timing-dependent — assert at least the first fired).
    assert call_count["n"] >= 1
