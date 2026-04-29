"""Tests for the tier-aware extensions to Mem0Memory (P0).

Covers:
- new add() kwargs (tier / importance / source_run_id / extra) flow into mem0's
  metadata via the per-coroutine `_PENDING_METADATA` ContextVar
- legacy add() callers (memory_type only, or no metadata at all) are unaffected
- search_records / get_all gain a `tier` filter that mirrors `memory_type`
- search_items / get_items decode hits into typed MemoryItems
- ContextVar isolation: concurrent add() calls don't leak metadata into
  each other, and a previous call's pending metadata is cleared before the
  next call sees the proxy
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from DefenseAgent.config import AgentProfile
from DefenseAgent.llm.types import Message
from DefenseAgent.memory.types import MemoryItem, MemoryTier


def _set_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_LAB_LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.example.com")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-chat")
    monkeypatch.setenv("EMBEDDING_API_KEY", "sk-test-emb")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://api.example.com")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("EMBEDDING_DIMS", "1536")


def _build_memory(profile: AgentProfile) -> Any:
    from DefenseAgent.memory.mem0_memory import Mem0Memory

    fake_mem0 = MagicMock(name="mem0.Memory")
    fake_mem0.add.return_value = None
    fake_mem0.search.return_value = {"results": []}
    fake_mem0.get_all.return_value = {"results": []}

    with patch.object(Mem0Memory, "_init_memory_obj", return_value=fake_mem0):
        memory = Mem0Memory(profile, load_env=False)
    memory._fake_mem0 = fake_mem0
    return memory


def _make_profile(tmp_path: Path) -> AgentProfile:
    profile = AgentProfile(
        id="test_agent", name="Tester", age=25,
        traits="t", backstory="b", initial_plan="p",
    )
    (tmp_path / "profile.yaml").write_text("agent: {}", encoding="utf-8")
    profile._source_path = (tmp_path / "profile.yaml").resolve()
    return profile


# ---------- add() new kwargs land in metadata ----------


async def test_add_tier_kwarg_routed_into_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    _set_env(monkeypatch)
    memory = _build_memory(_make_profile(tmp_path))

    await memory.add(
        [Message(role="user", content="x")],
        tier=MemoryTier.SEMANTIC,
    )

    metadata = memory._fake_mem0.add.call_args.kwargs["metadata"]
    assert metadata["tier"] == "semantic"


async def test_add_importance_kwarg_routed_into_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    _set_env(monkeypatch)
    memory = _build_memory(_make_profile(tmp_path))

    await memory.add(
        [Message(role="user", content="x")],
        importance=0.85,
    )

    metadata = memory._fake_mem0.add.call_args.kwargs["metadata"]
    assert metadata["importance"] == 0.85


async def test_add_source_run_id_and_extra_routed_into_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    _set_env(monkeypatch)
    memory = _build_memory(_make_profile(tmp_path))

    await memory.add(
        [Message(role="user", content="x")],
        source_run_id="run-7",
        extra={"attack_phase": "recon"},
    )

    metadata = memory._fake_mem0.add.call_args.kwargs["metadata"]
    assert metadata["source_run_id"] == "run-7"
    assert metadata["attack_phase"] == "recon"


async def test_add_combines_tier_and_legacy_memory_type(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """memory_type and tier are orthogonal dimensions — both should land in
    metadata when both are supplied."""
    _set_env(monkeypatch)
    memory = _build_memory(_make_profile(tmp_path))

    await memory.add(
        [Message(role="user", content="x")],
        memory_type="reflection",
        tier=MemoryTier.SEMANTIC,
        importance=0.7,
    )

    metadata = memory._fake_mem0.add.call_args.kwargs["metadata"]
    assert metadata["memory_type"] == "reflection"
    assert metadata["tier"] == "semantic"
    assert metadata["importance"] == 0.7


async def test_add_rejects_out_of_range_importance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    _set_env(monkeypatch)
    memory = _build_memory(_make_profile(tmp_path))

    with pytest.raises(ValueError, match="importance"):
        await memory.add([Message(role="user", content="x")], importance=2.0)


async def test_add_legacy_path_unchanged_by_tier_extension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """A caller that doesn't set any tier-aware kwarg must produce the same
    metadata shape as before P0 — no surprise tier/importance keys."""
    _set_env(monkeypatch)
    memory = _build_memory(_make_profile(tmp_path))

    await memory.add(
        [Message(role="user", content="x")],
        memory_type="trajectory",
    )

    metadata = memory._fake_mem0.add.call_args.kwargs["metadata"]
    assert metadata == {"memory_type": "trajectory"}


async def test_add_no_metadata_when_no_kwargs_provided(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Pure-legacy add() with no memory_type and no tier kwargs should not
    inject a metadata dict at all (mem0 native pathway)."""
    _set_env(monkeypatch)
    memory = _build_memory(_make_profile(tmp_path))

    await memory.add([Message(role="user", content="x")])

    kwargs = memory._fake_mem0.add.call_args.kwargs
    assert "metadata" not in kwargs


async def test_add_item_forwards_all_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    _set_env(monkeypatch)
    memory = _build_memory(_make_profile(tmp_path))

    item = MemoryItem(
        content="placeholder",  # add_item ignores content; messages drive ingest
        tier=MemoryTier.PROCEDURAL,
        memory_type="sop",
        importance=0.95,
        source_run_id="run-99",
        extra={"playbook": "isolate-host"},
    )
    await memory.add_item([Message(role="user", content="x")], item)

    metadata = memory._fake_mem0.add.call_args.kwargs["metadata"]
    assert metadata["tier"] == "procedural"
    assert metadata["memory_type"] == "sop"
    assert metadata["importance"] == 0.95
    assert metadata["source_run_id"] == "run-99"
    assert metadata["playbook"] == "isolate-host"


# ---------- ContextVar isolation ----------


async def test_pending_metadata_cleared_between_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """A first call setting tier=SEMANTIC must not leak into a second call that
    asks for legacy-only metadata."""
    _set_env(monkeypatch)
    memory = _build_memory(_make_profile(tmp_path))

    await memory.add(
        [Message(role="user", content="x")],
        tier=MemoryTier.SEMANTIC,
        importance=0.9,
    )
    await memory.add(
        [Message(role="user", content="y")],
        memory_type="trajectory",
    )

    second_metadata = memory._fake_mem0.add.call_args_list[-1].kwargs["metadata"]
    assert "tier" not in second_metadata
    assert "importance" not in second_metadata
    assert second_metadata == {"memory_type": "trajectory"}


async def test_concurrent_adds_do_not_leak_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """asyncio.gather'd add() calls each get their own ContextVar context, so
    A's tier=SEMANTIC must not appear on B's mem0.add call."""
    _set_env(monkeypatch)
    memory = _build_memory(_make_profile(tmp_path))

    # Mark each underlying add call so we can identify which Mem0Memory.add()
    # the captured metadata came from. Since ms-agent's add_single uses an
    # asyncio.Lock, the two calls serialize at the proxy boundary — but the
    # ContextVar must still keep the metadata bound to the right caller.
    async def run_a():
        await memory.add(
            [Message(role="user", content="A")],
            tier=MemoryTier.SEMANTIC,
            importance=0.9,
        )

    async def run_b():
        await memory.add(
            [Message(role="user", content="B")],
            memory_type="trajectory",
        )

    await asyncio.gather(run_a(), run_b())

    # Two add calls happened; identify them by the message content forwarded.
    call_metadata_by_content = {}
    for call in memory._fake_mem0.add.call_args_list:
        forwarded = call.args[0]
        if not forwarded:
            continue
        content = forwarded[0]["content"]
        call_metadata_by_content[content] = call.kwargs.get("metadata", {})

    assert call_metadata_by_content["A"].get("tier") == "semantic"
    assert call_metadata_by_content["A"].get("importance") == 0.9
    assert "tier" not in call_metadata_by_content["B"]
    assert call_metadata_by_content["B"].get("memory_type") == "trajectory"


# ---------- search_records / get_all tier filter ----------


def test_search_records_filters_by_tier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    _set_env(monkeypatch)
    memory = _build_memory(_make_profile(tmp_path))
    memory._fake_mem0.search.return_value = {
        "results": [
            {"memory": "ep", "metadata": {"tier": "episodic"}},
            {"memory": "se", "metadata": {"tier": "semantic"}},
            {"memory": "legacy", "metadata": {}},  # no tier → excluded
        ]
    }

    only_semantic = memory.search_records("anything", tier=MemoryTier.SEMANTIC)
    assert [r["memory"] for r in only_semantic] == ["se"]


def test_search_records_tier_filter_accepts_string(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    _set_env(monkeypatch)
    memory = _build_memory(_make_profile(tmp_path))
    memory._fake_mem0.search.return_value = {
        "results": [{"memory": "ep", "metadata": {"tier": "episodic"}}]
    }

    hits = memory.search_records("anything", tier="episodic")
    assert len(hits) == 1


def test_get_all_filters_by_tier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    _set_env(monkeypatch)
    memory = _build_memory(_make_profile(tmp_path))
    memory._fake_mem0.get_all.return_value = {
        "results": [
            {"memory": "p", "metadata": {"tier": "procedural"}},
            {"memory": "s", "metadata": {"tier": "semantic"}},
        ]
    }

    procedural = memory.get_all(tier=MemoryTier.PROCEDURAL)
    assert [r["memory"] for r in procedural] == ["p"]


# ---------- search_items / get_items return typed records ----------


def test_search_items_returns_memory_items(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    _set_env(monkeypatch)
    memory = _build_memory(_make_profile(tmp_path))
    memory._fake_mem0.search.return_value = {
        "results": [
            {
                "id": "u1",
                "memory": "first hit",
                "metadata": {
                    "tier": "semantic",
                    "importance": 0.7,
                    "memory_type": "reflection",
                },
            }
        ]
    }

    items = memory.search_items("anything")
    assert len(items) == 1
    assert isinstance(items[0], MemoryItem)
    assert items[0].tier == MemoryTier.SEMANTIC
    assert items[0].importance == 0.7
    assert items[0].memory_type == "reflection"
    assert items[0].record_id == "u1"


def test_get_items_returns_memory_items(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    _set_env(monkeypatch)
    memory = _build_memory(_make_profile(tmp_path))
    memory._fake_mem0.get_all.return_value = {
        "results": [
            {"memory": "legacy", "metadata": {}},  # decodes with defaults
        ]
    }

    items = memory.get_items()
    assert len(items) == 1
    assert items[0].tier == MemoryTier.EPISODIC  # default for legacy records
    assert items[0].importance == 0.5


# ---------- hybrid scoring (P1) ----------


def _make_profile_with_weights(
    tmp_path: Path,
    *,
    similarity: float,
    importance: float,
    recency: float = 0.0,
    frequency: float = 0.0,
) -> AgentProfile:
    """Like _make_profile but lets each test pin its own scoring weights so the
    hybrid re-rank behavior is unambiguous to assert against."""
    from DefenseAgent.config.profile import (
        MemoryConfig,
        ScoringWeights,
    )

    profile = AgentProfile(
        id="test_agent", name="Tester", age=25,
        traits="t", backstory="b", initial_plan="p",
        memory=MemoryConfig(
            scoring=ScoringWeights(
                similarity=similarity,
                importance=importance,
                recency=recency,
                frequency=frequency,
                recency_half_life_days=7.0,
            )
        ),
    )
    (tmp_path / "profile.yaml").write_text("agent: {}", encoding="utf-8")
    profile._source_path = (tmp_path / "profile.yaml").resolve()
    return profile


def test_search_records_default_scoring_is_vector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Back-compat: not setting `scoring=` keeps mem0's native ordering and
    does not attach `_hybrid_score`."""
    _set_env(monkeypatch)
    memory = _build_memory(_make_profile(tmp_path))
    memory._fake_mem0.search.return_value = {
        "results": [
            {"memory": "first", "score": 0.9, "metadata": {}},
            {"memory": "second", "score": 0.5, "metadata": {}},
        ]
    }

    hits = memory.search_records("anything")
    assert [r["memory"] for r in hits] == ["first", "second"]
    assert all("_hybrid_score" not in r for r in hits)


def test_search_records_invalid_scoring_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    _set_env(monkeypatch)
    memory = _build_memory(_make_profile(tmp_path))

    with pytest.raises(ValueError, match="scoring"):
        memory.search_records("anything", scoring="bogus")


def test_search_records_hybrid_reranks_by_importance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """With importance weighted high relative to similarity, a less-similar
    high-importance record should outrank a more-similar trivial one."""
    _set_env(monkeypatch)
    profile = _make_profile_with_weights(
        tmp_path, similarity=0.2, importance=0.8,
    )
    memory = _build_memory(profile)
    memory._fake_mem0.search.return_value = {
        "results": [
            {
                "memory": "trivial",
                "score": 0.9,
                "metadata": {"importance": 0.1, "tier": "episodic"},
            },
            {
                "memory": "important",
                "score": 0.6,
                "metadata": {"importance": 1.0, "tier": "episodic"},
            },
        ]
    }

    hits = memory.search_records("anything", scoring="hybrid")
    assert [r["memory"] for r in hits] == ["important", "trivial"]
    assert all("_hybrid_score" in r for r in hits)
    assert hits[0]["_hybrid_score"] > hits[1]["_hybrid_score"]


def test_search_records_hybrid_fetches_more_candidates_than_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Hybrid mode must request candidate_multiplier × limit from mem0 so
    the re-rank has room to surface a low-similarity-but-important hit."""
    _set_env(monkeypatch)
    memory = _build_memory(_make_profile(tmp_path))
    memory._fake_mem0.search.return_value = {"results": []}

    memory.search_records(
        "anything", limit=5, scoring="hybrid", candidate_multiplier=4,
    )
    fetched_limit = memory._fake_mem0.search.call_args.kwargs["limit"]
    assert fetched_limit == 20


def test_search_records_hybrid_clips_to_requested_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    _set_env(monkeypatch)
    profile = _make_profile_with_weights(
        tmp_path, similarity=1.0, importance=0.0,
    )
    memory = _build_memory(profile)
    memory._fake_mem0.search.return_value = {
        "results": [
            {"memory": str(i), "score": float(i) / 10, "metadata": {}}
            for i in range(10)
        ]
    }

    hits = memory.search_records("anything", limit=3, scoring="hybrid")
    assert len(hits) == 3
    # Highest similarities (9, 8, 7) win when only similarity is weighted.
    assert [r["memory"] for r in hits] == ["9", "8", "7"]


def test_search_items_hybrid_returns_typed_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    _set_env(monkeypatch)
    profile = _make_profile_with_weights(
        tmp_path, similarity=0.5, importance=0.5,
    )
    memory = _build_memory(profile)
    memory._fake_mem0.search.return_value = {
        "results": [
            {
                "memory": "x",
                "score": 0.7,
                "metadata": {"tier": "semantic", "importance": 0.9},
            }
        ]
    }

    items = memory.search_items("anything", scoring="hybrid")
    assert len(items) == 1
    assert isinstance(items[0], MemoryItem)
    assert items[0].tier == MemoryTier.SEMANTIC
    assert items[0].importance == 0.9
