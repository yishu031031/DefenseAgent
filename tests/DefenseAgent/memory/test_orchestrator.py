"""Tests for DefenseAgent.memory.orchestrator.MemoryOrchestrator (P2).

Strategy: stub the persistent layer with a `Mem0Memory` whose internal mem0
client is a MagicMock (same pattern as test_mem0_memory.py). Stub the working
layer with a tiny in-memory list for tests that exercise routing — the real
WorkingMemory lands in P4.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from DefenseAgent.config import AgentProfile
from DefenseAgent.llm.types import Message
from DefenseAgent.memory import (
    MemoryItem,
    MemoryOrchestrator,
    MemoryTier,
    WorkingMemoryProtocol,
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


def _make_profile(tmp_path: Path) -> AgentProfile:
    profile = AgentProfile(
        id="test_agent", name="Tester", age=25,
        traits="t", backstory="b", initial_plan="p",
    )
    (tmp_path / "profile.yaml").write_text("agent: {}", encoding="utf-8")
    profile._source_path = (tmp_path / "profile.yaml").resolve()
    return profile


def _build_persistent(profile: AgentProfile) -> Any:
    from DefenseAgent.memory.mem0_memory import Mem0Memory

    fake_mem0 = MagicMock(name="mem0.Memory")
    fake_mem0.add.return_value = None
    fake_mem0.search.return_value = {"results": []}
    fake_mem0.get_all.return_value = {"results": []}

    with patch.object(Mem0Memory, "_init_memory_obj", return_value=fake_mem0):
        memory = Mem0Memory(profile, load_env=False)
    memory._fake_mem0 = fake_mem0
    return memory


class _FakeWorking:
    """Minimal WorkingMemoryProtocol implementation used to exercise the
    orchestrator's routing without depending on P4's real impl."""

    def __init__(self) -> None:
        self.items: list[MemoryItem] = []

    def add(self, item: MemoryItem) -> None:
        self.items.append(item)

    def search(self, query: str, *, limit: int) -> list[MemoryItem]:
        # Naive substring match — the real impl can use anything; the
        # orchestrator only depends on the Protocol contract.
        hits = [i for i in self.items if query.lower() in i.content.lower()]
        return hits[:limit]

    def clear(self) -> None:
        self.items.clear()


# ---------- protocol & construction ----------


def test_fake_working_satisfies_protocol():
    """Sanity: the test stub matches the runtime-checkable Protocol."""
    assert isinstance(_FakeWorking(), WorkingMemoryProtocol)


def test_orchestrator_from_profile_constructs_persistent_and_working(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """P4: from_profile auto-creates a WorkingMemory from `profile.memory.tier_limits`
    so users get the full four-tier architecture without manual wiring."""
    _set_env(monkeypatch)
    profile = _make_profile(tmp_path)

    from DefenseAgent.memory.mem0_memory import Mem0Memory
    from DefenseAgent.memory.working import WorkingMemory

    fake_mem0 = MagicMock()
    fake_mem0.add.return_value = None
    fake_mem0.search.return_value = {"results": []}
    fake_mem0.get_all.return_value = {"results": []}
    with patch.object(Mem0Memory, "_init_memory_obj", return_value=fake_mem0):
        orch = MemoryOrchestrator.from_profile(profile, load_env=False)

    assert orch.profile is profile
    assert isinstance(orch.persistent, Mem0Memory)
    assert isinstance(orch.working, WorkingMemory)


def test_orchestrator_from_profile_with_working_false_skips_working(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Tests opting out of the auto-Working layer (e.g. when an external
    coordinator manages session memory separately)."""
    _set_env(monkeypatch)
    profile = _make_profile(tmp_path)

    from DefenseAgent.memory.mem0_memory import Mem0Memory

    fake_mem0 = MagicMock()
    fake_mem0.add.return_value = None
    fake_mem0.search.return_value = {"results": []}
    fake_mem0.get_all.return_value = {"results": []}
    with patch.object(Mem0Memory, "_init_memory_obj", return_value=fake_mem0):
        orch = MemoryOrchestrator.from_profile(
            profile, load_env=False, with_working=False,
        )

    assert orch.working is None


# ---------- write routing ----------


async def test_add_default_tier_is_episodic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Most agent traces are episodic; that's the safe default for callers
    who don't think about tiers."""
    _set_env(monkeypatch)
    profile = _make_profile(tmp_path)
    orch = MemoryOrchestrator(profile, _build_persistent(profile))

    await orch.add([Message(role="user", content="x")])

    metadata = orch.persistent._fake_mem0.add.call_args.kwargs["metadata"]
    assert metadata["tier"] == "episodic"


async def test_add_uses_profile_default_importance_when_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Default importance lives on the profile, not in code — the orchestrator
    must read it instead of hardcoding 0.5."""
    _set_env(monkeypatch)
    profile = _make_profile(tmp_path)
    profile.memory.default_importance = 0.65
    orch = MemoryOrchestrator(profile, _build_persistent(profile))

    await orch.add([Message(role="user", content="x")])

    metadata = orch.persistent._fake_mem0.add.call_args.kwargs["metadata"]
    assert metadata["importance"] == 0.65


async def test_add_explicit_importance_overrides_profile_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    _set_env(monkeypatch)
    profile = _make_profile(tmp_path)
    profile.memory.default_importance = 0.4
    orch = MemoryOrchestrator(profile, _build_persistent(profile))

    await orch.add([Message(role="user", content="x")], importance=0.95)

    metadata = orch.persistent._fake_mem0.add.call_args.kwargs["metadata"]
    assert metadata["importance"] == 0.95


async def test_add_episodic_convenience(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    _set_env(monkeypatch)
    profile = _make_profile(tmp_path)
    orch = MemoryOrchestrator(profile, _build_persistent(profile))

    await orch.add_episodic(
        [Message(role="user", content="x")], memory_type="trajectory",
    )

    metadata = orch.persistent._fake_mem0.add.call_args.kwargs["metadata"]
    assert metadata["tier"] == "episodic"
    assert metadata["memory_type"] == "trajectory"


async def test_add_semantic_convenience(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    _set_env(monkeypatch)
    profile = _make_profile(tmp_path)
    orch = MemoryOrchestrator(profile, _build_persistent(profile))

    await orch.add_semantic(
        [Message(role="user", content="x")], memory_type="reflection",
    )

    metadata = orch.persistent._fake_mem0.add.call_args.kwargs["metadata"]
    assert metadata["tier"] == "semantic"


async def test_add_procedural_convenience(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    _set_env(monkeypatch)
    profile = _make_profile(tmp_path)
    orch = MemoryOrchestrator(profile, _build_persistent(profile))

    await orch.add_procedural(
        [Message(role="user", content="x")], memory_type="sop",
    )

    metadata = orch.persistent._fake_mem0.add.call_args.kwargs["metadata"]
    assert metadata["tier"] == "procedural"


async def test_add_tier_string_is_coerced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Tier args from YAML or LLM tool calls arrive as strings — the
    orchestrator must accept either."""
    _set_env(monkeypatch)
    profile = _make_profile(tmp_path)
    orch = MemoryOrchestrator(profile, _build_persistent(profile))

    await orch.add([Message(role="user", content="x")], tier="semantic")

    metadata = orch.persistent._fake_mem0.add.call_args.kwargs["metadata"]
    assert metadata["tier"] == "semantic"


async def test_add_unknown_tier_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    _set_env(monkeypatch)
    profile = _make_profile(tmp_path)
    orch = MemoryOrchestrator(profile, _build_persistent(profile))

    with pytest.raises(ValueError, match="unknown memory tier"):
        await orch.add([Message(role="user", content="x")], tier="bogus")


# ---------- working tier routing ----------


async def test_add_working_routes_to_working_layer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    _set_env(monkeypatch)
    profile = _make_profile(tmp_path)
    working = _FakeWorking()
    orch = MemoryOrchestrator(
        profile, _build_persistent(profile), working=working,
    )

    await orch.add(
        [Message(role="user", content="hello")],
        tier=MemoryTier.WORKING,
        memory_type="scratch",
    )

    # mem0.add NOT called — the write went to working only.
    orch.persistent._fake_mem0.add.assert_not_called()
    assert len(working.items) == 1
    assert working.items[0].tier == MemoryTier.WORKING
    assert working.items[0].memory_type == "scratch"
    assert "hello" in working.items[0].content


async def test_add_working_without_layer_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Loud failure: if the user routes to WORKING but didn't supply a
    backend, they should know — silent success would lose the write."""
    _set_env(monkeypatch)
    profile = _make_profile(tmp_path)
    orch = MemoryOrchestrator(profile, _build_persistent(profile))

    with pytest.raises(RuntimeError, match="WORKING tier"):
        await orch.add(
            [Message(role="user", content="x")],
            tier=MemoryTier.WORKING,
        )


# ---------- recall ----------


def test_recall_default_uses_hybrid_scoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """The orchestrator's API consciously diverges from mem0's default — hybrid
    scoring is the new normal because the agent benefits from importance and
    recency weighting."""
    _set_env(monkeypatch)
    profile = _make_profile(tmp_path)
    profile.memory.scoring.similarity = 0.2
    profile.memory.scoring.importance = 0.8
    persistent = _build_persistent(profile)
    persistent._fake_mem0.search.return_value = {
        "results": [
            {
                "memory": "trivial", "score": 0.9,
                "metadata": {"importance": 0.1, "tier": "episodic"},
            },
            {
                "memory": "important", "score": 0.6,
                "metadata": {"importance": 1.0, "tier": "episodic"},
            },
        ]
    }
    orch = MemoryOrchestrator(profile, persistent)

    items = orch.recall("anything", limit=2)
    assert [i.content for i in items] == ["important", "trivial"]


def test_recall_specific_tier_narrows_to_that_tier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    _set_env(monkeypatch)
    profile = _make_profile(tmp_path)
    persistent = _build_persistent(profile)
    persistent._fake_mem0.search.return_value = {
        "results": [
            {"memory": "ep", "score": 0.5, "metadata": {"tier": "episodic"}},
            {"memory": "se", "score": 0.5, "metadata": {"tier": "semantic"}},
        ]
    }
    orch = MemoryOrchestrator(profile, persistent)

    items = orch.recall("anything", tier=MemoryTier.SEMANTIC)
    assert [i.content for i in items] == ["se"]


def test_recall_working_tier_only_queries_working_layer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    _set_env(monkeypatch)
    profile = _make_profile(tmp_path)
    persistent = _build_persistent(profile)
    working = _FakeWorking()
    working.add(MemoryItem(content="user said hello", tier=MemoryTier.WORKING))
    orch = MemoryOrchestrator(profile, persistent, working=working)

    items = orch.recall("hello", tier=MemoryTier.WORKING, limit=5)
    assert len(items) == 1
    assert items[0].content == "user said hello"
    persistent._fake_mem0.search.assert_not_called()


def test_recall_with_no_tier_merges_working_in_front(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Working items are session-fresh — they should land in front of older
    persistent hits when recall spans all tiers."""
    _set_env(monkeypatch)
    profile = _make_profile(tmp_path)
    persistent = _build_persistent(profile)
    persistent._fake_mem0.search.return_value = {
        "results": [
            {"memory": "old persistent", "score": 0.9, "metadata": {}},
        ]
    }
    working = _FakeWorking()
    working.add(MemoryItem(content="fresh working", tier=MemoryTier.WORKING))
    orch = MemoryOrchestrator(profile, persistent, working=working)

    items = orch.recall("fresh", limit=5)
    assert items[0].content == "fresh working"
    assert any(i.content == "old persistent" for i in items)


def test_recall_empty_query_returns_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    _set_env(monkeypatch)
    profile = _make_profile(tmp_path)
    orch = MemoryOrchestrator(profile, _build_persistent(profile))

    assert orch.recall("") == []
    assert orch.recall("   ".strip()) == []  # whitespace-only also empty


def test_recall_unknown_tier_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    _set_env(monkeypatch)
    profile = _make_profile(tmp_path)
    orch = MemoryOrchestrator(profile, _build_persistent(profile))

    with pytest.raises(ValueError, match="unknown memory tier"):
        orch.recall("anything", tier="bogus")


# ---------- back-compat shim ----------


def test_search_records_shim_delegates_to_persistent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Existing agent code calls `.search_records()` on `self.memory`; the
    orchestrator must offer the same method so the agent can adopt it without
    changing call sites."""
    _set_env(monkeypatch)
    profile = _make_profile(tmp_path)
    persistent = _build_persistent(profile)
    persistent._fake_mem0.search.return_value = {
        "results": [{"memory": "x", "score": 0.5, "metadata": {}}]
    }
    orch = MemoryOrchestrator(profile, persistent)

    hits = orch.search_records("anything", limit=1)
    assert len(hits) == 1
    # Hybrid scoring is on by default in the shim — surface check.
    assert "_hybrid_score" in hits[0]
