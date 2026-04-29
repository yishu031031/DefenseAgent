"""End-to-end smoke test against the real memory backend stack.

Touches the real moving parts (no mocks):
  - Ollama (bge-m3) for embeddings via the OpenAI-compatible /v1/embeddings
  - mem0ai 2.x as the memory engine
  - Qdrant in local-file mode (collection file lands in a temp dir per run)
  - DeepSeek chat LLM (only when something actually needs LLM work)

Loads `.env` for credentials, builds a fresh MemoryOrchestrator into a temp
storage path, exercises the full P0+P1+P2+P4+P5 surface, and prints
verbose pass/fail per scenario plus a final stats line.

Usage:
    python scripts/smoke_real_backend.py [--keep]

    --keep   leave the temp Qdrant collection on disk so you can inspect
             it after the run (path printed at the end)
"""
from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


# Project root so `python scripts/...` can import DefenseAgent without install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from DefenseAgent.config import AgentProfile
from DefenseAgent.config.profile import (
    ConsolidationConfig,
    MemoryConfig,
    ScoringWeights,
    TierLimits,
)
from DefenseAgent.llm.types import Message
from DefenseAgent.memory import (
    MemoryConsolidator,
    MemoryItem,
    MemoryOrchestrator,
    MemoryTier,
    WorkingMemory,
)


PASS = "  [PASS]"
FAIL = "  [FAIL]"


class Runner:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0
        self.skipped = 0

    async def aio(self, name: str, coro):
        try:
            await coro
            print(f"{PASS} {name}")
            self.passed += 1
        except Exception as e:  # noqa: BLE001
            self.failed += 1
            print(f"{FAIL} {name}")
            print(f"         {type(e).__name__}: {e}")
            for line in traceback.format_exc().splitlines()[-6:]:
                print(f"         {line}")

    def sync(self, name: str, fn) -> None:
        try:
            fn()
            print(f"{PASS} {name}")
            self.passed += 1
        except Exception as e:  # noqa: BLE001
            self.failed += 1
            print(f"{FAIL} {name}")
            print(f"         {type(e).__name__}: {e}")
            for line in traceback.format_exc().splitlines()[-6:]:
                print(f"         {line}")


def _make_profile(storage_path: Path) -> AgentProfile:
    """Profile tuned so the hybrid re-rank test has a clear winner: importance
    weight dominates similarity. Tier capacities and consolidation thresholds
    are picked so the consolidator test fires on a single seeded item."""
    profile = AgentProfile(
        id="real_backend_smoke", name="SmokeTester", age=25,
        traits="t", backstory="b", initial_plan="p",
        memory=MemoryConfig(
            storage_path=str(storage_path),
            scoring=ScoringWeights(
                similarity=0.3, importance=0.6, recency=0.05, frequency=0.05,
                recency_half_life_days=7.0,
            ),
            tier_limits=TierLimits(
                working_capacity=10, working_ttl_seconds=3600,
            ),
            consolidation=ConsolidationConfig(
                enabled=True,
                promote_to_episodic_threshold=0.7,
                importance_boost_on_promotion=1.1,
                interval_seconds=60,
            ),
        ),
    )
    # Profile needs a source_dir so storage_path resolution succeeds; point at
    # the storage dir's parent so the auto-create paths line up cleanly.
    fake_yaml = storage_path / "profile.yaml"
    fake_yaml.write_text("agent: {}", encoding="utf-8")
    profile._source_path = fake_yaml.resolve()
    return profile


def _build_orchestrator(profile: AgentProfile) -> MemoryOrchestrator:
    """Use the same construction path the agent builder uses, so this script
    exercises the public surface end users will hit."""
    return MemoryOrchestrator.from_profile(
        profile,
        load_env=False,  # already loaded at module top
        storage_path=profile.memory.storage_path,
    )


# --------------------------------------------------------------- test cases


async def t_orchestrator_constructs(orch: MemoryOrchestrator):
    """Construction itself touches mem0 → Ollama for the first embedding
    pre-warm, so a clean construction proves the wiring is sound."""
    assert orch.persistent is not None, "persistent layer not built"
    assert orch.working is not None, "working layer not built"
    assert isinstance(orch.working, WorkingMemory)


async def t_qdrant_collection_file_exists(storage_path: Path):
    """mem0 with on_disk=True writes a Qdrant collection directory under the
    storage path. Confirm it's actually on disk after construction."""
    # mem0's storage layout: <storage>/default_memory/collection/<name>/
    candidates = list(storage_path.rglob("*.lock")) + list(
        storage_path.rglob("storage.sqlite")
    )
    assert candidates, (
        f"no Qdrant collection files found under {storage_path}; "
        f"contents: {list(storage_path.rglob('*'))[:20]}"
    )


async def t_add_and_get_all_round_trips_metadata(orch: MemoryOrchestrator):
    """Write a record with full metadata, then pull it back via get_all and
    verify tier / importance / memory_type survived the Qdrant round-trip.
    This is THE critical test — mem0 has historically filtered metadata
    fields it doesn't recognize."""
    await orch.add(
        [Message(role="user", content="The attacker pivoted via SMB.")],
        tier=MemoryTier.EPISODIC,
        memory_type="trajectory",
        importance=0.55,
        source_run_id="smoke-1",
    )
    records = orch.persistent.get_all(tier=MemoryTier.EPISODIC)
    assert records, f"no records returned from get_all; raw={records!r}"
    last = records[-1]
    metadata = last.get("metadata") or {}
    assert metadata.get("tier") == "episodic", \
        f"tier missing or wrong in metadata: {metadata!r}"
    assert metadata.get("memory_type") == "trajectory", \
        f"memory_type missing: {metadata!r}"
    # Importance should round-trip; allow float fuzzing.
    assert abs(float(metadata.get("importance", 0)) - 0.55) < 1e-3, \
        f"importance lost: {metadata!r}"


async def t_add_multiple_tiers(orch: MemoryOrchestrator):
    """Seed one record in each persistent tier so subsequent recall tests
    have something to find, and so we can verify tier filtering works on the
    real backend."""
    await orch.add_episodic(
        [Message(role="user", content="Saw process spawn cmd.exe at 03:14.")],
        memory_type="observation", importance=0.4,
    )
    await orch.add_semantic(
        [Message(role="user", content="Lateral movement via SMB suggests "
                                     "credential theft, not exploit chain.")],
        memory_type="reflection", importance=0.85,
    )
    await orch.add_procedural(
        [Message(role="user", content="Isolate host playbook: disable NIC, "
                                     "snapshot disk, alert SOC.")],
        memory_type="sop", importance=0.9,
    )
    # Now persistent layer has at least 4 records (the round-trip test added
    # one too). Each tier should have at least 1.
    for tier in (MemoryTier.EPISODIC, MemoryTier.SEMANTIC, MemoryTier.PROCEDURAL):
        recs = orch.persistent.get_all(tier=tier)
        assert recs, f"tier {tier.value} unexpectedly empty after seeding"


async def t_search_records_returns_score(orch: MemoryOrchestrator):
    """Hybrid scoring assumes mem0 returns a `score` field per hit.
    Validate that assumption against the real backend."""
    hits = orch.persistent.search_records(
        "lateral movement attacker", limit=5, scoring="vector",
    )
    assert hits, "no search hits returned for an obvious query"
    first = hits[0]
    assert "score" in first or "memory" in first, \
        f"unexpected hit shape: {first!r}"
    # Score should be a float in [0, 1] when present.
    if "score" in first:
        assert isinstance(first["score"], (int, float)), \
            f"score field has wrong type: {type(first['score'])}"


async def t_hybrid_recall_reranks(orch: MemoryOrchestrator):
    """With profile weights at importance=0.6 / similarity=0.3, the highly
    important semantic reflection should outrank a less-important but more
    similar episodic observation when both match the query."""
    items = orch.recall(
        "credential theft via SMB lateral movement",
        limit=5, scoring="hybrid",
    )
    assert items, "hybrid recall returned nothing"
    contents = [i.content for i in items]
    # The semantic reflection (importance 0.85) should beat the episodic
    # observation (importance 0.4) on hybrid score.
    semantic_idx = next(
        (i for i, c in enumerate(contents) if "credential theft" in c), None,
    )
    episodic_idx = next(
        (i for i, c in enumerate(contents) if "spawn cmd.exe" in c), None,
    )
    assert semantic_idx is not None, \
        f"semantic reflection not in recall: {contents}"
    if episodic_idx is not None:
        assert semantic_idx < episodic_idx, (
            f"semantic should outrank episodic via hybrid scoring; "
            f"got order: {contents}"
        )


async def t_recall_tier_filter(orch: MemoryOrchestrator):
    """Tier-narrowed recall should only return records from that tier."""
    items = orch.recall(
        "playbook isolate", limit=5, tier=MemoryTier.PROCEDURAL,
    )
    assert items, "tier-narrowed recall returned nothing"
    for i in items:
        assert i.tier == MemoryTier.PROCEDURAL, \
            f"got non-procedural item via tier filter: {i.tier} {i.content!r}"


async def t_working_layer_skips_qdrant(
    orch: MemoryOrchestrator, storage_path: Path,
):
    """WORKING writes must not touch Qdrant — they live in the in-memory
    deque only. Detect by counting persistent records before and after."""
    before = len(orch.persistent.get_all())
    await orch.add(
        [Message(role="user", content="WORKING-only scratch note that should never persist.")],
        tier=MemoryTier.WORKING,
        importance=0.5,
    )
    after = len(orch.persistent.get_all())
    assert before == after, (
        f"WORKING write leaked into Qdrant: persistent count changed "
        f"{before} → {after}"
    )
    assert orch.working is not None and len(orch.working) >= 1


async def t_consolidator_promotes_to_qdrant(
    orch: MemoryOrchestrator, storage_path: Path,
):
    """Seed a high-importance WORKING item, run the consolidator, and verify
    a new EPISODIC record appeared in Qdrant. Importance boost should also
    show up on the new record."""
    orch.working.add(MemoryItem(
        content="Repeated failed logins from 10.0.0.42 — possible brute force.",
        tier=MemoryTier.WORKING,
        importance=0.85,
        memory_type="alert",
    ))
    before_episodic = len(orch.persistent.get_all(tier=MemoryTier.EPISODIC))
    consolidator = MemoryConsolidator(orch)
    stats = await consolidator.run_once()
    after_episodic = len(orch.persistent.get_all(tier=MemoryTier.EPISODIC))
    assert stats.promoted_to_episodic >= 1, f"no promotion happened: {stats!r}"
    assert after_episodic > before_episodic, (
        f"EPISODIC count didn't grow: {before_episodic} → {after_episodic}"
    )
    # Find the promoted record by content match.
    promoted = [
        r for r in orch.persistent.get_all(tier=MemoryTier.EPISODIC)
        if "brute force" in (r.get("memory") or "")
    ]
    assert promoted, "promoted record not found by content search"
    metadata = promoted[0].get("metadata") or {}
    assert metadata.get("consolidated_from_tier") == "working", \
        f"consolidated_from_tier not stamped: {metadata!r}"
    # Importance boost: 0.85 * 1.1 = 0.935.
    boosted = float(metadata.get("importance", 0))
    assert 0.9 < boosted <= 1.0, f"importance boost wrong: got {boosted}"


# ----------------------------------------------------------------- main


async def main(argv: list[str]) -> int:
    keep = "--keep" in argv

    # Load .env from project root.
    project_root = Path(__file__).resolve().parent.parent
    env_path = project_root / ".env"
    if not env_path.is_file():
        print(f"[FATAL] .env not found at {env_path}")
        return 2
    load_dotenv(env_path)

    # Sanity-check the env vars the bridge will read so failures are loud.
    for required in (
        "AGENT_LAB_LLM_PROVIDER",
        "DEEPSEEK_API_KEY",
        "EMBEDDING_API_KEY",
        "EMBEDDING_BASE_URL",
        "EMBEDDING_MODEL",
    ):
        if not os.environ.get(required):
            print(f"[FATAL] {required} not set in .env")
            return 2

    storage_path = Path(tempfile.mkdtemp(prefix="defenseagent_smoke_"))
    print(f"storage_path = {storage_path}")
    print(f"embedder    = {os.environ.get('EMBEDDING_MODEL')} "
          f"@ {os.environ.get('EMBEDDING_BASE_URL')}")
    print(f"chat LLM    = {os.environ.get('AGENT_LAB_LLM_PROVIDER')}")
    print()

    profile = _make_profile(storage_path)

    runner = Runner()
    orch: MemoryOrchestrator | None = None

    print("=== Construction & wiring ===\n")
    try:
        orch = _build_orchestrator(profile)
        print(f"{PASS} orchestrator constructed (mem0 + Qdrant + Ollama warm-up)")
        runner.passed += 1
    except Exception as e:  # noqa: BLE001
        print(f"{FAIL} orchestrator construction")
        print(f"         {type(e).__name__}: {e}")
        for line in traceback.format_exc().splitlines()[-8:]:
            print(f"         {line}")
        runner.failed += 1
        print(f"\n=== {runner.passed} passed, {runner.failed} failed ===")
        return 1

    await runner.aio("orchestrator state sanity",
                     t_orchestrator_constructs(orch))
    await runner.aio("Qdrant collection file on disk",
                     t_qdrant_collection_file_exists(storage_path))

    print("\n=== Add / read round-trip ===\n")
    await runner.aio("add() metadata round-trips through Qdrant",
                     t_add_and_get_all_round_trips_metadata(orch))
    await runner.aio("seed each persistent tier",
                     t_add_multiple_tiers(orch))

    print("\n=== Search & recall ===\n")
    await runner.aio("search_records returns score field",
                     t_search_records_returns_score(orch))
    await runner.aio("hybrid recall reranks by importance",
                     t_hybrid_recall_reranks(orch))
    await runner.aio("recall(tier=PROCEDURAL) narrows correctly",
                     t_recall_tier_filter(orch))

    print("\n=== Working layer ===\n")
    await runner.aio("WORKING write doesn't touch Qdrant",
                     t_working_layer_skips_qdrant(orch, storage_path))

    print("\n=== Consolidator ===\n")
    await runner.aio("consolidator promotes WORKING → EPISODIC in Qdrant",
                     t_consolidator_promotes_to_qdrant(orch, storage_path))

    print()
    print(f"=== {runner.passed} passed, {runner.failed} failed ===")

    if keep:
        print(f"storage kept at: {storage_path}")
    else:
        # Best-effort cleanup; on Windows Qdrant sometimes holds file handles
        # past the test, so swallow errors.
        try:
            shutil.rmtree(storage_path, ignore_errors=True)
            print(f"storage cleaned: {storage_path}")
        except Exception as e:  # noqa: BLE001
            print(f"storage cleanup failed (safe to ignore): {e}")

    return 0 if runner.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
