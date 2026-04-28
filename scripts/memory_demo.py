"""Memory module demo — a day in the example agent's life.

Builds up a stream of memories spanning every kind (observation / fact /
preference / plan / reflection), then runs three queries to show the hybrid
retriever's behavior. The retriever's score components are printed so the
ranking is inspectable.

Uses real embeddings (the configured EMBEDDING_* provider in .env), and
exercises all four module front doors:
  • AgentProfile (Module 2) loads the example agent.
  • AgentLogger  (Module 3) records each add()/recall() event.
  • Memory       (Module 4) is the unified memory surface.

Usage (from project root, conda env active, EMBEDDING_API_KEY filled in):
    python scripts/memory_demo.py

If EMBEDDING_API_KEY is blank, the script exits 2 with a clear message.
"""
import asyncio
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from DefenseAgent.config import AgentProfile, ConfigError            # ← front-door class
from DefenseAgent.memory import Memory                                # ← front-door class
from DefenseAgent.memory import (
    EmbeddingConfigError,
    EmbeddingProviderError,
)
from DefenseAgent.ops import AgentLogger                              # ← front-door class


from DefenseAgent.examples import EXAMPLE_PROFILE_PATH as DEFAULT_PROFILE_PATH

_NOW = datetime(2026, 4, 22, 18, 0, tzinfo=timezone.utc)  # reference "now"

# Each entry: (content, kind, importance, hours_ago[, metadata])
SEED_MEMORIES = [
    # Observations from today
    ("Attended the 9 AM data structures lecture, covered binary search trees.", "observation", 6.0, 9),
    ("Grabbed coffee with my roommate before class.", "observation", 3.0, 10),
    ("Spent two hours in the library working on the BST homework set.", "observation", 7.0, 4),
    ("Got stuck on problem 3 for an hour, finally worked it out with the TA.", "observation", 8.0, 2),
    ("Reviewed my Spanish vocabulary on the bus home.", "observation", 4.0, 1),
    # Facts — stable, no recency decay
    ("I'm a second-year Computer Science major.", "fact", 7.0, 24 * 30),
    ("I'm bilingual in Spanish and English.", "fact", 6.0, 24 * 30),
    ("I work part-time at the campus library.", "fact", 7.0, 24 * 30),
    # Preferences — stable personal dispositions
    ("I prefer working in the library over my dorm; it's quieter.", "preference", 6.0, 24 * 7),
    ("I hate 8 AM classes; my brain needs coffee first.", "preference", 5.0, 24 * 7),
    # Plans — with status metadata
    ("Finish the BST homework set by Friday.", "plan", 8.0, 12),
    ("Meet the study group at 5 PM in the library.", "plan", 7.0, 8),
    ("Pick up groceries after class.", "plan", 4.0, 24, {"status": "done"}),
    # Reflection (Module 5 would normally produce these; seeded here for demo)
    ("I learn algorithms faster when I struggle first and ask questions after.", "reflection", 9.0, 6),
]


def _format_scored(s, width: int = 72) -> str:
    content = s.record.content
    if len(content) > width:
        content = content[: width - 1] + "…"
    return (
        f"  [{s.record.kind:12s} imp={s.record.importance:>4.1f}] "
        f"score={s.score:.3f}  "
        f"rec={s.recency_score:.2f} imp_n={s.importance_score:.2f} rel={s.relevance_score:.2f}  "
        f"(dense#{s.dense_rank} bm25#{s.sparse_rank})\n"
        f"    {content}"
    )


async def main() -> int:
    # --- Step 1: profile (Module 2) --------------------------------------
    profile_path = (
        Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PROFILE_PATH
    )
    try:
        profile = AgentProfile.from_yaml(profile_path)
    except ConfigError as e:
        print(f"[demo] profile load failed: {type(e).__name__}: {e}")
        return 2
    print(f"[demo] profile : {profile.name} (id={profile.id})")

    # --- Step 2: logger (Module 3) ---------------------------------------
    log_file = (
        Path(__file__).resolve().parent.parent / "logs" / f"{profile.id}.memory.log"
    )
    logger = AgentLogger.from_profile(
        profile, log_file=log_file, stream=None, level=logging.INFO,
    )

    # --- Step 3: memory (Module 4) — the unified front door --------------
    try:
        memory = Memory.from_env(profile)
    except EmbeddingConfigError as e:
        print(f"[demo] embedding adapter not configured: {e}")
        print("[demo] fill EMBEDDING_API_KEY (and others) in .env to run this demo.")
        return 2
    print(
        f"[demo] memory  : {type(memory).__name__} "
        f"(embedder={type(memory.embedding_adapter).__name__}, "
        f"model={memory.embedding_adapter._model})"
    )

    # --- Step 4: seed memories with back-dated timestamps ----------------
    # memory.stream exposes the underlying MemoryStream so we can shift the
    # clock per-entry; in normal use the default clock (UTC now) is fine.
    print(f"\n[demo] seeding {len(SEED_MEMORIES)} memories of 5 kinds…")
    for entry in SEED_MEMORIES:
        if len(entry) == 5:
            content, kind, importance, hours_ago, metadata = entry
        else:
            content, kind, importance, hours_ago = entry
            metadata = None
        memory.stream._clock = lambda h=hours_ago: _NOW - timedelta(hours=h)
        try:
            record = await memory.remember(
                content, kind=kind, importance=importance, metadata=metadata,
            )
            logger.info(
                "memory.added", "stored record",
                record_id=record.id, kind=record.kind,
                importance=record.importance, hours_ago=hours_ago,
            )
        except EmbeddingProviderError as e:
            print(f"[demo] embedding failed for {content!r}: {e}")
            return 1
    # Restore both clocks to "now" for retrieval.
    memory.stream._clock = lambda: _NOW
    memory.retriever._clock = lambda: _NOW
    print(f"[demo] memory now holds {len(memory)} records")

    # --- Step 5: three queries covering different retrieval styles -------
    queries = [
        "How am I doing on the data structures homework?",
        "Where do I like to study, and why?",
        "What should I do next this evening?",
    ]

    for q in queries:
        logger.info("memory.recall", "querying", query=q)
        results = await memory.recall(q, top_k=5)
        logger.info(
            "memory.recalled", "done",
            query=q, hits=len(results),
            top_score=results[0].score if results else None,
        )

        print("\n" + "=" * 72)
        print(f"QUERY: {q}")
        print("=" * 72)
        if not results:
            print("  (no memories)")
            continue
        for s in results:
            print(_format_scored(s))

    # --- Step 6: summary -------------------------------------------------
    n_lines = sum(1 for _ in log_file.open(encoding="utf-8") if _.strip())
    print("\n" + "=" * 72)
    print(f"[demo] wrote {n_lines} JSON log line(s) to {log_file}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
