"""Reflection module demo — The example agent observes, then reflects.

Exercises all five front-door classes together:
  • AgentProfile (Module 2) — the agent's identity + cognitive.reflection_threshold.
  • LLM          (Module 1) — rates importance + synthesizes insights.
  • Memory       (Module 4) — holds the agent's day.
  • AgentLogger  (Module 3) — records each reflect_now() call.
  • Reflector    (Module 5) — the new piece: reads memories, prompts LLM, writes back.

Flow:
  1. Load profile.
  2. Score importance for each observation via LLM (Park §3.2.1).
  3. Store the observation with that LLM-rated importance.
  4. After all observations, force a reflection via LLM (Park §3.2.2).
  5. Show the emergent reflections.
  6. Query memory for a thematic question — reflections surface alongside raw observations.

Requires both the LLM (AGENT_LAB_LLM_PROVIDER) and the embedding adapter
(EMBEDDING_*) to be configured in `.env`. Exits 2 with a clear message
if either is missing.

Usage:
    python scripts/reflection_demo.py
    python scripts/reflection_demo.py path/to/other_profile.yaml
"""
import asyncio
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from DefenseAgent.config import AgentProfile, ConfigError
from DefenseAgent.llm import LLM
from DefenseAgent.llm import LLMError, LLMProviderError
from DefenseAgent.memory import Memory
from DefenseAgent.memory import EmbeddingConfigError, EmbeddingProviderError
from DefenseAgent.ops import AgentLogger
from DefenseAgent.reflection import Reflector


from DefenseAgent.examples import EXAMPLE_PROFILE_PATH as DEFAULT_PROFILE_PATH

_NOW = datetime(2026, 4, 22, 18, 0, tzinfo=timezone.utc)

# Observations from the agent's day. hours_ago is used to back-date the memory
# so the reflection has a realistic chronological spread.
OBSERVATIONS = [
    ("Attended the 9 AM data structures lecture, covered binary search trees.", 9),
    ("Grabbed coffee with my roommate before class.", 10),
    ("Spent two hours in the library working on the BST homework set.", 4),
    ("Got stuck on problem 3 for an hour, finally worked it out with the TA.", 2),
    ("Met the study group at 5 PM and walked them through problem 3.", 1),
    ("Realized I learn faster when I struggle alone first.", 0),
]


async def main() -> int:
    # --- Step 1: profile (Module 2) -------------------------------------
    profile_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PROFILE_PATH
    try:
        profile = AgentProfile.from_yaml(profile_path)
    except ConfigError as e:
        print(f"[demo] profile load failed: {type(e).__name__}: {e}")
        return 2
    threshold = profile.cognitive.reflection_threshold
    print(f"[demo] profile  : {profile.name} (id={profile.id})")
    print(f"[demo] threshold: {threshold} observations per reflection")

    # --- Step 2: logger (Module 3) --------------------------------------
    log_file = (
        Path(__file__).resolve().parent.parent / "logs" / f"{profile.id}.reflection.log"
    )
    logger = AgentLogger.from_profile(
        profile, log_file=log_file, stream=None, level=logging.INFO,
    )

    # --- Step 3: LLM (Module 1) + Memory (Module 4) ---------------------
    try:
        llm = LLM.from_env()
    except LLMError as e:
        print(f"[demo] LLM configuration: {type(e).__name__}: {e}")
        return 2
    try:
        memory = Memory.from_env(profile, clock=lambda: _NOW)
    except EmbeddingConfigError as e:
        print(f"[demo] embedding configuration: {e}")
        print("[demo] fill EMBEDDING_API_KEY (and others) in .env to run this demo.")
        return 2
    print(f"[demo] LLM      : {type(llm.adapter).__name__} "
          f"(model={getattr(llm.adapter, '_model', '?')})")
    print(f"[demo] embedder : {type(memory.embedding_adapter).__name__} "
          f"(model={memory.embedding_adapter._model})")

    # --- Step 4: the new piece — Reflector (Module 5) -------------------
    reflector = Reflector(
        memory, llm,
        num_insights=3,
        reflection_importance=8.5,
        clock=lambda: _NOW,
    )
    print(f"[demo] reflector: {type(reflector).__name__}")

    # --- Step 5: seed observations with LLM-scored importance -----------
    print(f"\n[demo] recording {len(OBSERVATIONS)} observations "
          f"(LLM scores importance for each)…")
    try:
        for content, hours_ago in OBSERVATIONS:
            score = await reflector.score_importance(content)
            # Back-date this observation by overriding the stream clock
            # for just this one call. (When Memory.remember(timestamp=...)
            # lands, this shim goes away.)
            memory.stream._clock = lambda h=hours_ago: _NOW - timedelta(hours=h)
            record = await memory.remember(
                content, kind="observation", importance=score,
            )
            logger.info(
                "memory.observed", "stored observation",
                importance=score, hours_ago=hours_ago, record_id=record.id,
            )
            print(f"  imp={score:>4.1f}  (−{hours_ago}h)  {content}")
    except LLMProviderError as e:
        print(f"[demo] LLM error during importance scoring: {e}")
        return 1
    except EmbeddingProviderError as e:
        print(f"[demo] embedding error: {e}")
        return 1

    # Restore clock to NOW for the reflection + retrieval steps.
    memory.stream._clock = lambda: _NOW

    # --- Step 6: reflect -----------------------------------------------
    print(f"\n[demo] unreflected count: {reflector.unreflected_count} "
          f"(threshold is {threshold})")
    print("[demo] forcing a reflection via reflector.reflect_now()…")
    logger.info("reflection.started", "synthesizing insights",
                unreflected=reflector.unreflected_count)

    try:
        new_reflections = await reflector.reflect_now()
    except LLMProviderError as e:
        print(f"[demo] LLM error during reflection: {e}")
        return 1

    logger.info(
        "reflection.finished", "insights stored", count=len(new_reflections),
    )

    print(f"\n[demo] The agent's reflections ({len(new_reflections)}):")
    print("=" * 72)
    for r in new_reflections:
        print(f"  • {r.content}")
    print("=" * 72)

    # --- Step 7: retrieval now includes reflections ---------------------
    query = "what patterns are emerging in how the agent works?"
    print(f"\n[demo] query: {query!r}")
    results = await memory.recall(query, top_k=5)
    for s in results:
        marker = "★" if s.record.kind == "reflection" else " "
        print(f"  {marker} [{s.record.kind:11s} imp={s.record.importance:>4.1f} "
              f"score={s.score:.3f}]  {s.record.content}")

    print(f"\n[demo] wrote log lines to {log_file}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
