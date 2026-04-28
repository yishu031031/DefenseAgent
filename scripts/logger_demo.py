"""Comprehensive logger demo — every logger capability, end-to-end.

What this demonstrates (in order):
  1. AgentLogger.from_profile — bind the logger to an AgentProfile from Module 2.
  2. All five log levels with level filtering.
  3. Structured kwargs going into the `data` field.
  4. Reserved-kwarg protection — the logger refuses to shadow top-level keys.
  5. File sink appends JSON lines to logs/<agent_id>.log.
  6. Integration with Module 1 (LLM front-door class): wrap llm.chat() with logs.
  7. Error-path integration: catch LLMProviderError and log it, without crashing.

Usage:
    python scripts/logger_demo.py
    python scripts/logger_demo.py path/to/other_profile.yaml
"""
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from DefenseAgent.config import AgentProfile, ConfigError            # ← front-door class
from DefenseAgent.llm import LLM, Message                            # ← front-door class
from DefenseAgent.llm import LLMError, LLMProviderError
from DefenseAgent.ops import AgentLogger                             # ← front-door class


from DefenseAgent.examples import EXAMPLE_PROFILE_PATH as DEFAULT_PROFILE_PATH


# -------------------- demo sections --------------------


def demo_level_filtering(logger: AgentLogger) -> None:
    """Section 1: show that only INFO+ emits on a default logger."""
    print("\n[section 1] Level filtering (default level = INFO)\n" + "-" * 60)
    logger.debug("demo.level", "this DEBUG line is suppressed")
    logger.info("demo.level", "INFO passes through")
    logger.warning("demo.level", "WARNING passes through")
    logger.error("demo.level", "ERROR passes through")
    logger.critical("demo.level", "CRITICAL passes through")


def demo_structured_data(logger: AgentLogger) -> None:
    """Section 2: arbitrary kwargs become the structured `data` field."""
    print("\n[section 2] Structured kwargs -> data field\n" + "-" * 60)
    logger.info(
        "demo.data",
        "A log line with rich payload",
        request_id="req-abc-123",
        retry_count=0,
        nested={"k": "v", "list": [1, 2, 3]},
    )


def demo_reserved_key_rejection(logger: AgentLogger) -> None:
    """Section 3: demonstrate the reserved-key ValueError."""
    print("\n[section 3] Reserved-kwarg rejection\n" + "-" * 60)
    try:
        logger.info("demo.reserved", "trying to shadow agent_id", agent_id="other")
    except ValueError as e:
        print(f"[demo] caught expected ValueError: {e}")


async def demo_successful_llm_call(
    logger: AgentLogger, llm: LLM, user_question: str, system_prompt: str,
) -> None:
    """Section 4: wrap a real llm.chat() call with request/response logs."""
    print("\n[section 4] Wrapping a real LLM call with log events\n" + "-" * 60)
    logger.info(
        "llm.request",
        "Sending chat request",
        adapter=type(llm.adapter).__name__,
        model=getattr(llm.adapter, "_model", "?"),
        messages_count=1,
        max_tokens=160,
    )
    try:
        resp = await llm.chat(
            [Message(role="user", content=user_question)],
            system=system_prompt,
            temperature=0.7,
            max_tokens=160,
        )
    except LLMProviderError as e:
        logger.error(
            "llm.error",
            "Provider returned error",
            provider=e.provider,
            status_code=e.status_code,
        )
        raise

    logger.info(
        "llm.response",
        "Received chat response",
        stop_reason=resp.stop_reason,
        prompt_tokens=resp.usage.prompt_tokens,
        completion_tokens=resp.usage.completion_tokens,
        total_tokens=resp.usage.total_tokens,
        content_preview=resp.content[:80],
    )
    print(f"[demo] assistant reply: {resp.content}")


def demo_error_path_logging(logger: AgentLogger) -> None:
    """Section 5: wire an in-process stub that always raises, log the error."""
    print("\n[section 5] Logging an error path (stubbed failure)\n" + "-" * 60)

    simulated = LLMProviderError(
        provider="demo-stub", status_code=503, message="simulated outage",
    )
    try:
        raise simulated
    except LLMProviderError as e:
        logger.error(
            "llm.error",
            "Stubbed provider error for demo purposes",
            provider=e.provider,
            status_code=e.status_code,
            retry_scheduled=True,
        )
    print("[demo] simulated LLMProviderError was logged but did not propagate")


# -------------------- main --------------------


async def main() -> int:
    profile_path = (
        Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PROFILE_PATH
    )

    # Module 2: profile front door
    try:
        profile = AgentProfile.from_yaml(profile_path)
    except ConfigError as e:
        print(f"[demo] profile load failed: {type(e).__name__}: {e}")
        return 2

    # Module 3: logger front door, bound to the profile
    log_file = (
        Path(__file__).resolve().parent.parent
        / "logs"
        / f"{profile.id}.log"
    )
    logger = AgentLogger.from_profile(
        profile, log_file=log_file, level=logging.INFO,
    )
    print(f"[demo] profile   : {profile.name} (id={profile.id})")
    print(f"[demo] log file  : {log_file}")

    # Sections 1–3 are pure logger demos, no LLM needed.
    demo_level_filtering(logger)
    demo_structured_data(logger)
    demo_reserved_key_rejection(logger)

    # Section 4 needs Module 1's LLM front door.
    system_prompt = (
        f"You are {profile.name}, a {profile.age}-year-old.\n"
        f"Traits: {profile.traits}\n"
        f"Backstory: {profile.backstory.strip()}\n"
        "Answer briefly and stay in character."
    )
    try:
        llm = LLM.from_env()
        await demo_successful_llm_call(
            logger, llm,
            user_question="In one sentence, what class do you have next?",
            system_prompt=system_prompt,
        )
    except LLMError as e:
        print(f"[demo] LLM call skipped/failed: {type(e).__name__}: {e}")

    demo_error_path_logging(logger)

    # Summarize what landed on disk.
    print("\n" + "=" * 60)
    if log_file.exists():
        n = sum(1 for _ in log_file.open(encoding="utf-8") if _.strip())
        print(f"[demo] wrote {n} JSON-line(s) to {log_file}")
        print("[demo] tail of log file:")
        with log_file.open(encoding="utf-8") as f:
            for line in f.readlines()[-3:]:
                print("   " + line.rstrip())
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
