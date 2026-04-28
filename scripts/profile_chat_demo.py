"""End-to-end demo: load the student agent profile and talk to it via the LLM.

Combines Module 1 (LLM) and Module 2 (AgentProfile).
Uses the provider configured in .env (e.g., AGENT_LAB_LLM_PROVIDER=deepseek).

Usage (from project root, conda env active):
    python scripts/profile_chat_demo.py
    python scripts/profile_chat_demo.py path/to/other_profile.yaml
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from DefenseAgent.config import AgentProfile, ConfigError            # ← front-door class
from DefenseAgent.llm import LLM, Message                            # ← front-door class
from DefenseAgent.llm import LLMError


from DefenseAgent.examples import EXAMPLE_PROFILE_PATH as DEFAULT_PROFILE_PATH

# The user-facing question the agent will answer in character.
USER_QUESTION = "It's 2 PM. What have you been doing this morning, and what's next?"


def build_system_prompt(profile: AgentProfile) -> str:
    """Collapse the profile's identity fields into a role-play system prompt."""
    return (
        f"You are {profile.name}, a {profile.age}-year-old.\n"
        f"Traits: {profile.traits}\n"
        f"Backstory: {profile.backstory.strip()}\n"
        f"Today's plan: {profile.initial_plan.strip()}\n"
        "Stay fully in character. Reply in the first person, casually, "
        "and keep your answer under 80 words."
    )


async def main() -> int:
    profile_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PROFILE_PATH

    # --- Step 1: load and validate the agent profile (Module 2) -----------
    try:
        profile = AgentProfile.from_yaml(profile_path)
    except ConfigError as e:
        print(f"[demo] profile load failed: {type(e).__name__}: {e}")
        return 2
    print(f"[demo] profile: {profile.name}, age {profile.age}, id={profile.id}")
    print(f"[demo] profile source: {profile_path}")

    # --- Step 2: build the LLM from .env (Module 1) -----------------------
    try:
        llm = LLM.from_env()
    except LLMError as e:
        print(f"[demo] LLM config failed: {type(e).__name__}: {e}")
        return 2
    print(f"[demo] adapter: {type(llm.adapter).__name__} "
          f"(model={getattr(llm.adapter, '_model', '?')})")

    # --- Step 3: compose — profile becomes the system prompt --------------
    system = build_system_prompt(profile)
    print("---")
    print("[demo] system prompt:")
    print(system)
    print("---")
    print(f"[demo] user: {USER_QUESTION}")

    # --- Step 4: send one turn of conversation ----------------------------
    try:
        resp = await llm.chat(
            [Message(role="user", content=USER_QUESTION)],
            system=system,
            temperature=0.7,
            max_tokens=256,
        )
    except LLMError as e:
        print(f"[demo] LLM call failed: {type(e).__name__}: {e}")
        return 1

    # --- Step 5: show the result ------------------------------------------
    print(f"[demo] {profile.name}: {resp.content}")
    print("---")
    print(
        f"[demo] usage: prompt={resp.usage.prompt_tokens} "
        f"completion={resp.usage.completion_tokens} "
        f"total={resp.usage.total_tokens}  "
        f"stop_reason={resp.stop_reason}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
