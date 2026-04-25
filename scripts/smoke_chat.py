"""Smoke test: send one message through whatever provider .env is configured for.

Usage (from project root, with the conda env active):
    python scripts/smoke_chat.py
"""
import asyncio
import os
import sys
from pathlib import Path

# Allow running as a script without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from DefenseAgent.llm import LLM, Message                              # ← front-door class
from DefenseAgent.llm import LLMError


async def main() -> int:
    # Module 1 — the `LLM` front-door class reads .env and builds the right adapter.
    try:
        llm = LLM.from_env()
    except LLMError as e:
        print(f"[smoke] configuration error: {e}")
        return 2

    provider = os.environ.get("AGENT_LAB_LLM_PROVIDER", "<unset>").strip().lower()
    model = getattr(llm.adapter, "_model", "<unknown>")
    print(f"[smoke] provider={provider}  model={model}")
    print(f"[smoke] adapter={type(llm.adapter).__name__}")
    print("[smoke] sending: 'Say hello in 5 words or fewer.'")

    try:
        resp = await llm.chat( #llm的核心函数有chat,chat_stream
            [Message(role="user", content="Say hello in 5 words or fewer.")],
            max_tokens=64,
            temperature=0.2,
        )
    except LLMError as e:
        print(f"[smoke] LLM error: {e}")
        return 1

    print("---")
    print(f"content:     {resp.content!r}")
    print(f"stop_reason: {resp.stop_reason}")
    print(
        f"usage:       prompt={resp.usage.prompt_tokens} "
        f"completion={resp.usage.completion_tokens} "
        f"total={resp.usage.total_tokens}"
    )
    print(f"tool_calls:  {len(resp.tool_calls)}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
