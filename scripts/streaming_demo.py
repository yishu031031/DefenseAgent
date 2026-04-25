"""Streaming demo — watch an LLM response type itself out.

Uses the provider configured in .env. Prints each text delta as it arrives,
then a one-line summary with stop_reason and token usage at the end.

Usage (from project root, conda env active):
    python scripts/streaming_demo.py
"""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from DefenseAgent.llm import LLM, Message, StreamEnd, TextDelta
from DefenseAgent.llm import LLMError


USER_QUESTION = (
    "Explain in 2-3 short sentences how binary search works, "
    "like you're telling a classmate over coffee."
)


async def main() -> int:
    try:
        llm = LLM.from_env()
    except LLMError as e:
        print(f"[stream] configuration error: {e}")
        return 2

    model = getattr(llm.adapter, "_model", "<unknown>")
    print(f"[stream] adapter={type(llm.adapter).__name__}  model={model}")
    print(f"[stream] user: {USER_QUESTION}")
    print("---")
    print("[stream] assistant: ", end="", flush=True)

    t0 = time.monotonic()
    first_token_at = None
    total_chars = 0
    final: StreamEnd | None = None

    try:
        async for chunk in llm.chat_stream(
            [Message(role="user", content=USER_QUESTION)],
            temperature=0.5,
            max_tokens=180,
        ):
            if isinstance(chunk, TextDelta):
                if first_token_at is None:
                    first_token_at = time.monotonic()
                print(chunk.text, end="", flush=True)
                total_chars += len(chunk.text)
            elif isinstance(chunk, StreamEnd):
                final = chunk
    except LLMError as e:
        print()
        print(f"[stream] LLM error: {e}")
        return 1

    t1 = time.monotonic()
    print()
    print("---")
    ttft = (first_token_at - t0) if first_token_at else None
    total = t1 - t0
    ttft_str = f"{ttft*1000:.0f}ms" if ttft is not None else "n/a"
    print(f"[stream] chars streamed : {total_chars}")
    print(f"[stream] time to first  : {ttft_str}")
    print(f"[stream] total time     : {total*1000:.0f}ms")
    if final is not None:
        print(
            f"[stream] stop_reason    : {final.stop_reason}"
        )
        print(
            f"[stream] usage          : "
            f"prompt={final.usage.prompt_tokens} "
            f"completion={final.usage.completion_tokens} "
            f"total={final.usage.total_tokens}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
