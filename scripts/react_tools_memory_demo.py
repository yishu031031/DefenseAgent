"""End-to-end ReActAgent demo: chained calculator + Tavily web search + memory recall.

Three multi-step turns against ONE agent instance, sharing memory across
the turns. Each turn requires the LLM to invoke multiple tools in
sequence and pass intermediate values forward — the classic ReAct
Thought → Action → Observation → Thought → … loop.

  Turn 1 — chained calculator (3 sequential calculator calls).
           The LLM must compute a = 47*89, then b = sqrt(a+122) using
           the value of a, then result = (a-100)/b using both. Each
           call's output feeds the next.

  Turn 2 — web research × 2 + arithmetic × 2 (4 tool calls).
           Two web_search calls pull two birth years; two calculator
           calls reduce them to age difference + average.

  Turn 3 — memory_recall + web_search + calculator (3 tool calls).
           memory_recall pulls the result from Turn 1 out of mem0,
           web_search pulls a current fact, calculator combines them.

The trace printer surfaces every event the agent emits — its
intermediate `thought:` text, every tool_call with its arguments, every
tool result, and the final answer — so you can read the full ReAct loop
end-to-end.

Memory is rooted in a fresh tempdir so each run starts clean — nothing
persists across invocations of this script.

Usage (from project root, conda env active):
    python scripts/react_tools_memory_demo.py

Required env (.env at the project root):
    AGENT_LAB_LLM_PROVIDER=...   (e.g. deepseek)
    <PROVIDER>_API_KEY=...       (per-provider block)
    <PROVIDER>_MODEL=...
    EMBEDDING_API_KEY=...        (mem0 needs an embedder)
    EMBEDDING_BASE_URL=...
    EMBEDDING_MODEL=...
    TAVILY_API_KEY=...
"""
import argparse
import ast
import asyncio
import logging
import math
import operator
import os
import sys
import tempfile
from pathlib import Path

import httpx
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Quiet mem0's optional-dependency notices ("Failed to load spaCy …", BM25 hint)
# — they are informational, not failures, and only fire when the optional
# packages are absent. Run with --verbose to see them.
for _name in ("mem0.utils.spacy_models", "mem0.utils.factory", "mem0.memory.main"):
    logging.getLogger(_name).setLevel(logging.ERROR)

from DefenseAgent import AgentConfig, ReActAgent
from DefenseAgent.config import AgentProfile


from DefenseAgent.examples import EXAMPLE_PROFILE_PATH as EXAMPLE_PROFILE
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------- Tool 1: safe arithmetic calculator ----------

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {
    "sqrt": math.sqrt, "log": math.log, "log10": math.log10, "exp": math.exp,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "abs": abs, "round": round, "min": min, "max": max,
}


def _eval_node(node: ast.AST) -> float:
    """Recursively evaluate a parsed arithmetic AST; rejects anything outside the whitelist."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCS
        and not node.keywords
    ):
        return _FUNCS[node.func.id](*[_eval_node(a) for a in node.args])
    raise ValueError(f"unsupported expression node: {ast.dump(node)}")


def calculator(expression: str) -> str:
    """Evaluate a Python-style arithmetic expression. Supports + - * / // % **, unary +/-, and the functions sqrt, log, log10, exp, sin, cos, tan, abs, round, min, max. Returns the numeric result as a string, or an error message."""
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_eval_node(tree.body))
    except Exception as e:
        return f"calculator error: {type(e).__name__}: {e}"


# ---------- Tool 2: Tavily web search ----------

_TAVILY_URL = "https://api.tavily.com/search"


async def web_search(query: str) -> str:
    """Search the web via Tavily and return a compact summary (Tavily's `answer` plus the top 3 result titles + URLs + snippets)."""
    api_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not api_key:
        return "web_search error: TAVILY_API_KEY is not set in .env"
    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": 3,
        "include_answer": True,
        "search_depth": "basic",
    }
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(_TAVILY_URL, json=payload)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPError as e:
        return f"web_search error: {type(e).__name__}: {e}"

    lines: list[str] = []
    answer = data.get("answer")
    if answer:
        lines.append(f"Tavily answer: {answer}")
    for i, hit in enumerate(data.get("results", []), 1):
        title = hit.get("title", "(untitled)")
        url = hit.get("url", "")
        snippet = (hit.get("content") or "").strip().replace("\n", " ")
        lines.append(f"{i}. {title} — {url}")
        if snippet:
            lines.append(f"   {snippet[:240]}")
    return "\n".join(lines) if lines else "(no results)"


# ---------- Demo orchestration ----------

def _banner(title: str) -> None:
    """Print a wide visual divider."""
    line = "=" * 72
    print(f"\n{line}\n{title}\n{line}")


def _truncate(text: str, n: int) -> str:
    """Trim `text` to `n` chars, collapsing newlines and adding an ellipsis."""
    flat = " ".join((text or "").split())
    return flat if len(flat) <= n else flat[: n - 1] + "…"


def _format_args(args: dict) -> str:
    """Render tool arguments as `key=value` pairs, truncated for readability."""
    return ", ".join(f"{k}={_truncate(str(v), 80)!r}" for k, v in args.items())


def _print_step_trace(steps) -> None:
    """Print one line per ReAct event — the LLM's intermediate reasoning, every tool call with its args, every tool result, and the final answer banner — so the multi-step ReAct loop is visible end-to-end."""
    tool_calls_count = 0
    for s in steps:
        if s.kind == "tool_call":
            if s.content:
                print(f"   [step {s.index}] thought   {_truncate(s.content, 180)}")
            for tc in s.tool_calls:
                tool_calls_count += 1
                print(f"   [step {s.index}] tool_call {tc.name}({_format_args(tc.arguments)})")
        elif s.kind == "tool_result":
            for tr in s.tool_results:
                print(f"   [step {s.index}] result    [{tr.name}] {_truncate(tr.content or '', 180)}")
        elif s.kind == "answer":
            tokens = s.usage.total_tokens if s.usage else 0
            print(f"   [step {s.index}] answer    ({tokens} tok)")
    print(f"   ── {tool_calls_count} tool call(s) total ──")


async def _run_turn(agent: ReActAgent, turn: int, task: str) -> None:
    """Run a single turn end-to-end, print the answer + a compact trace, surface failures inline."""
    print(f"\n--- Turn {turn} ---")
    print(f"User : {task}")
    try:
        result = await agent.run(task, max_steps=12)
    except Exception as e:
        print(f"[demo] turn {turn} raised {type(e).__name__}: {e}")
        return
    print(f"{agent.profile.name} : {result.final_answer}")
    print("Trace:")
    _print_step_trace(result.steps)
    print(
        f"Total tokens: {result.usage.total_tokens} "
        f"(prompt={result.usage.prompt_tokens}, completion={result.usage.completion_tokens})"
    )


async def main() -> int:
    """Build the agent with calculator + Tavily tools wired in, run three turns, then close."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep-memory",
        action="store_true",
        help="reuse the profile's default memory dir instead of a fresh tempdir",
    )
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env", override=False)

    if not os.environ.get("TAVILY_API_KEY"):
        print("[demo] TAVILY_API_KEY missing in .env — Tavily turn will return an error string.")

    profile = AgentProfile.from_yaml(EXAMPLE_PROFILE)

    if args.keep_memory:
        memory_dir: Path | None = None
        print("[demo] keeping memory at the profile's default location")
    else:
        tmp_root = Path(tempfile.mkdtemp(prefix="agent_lab_demo_"))
        memory_dir = tmp_root / "memory"
        print(f"[demo] using fresh memory dir: {memory_dir}")

    _banner("Build the agent (one AgentConfig → ReActAgent(config))")
    config = AgentConfig(
        profile=profile,
        load_env=False,             # we already loaded .env above
        tools=[calculator, web_search],
        storage_path=memory_dir,
        reflect_after_run=False,    # keep the demo cheap; no extra LLM call after each turn
    )
    async with ReActAgent(config) as agent:
        print(f"adapter: {type(agent.llm.adapter).__name__} (model={agent.llm.adapter.model})")
        print(f"tools  : {agent.tools.names()}  (+ memory_recall via BaseAgent)")
        _banner("Turn 1 — chained calculator calls (intermediate values feed the next)")
        await _run_turn(
            agent, 1,
            "Walk through this step by step. Use the calculator tool for every arithmetic "
            "operation — never do math in your head:\n"
            "  Step 1: a = 47 * 89\n"
            "  Step 2: b = sqrt(a + 122)        (use the value of a from step 1)\n"
            "  Step 3: result = (a - 100) / b   (use both a and b)\n"
            "Round the final result to 2 decimals. Report a, b, and result.",
        )

        _banner("Turn 2 — web research × 2 + arithmetic")
        await _run_turn(
            agent, 2,
            "Use the web_search tool TWICE — once for each fact — and then the calculator:\n"
            "  (1) Find the year Geoffrey Hinton was born.\n"
            "  (2) Find the year John J. Hopfield was born.\n"
            "  (3) Compute the absolute age difference (use calculator).\n"
            "  (4) Compute the average of the two birth years (use calculator).\n"
            "Report all four numbers (the two birth years, the difference, the average).",
        )

        _banner("Turn 3 — memory_recall + web search + arithmetic, chained")
        await _run_turn(
            agent, 3,
            "Three steps, in order:\n"
            "  (1) Use memory_recall to find the FINAL numeric result you computed in our "
            "first exchange (the rounded result from step 3 of the chained-calculator turn).\n"
            "  (2) Use web_search to find the population of Iceland in 2024.\n"
            "  (3) Use the calculator to divide Iceland's population by that earlier "
            "result, rounded to the nearest integer.\n"
            "Report the recalled number, Iceland's population, and the final ratio.",
        )

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
