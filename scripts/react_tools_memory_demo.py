"""End-to-end ReActAgent demo: calculator tool + Tavily web search + memory recall.

Three turns against ONE agent instance, sharing memory across the turns:

  Turn 1 — math question. The LLM picks the `calculator` tool, gets a
           number, answers in plain text. The Q->A is persisted to mem0
           as memory_type='outcome'.

  Turn 2 — open-domain question. The LLM picks the `web_search` tool
           (Tavily REST), summarises the top results, answers. Again
           persisted as 'outcome'.

  Turn 3 — recall question that points back at Turn 1 ("what math
           problem did I ask earlier?"). The LLM picks `memory_recall`
           (built into BaseAgent), retrieves the Turn-1 outcome record,
           and answers from it.

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
import math
import operator
import os
import sys
import tempfile
from pathlib import Path

import httpx
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from DefenseAgent.agent import ReActAgent
from DefenseAgent.config import AgentProfile
from DefenseAgent.llm.llm import LLM
from DefenseAgent.memory import ContextCompressor, DefaultMemory
from DefenseAgent.reflection import Reflector
from DefenseAgent.tools import ToolRegistry


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAYA_PROFILE = PROJECT_ROOT / "agents" / "maya_rodriguez" / "profile.yaml"


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


def _print_step_trace(steps) -> None:
    """Print one short line per AgentStep so you can see the LLM's path through tools."""
    for s in steps:
        if s.kind == "tool_call":
            names = ", ".join(tc.name for tc in s.tool_calls)
            print(f"   [step {s.index}] tool_call → {names}")
        elif s.kind == "tool_result":
            for tr in s.tool_results:
                preview = (tr.content or "").splitlines()[0][:120]
                print(f"   [step {s.index}] tool_result ({tr.name}) → {preview}")
        elif s.kind == "answer":
            print(f"   [step {s.index}] answer ({s.usage.total_tokens if s.usage else 0} tok)")


async def _run_turn(agent: ReActAgent, turn: int, task: str) -> None:
    """Run a single turn, print the answer + a compact trace, and surface failures inline."""
    print(f"\n--- Turn {turn} ---")
    print(f"User : {task}")
    try:
        result = await agent.run(task, max_steps=6)
    except Exception as e:
        print(f"[demo] turn {turn} raised {type(e).__name__}: {e}")
        return
    print(f"Maya : {result.final_answer}")
    print(f"Trace:")
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

    profile = AgentProfile.from_yaml(MAYA_PROFILE)

    if args.keep_memory:
        memory_dir: Path | None = None
        print("[demo] keeping memory at the profile's default location")
    else:
        tmp_root = Path(tempfile.mkdtemp(prefix="agent_lab_demo_"))
        memory_dir = tmp_root / "memory"
        print(f"[demo] using fresh memory dir: {memory_dir}")

    _banner("Build the agent (LLM + DefaultMemory + ToolRegistry + Reflector)")
    llm = LLM.from_env(load_env=False)
    memory = DefaultMemory(
        profile, storage_path=memory_dir, load_env=False,
    )
    compactor = ContextCompressor(profile, load_env=False)
    reflector = Reflector(memory, llm)
    tools = ToolRegistry()
    tools.tool(calculator)
    tools.tool(web_search)
    print(f"adapter: {type(llm.adapter).__name__} (model={llm.adapter.model})")
    print(f"tools  : {tools.names()}  (+ memory_recall via BaseAgent)")

    async with ReActAgent(
        profile,
        llm=llm,
        memory=memory,
        tools=tools,
        reflector=reflector,
        compactor=compactor,
        reflect_after_run=False,  # keep the demo cheap; no extra LLM call after each turn
    ) as agent:
        _banner("Turn 1 — exercises the calculator tool")
        await _run_turn(
            agent, 1,
            "What is 47 * 89 + sqrt(144)? Use a tool — don't do the arithmetic in your head.",
        )

        _banner("Turn 2 — exercises the Tavily web search tool")
        await _run_turn(
            agent, 2,
            "Use the web_search tool to find out who won the Nobel Prize in Physics in 2024, "
            "then tell me their name and what they were awarded for.",
        )

        _banner("Turn 3 — exercises memory_recall across turns")
        await _run_turn(
            agent, 3,
            "Earlier I asked you to compute a math expression. "
            "Look in your memory and tell me what the expression was and the answer you gave.",
        )

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
