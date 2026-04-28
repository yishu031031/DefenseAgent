#!/usr/bin/env python3
"""Trigger-accuracy eval for a skill's frontmatter `description`.

Usage:
    eval_description.py <skill-path> --queries "q1" "q2" ... [--negative "..."]
                                     [--dotenv PATH]

Each `--queries` entry is a phrase a user might say. The script asks the
agent's LLM (built from the ambient .env via `DefenseAgent.llm.LLM.from_env`)
to judge — given only the skill's name + description — whether that query
should select this skill. `--negative` queries are the inverse: phrases that
look related but should NOT trigger.

The active LLM is whatever `AGENT_LAB_LLM_PROVIDER` plus the matching
`<PROVIDER>_*` block in the loaded .env resolves to. Override at the env
level rather than the CLI to keep this script stateless.

Exit code 0 when at least 80% of judgements match expectations, 1 otherwise.
The output is one line per query plus an aggregate hit-rate line at the end.
"""
from __future__ import annotations

import argparse
import asyncio
import re
import sys
from pathlib import Path

import yaml


_FRONTMATTER_RE = re.compile(r"^---\s*\n(?P<body>.*?)\n---\s*\n", re.DOTALL)

_PROMPT = """\
You are a skill-selection oracle. Given the skill below and a user query,
answer whether the skill should be invoked for that query.

Skill name: {name}
Skill description: {description}

User query: {query}

Reply with exactly one word: YES or NO.
"""


def _load_frontmatter(skill_dir: Path) -> dict:
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.is_file():
        raise SystemExit(f"error: {skill_md} not found")
    content = skill_md.read_text(encoding="utf-8")
    match = _FRONTMATTER_RE.match(content)
    if match is None:
        raise SystemExit("error: SKILL.md has no leading YAML frontmatter")
    data = yaml.safe_load(match.group("body"))
    if not isinstance(data, dict):
        raise SystemExit("error: frontmatter is not a mapping")
    if "name" not in data or "description" not in data:
        raise SystemExit("error: frontmatter missing name or description")
    return data


async def _judge(llm, name: str, description: str, query: str) -> bool:
    from DefenseAgent.llm.types import Message

    prompt = _PROMPT.format(name=name, description=description, query=query)
    response = await llm.chat(
        [Message(role="user", content=prompt)],
        temperature=0.0,
        max_tokens=8,
    )
    text = (response.content or "").strip().upper()
    return text.startswith("YES")


async def _run(
    skill_dir: Path,
    positive: list[str],
    negative: list[str],
    dotenv_path: str | None,
) -> int:
    from DefenseAgent.llm.llm import LLM

    data = _load_frontmatter(skill_dir)
    name = str(data["name"])
    description = str(data["description"])

    llm = LLM.from_env(dotenv_path=dotenv_path)

    results: list[tuple[str, bool, bool]] = []  # (query, expected, judged)
    for query in positive:
        judged = await _judge(llm, name, description, query)
        results.append((query, True, judged))
    for query in negative:
        judged = await _judge(llm, name, description, query)
        results.append((query, False, judged))

    correct = 0
    for query, expected, judged in results:
        ok = expected == judged
        correct += int(ok)
        marker = "OK" if ok else "MISS"
        verdict = "YES" if judged else "NO"
        want = "YES" if expected else "NO"
        print(f"[{marker}] expected={want} judged={verdict}  {query}")

    total = len(results) or 1
    rate = correct / total
    print(f"hit-rate: {correct}/{total} = {rate:.0%}")
    return 0 if rate >= 0.8 else 1


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="eval_description.py",
        description="Trigger-accuracy eval for a skill's description.",
    )
    parser.add_argument("path", help="Path to the skill directory.")
    parser.add_argument(
        "--queries",
        nargs="+",
        default=[],
        help="Positive queries (should trigger).",
    )
    parser.add_argument(
        "--negative",
        nargs="*",
        default=[],
        help="Negative queries (should NOT trigger).",
    )
    parser.add_argument(
        "--dotenv",
        default=None,
        help="Path to .env (defaults to ambient discovery).",
    )
    args = parser.parse_args(argv)

    skill_dir = Path(args.path).expanduser().resolve()
    if not skill_dir.is_dir():
        print(f"error: {skill_dir} is not a directory", file=sys.stderr)
        return 1
    if not args.queries:
        print(
            "error: provide at least one --queries entry",
            file=sys.stderr,
        )
        return 2

    return asyncio.run(
        _run(skill_dir, args.queries, args.negative, args.dotenv)
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
