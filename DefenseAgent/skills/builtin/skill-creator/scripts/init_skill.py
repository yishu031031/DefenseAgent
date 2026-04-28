#!/usr/bin/env python3
"""Scaffold a new DefenseAgent skill directory under the chosen scope root.

Usage:
    init_skill.py <name> [--scope project|user] [--with-scripts] [--root PATH]

`<name>` must be kebab-case (matches the directory name and the SKILL.md
`name` field). `--scope` picks the default root: `project` → `./skills/`,
`user` → `~/.defense-agent/skills/`. `--root` overrides both. The created
SKILL.md ships placeholder frontmatter and a body skeleton aligned with
`writing-skills`. Re-running on an existing skill name is refused so we never
clobber user-authored content.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

_SKILL_TEMPLATE = """---
name: {name}
description: "Use when <trigger conditions with concrete keywords>. <One sentence summary of the behaviour>. <Optionally, what NOT to use this for>."
---

# {title}

<One-sentence purpose. Replace this paragraph.>

## Step 1 — <first action>

<What to do, in imperative voice.>

## Step 2 — <next action>

<...>

## Anti-patterns

- <Known way to misapply this skill.>
"""

_SCRIPT_PLACEHOLDER = """#!/usr/bin/env python3
\"\"\"<One-line purpose. Replace.>\"\"\"
from __future__ import annotations

import sys


def main(argv: list[str]) -> int:
    print("hello from {name}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
"""


def _resolve_root(scope: str, root_override: str | None) -> Path:
    if root_override is not None:
        return Path(root_override).expanduser().resolve()
    if scope == "project":
        return (Path.cwd() / "skills").resolve()
    return (Path.home() / ".defense-agent" / "skills").resolve()


def _kebab_to_title(name: str) -> str:
    return " ".join(part.capitalize() for part in name.split("-"))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="init_skill.py",
        description="Scaffold a new DefenseAgent skill directory.",
    )
    parser.add_argument("name", help="Skill name (kebab-case).")
    parser.add_argument(
        "--scope",
        choices=("project", "user"),
        default="project",
        help="Default root: project=./skills/, user=~/.defense-agent/skills/.",
    )
    parser.add_argument(
        "--with-scripts",
        action="store_true",
        help="Also create scripts/ with a placeholder Python helper.",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Explicit root directory; overrides --scope.",
    )
    args = parser.parse_args(argv)

    name: str = args.name
    if not _NAME_RE.fullmatch(name):
        print(
            f"error: name {name!r} is not kebab-case "
            "(lowercase letters, digits, single hyphens between segments)",
            file=sys.stderr,
        )
        return 2

    root = _resolve_root(args.scope, args.root)
    root.mkdir(parents=True, exist_ok=True)

    skill_dir = root / name
    if skill_dir.exists():
        print(
            f"error: {skill_dir} already exists; refusing to overwrite",
            file=sys.stderr,
        )
        return 1

    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        _SKILL_TEMPLATE.format(name=name, title=_kebab_to_title(name)),
        encoding="utf-8",
    )
    if args.with_scripts:
        scripts_dir = skill_dir / "scripts"
        scripts_dir.mkdir()
        (scripts_dir / "main.py").write_text(
            _SCRIPT_PLACEHOLDER.format(name=name),
            encoding="utf-8",
        )

    print(str(skill_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
