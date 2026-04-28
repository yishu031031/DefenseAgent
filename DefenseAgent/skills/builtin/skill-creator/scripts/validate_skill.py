#!/usr/bin/env python3
"""Validate a DefenseAgent skill directory against the loader's requirements.

Usage:
    validate_skill.py <skill-path>

Checks performed:
  * `SKILL.md` exists at the skill path.
  * Leading YAML frontmatter parses cleanly.
  * Required fields (`name`, `description`) are present and non-empty.
  * `name` matches the directory basename.
  * Description starts with `Use when` (the trigger-clarity convention from
    `writing-skills`).
  * `ms_agent.skill.SkillSchemaParser` accepts the directory — same parser
    the runtime loader uses, so a green run here means the skill will load.

Exit code 0 on clean, 1 on any reported issue. Output is human-readable
findings on stdout.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml


_FRONTMATTER_RE = re.compile(r"^---\s*\n(?P<body>.*?)\n---\s*\n", re.DOTALL)


def _check_frontmatter(content: str, findings: list[str]) -> dict | None:
    match = _FRONTMATTER_RE.match(content)
    if match is None:
        findings.append("missing leading YAML frontmatter (--- ... --- block)")
        return None
    try:
        data = yaml.safe_load(match.group("body"))
    except yaml.YAMLError as e:
        findings.append(f"frontmatter YAML parse error: {e}")
        return None
    if not isinstance(data, dict):
        findings.append(
            f"frontmatter must be a mapping, got {type(data).__name__}"
        )
        return None
    return data


def _check_required_fields(data: dict, findings: list[str]) -> None:
    for field in ("name", "description"):
        value = data.get(field)
        if not isinstance(value, str) or not value.strip():
            findings.append(f"missing or empty required field: {field!r}")


def _check_name_match(
    data: dict, skill_dir: Path, findings: list[str],
) -> None:
    name = data.get("name")
    if isinstance(name, str) and name != skill_dir.name:
        findings.append(
            f"frontmatter name={name!r} does not match directory "
            f"basename {skill_dir.name!r}"
        )


def _check_description_shape(data: dict, findings: list[str]) -> None:
    desc = data.get("description")
    if isinstance(desc, str) and not desc.strip().lower().startswith("use when"):
        findings.append(
            "description should start with 'Use when ...' for triggering "
            "clarity (see writing-skills)"
        )


def _try_schema_parser(skill_dir: Path, findings: list[str]) -> None:
    """Round-trip the skill through `ms_agent.skill.SkillLoader.load_skills` — exactly what `DefenseAgent.skills.SkillLoader` invokes at runtime — so a clean validate here means the loader will also accept it. When ms_agent is not importable (test environment without the framework's full dep tree) we print a note to stderr and skip; the package is a hard dep at runtime so this only fires in CI fixtures."""
    try:
        from ms_agent.skill.loader import SkillLoader
    except ImportError as e:
        print(
            f"note: skipping ms_agent SkillLoader round-trip — ms_agent "
            f"not importable here ({e})",
            file=sys.stderr,
        )
        return
    try:
        loaded = SkillLoader().load_skills(str(skill_dir))
    except Exception as e:  # noqa: BLE001 — surface anything the loader raises
        findings.append(f"ms_agent SkillLoader rejected the skill: {e}")
        return
    if not loaded:
        findings.append(
            "ms_agent SkillLoader returned no schemas (directory may be "
            "missing SKILL.md or all candidates failed silently)"
        )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="validate_skill.py",
        description="Validate a DefenseAgent skill directory.",
    )
    parser.add_argument("path", help="Path to the skill directory.")
    args = parser.parse_args(argv)

    skill_dir = Path(args.path).expanduser().resolve()
    findings: list[str] = []

    if not skill_dir.is_dir():
        print(f"error: {skill_dir} is not a directory", file=sys.stderr)
        return 1

    skill_md = skill_dir / "SKILL.md"
    if not skill_md.is_file():
        print(f"error: {skill_md} not found", file=sys.stderr)
        return 1

    content = skill_md.read_text(encoding="utf-8")
    data = _check_frontmatter(content, findings)
    if data is not None:
        _check_required_fields(data, findings)
        _check_name_match(data, skill_dir, findings)
        _check_description_shape(data, findings)

    _try_schema_parser(skill_dir, findings)

    if findings:
        print(f"FAIL: {skill_dir}")
        for line in findings:
            print(f"  - {line}")
        return 1

    print(f"OK: {skill_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
