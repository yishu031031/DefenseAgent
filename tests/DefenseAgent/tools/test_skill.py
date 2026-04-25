"""Tests for DefenseAgent.tools.skill.

Covers the three-layer progressive disclosure of an Anthropic-style Agent Skill:
  Layer 1 — frontmatter (name + description) is eagerly parsed.
  Layer 2 — SKILL.md body is returned when the handler is called with no `file`.
  Layer 3 — any file inside the skill root is returned when called with `{"file": ...}`.
Also covers error cases: missing SKILL.md, bad frontmatter, path escapes.
"""
import asyncio
from pathlib import Path

import pytest

from DefenseAgent.tools import Skill, SkillLoadError


def _write_skill(
    root: Path,
    *,
    name: str = "tabular-report",
    description: str = "Produces a markdown table from a row list.",
    body: str = "# Tabular Report\n\nSee scripts/generate.py for the full implementation.\n",
    extras: dict[str, str] | None = None,
) -> Path:
    """Create a minimal Skill dir with SKILL.md + optional extra files; returns the dir path."""
    root.mkdir(parents=True, exist_ok=True)
    frontmatter = f"---\nname: {name}\ndescription: {description}\n---\n\n"
    (root / "SKILL.md").write_text(frontmatter + body, encoding="utf-8")
    if extras is not None:
        for rel_path, content in extras.items():
            target = root / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
    return root


# ---------- Layer 1: frontmatter ----------


def test_skill_reads_name_and_description_from_frontmatter(tmp_path: Path) -> None:
    skill_dir = _write_skill(tmp_path / "s")
    skill = Skill(skill_dir)
    assert skill.name == "tabular-report"
    assert skill.description == "Produces a markdown table from a row list."


def test_skill_to_tool_exposes_only_frontmatter(tmp_path: Path) -> None:
    skill_dir = _write_skill(tmp_path / "s", body="SECRET LAYER 2 BODY\n")
    tool = Skill(skill_dir).to_tool()
    assert tool.name == "tabular-report"
    assert tool.description == "Produces a markdown table from a row list."
    assert tool.source == "skill"
    assert "SECRET" not in tool.description
    assert tool.input_schema["type"] == "object"
    assert "file" in tool.input_schema["properties"]


def test_skill_description_is_trimmed(tmp_path: Path) -> None:
    skill_dir = _write_skill(tmp_path / "s", name="  padded  ", description="  hello  ")
    skill = Skill(skill_dir)
    assert skill.name == "padded"
    assert skill.description == "hello"


# ---------- Layer 2: body ----------


def test_handler_returns_body_when_file_arg_is_absent(tmp_path: Path) -> None:
    body = "# Instructions\n\nStep 1.\nStep 2.\n"
    skill_dir = _write_skill(tmp_path / "s", body=body)
    skill = Skill(skill_dir)
    result = asyncio.run(skill._handle({}))
    assert result == body


def test_handler_returns_body_when_file_arg_is_empty_string(tmp_path: Path) -> None:
    body = "# Instructions\n"
    skill_dir = _write_skill(tmp_path / "s", body=body)
    skill = Skill(skill_dir)
    result = asyncio.run(skill._handle({"file": ""}))
    assert result == body


def test_body_property_matches_handler_body(tmp_path: Path) -> None:
    body = "# Title\n\nBody line.\n"
    skill_dir = _write_skill(tmp_path / "s", body=body)
    skill = Skill(skill_dir)
    assert skill.body == body


# ---------- Layer 3: on-demand file reads ----------


def test_handler_reads_subfile_inside_skill(tmp_path: Path) -> None:
    skill_dir = _write_skill(
        tmp_path / "s",
        extras={"scripts/generate.py": "print('hello')\n"},
    )
    skill = Skill(skill_dir)
    content = asyncio.run(skill._handle({"file": "scripts/generate.py"}))
    assert content == "print('hello')\n"


def test_read_file_direct(tmp_path: Path) -> None:
    skill_dir = _write_skill(
        tmp_path / "s", extras={"templates/row.md": "| {name} |\n"}
    )
    skill = Skill(skill_dir)
    assert skill.read_file("templates/row.md") == "| {name} |\n"


def test_read_file_requesting_skill_md_returns_body(tmp_path: Path) -> None:
    body = "# Body\n"
    skill_dir = _write_skill(tmp_path / "s", body=body)
    skill = Skill(skill_dir)
    content = asyncio.run(skill._handle({"file": "SKILL.md"}))
    assert content == body


# ---------- security + errors ----------


def test_absolute_path_rejected(tmp_path: Path) -> None:
    skill_dir = _write_skill(tmp_path / "s")
    skill = Skill(skill_dir)
    with pytest.raises(SkillLoadError):
        skill.read_file("/etc/passwd")


def test_parent_escape_rejected(tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    skill_dir = _write_skill(tmp_path / "s")
    skill = Skill(skill_dir)
    with pytest.raises(SkillLoadError):
        skill.read_file("../outside.txt")


def test_missing_file_raises(tmp_path: Path) -> None:
    skill_dir = _write_skill(tmp_path / "s")
    skill = Skill(skill_dir)
    with pytest.raises(SkillLoadError):
        skill.read_file("nope.py")


def test_missing_skill_md_raises(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SkillLoadError):
        Skill(empty)


def test_path_is_not_a_directory_raises(tmp_path: Path) -> None:
    not_a_dir = tmp_path / "file.txt"
    not_a_dir.write_text("hi", encoding="utf-8")
    with pytest.raises(SkillLoadError):
        Skill(not_a_dir)


def test_frontmatter_without_opening_delim_raises(tmp_path: Path) -> None:
    d = tmp_path / "s"
    d.mkdir()
    (d / "SKILL.md").write_text("no frontmatter here\n", encoding="utf-8")
    with pytest.raises(SkillLoadError):
        Skill(d)


def test_frontmatter_without_closing_delim_raises(tmp_path: Path) -> None:
    d = tmp_path / "s"
    d.mkdir()
    (d / "SKILL.md").write_text("---\nname: x\ndescription: y\n", encoding="utf-8")
    with pytest.raises(SkillLoadError):
        Skill(d)


def test_frontmatter_missing_name_raises(tmp_path: Path) -> None:
    d = tmp_path / "s"
    d.mkdir()
    (d / "SKILL.md").write_text(
        "---\ndescription: x\n---\nbody\n", encoding="utf-8"
    )
    with pytest.raises(SkillLoadError):
        Skill(d)


def test_frontmatter_missing_description_raises(tmp_path: Path) -> None:
    d = tmp_path / "s"
    d.mkdir()
    (d / "SKILL.md").write_text(
        "---\nname: x\n---\nbody\n", encoding="utf-8"
    )
    with pytest.raises(SkillLoadError):
        Skill(d)


def test_bad_yaml_raises_skill_load_error(tmp_path: Path) -> None:
    d = tmp_path / "s"
    d.mkdir()
    (d / "SKILL.md").write_text(
        "---\nname: [invalid\n---\nbody\n", encoding="utf-8"
    )
    with pytest.raises(SkillLoadError):
        Skill(d)


def test_non_string_file_arg_raises(tmp_path: Path) -> None:
    skill_dir = _write_skill(tmp_path / "s")
    skill = Skill(skill_dir)
    with pytest.raises(SkillLoadError):
        asyncio.run(skill._handle({"file": 42}))
