"""Tests for DefenseAgent.skills — the SkillLoader subclass and the SkillSchema → Tool conversion.

We inherit ms-agent's `SkillLoader` for the parsing/discovery surface, so its existing tests cover frontmatter handling, registry semantics, etc. These tests focus on what DefenseAgent adds:
  • The ms-agent loader is reachable through our subclass (load + scan trees + registry).
  • `to_tools()` converts every loaded `SkillSchema` into a Tool with the right metadata.
  • The runtime tool handler returns the body when called with no `file`, looks up files by basename, by relative path, and rejects path-escape / non-string args.
"""
import asyncio
from pathlib import Path

import pytest

from DefenseAgent.skills import SkillLoader, SkillLoadError, SkillSchema


def _write_skill(
    directory: Path,
    *,
    name: str = "tabular-report",
    description: str = "Produces a markdown table from a row list.",
    body: str = "# Tabular Report\n\nSee scripts/generate.py for the full implementation.\n",
    extras: dict[str, str] | None = None,
) -> Path:
    """Write a minimal SKILL.md skill (plus optional extra files) and return its directory."""
    directory.mkdir(parents=True, exist_ok=True)
    frontmatter = f"---\nname: {name}\ndescription: {description}\n---\n\n"
    (directory / "SKILL.md").write_text(frontmatter + body, encoding="utf-8")
    if extras is not None:
        for rel_path, content in extras.items():
            target = directory / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
    return directory


# ---------- inherited loader surface ----------


def test_loader_loads_single_skill_directory(tmp_path: Path) -> None:
    skill_dir = _write_skill(tmp_path / "tabular")
    loader = SkillLoader()
    loaded = loader.load_skills(str(skill_dir))
    assert len(loaded) == 1
    schema = next(iter(loaded.values()))
    assert isinstance(schema, SkillSchema)
    assert schema.name == "tabular-report"
    assert schema.description.startswith("Produces a markdown table")
    # SkillLoader's internal registry is keyed by `{skill_id}@{version}`.
    assert any(key.endswith("@latest") for key in loader.list_skills())


def test_loader_walks_immediate_subdirs_when_root_has_no_skill_md(tmp_path: Path) -> None:
    """Mirrors ms-agent's `SkillLoader._scan_and_load_skills` — if root has no SKILL.md, walk children."""
    root = tmp_path / "skills"
    _write_skill(root / "alpha", name="alpha")
    _write_skill(root / "beta", name="beta")
    (root / "not_a_skill").mkdir()

    loader = SkillLoader()
    loaded = loader.load_skills(str(root))
    names = sorted(s.name for s in loaded.values())
    assert names == ["alpha", "beta"]


def test_loader_skips_invalid_skills_silently(tmp_path: Path) -> None:
    """ms-agent's parser returns None on bad SKILL.md; the wrapper just skips. Unlike our previous strict mode, no SkillLoadError is raised here."""
    root = tmp_path / "skills"
    _write_skill(root / "good", name="good")
    bad = root / "broken"
    bad.mkdir(parents=True)
    (bad / "SKILL.md").write_text("not yaml at all", encoding="utf-8")

    loader = SkillLoader()
    loaded = loader.load_skills(str(root))
    names = sorted(s.name for s in loaded.values())
    assert names == ["good"]


# ---------- SkillSchema → Tool conversion ----------


def test_to_tools_returns_one_tool_per_loaded_skill(tmp_path: Path) -> None:
    skill_dir = _write_skill(
        tmp_path / "tabular",
        extras={"scripts/generate.py": "print('hi')\n"},
    )
    loader = SkillLoader()
    loader.load_skills(str(skill_dir))

    tools = loader.to_tools()
    assert len(tools) == 1
    tool = tools[0]
    assert tool.name == "tabular-report"
    assert tool.description.startswith("Produces a markdown table")
    assert tool.source == "skill"
    assert tool.metadata["skill_id"]
    assert tool.metadata["version"] == "latest"
    assert tool.metadata["skill_path"] == str(skill_dir.resolve())
    assert "file" in tool.input_schema["properties"]


# ---------- runtime handler ----------


def test_handler_returns_skill_body_when_no_file_arg(tmp_path: Path) -> None:
    skill_dir = _write_skill(
        tmp_path / "tabular",
        body="# How to write a table\n\nUse generate.py.\n",
    )
    loader = SkillLoader()
    loader.load_skills(str(skill_dir))
    tool = loader.to_tools()[0]

    body = asyncio.run(tool.handler({}))
    assert "How to write a table" in body
    # The frontmatter must be stripped.
    assert "---" not in body.splitlines()[0]
    assert "name: tabular-report" not in body


def test_handler_returns_skill_body_when_file_is_skill_md(tmp_path: Path) -> None:
    skill_dir = _write_skill(tmp_path / "s", body="BODY-ONE\n")
    loader = SkillLoader()
    loader.load_skills(str(skill_dir))
    tool = loader.to_tools()[0]

    out = asyncio.run(tool.handler({"file": "SKILL.md"}))
    assert "BODY-ONE" in out


def test_handler_resolves_file_by_basename(tmp_path: Path) -> None:
    skill_dir = _write_skill(
        tmp_path / "s",
        extras={"scripts/generate.py": "print('go')\n"},
    )
    loader = SkillLoader()
    loader.load_skills(str(skill_dir))
    tool = loader.to_tools()[0]

    contents = asyncio.run(tool.handler({"file": "generate.py"}))
    assert contents == "print('go')\n"


def test_handler_resolves_file_by_relative_path(tmp_path: Path) -> None:
    skill_dir = _write_skill(
        tmp_path / "s",
        extras={"templates/header.md": "# Header\n"},
    )
    loader = SkillLoader()
    loader.load_skills(str(skill_dir))
    tool = loader.to_tools()[0]

    contents = asyncio.run(tool.handler({"file": "templates/header.md"}))
    assert contents == "# Header\n"


def test_handler_raises_on_absolute_path(tmp_path: Path) -> None:
    skill_dir = _write_skill(tmp_path / "s")
    loader = SkillLoader()
    loader.load_skills(str(skill_dir))
    tool = loader.to_tools()[0]

    with pytest.raises(SkillLoadError):
        asyncio.run(tool.handler({"file": "/etc/passwd"}))


def test_handler_raises_on_path_escape_attempt(tmp_path: Path) -> None:
    skill_dir = _write_skill(tmp_path / "s")
    loader = SkillLoader()
    loader.load_skills(str(skill_dir))
    tool = loader.to_tools()[0]

    with pytest.raises(SkillLoadError):
        asyncio.run(tool.handler({"file": "../../etc/passwd"}))


def test_handler_raises_on_non_string_file_arg(tmp_path: Path) -> None:
    skill_dir = _write_skill(tmp_path / "s")
    loader = SkillLoader()
    loader.load_skills(str(skill_dir))
    tool = loader.to_tools()[0]

    with pytest.raises(SkillLoadError):
        asyncio.run(tool.handler({"file": 42}))


def test_handler_raises_when_file_not_in_skill(tmp_path: Path) -> None:
    skill_dir = _write_skill(tmp_path / "s")
    loader = SkillLoader()
    loader.load_skills(str(skill_dir))
    tool = loader.to_tools()[0]

    with pytest.raises(SkillLoadError):
        asyncio.run(tool.handler({"file": "missing.md"}))
