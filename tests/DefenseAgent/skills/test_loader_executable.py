"""Tests for SkillLoader.to_tools(container=...) — every script bundled inside a skill becomes an executable Tool.

Same offline footprint as test_container.py: real subprocesses, no Docker, no network.
"""
import asyncio
from pathlib import Path

import pytest

from DefenseAgent.skills import SkillContainer, SkillLoader
from DefenseAgent.tools import ToolRegistry
from DefenseAgent.tools.types import SkillLoadError


def _write_skill_with_script(
    root: Path,
    *,
    name: str,
    script_name: str,
    script_body: str,
    skill_body: str = "Use scripts/X for the work.\n",
) -> Path:
    """Write a SKILL.md plus one script under `scripts/`; return the skill directory path."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: a {name} skill\n---\n\n{skill_body}",
        encoding="utf-8",
    )
    scripts_dir = root / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    (scripts_dir / script_name).write_text(script_body, encoding="utf-8")
    return root


# ---------- to_tools(container=...) yields read + execute Tools ----------


def test_to_tools_without_container_omits_executable_tools(tmp_path: Path) -> None:
    skill_dir = _write_skill_with_script(
        tmp_path / "calc",
        name="calc",
        script_name="add.py",
        script_body="print('go')\n",
    )
    loader = SkillLoader()
    loader.load_skills(str(skill_dir))

    tools = loader.to_tools()
    assert [t.name for t in tools] == ["calc"]


def test_to_tools_with_container_adds_one_tool_per_script(tmp_path: Path) -> None:
    skill_dir = _write_skill_with_script(
        tmp_path / "calc",
        name="calc",
        script_name="add.py",
        script_body="print('hi')\n",
    )
    container = SkillContainer(workspace_dir=tmp_path / "ws")
    loader = SkillLoader()
    loader.load_skills(str(skill_dir))

    tools = loader.to_tools(container=container)
    names = sorted(t.name for t in tools)
    assert names == ["calc", "calc__add"]


def test_executable_tool_handler_runs_the_script(tmp_path: Path) -> None:
    skill_dir = _write_skill_with_script(
        tmp_path / "calc",
        name="calc",
        script_name="add.py",
        script_body=(
            "import sys\n"
            "a, b = int(sys.argv[1]), int(sys.argv[2])\n"
            "print(a + b)\n"
        ),
    )
    container = SkillContainer(workspace_dir=tmp_path / "ws")
    loader = SkillLoader()
    loader.load_skills(str(skill_dir))
    tools = loader.to_tools(container=container)

    add_tool = next(t for t in tools if t.name == "calc__add")
    out = asyncio.run(add_tool.handler({"args": ["3", "4"]}))
    assert "7" in out
    assert "exit_code=0" in out


def test_executable_tool_renders_stderr_and_nonzero_exit(tmp_path: Path) -> None:
    skill_dir = _write_skill_with_script(
        tmp_path / "calc",
        name="calc",
        script_name="boom.py",
        script_body=(
            "import sys\n"
            "print('hello')\n"
            "print('whoops', file=sys.stderr)\n"
            "sys.exit(2)\n"
        ),
    )
    container = SkillContainer(workspace_dir=tmp_path / "ws")
    loader = SkillLoader()
    loader.load_skills(str(skill_dir))
    add_tool = next(t for t in loader.to_tools(container=container) if "__" in t.name)

    out = asyncio.run(add_tool.handler({}))
    assert "hello" in out
    assert "whoops" in out
    assert "exit_code=2" in out


def test_executable_tool_rejects_non_string_args(tmp_path: Path) -> None:
    skill_dir = _write_skill_with_script(
        tmp_path / "calc",
        name="calc",
        script_name="echo.py",
        script_body="print('hi')\n",
    )
    container = SkillContainer(workspace_dir=tmp_path / "ws")
    loader = SkillLoader()
    loader.load_skills(str(skill_dir))
    tool = next(t for t in loader.to_tools(container=container) if "__" in t.name)

    with pytest.raises(SkillLoadError):
        asyncio.run(tool.handler({"args": [1, 2, 3]}))


def test_executable_tool_blocks_dangerous_script(tmp_path: Path) -> None:
    skill_dir = _write_skill_with_script(
        tmp_path / "calc",
        name="calc",
        script_name="evil.py",
        script_body="import os\nos.system('echo nope')\n",
    )
    container = SkillContainer(workspace_dir=tmp_path / "ws")
    loader = SkillLoader()
    loader.load_skills(str(skill_dir))
    tool = next(t for t in loader.to_tools(container=container) if "__" in t.name)

    out = asyncio.run(tool.handler({}))
    assert "Security check failed" in out
    assert "exit_code=-1" in out


# ---------- ToolRegistry plumbing ----------


def test_registry_add_skill_with_container_registers_executable_tools(tmp_path: Path) -> None:
    skill_dir = _write_skill_with_script(
        tmp_path / "calc",
        name="calc",
        script_name="add.py",
        script_body="print(2 + 2)\n",
    )
    container = SkillContainer(workspace_dir=tmp_path / "ws")
    registry = ToolRegistry()
    registry.add_skill(skill_dir, container=container)
    assert sorted(registry.names()) == ["calc", "calc__add"]


def test_registry_add_skill_without_container_only_registers_read_only_tool(tmp_path: Path) -> None:
    skill_dir = _write_skill_with_script(
        tmp_path / "calc",
        name="calc",
        script_name="add.py",
        script_body="print('hi')\n",
    )
    registry = ToolRegistry()
    registry.add_skill(skill_dir)
    assert registry.names() == ["calc"]
