"""Tests for the self-evolution skill auto-discovery added in 0.1.5.

Covers:
  * `EvolutionConfig` schema (defaults, overrides, validation).
  * `discover_skill_dirs()` ordering and override semantics.
  * `SkillLoader.load_dirs_tolerant()` — loads existing layers, silently skips
    missing layers, never raises on bad input.
  * The five bundled methodology skills are loadable end to end and their
    `description` fields follow the "Use when ..." convention enforced by
    `writing-skills`.
  * `_autoload_evolution_skills` populates a `ToolRegistry` with the five
    builtins when no profile-driven overrides are set.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from DefenseAgent.config.profile import AgentProfile, EvolutionConfig
from DefenseAgent.skills import (
    SkillLoader,
    builtin_skills_path,
    default_project_skills_path,
    default_user_skills_path,
    discover_skill_dirs,
)


_BUILTIN_NAMES = {
    "skill-creator",
    "using-skills",
    "writing-skills",
    "reflect-and-distill",
    "promote-memory-to-skill",
}


# ---------- EvolutionConfig schema ----------


def test_evolution_config_defaults() -> None:
    ec = EvolutionConfig()
    assert ec.use_builtin is True
    assert ec.user_skills_dir is None
    assert ec.project_skills_dir is None
    assert ec.default_scope == "project"


def test_evolution_config_attaches_to_profile() -> None:
    p = AgentProfile(id="x", name="x")
    assert isinstance(p.evolution, EvolutionConfig)
    assert p.evolution.use_builtin is True


def test_evolution_config_rejects_bad_scope() -> None:
    with pytest.raises(ValidationError):
        EvolutionConfig(default_scope="bogus")


def test_evolution_config_accepts_overrides() -> None:
    ec = EvolutionConfig(
        use_builtin=False,
        user_skills_dir="",
        project_skills_dir="/tmp/x",
        default_scope="user",
    )
    assert ec.use_builtin is False
    assert ec.user_skills_dir == ""
    assert ec.project_skills_dir == "/tmp/x"
    assert ec.default_scope == "user"


# ---------- path helpers + discover_skill_dirs ----------


def test_builtin_skills_path_exists_in_checkout() -> None:
    p = builtin_skills_path()
    assert p.is_dir()
    assert (p / "skill-creator" / "SKILL.md").is_file()


def test_default_user_skills_path_under_home() -> None:
    p = default_user_skills_path()
    assert p == Path.home() / ".defense-agent" / "skills"


def test_default_project_skills_path_under_cwd() -> None:
    p = default_project_skills_path()
    assert p == Path.cwd() / "skills"


def test_discover_skill_dirs_default_order() -> None:
    dirs = discover_skill_dirs(EvolutionConfig())
    assert dirs[0] == builtin_skills_path()
    assert dirs[1] == default_user_skills_path()
    assert dirs[2] == default_project_skills_path()


def test_discover_skill_dirs_drops_builtin_when_off() -> None:
    dirs = discover_skill_dirs(EvolutionConfig(use_builtin=False))
    assert builtin_skills_path() not in dirs


def test_discover_skill_dirs_drops_layer_on_empty_string() -> None:
    dirs = discover_skill_dirs(
        EvolutionConfig(user_skills_dir="", project_skills_dir="")
    )
    assert all(p == builtin_skills_path() for p in dirs)


def test_discover_skill_dirs_honours_path_override(tmp_path: Path) -> None:
    custom_user = tmp_path / "user-skills"
    custom_project = tmp_path / "proj-skills"
    dirs = discover_skill_dirs(
        EvolutionConfig(
            user_skills_dir=str(custom_user),
            project_skills_dir=str(custom_project),
        )
    )
    assert custom_user in dirs
    assert custom_project in dirs


def test_discover_skill_dirs_handles_none_evolution() -> None:
    dirs = discover_skill_dirs(None)
    assert dirs[0] == builtin_skills_path()


# ---------- SkillLoader.load_dirs_tolerant ----------


def test_load_dirs_tolerant_loads_all_builtins() -> None:
    loader = SkillLoader()
    added = loader.load_dirs_tolerant([builtin_skills_path()])
    names = {s.name for s in added}
    assert names == _BUILTIN_NAMES


def test_load_dirs_tolerant_silent_on_missing_dir(tmp_path: Path) -> None:
    nonexistent = tmp_path / "never_exists"
    loader = SkillLoader()
    added = loader.load_dirs_tolerant([nonexistent])
    assert added == []  # no exception, no skills loaded


def test_load_dirs_tolerant_silent_on_empty_dir(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    loader = SkillLoader()
    # An existing directory with no SKILL.md inside — ms-agent's loader treats
    # this as a hard failure; tolerant wrapper must downgrade to a warning.
    added = loader.load_dirs_tolerant([empty])
    assert added == []


def test_load_dirs_tolerant_mixed_layers(tmp_path: Path) -> None:
    """Real builtin + nonexistent + empty — the existing layer still loads."""
    nonexistent = tmp_path / "nope"
    empty = tmp_path / "empty"
    empty.mkdir()
    loader = SkillLoader()
    added = loader.load_dirs_tolerant(
        [builtin_skills_path(), nonexistent, empty]
    )
    assert {s.name for s in added} == _BUILTIN_NAMES


# ---------- builtin SKILL.md health ----------


def test_every_builtin_description_starts_with_use_when() -> None:
    """`writing-skills` mandates this convention; enforce it on the bundled set."""
    loader = SkillLoader()
    loader.load_dirs_tolerant([builtin_skills_path()])
    for schema in loader.all_skills().values():
        assert schema.description.lower().startswith("use when"), (
            f"{schema.name}: description does not start with 'Use when ...': "
            f"{schema.description[:80]!r}"
        )


def test_skill_creator_bundles_three_helper_scripts() -> None:
    loader = SkillLoader()
    loader.load_dirs_tolerant([builtin_skills_path()])
    skill_creator = next(
        s for s in loader.all_skills().values() if s.name == "skill-creator"
    )
    script_names = {f.name for f in skill_creator.scripts}
    assert script_names == {
        "init_skill.py",
        "validate_skill.py",
        "eval_description.py",
    }


def test_to_tools_exposes_each_builtin_as_one_tool() -> None:
    loader = SkillLoader()
    loader.load_dirs_tolerant([builtin_skills_path()])
    tools = loader.to_tools()
    tool_names = {t.name for t in tools}
    assert tool_names == _BUILTIN_NAMES


# ---------- _autoload_evolution_skills wiring ----------


def test_autoload_evolution_skills_populates_registry() -> None:
    """Manually invoke the builder helper and confirm the five builtins land in a fresh ToolRegistry. We disable the user/project layers (empty string) so the test does not depend on the developer's cwd or home."""
    from DefenseAgent.agent._builder import _autoload_evolution_skills
    from DefenseAgent.tools import ToolRegistry

    profile = AgentProfile(id="x", name="x")
    profile.evolution = EvolutionConfig(
        user_skills_dir="",
        project_skills_dir="",
    )
    registry = ToolRegistry()
    _autoload_evolution_skills(registry, profile)

    tool_names = {spec["name"] for spec in registry.specs()}
    assert _BUILTIN_NAMES.issubset(tool_names)


def test_autoload_evolution_skills_no_op_when_all_layers_off(
    tmp_path: Path,
) -> None:
    """`use_builtin=False` plus suppressed user/project layers should leave the registry empty (the autoload helper must be safe to call regardless)."""
    from DefenseAgent.agent._builder import _autoload_evolution_skills
    from DefenseAgent.tools import ToolRegistry

    profile = AgentProfile(id="x", name="x")
    profile.evolution = EvolutionConfig(
        use_builtin=False,
        user_skills_dir="",
        project_skills_dir="",
    )
    registry = ToolRegistry()
    _autoload_evolution_skills(registry, profile)
    assert registry.specs() == []
