"""Tests for the four Tier-1 parity gaps closed against ms-agent's skill module:

  1. `SkillContext` is re-exported via `DefenseAgent.skills` (was only reachable
     via the upstream `ms_agent.skill.schema` import path).
  2. Module-level `load_skills()` convenience function mirrors ms-agent's
     `loader.py:230` one-line wrapper.
  3. `author` and `tags` from the SKILL.md frontmatter are carried into the
     resulting Tool's `metadata` dict.
  4. The Tool's description is augmented with a `Bundled files — scripts: ...;
     references: ...; resources: ...` line whenever the skill has any non-empty
     bucket (mirrors ms-agent's `', '.join(context.get_*_list()) or 'None'`
     formatting in PROMPT_SKILL_ANALYSIS_PLAN).
"""
from pathlib import Path

from DefenseAgent.skills import SkillContext, SkillLoader, load_skills


def _write_skill(
    root: Path,
    *,
    name: str = "tabular-report",
    description: str = "Produces a markdown table from a row list.",
    body: str = "Use scripts/generate.py.\n",
    author: str | None = None,
    tags: list[str] | None = None,
    extras: dict[str, str] | None = None,
) -> Path:
    """Write a SKILL.md plus optional extra files; supports `author` and `tags` frontmatter so we can assert on schema/tool metadata."""
    root.mkdir(parents=True, exist_ok=True)
    frontmatter = ["---", f"name: {name}", f"description: {description}"]
    if author is not None:
        frontmatter.append(f"author: {author}")
    if tags is not None:
        frontmatter.append("tags:")
        for tag in tags:
            frontmatter.append(f"  - {tag}")
    frontmatter.extend(["---", "", body])
    (root / "SKILL.md").write_text("\n".join(frontmatter), encoding="utf-8")
    if extras is not None:
        for rel_path, content in extras.items():
            target = root / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
    return root


# ---------- (1) SkillContext is re-exported ----------


def test_skill_context_is_reachable_from_skills_package(tmp_path: Path) -> None:
    """`SkillContext` should be importable from `DefenseAgent.skills` directly. Sanity check that the re-exported class is the actual ms-agent class (not a stand-in)."""
    from ms_agent.skill.schema import SkillContext as UpstreamSkillContext
    assert SkillContext is UpstreamSkillContext


def test_skill_context_can_enumerate_files_for_a_loaded_skill(tmp_path: Path) -> None:
    """Round-trip: load a skill, build a SkillContext from its schema, get the file lists. Mirrors ms-agent's `auto_skills.py:210` usage exactly."""
    skill_dir = _write_skill(
        tmp_path / "rep",
        extras={
            "scripts/generate.py": "print('go')\n",
            "references/cheatsheet.md": "# Cheatsheet\n",
        },
    )
    schemas = load_skills(str(skill_dir))
    schema = next(iter(schemas.values()))
    ctx = SkillContext(skill=schema)
    assert "generate.py" in ctx.get_scripts_list()
    assert "cheatsheet.md" in ctx.get_references_list()


# ---------- (2) module-level load_skills convenience ----------


def test_module_level_load_skills_returns_same_mapping_as_loader(tmp_path: Path) -> None:
    """`load_skills(path)` is shorthand for `SkillLoader().load_skills(path)` — keys + name should match."""
    skill_dir = _write_skill(tmp_path / "rep")
    via_function = load_skills(str(skill_dir))
    via_class = SkillLoader().load_skills(str(skill_dir))
    assert set(via_function.keys()) == set(via_class.keys())
    assert next(iter(via_function.values())).name == "tabular-report"


def test_module_level_load_skills_walks_a_directory_tree(tmp_path: Path) -> None:
    """ms-agent's loader auto-detects single-skill vs tree (`_scan_and_load_skills`); the convenience function inherits the same behaviour."""
    root = tmp_path / "skills"
    _write_skill(root / "alpha", name="alpha")
    _write_skill(root / "beta", name="beta")
    schemas = load_skills(str(root))
    assert sorted(s.name for s in schemas.values()) == ["alpha", "beta"]


# ---------- (3) author + tags surface in tool metadata ----------


def test_tool_metadata_includes_author_and_tags(tmp_path: Path) -> None:
    skill_dir = _write_skill(
        tmp_path / "rep",
        author="Maya Rodriguez",
        tags=["analytics", "reporting"],
    )
    loader = SkillLoader()
    loader.load_skills(str(skill_dir))
    tool = next(t for t in loader.to_tools() if t.source == "skill")
    assert tool.metadata["author"] == "Maya Rodriguez"
    assert tool.metadata["tags"] == ["analytics", "reporting"]


def test_tool_metadata_handles_missing_author_and_tags(tmp_path: Path) -> None:
    """When the frontmatter omits `author` / `tags`, metadata still has stable keys so downstream filters can rely on the shape."""
    skill_dir = _write_skill(tmp_path / "rep")
    loader = SkillLoader()
    loader.load_skills(str(skill_dir))
    tool = next(t for t in loader.to_tools() if t.source == "skill")
    assert tool.metadata["author"] is None
    assert tool.metadata["tags"] == []


# ---------- (4) tool description lists bundled files ----------


def test_tool_description_lists_bundled_files(tmp_path: Path) -> None:
    """Skills with scripts/references/resources should advertise them in the description so the LLM can decide whether to read or execute without an extra round-trip."""
    skill_dir = _write_skill(
        tmp_path / "rep",
        description="Produces a markdown table.",
        extras={
            "scripts/generate.py": "print('go')\n",
            "scripts/format.py": "print('fmt')\n",
            "references/cheatsheet.md": "# Cheatsheet\n",
            "data/template.txt": "row\n",
        },
    )
    loader = SkillLoader()
    loader.load_skills(str(skill_dir))
    tool = next(t for t in loader.to_tools() if t.source == "skill")

    desc = tool.description
    # original description is preserved at the top
    assert desc.startswith("Produces a markdown table.")
    # all three buckets land in one structured line, ms-agent style
    assert "scripts: " in desc
    assert "generate.py" in desc and "format.py" in desc
    assert "references: " in desc
    assert "cheatsheet.md" in desc
    assert "resources: " in desc
    assert "template.txt" in desc
    # and we drop SKILL.md / LICENSE.txt from the resources list (ms-agent filter)
    assert "SKILL.md" not in desc.split("Bundled files")[1]


def test_tool_description_omits_inventory_when_no_extra_files(tmp_path: Path) -> None:
    """A skill with only SKILL.md doesn't deserve an inventory suffix — keeps the description clean."""
    skill_dir = _write_skill(tmp_path / "rep", description="Just instructions.")
    loader = SkillLoader()
    loader.load_skills(str(skill_dir))
    tool = next(t for t in loader.to_tools() if t.source == "skill")
    assert tool.description == "Just instructions."
    assert "Bundled files" not in tool.description


def test_tool_description_shows_none_for_empty_bucket(tmp_path: Path) -> None:
    """When at least one bucket has files, all three are listed; empty ones show 'None' (matches ms-agent's `... or 'None'` in PROMPT_SKILL_ANALYSIS_PLAN)."""
    skill_dir = _write_skill(
        tmp_path / "rep",
        extras={"scripts/only.py": "print('hi')\n"},
    )
    loader = SkillLoader()
    loader.load_skills(str(skill_dir))
    tool = next(t for t in loader.to_tools() if t.source == "skill")
    assert "scripts: only.py" in tool.description
    assert "references: None" in tool.description
    assert "resources: None" in tool.description
