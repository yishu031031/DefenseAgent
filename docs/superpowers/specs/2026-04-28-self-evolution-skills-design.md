# Self-Evolution Skills — Design

**Status:** Approved (2026-04-28)
**Scope:** Add Claude-Code-style self-evolution to DefenseAgent via static SKILL.md packs, a skill-creator skill, and lightweight runtime auto-discovery — no Python evolution engine, no autonomous closed-loop.

## Decisions

| # | Decision | Choice |
|---|----------|--------|
| 1 | Evolution mechanism | **A** Static SKILL.md packs only |
| 2 | Skill source layers | **builtin** (in package) + **user** (`~/.defense-agent/skills/`) + **project** (`./skills/`) |
| 3 | `skill-creator` form | **Y** Conversational SKILL.md + 3 helper scripts |
| 4 | Builtin roster | **Standard** 5 skills |
| 5 | Auto-discovery | **Q** Default-load all three layers, fall back silently when missing |
| 6 | Layer precedence | **M3** project > user > builtin (later overrides earlier on name collision) |
| 7 | Reflection bridge | **N1** Pure SKILL.md; do not modify `Reflector` |

## Architecture

Three skill-source layers fed into the existing `SkillLoader`:

1. **builtin** — `DefenseAgent/skills/builtin/`, ships in the wheel (`importlib.resources` anchor). Five framework methodology skills live here.
2. **user** — `~/.defense-agent/skills/`, cross-project user skills (path overridable via profile).
3. **project** — `./skills/` relative to the agent's working dir (path overridable via profile).

Layers load in order builtin → user → project. Same-name skills override (later wins). Each layer's failures are logged but never abort agent startup. The existing `to_tools()` converts every loaded skill into a tool whose name equals the skill name.

`skill-creator` writes new skill directories. Default scope is **project** (`./skills/<name>/`); pass `--scope=user` to write to `~/.defense-agent/skills/<name>/`.

`Reflector` is **not modified**. The two reflection-related skills (`reflect-and-distill`, `promote-memory-to-skill`) drive the LLM through `memory_recall` only.

## Skill roster (`DefenseAgent/skills/builtin/`)

| Skill | Purpose | Scripts? |
|-------|---------|----------|
| `using-skills` | Tells the LLM how to discover and invoke skills; lists the four other built-ins | none |
| `writing-skills` | Methodology for authoring skills (frontmatter discipline, single responsibility, YAGNI) | none |
| `skill-creator` | Walks user + LLM through creating a new skill | `init_skill.py`, `validate_skill.py`, `eval_description.py` |
| `reflect-and-distill` | Use `memory_recall` to consolidate recent records into the four auto-memory categories | none |
| `promote-memory-to-skill` | When a recurring pattern surfaces, draft a skill candidate and hand off to `skill-creator` | none |

### Frontmatter contract

Every SKILL.md begins with YAML frontmatter:

```yaml
---
name: <kebab-case>
description: Use when <trigger>. <one-line summary of behaviour>.
---
```

`description` must lead with `Use when …` so triggering remains predictable.

### Helper script contracts

| Script | Args | Behaviour |
|--------|------|-----------|
| `init_skill.py` | `<name> [--scope project\|user] [--with-scripts]` | Create `<root>/<name>/SKILL.md` (and optional `scripts/`) with placeholder frontmatter + body |
| `validate_skill.py` | `<skill_path>` | Parse YAML, check required fields, attempt `SkillSchemaParser` load, print findings |
| `eval_description.py` | `<skill_path> --queries "..." [...]` | Use the agent's LLM to judge whether each query should trigger the skill given its description |

## Runtime integration

### `DefenseAgent/config/profile.py`
Add `EvolutionConfig`:

```python
class EvolutionConfig(BaseModel):
    use_builtin: bool = True
    user_skills_dir: str | None = None      # default: ~/.defense-agent/skills
    project_skills_dir: str | None = None   # default: ./skills
    default_scope: str = "project"          # "project" | "user"
```

Mount as `AgentProfile.evolution`.

### `DefenseAgent/skills/loader.py`
Add module-level helpers:

- `builtin_skills_path() -> Path` — anchored on `__file__`, returns `<package>/skills/builtin/`
- `default_user_skills_path() -> Path` — `Path.home() / ".defense-agent" / "skills"`
- `default_project_skills_path() -> Path` — `Path.cwd() / "skills"`
- `discover_skill_dirs(evolution: EvolutionConfig) -> list[Path]` — returns ordered list (builtin → user → project), filtering non-existent layers
- `SkillLoader.load_dirs_tolerant(dirs)` — wrap `load_skills` per dir; per-skill failures are logged and skipped, not raised

### `DefenseAgent/agent/_builder.py`
After building the user `ToolRegistry` (when `use_tools=True` and no injected registry), call `discover_skill_dirs(profile.evolution)` and load each layer with `load_dirs_tolerant`, then merge resulting tools into the registry. A custom-injected `tool_registry` is left alone (caller manages everything).

### `pyproject.toml`
`[tool.hatch.build.targets.wheel]` already packages `DefenseAgent/`. Confirm SKILL.md and scripts directories ride along (Python files already do; markdown needs `force-include` if hatch trims it).

### `.gitignore`
Project-level `./skills/` is conventionally a candidate for exclusion (evolution products are user-local). Leave choice to the repo owner; we will not auto-add.

## Error handling

- Missing layer directories → silent (don't warn — they're the common-case empty state).
- Per-skill load failure (bad frontmatter, parser error) → warn via the agent's logger; continue.
- Same-name collision across layers → silent (this is the intended override mechanism).
- Helper-script failures inside `skill-creator` → bubble up as the script's exit code via `SkillContainer`.

## Out of scope (deferred)

- A `reflect_now` agent built-in tool (would belong with N2 in the brainstorm).
- Categorised reflection output from `Reflector` (N3).
- Autonomous evolution loop (C in the brainstorm).
- Skill versioning / rollback / sandbox eval.
- A skill marketplace registry.

## Tests / smoke checks

1. `SkillLoader().load_dirs_tolerant([builtin])` returns 5 schemas with the expected names.
2. `validate_skill.py` round-trips on each builtin SKILL.md without errors.
3. Smoke test: build an agent without a profile YAML; confirm five tool names show up in `tools.specs()`.

## Files touched

- New: `DefenseAgent/skills/builtin/{using-skills,writing-skills,skill-creator,reflect-and-distill,promote-memory-to-skill}/SKILL.md`
- New: `DefenseAgent/skills/builtin/skill-creator/scripts/{init_skill.py,validate_skill.py,eval_description.py}`
- New: `DefenseAgent/skills/builtin/__init__.py` (empty `importlib.resources` anchor)
- Edited: `DefenseAgent/config/profile.py` (+ `EvolutionConfig`)
- Edited: `DefenseAgent/skills/loader.py` (+ default-path helpers, tolerant loader)
- Edited: `DefenseAgent/agent/_builder.py` (auto-load wiring)
- Edited: `pyproject.toml` (package data confirmation)
- New: `tests/DefenseAgent/skills/test_builtin_skills.py`
