---
name: using-skills
description: Use when starting any conversation or task with DefenseAgent. Establishes how to discover and invoke skills before responding, lists the bundled methodology skills (skill-creator, writing-skills, reflect-and-distill, promote-memory-to-skill), and documents the tool-call shape — each skill is a tool whose name equals the skill name; calling that tool with no arguments returns the skill's instructions.
---

# Using Skills

Skills are reusable procedures stored as markdown. They live in three layers and are auto-discovered when the agent starts:

| Layer | Path | Purpose |
|-------|------|---------|
| builtin | `DefenseAgent/skills/builtin/` (in the wheel) | Framework methodology — what you are reading now |
| user | `~/.defense-agent/skills/` | Cross-project skills the user authored |
| project | `./skills/` (cwd) | Skills versioned with the current project |

Later layers override earlier ones on name collision (`project > user > builtin`).

## The rule

**Before responding to the user, evaluate whether a skill applies.** If there is even a small chance one fits, call its tool first to read its instructions, then follow them.

This applies to *every* user request, including ones that look trivial. "Simple" tasks are where unexamined assumptions cause the most wasted work.

## How to invoke a skill

Each loaded skill is a tool whose name equals the skill name. Two call shapes:

- No arguments → returns the SKILL.md body (the procedure to follow).
- `file=<basename-or-relative-path>` → returns a bundled file inside the skill directory (e.g., a script source, a reference doc, a template).

Skills with executable scripts also expose them as additional tools named `<skill-name>__<script-stem>` (e.g., `skill-creator__init_skill`). Call those tools with `args` (positional CLI args) and optional `stdin` / `timeout`.

## The other built-ins

- **`writing-skills`** — Methodology for authoring a skill: frontmatter discipline, single responsibility, YAGNI. Read this before creating or refining a skill.
- **`skill-creator`** — Walks you and the user through creating a new skill end to end, including frontmatter design, scaffolding, validation, and trigger evals. Calls `writing-skills` for methodology.
- **`reflect-and-distill`** — Use when the user asks you to reflect, summarise lessons, or consolidate memory. Calls `memory_recall` to gather records, then groups them into the four auto-memory categories (user, feedback, project, reference).
- **`promote-memory-to-skill`** — Use when memories show a recurring pattern worth saving as a reusable skill. Drafts the skill candidate, then hands off to `skill-creator`.

## Anti-patterns

- "I remember what this skill says" → No. Skills evolve; read the current version.
- "This is too simple to need a skill" → If a skill exists for the situation, use it.
- "I'll skip the skill and just do the task" → The skill might tell you the task is the wrong shape. Read first.
- "I'll respond first and check skills later" → Skills inform the response; check before, not after.
