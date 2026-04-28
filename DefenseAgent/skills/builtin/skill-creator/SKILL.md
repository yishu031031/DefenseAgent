---
name: skill-creator
description: Use when the user asks to create a new skill, save a workflow as a skill, distil a procedure into a reusable skill, or extract a recurring pattern into one. Walks through brainstorming the trigger, drafting the description, scaffolding files, and validating. Defaults to the project skill directory; pass scope=user for cross-project skills.
---

# Skill Creator

Author a new DefenseAgent skill end to end. The output is a `<skill-name>/` directory under either `./skills/` (project scope, default) or `~/.defense-agent/skills/` (user scope).

## Step 1 — Read the methodology

Before writing anything, read `writing-skills` (call the `writing-skills` tool with no arguments). It defines the frontmatter rules and structure conventions you must follow.

## Step 2 — Brainstorm with the user

Drive a short conversation to nail down four things:

1. **Trigger condition** — what user phrase, situation, or task type should select this skill? List 3–5 concrete examples.
2. **Procedure** — what are the steps the LLM should follow once the skill is selected? Be specific.
3. **Scope** — is this useful only for the current project (project scope) or across every project the user touches (user scope)?
4. **Scripts?** — does the procedure include mechanical steps that would benefit from being a Python/shell helper, or is the whole thing reasoning + tool use?

Don't skip this step even when the user says "just make a skill that does X". Vague triggers produce wasted invocations later.

## Step 3 — Draft the frontmatter

Write the `name` (kebab-case) and `description`. Description shape (from `writing-skills`):

```
Use when <trigger conditions, with concrete keywords>. <One sentence on behaviour>. [Optional: what NOT to use this for.]
```

Show the draft to the user before scaffolding. A bad description means bad triggering — it's the field most worth iterating on.

## Step 4 — Scaffold

Call the `skill-creator__init_skill` tool with `args=["<name>", "--scope", "<project|user>"]` (add `"--with-scripts"` if step 2 decided you need them). This creates:

```
<root>/<name>/
├── SKILL.md                 (placeholder frontmatter + body skeleton)
└── scripts/                 (only if --with-scripts)
```

`<root>` is `./skills/` for project scope, `~/.defense-agent/skills/` for user scope.

## Step 5 — Fill in the body

Use the `Write` tool (or whatever file-writing tool the host exposes) to replace the placeholder content. Body structure (from `writing-skills`):

1. One-sentence purpose at the top.
2. Numbered or `##`-headed steps.
3. Anti-pattern callouts where relevant.
4. References to sibling skills if they participate in the procedure.

## Step 6 — Validate frontmatter and structure

Call `skill-creator__validate_skill` with `args=["<absolute-or-relative-path-to-skill-dir>"]`. It reports:

- YAML parse errors
- Missing required fields (`name`, `description`)
- `name` / directory mismatch
- `SkillSchemaParser` load failures (the same parser the loader uses)

Fix anything reported; re-run until clean.

## Step 7 — Trigger evaluation (recommended)

Call `skill-creator__eval_description` with `args=["<skill-path>", "--queries", "q1", "q2", ...]`. Provide:

- 3–5 **positive queries** — phrases that should trigger this skill.
- 2–3 **negative queries** — phrases that look related but should not trigger it.

The script asks the agent's LLM to judge each query against the description. Aim for >= 8/10 correct decisions; if not, tighten the description (Step 3) and re-eval.

## Step 8 — Tell the user it's ready

Summarise:

- Where the skill landed (full path).
- Whether they need to restart the agent or call `SkillLoader.reload()` to pick it up. (Default: yes — the loader caches at startup.)
- Suggest committing project-scope skills to git.

## Anti-patterns

- **Skipping Step 1** — `writing-skills` is short; reading it once per author session is cheap insurance.
- **Skipping Step 7** — descriptions feel right but trigger wrong. Eval is the only honest check.
- **Putting business logic in `scripts/` "just in case"** — if the body alone works, leave scripts out (`writing-skills` YAGNI rule).
- **Authoring inside `DefenseAgent/skills/builtin/`** — that directory is framework-owned. New skills go to project or user scope.
