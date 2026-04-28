---
name: promote-memory-to-skill
description: Use when memories reveal a recurring pattern, repeated user feedback, or a frequently needed procedure that would benefit from being a reusable skill. Trigger phrases include "this keeps coming up", "we always do X", "save this approach", "make this a skill". Validates the pattern, drafts a skill candidate, then hands off to skill-creator.
---

# Promote Memory to Skill

Identify a recurring pattern in memory and turn it into a real skill. This skill is the *bridge* between `reflect-and-distill` (which surfaces patterns) and `skill-creator` (which authors files).

## Step 1 — Identify the candidate pattern

The user (or your own observation) names something that "keeps happening". Examples:

- "Every time we touch the migration scripts we forget to update the schema doc."
- "I always want PRs to be squashed with a Co-Authored-By line."
- "We've debugged this same TypeError four times now."

Anchor on the specific phrase the user used and the context.

## Step 2 — Gather supporting evidence

Use `memory_recall` to find at least 5 records that support the candidate pattern. Vary the queries:

- Direct keyword from the user's phrasing.
- Adjacent terms (synonyms, related concepts).
- The opposite ("times this did NOT happen") — to check whether the pattern is real or a hot-take.

If you can't find 5 records, the pattern may not be ready for promotion. Tell the user and stop. A skill built on one or two anecdotes is over-fit and will trigger wrong.

## Step 3 — Sanity-check the pattern

Ask yourself:

- **Is it consistent?** Do the supporting records show the same shape, or are they a grab-bag of vaguely related things?
- **Is it actionable?** A skill needs concrete steps. "User likes clean code" is not a skill; "When opening a PR for a Python file, run black before committing" is.
- **Is it stable?** A pattern from this week's project that won't apply next week probably belongs in memory, not a skill.

If any of these fails, prefer logging the insight as a feedback/project memory instead.

## Step 4 — Draft the skill candidate

Sketch four fields and show the user:

- **Name** (kebab-case, action-flavoured).
- **Trigger** — one sentence of "Use when …" with the keywords from Step 1.
- **Procedure** — 3–7 numbered steps.
- **Scope** — `project` (only this repo) or `user` (everywhere). Default to project unless the user explicitly says "everywhere".

Example:

```
Name: ensure-schema-doc-updated
Trigger: Use when editing files under migrations/ or schema/.
Procedure:
  1. Identify which schema file changed.
  2. Open docs/schema/<name>.md.
  3. Update the affected sections.
  4. Stage both files together.
Scope: project
```

## Step 5 — Confirm before authoring

Ask the user: "Does this look right? Anything to add, drop, or rename?"

Iterate until they approve. **Do not skip this step** — once written, a poorly scoped skill will mis-fire on every future task that brushes its trigger zone.

## Step 6 — Hand off to skill-creator

Call the `skill-creator` tool with no arguments to load its instructions, then follow them, feeding in the approved draft. Skill-creator owns Steps 3 onward (frontmatter formatting, scaffolding, validation, eval).

## Step 7 — Confirm the new skill loads

After `skill-creator` finishes, tell the user:

- Where the skill landed.
- Whether the agent needs a restart or `loader.reload()` for it to take effect.
- (Optional) suggest a quick test: ask the user to phrase their original "this keeps happening" in a fresh session and see whether the new skill triggers.

## Anti-patterns

- **Promoting on a single anecdote** — Step 2 minimum of 5 supporting records exists for a reason.
- **Authoring the SKILL.md yourself instead of calling skill-creator** — `skill-creator` enforces validation + eval. Bypassing it produces skills that look fine but trigger wrong.
- **Promoting feedback-shaped content** — "User prefers terse output" is feedback memory, not a skill. Skills are *procedures*, not *preferences*.
- **Promoting before the pattern is mature** — when in doubt, file it as a feedback/project memory now. You can always promote it later.
