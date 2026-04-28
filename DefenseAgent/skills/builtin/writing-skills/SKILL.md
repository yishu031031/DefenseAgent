---
name: writing-skills
description: Use when creating or refining a DefenseAgent skill, before drafting frontmatter or scaffolding files. Defines the frontmatter rules (especially description tuning so skills trigger correctly), file structure conventions, single-responsibility and YAGNI rules. Referenced by skill-creator step 1.
---

# Writing Skills

A good skill is a focused, testable procedure that reads cleanly to both a human and an LLM. This document is the methodology.

## File layout

```
<skill-name>/
├── SKILL.md                 (required — the procedure)
├── scripts/                 (optional — executable helpers)
│   └── *.py / *.sh / *.js
├── references/              (optional — read-only reference docs)
└── resources/               (optional — assets like templates, images)
```

`SKILL.md` is the spine. Everything else is opt-in. Don't create a `scripts/` directory until you actually need an executable helper — you can always add one later.

## Frontmatter

Every SKILL.md begins with YAML frontmatter:

```yaml
---
name: <kebab-case-name>
description: Use when <trigger>. <one-line summary of behaviour>.
---
```

### Field rules

- **`name`** — kebab-case, action-flavoured (`promote-memory-to-skill`, not `MemoryPromoter`). Must match the directory name.
- **`description`** — the most load-bearing field. The LLM picks skills by reading descriptions, so be precise. Mandatory shape:
  - First sentence starts with `Use when …` and lists the *trigger conditions* (what user phrases or situations should select this skill).
  - Second sentence summarises the *behaviour* (what the skill does once selected).
  - If meaningful, a third sentence covers *what NOT to use this for* (negative trigger).
- **`type`** *(optional)* — free-form tag (e.g., `methodology`, `domain`).
- **`author`**, **`tags`**, **`version`** *(optional)* — pass through to the schema for filtering.

### Description tuning

Every wasted invocation is a description failure. Two failure modes:

- **Over-triggers** (selected when irrelevant) → description is too generic. Add specificity ("when the user mentions X", "after step Y of …").
- **Under-triggers** (missed when relevant) → description doesn't list the situation's keywords. Add the actual phrases users say.

Validate with `skill-creator__eval_description` (see `skill-creator`). Aim for >= 8/10 hit-rate on five plausible queries.

## Body structure

The SKILL.md body should be optimised for **scan + execute**, not narrative:

1. **One-sentence purpose** at the top (often a `## Purpose` or `## What this does`).
2. **Numbered or `##`-headed steps** — the LLM follows these literally. Don't bury actions inside paragraphs.
3. **Anti-pattern / red-flag callouts** when there's a known way to misapply the skill.
4. **References** to other skills with their tool names in code spans (e.g., `` `writing-skills` ``).

Avoid:
- Long preambles or motivation sections — put motivation in the description, not the body.
- Conditional flowcharts past two levels — split into a separate skill instead.
- Code blocks longer than ~30 lines — move to `scripts/` or `references/`.

## Single responsibility

One skill, one job. Triggers should not overlap. If two skills could both fire on the same query, either:
- Make one delegate to the other (the way `skill-creator` calls `writing-skills`), or
- Tighten the descriptions so each owns a distinct trigger zone.

## YAGNI

- Don't add `scripts/` until the procedure mechanically repeats and is worth automating.
- Don't pre-create `references/` for "future" docs.
- Don't generalise on the first version. Two concrete variants beat one premature abstraction.

## Self-check before publishing

1. `skill-creator__validate_skill <path>` — frontmatter + parser round-trip clean.
2. `skill-creator__eval_description <path> --queries "..." "..."` — trigger eval on at least 3 positive + 2 negative queries.
3. Read the body aloud. If the steps don't compose into "do A, do B, …" without backtracking, the structure is wrong.
