---
name: reflect-and-distill
description: Use when the user asks to reflect, summarise lessons, consolidate memory, recap a session, or take stock after a long working stretch. Pulls recent records via memory_recall, groups them into the four auto-memory categories (user, feedback, project, reference), and asks the user which insights to retain. Does not bypass or reconfigure the existing Reflector.
---

# Reflect and Distill

Drive a structured reflection pass over the agent's mem0-backed memory using only `memory_recall` and the LLM's judgement. The output is a categorised summary the user can review.

## Step 1 — Gather raw material

Call `memory_recall` multiple times with different framings, capping each at `top_k=10`. Suggested queries:

- `recent decisions` — what was decided lately, by whom, why.
- `user preferences` — how the user wants to work.
- `failures` / `things that did not work` — known mistakes to avoid.
- `project state` — current ongoing work.

If the user mentions a specific theme, add a query around it.

## Step 2 — Categorise into four buckets

Read each retrieved record and decide which of the four auto-memory categories it belongs to (these mirror the categories used by Claude Code's auto memory system):

| Category | Holds |
|----------|-------|
| **user** | Who the user is, role, expertise, recurring goals. Not the user's behaviour in this session — that goes to feedback. |
| **feedback** | Corrections, approvals, working preferences. "Don't do X." "Yes, exactly like that." Always include the *why* if known. |
| **project** | Decisions and context not derivable from code: in-flight initiatives, deadlines, stakeholder asks. |
| **reference** | Pointers to external systems: dashboards, channels, ticket trackers, docs URLs. |

A record that doesn't fit any of the four is probably an ephemeral observation; drop it.

## Step 3 — Present the distilled summary

Show the user the four categories with bullet points under each. Format:

```
## User
- <fact 1>
- <fact 2>

## Feedback
- <rule 1> (Why: <reason>)
- <rule 2>

## Project
- <fact 1>

## Reference
- <pointer 1>
```

Keep each bullet to one line. If a category is empty for this run, omit the heading rather than printing "(none)".

## Step 4 — Ask which to retain

Ask the user explicitly: "Which of these should I keep, refine, or drop?" Don't assume — surfacing the choice is the value of this skill over a passive `Reflector` run.

## Step 5 — Persist the keepers

For each retained insight, the agent's normal memory write path takes over: when the next `agent.run()` saves its outcome, mem0 picks up the conversation context. **Do not** invoke the `Reflector` directly — its threshold-gated cycle continues to run on its own schedule and shouldn't be double-fired.

If the host application provides an explicit "save to memory" tool, use that for items the user wants pinned now rather than next-run.

## Anti-patterns

- **Querying `memory_recall` once with a generic string** — you'll miss most of the relevant material. Three to five framings is typical.
- **Inventing "reflection insights"** that aren't grounded in retrieved records. Every bullet must trace to at least one record.
- **Bypassing the Reflector** — this skill is *additive* to the agent's existing reflection cycle, not a replacement. The Reflector still runs on its own threshold.
- **Trying to write directly to memory_type=reflection** — that's the Reflector's tag. Distilled-and-retained items use whichever memory_type the host application already uses for user-confirmed facts.
