---
name: tabular-report
description: Render a list of row dictionaries as a GitHub-flavored Markdown table, with an optional header block.
---

# Tabular Report

Use this skill when you have a list of dict-shaped rows and need to present
them to the user as a clean Markdown table.

## How to use it

1. Collect your rows as a list of dicts. Every dict should have the same keys.
2. Decide the column order you want. The skill does **not** infer ordering —
   pass the column names explicitly.
3. Call the helper in `scripts/generate.py`. Request that file via the skill's
   `file` argument to read its source, then invoke it in your own code.
4. If the report needs a title or framing paragraph, prepend the block in
   `templates/header.md` (also available via `file`).

## Example

```
rows    = [{"name": "Maya", "score": 94}, {"name": "Chloe", "score": 87}]
columns = ["name", "score"]

→  render_table(rows, columns)

| name  | score |
| ---   | ---   |
| Maya  | 94    |
| Chloe | 87    |
```

## Assets

- `scripts/generate.py` — the `render_table(rows, columns)` helper (Layer 3).
- `templates/header.md` — the optional report header block (Layer 3).

To fetch either, the model calls this same tool again with
`{"file": "scripts/generate.py"}` or `{"file": "templates/header.md"}`.
