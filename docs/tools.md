# Tools

Three tool sources, all merged into a single `ToolRegistry` the LLM sees as a flat namespace. Plus the built-in tools the agent auto-registers when their subsystem is on.

## Overview

```yaml
tools:
  skills:                                 # list[str]. SKILL.md-style bundles (read-only by default).
    - skills/tabular-report
  mcp:                                    # list[MCPServerConfig]. External MCP tool servers.
    - command: uvx
      args: [mcp-server-filesystem, /tmp]
  python:                                 # list[str]. Python entry-point strings.
    - python_tools/calc.py:calculator
    - my_pkg.search:web_search
  allow_skill_execution: false            # bool, default false. Promote skill scripts to executable tools.
  skill_execution_timeout: 300            # int ≥ 1, default 300. Subprocess timeout (seconds).
```

When `run()` starts, the registry is the union of all three sources plus the auto-registered `memory_recall` and `rag_search` (when enabled). **Each tool name must be globally unique** — collisions fail loud at construction.

---

## `tools.skills:` — local SKILL.md bundles

A skill is a directory anywhere under (or pointed at by) the profile, with `SKILL.md` at its root. The reference bundle [`DefenseAgent/examples/example_agent/skills/tabular-report/`](../DefenseAgent/examples/example_agent/skills/tabular-report) is the canonical shape:

```
skills/tabular-report/
├── SKILL.md                   # required — frontmatter + body
├── scripts/                   # optional — runnable scripts
│   └── generate.py
├── references/                # optional — long reference docs
└── templates/                 # optional — supporting resource files
    └── header.md
```

`SKILL.md` opens with YAML frontmatter, then a free-form Markdown body the LLM reads:

```markdown
---
name: tabular-report
description: Render a list of row dictionaries as a GitHub-flavored Markdown table.
author: kevin                  # optional, surfaces in tool metadata
tags: [reporting, table]       # optional, surfaces in tool metadata
---

# Tabular Report

Use this skill when you have row dicts and need a Markdown table.

## How to use it

1. Collect rows as a list of dicts with the same keys.
2. Pass column names explicitly — the skill won't infer them.
3. Read `scripts/generate.py` via this tool's `file=` argument, then call
   `render_table(rows, columns)` from your own code.
```

When the agent loads this skill, **one read-only tool** appears in the registry, named after the skill (`tabular-report`):

```json
{
  "name": "tabular-report",
  "description": "Render a list of row dictionaries as a GitHub-flavored Markdown table.\n\nBundled files — scripts: generate.py; references: None; resources: header.md.",
  "input_schema": {"file": "string (optional)"}
}
```

The description is the frontmatter `description:` plus a one-line inventory of bundled files (so the LLM can ask for them by name without guessing).

### How the LLM uses it

| Call | Returns |
|---|---|
| `tabular-report({})` (or `file=""`) | The SKILL.md body, frontmatter stripped — i.e. the LLM gets the prompt-style docs |
| `tabular-report({"file": "scripts/generate.py"})` | Raw text of that file |
| `tabular-report({"file": "templates/header.md"})` | Raw text of that file |
| `tabular-report({"file": "../../etc/passwd"})` | `SkillLoadError("path escapes skill directory ...")` — path-escape-guarded |

Skill metadata (`skill_id`, `version`, `author`, `tags`) rides along on the `Tool` object's `metadata` dict for downstream filtering or audit.

### Promoting scripts to executable tools

By default, scripts are *readable* but not *runnable* — the LLM has to paste their contents into its own reasoning. Flip `allow_skill_execution: true` and **each script becomes a separate executable tool** named `<skill>__<stem>`:

```yaml
tools:
  skills:
    - skills/tabular-report
  allow_skill_execution: true
  skill_execution_timeout: 300            # subprocess timeout (seconds)
```

Now the registry also exposes `tabular-report__generate` with input schema `{args?: list[str], stdin?: string, timeout?: int}`. Each call runs the script as a fresh subprocess via `SkillContainer` (inheriting ms-agent's dangerous-pattern guard against `rm -rf`-style payloads). Stdout, stderr and exit code are returned to the LLM as a single string.

Recognised script extensions: `.py`, `.sh`, `.js`. Scripts in subdirectories of `scripts/` are NOT recursively included — only top-level scripts get promoted.

---

## `tools.mcp:` — external MCP servers

[Model Context Protocol](https://modelcontextprotocol.io) servers are external processes that expose their own tool catalogues. DefenseAgent's `MCPClient` extends ms-agent's multi-server client and supports four transports:

| `transport:` | When to use | Required field |
|---|---|---|
| `stdio` (default) | Locally-launched server processes (`uvx`, `npx`, `python`, ...) | `command:` |
| `sse` | Long-lived HTTP server-sent-events endpoints | `url:` |
| `websocket` | WS-based servers | `url:` |
| `streamable_http` | HTTP streaming-style endpoints | `url:` |

Each entry **must set exactly one** of `command:` or `url:` — never both. Servers are connected lazily on the first `agent.run()` call (the connection is async and only spun up when a tool actually fires).

### stdio example — local filesystem server

```yaml
tools:
  mcp:
    - command: uvx                        # binary on PATH
      args: [mcp-server-filesystem, /tmp/sandbox]
      env:
        DEBUG: "1"
        GITHUB_TOKEN: ""                  # empty value → looked up in process env at connect()
      cwd: /workspace                     # optional working directory
      include: [read_file, list_dir]      # whitelist — only these tool names exposed
      # exclude: [delete_file]            # alternative: blacklist; mutually exclusive with include
```

Behaviour:

- Each tool the server advertises becomes a `Tool` in the registry, **named after the server's tool name** (no prefix). The originating server name is recorded in `tool.metadata["server"]` for traceability.
- `include:` / `exclude:` are mutually exclusive per server. Use them to scope down a chatty server (e.g. `mcp-server-filesystem` exposes ~10 tools — restrict to read-only with `include: [read_file, list_dir]`).
- Empty `env:` values (e.g. `GITHUB_TOKEN: ""`) are interpolated from the process environment at connect time — write `""` instead of hardcoding the key.

### Network transport example — SSE

```yaml
tools:
  mcp:
    - transport: sse
      url: https://mcp.example.com/sse
      headers:
        Authorization: "Bearer ${MCP_API_TOKEN}"  # not auto-interpolated; expand yourself
      timeout: 30                                  # connection timeout in seconds
      sse_read_timeout: 300                        # long-poll read timeout
      include: [search]
```

Header values are passed verbatim — DefenseAgent does **not** expand `${VAR}` for you. If you want env-var substitution, do it programmatically before constructing `AgentConfig`, or store the resolved value in `.env` and inline it.

### Multiple servers

```yaml
tools:
  mcp:
    - command: uvx
      args: [mcp-server-filesystem, /tmp]
      include: [read_file]
    - transport: sse
      url: https://mcp.example.com/sse
      headers: { Authorization: "Bearer secret" }
```

Both servers' tools end up in the same flat registry. Tool-name collisions across servers fail at registry build, so name discipline matters when you compose many servers.

Install with `defense-agent[mcp]` (the official `mcp>=1.0` Python SDK).

---

## `tools.python:` — your own Python functions

Two forms, both pointed at by an entry-point string `<module-or-file>:<function-name>`:

### 1. Relative file path

No packaging needed. Resolved against the profile's directory and loaded via `importlib.util.spec_from_file_location`. The interpreter doesn't need `sys.path` set up.

```
my_profile/
├── profile.yaml              # tools.python: ["python_tools/calc.py:calculator"]
└── python_tools/
    └── calc.py               # def calculator(expression: str) -> str: ...
```

### 2. Dotted module path

When your tool lives in an installed package. Resolved via `importlib.import_module`. The module must be importable from the running interpreter — installed via `pip install -e .` or already on `sys.path`.

```
my_pkg/
├── __init__.py
└── search.py                 # def web_search(query: str) -> str: ...
```

Profile entry: `my_pkg.search:web_search`.

### Tool schema is auto-derived

For both forms the **function signature** becomes the tool's input schema and the **docstring** becomes the description. The LLM never sees your code body — only this synthesised metadata:

```python
def calculator(expression: str, precision: int = 4) -> str:
    """Evaluate a Python arithmetic expression and return the result.

    Supports +, -, *, /, **, parentheses, and the math module.
    """
    ...
```

Becomes:

```json
{
  "name": "calculator",
  "description": "Evaluate a Python arithmetic expression...",
  "input_schema": {
    "type": "object",
    "properties": {
      "expression": {"type": "string"},
      "precision":  {"type": "integer", "default": 4}
    },
    "required": ["expression"]
  }
}
```

Type-hint coverage: `str`, `int`, `float`, `bool`, `list[T]`, `dict`, `Optional[T]`, plain `Path`. Any complex type without a clean JSON-schema fallback raises `ToolRegistrationError` at load — name issues surface immediately, not on first call.

### Inline tool in code (no profile entry)

If you don't want to put the tool in `profile.yaml`, register it programmatically:

```python
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression."""
    ...

config = AgentConfig(profile="…", tools=[calculator])
```

The `tools=` kwarg accepts plain callables — same auto-derivation applies. Use this for one-off tools, tests, or tools whose definition only makes sense at runtime (closures over an open DB connection, etc.).

---

## Built-in tools

In addition to anything you register under `tools:`, the agent automatically exposes these to the LLM:

| Tool | When registered | Input schema | What it does |
|---|---|---|---|
| `memory_recall` | When `memory.is_retrieve: true` | `{query: string, top_k?: int (1–20, default 5)}` | Tier-aware semantic search via `MemoryOrchestrator`. Returns up to top_k records as `- [<tier>/<memory_type>] <content>` bullets. Hybrid scoring by default (since 0.2.0). |
| `rag_search` | When `rag.enabled: true` | `{query: string, top_k?: int}` | Vector search over the RAG index. Returns ranked chunks above `score_threshold`. |
| `<skill>` (one per skill) | One per `tools.skills:` entry | `{file?: string}` | No `file` → returns SKILL.md body. With `file` → returns the named file. Path-escape-guarded. |
| `<skill>__<script>` (one per script) | When `allow_skill_execution: true` | `{args?: list[str], stdin?: string, timeout?: int}` | Runs the script as a subprocess. Returns stdout + stderr + exit code as a single string. |
