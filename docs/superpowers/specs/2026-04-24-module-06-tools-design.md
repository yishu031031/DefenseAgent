# Module 6 — Tools (Registry + Adapters) Design

**Date:** 2026-04-24
**Status:** Draft
**Module position:** 6 of N. The first module that lets the LLM *act* on the world (beyond text output).

## Purpose

Give the harness a single registry that can present a heterogeneous set of tools to the LLM and dispatch tool calls back to their right handler. Three registration paths, one unified interface:

1. **User-defined Python functions** — via a `@registry.tool` decorator that introspects the function signature to build the JSON schema.
2. **Anthropic-style Agent Skills** — directories containing a `SKILL.md` (frontmatter + body) plus optional asset files, served with **progressive disclosure** so skill content never bloats the system prompt.
3. **MCP (Model Context Protocol) servers** — stdio-launched subprocesses whose tool lists are discovered at connect time and whose calls are forwarded over the MCP session.

All three produce the same in-memory shape (`Tool` dataclass) and go through the same `execute(tool_calls) → list[Message]` path. The LLM module and the reflection module stay unaware of tool provenance.

A fourth, profile-driven entry point builds the whole registry for an agent in one call:

4. **`ToolRegistry.from_profile(profile)`** — reads `profile.tools.skills` and `profile.tools.mcp`, resolving skill paths relative to the profile's directory (`profile.source_dir`), and returns a fully populated `ToolRegistry`. This is how an agent declares its tool set in YAML.

## Scope

**In:**
- `ToolRegistry` facade with three registration paths + a canonical `spec()` emitter + a concurrent `execute()` dispatcher.
- Progressive-disclosure skill loading (Layer 1 frontmatter / Layer 2 body / Layer 3 asset files).
- MCP stdio adapter (via the official `mcp` Python SDK) with explicit lifecycle.
- Canonical `Tool` dataclass + error hierarchy (`ToolError`, `ToolRegistrationError`, `ToolNotFoundError`, `ToolExecutionError`, `SkillLoadError`).

**Out (deferred):**
- MCP **SSE / HTTP** transports — stdio first.
- MCP **prompts / resources** — only `tools` discovery for now.
- Per-tool **permission gates** (allow-list, user-confirmation hooks) — add when we integrate into an agent loop.
- **Tool result caching** / memoization.
- **Streaming** tool results.

## The three registration paths

### 1. `@registry.tool` — user-defined Python functions

```python
registry = ToolRegistry()

@registry.tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

# also:  @registry.tool(name="plus", description="Adds.")
```

- **Name** defaults to the function's `__name__`; overridable.
- **Description** defaults to the function's docstring; overridable.
- **Input schema** is derived by walking the signature: annotations → JSON types, defaults → optional fields, no default → `required`.
- Sync and async functions are both supported. Sync functions are off-loaded to a worker thread via `asyncio.to_thread` so blocking I/O doesn't stall the event loop.

### 2. `registry.add_skill(directory)` — Anthropic Agent Skills with progressive disclosure

Anthropic Agent Skills exist because context is scarce. A skill is a *directory*, not a file. The design insists on three layers:

| Layer | What | When it enters context |
|---|---|---|
| **1 — metadata** | `name` + `description` from SKILL.md frontmatter | Always; shown in `registry.spec()` in every turn |
| **2 — instructions** | SKILL.md body (markdown after frontmatter) | Only when the LLM invokes the skill tool with no `file` arg |
| **3 — assets** | Any other file in the skill directory (scripts, templates, extended refs) | Only when the LLM re-invokes the skill tool with `{"file": "rel/path"}` |

The registry exposes a skill as **one tool** with the input schema

```json
{
  "type": "object",
  "properties": {
    "file": {
      "type": "string",
      "description": "Optional POSIX-style relative path to an additional file inside the skill directory (Layer 3). Omit to load the skill's main instructions (Layer 2)."
    }
  }
}
```

The LLM learns a single protocol: "call the skill with no args to get instructions; call it again with `file=...` to pull referenced assets." The Layer-2 body is free to reference Layer-3 files by relative path, and those references work naturally because the same tool name + a new `file` argument fetches them.

**Security:** Layer-3 reads resolve the supplied path against the skill root and raise `SkillLoadError` on any escape attempt (`..`, absolute paths). Requests for `SKILL.md` route to the cached body, not to the file on disk.

### 3. `ToolRegistry.from_profile(profile, *, base_dir=None)` — profile-driven

Each agent lives in its own bundle under `agents/<agent_id>/` containing `profile.yaml` + `skills/` + (optionally) whatever else the agent needs. The profile declares its tool set:

```yaml
agent:
  id: maya_rodriguez
  name: Maya Rodriguez
  # ...identity + cognitive + memory blocks...
  tools:
    skills:
      - skills/tabular-report                # resolved against agents/maya_rodriguez/
      - ../../shared/skills/common-toolkit   # explicit opt-in to a shared skill
    mcp:
      - command: uvx
        args: [mcp-server-filesystem, /Users/maya/workspace]
      - command: python
        args: [-m, my.custom_mcp]
        env:
          API_KEY: secret
```

Code path:

```python
profile = AgentProfile.from_yaml("agents/maya_rodriguez/profile.yaml")
async with await ToolRegistry.from_profile(profile) as registry:
    ...   # registry is populated with Maya's skills + MCP servers — nothing else
```

Skill paths resolve relative to `profile.source_dir`, which `AgentProfile.from_yaml` sets to the profile's parent directory. This makes agent bundles self-contained and portable (copy the bundle elsewhere and the relative paths still work). In-memory profiles (no `source_dir`) raise `ToolRegistrationError` unless `base_dir` is passed explicitly.

**Isolation property:** two agents using `from_profile` cannot accidentally see each other's skills — each profile only reads paths below (or explicitly above, via `../`) its own directory.

### 4. `registry.add_mcp(command=..., args=[...])` — MCP stdio servers

```python
async with ToolRegistry() as registry:
    await registry.add_mcp(command="uvx", args=["mcp-server-filesystem", "/tmp"])
    # all filesystem-server tools are now in registry.spec()
```

- Opens the stdio connection + `ClientSession` via the official `mcp` SDK, initializes, and calls `list_tools()`.
- Each discovered tool is wrapped as a `Tool` whose handler forwards calls to that server's `session.call_tool(name, arguments)`.
- Results flatten `CallToolResult.content[].text` into a single string for the LLM. `isError=True` raises `ToolExecutionError`.
- Lifecycle is managed through an `AsyncExitStack` inside `MCPClient` — `registry.close()` (or the async `__aexit__`) tears down every MCP client the registry opened.

## Data model

```python
ToolSource  = Literal["python", "skill", "mcp"]
ToolHandler = Callable[[dict[str, Any]], Awaitable[str]]

@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    source: ToolSource
    handler: ToolHandler
    metadata: dict[str, Any]
```

Every registration path produces one of these; the facade's dispatcher treats all three uniformly.

### Error hierarchy

```
ToolError                  # base
├── ToolRegistrationError  # name collision, duplicate registration
├── ToolNotFoundError      # execute() called on unknown tool name
├── ToolExecutionError     # tool handler raised; __cause__ preserved
└── SkillLoadError         # missing SKILL.md, bad frontmatter, path escape
```

## Execution semantics

```python
messages = await registry.execute(tool_calls)
```

- `tool_calls: list[ToolCall]` (the `ToolCall` dataclass from `DefenseAgent.llm.types`, already emitted by every LLM adapter).
- Returns `list[Message]`, one per call, with `role="tool"`, `tool_call_id` preserved, `name` set to the tool name, `content` set to the tool's string output.
- **Concurrency:** `execute()` gathers all calls with `asyncio.gather`, so independent tool calls run in parallel.
- **Errors never crash dispatch.** A missing tool, a raised `ToolError`, or an unexpected exception become the `content` of an otherwise well-formed `Message`. The LLM sees the error as a normal tool result and can decide how to recover.

## File layout

```
DefenseAgent/tools/                          # 5 files, one concern per file
├── __init__.py                               # re-exports ToolRegistry + Skill + MCPClient + errors
├── tools.py                                  # ToolRegistry facade (decorator + add_skill + add_mcp + spec + execute)
├── types.py                                  # Tool dataclass + error hierarchy + ToolSource + ToolHandler
├── skill.py                                  # Skill (directory loader, progressive disclosure, Layer-3 reads)
└── mcp.py                                    # MCPClient (stdio adapter via AsyncExitStack)

tests/DefenseAgent/tools/
├── __init__.py
├── test_registry.py                          # decorator, add_skill, register, spec, execute, concurrency, context manager
├── test_skill.py                             # frontmatter parse, Layer 2 body, Layer 3 reads, security
├── test_mcp.py                               # stdio + session patched with fakes; discovery, forwarding, error cases
└── test_from_profile.py                      # ToolRegistry.from_profile against real + synthetic agent bundles

agents/                                       # per-agent bundles (user data, not code)
├── maya_rodriguez/
│   ├── profile.yaml                          # declares tools: section
│   └── skills/
│       └── tabular-report/
│           ├── SKILL.md
│           └── ...
└── alice_chen/
    └── profile.yaml
```

Dependency graph: `types` ← `skill`, `mcp` ← `tools`; `tools` also depends on `DefenseAgent.config.profile.AgentProfile` (one-way, leaf module).

## Dependencies

**One new runtime dep**: `mcp>=1.0.0` (the official Anthropic MCP Python SDK, Apache-2.0).

Everything else is stdlib (`asyncio`, `inspect`, `contextlib.AsyncExitStack`, `pathlib`) or already present (`pyyaml` for frontmatter, `pydantic` indirectly via the MCP SDK).

## Integration with earlier modules

| Module | How it's used | Do we modify that module? |
|---|---|---|
| Module 1 (LLM) | Reuses `Message` and `ToolCall` from `DefenseAgent.llm.types`. Later: LLM adapters will consume `registry.spec()` when `chat(..., tools=...)` is wired. | No |
| Module 2 (Config) | `AgentProfile.tools` (`ToolsConfig` + `MCPServerConfig`) declares every skill + MCP server an agent needs; `ToolRegistry.from_profile` consumes it. Skill paths resolve against `profile.source_dir`. | Yes — added on 2026-04-24 |
| Module 3 (Ops) | A logging agent can wrap `registry.execute()` to emit `tool.call_started` / `tool.call_finished` events. Optional. | No |
| Module 4 (Memory) | Not used. Tool results could be stored as `observation` records via `memory.remember` at a higher level. | No |
| Module 5 (Reflection) | Not used directly. | No |

Zero retrofits.

## Testing strategy

- **Skill** tests use a `tmp_path`-backed skill directory built fresh per test; all three disclosure layers and every security guard are exercised.
- **Registry** tests cover the decorator's two forms (bare `@registry.tool` and `@registry.tool(...)`), name collisions, schema derivation, `execute()` happy paths, error messages, concurrency (a `time.monotonic` assertion), and the async context manager.
- **MCP** tests never launch a real server. They patch `DefenseAgent.tools.mcp.stdio_client` and `DefenseAgent.tools.mcp.ClientSession` with `asynccontextmanager`-returning fakes that yield a `_FakeSession` recording every call. This validates the adapter's wire contract (initialize → list_tools → call_tool) without the subprocess fragility.

All tests stay fully offline.

## Open questions

- **Skill loader watch mode?** If a SKILL.md body changes on disk, we don't pick it up (body is cached at load time). Likely fine — skills are versioned in the filesystem, and re-registering is cheap. Re-open if this bites.
- **MCP environment passing.** Right now `env` is plumbed through `StdioServerParameters`, but we pass `None` to inherit the parent env. A future "only pass allow-listed env vars" policy may be needed.
- **Schema derivation for complex annotations.** We handle the simple primitive types; anything else falls through to `"string"`. A later pass can add `Optional`, `Literal`, and Pydantic model support when real users hit the edge.
