# Module 6 Walkthrough — Tools

> Companion to the [design spec](../superpowers/specs/2026-04-24-module-06-tools-design.md). The spec records **what** we decided and **why**; this walkthrough explains **how** each file implements those decisions.

---

## CORE CLASS: `ToolRegistry`

Start here. The module's single public entry point:

```python
from DefenseAgent.tools import ToolRegistry

# Typical case — build the whole registry from the agent's profile:
profile = AgentProfile.from_yaml("agents/maya_rodriguez/profile.yaml")
async with await ToolRegistry.from_profile(profile) as registry:
    tools_for_llm = registry.specs()                 # → [{name, description, input_schema}, ...]
    results = await registry.execute(tool_calls)     # → [Message(role="tool", ...)]

# Or register things by hand:
async with ToolRegistry() as registry:

    # 1. user-defined Python function
    @registry.tool
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    # 2. Anthropic Agent Skill (a directory)
    registry.add_skill("agents/maya_rodriguez/skills/tabular-report")

    # 3. MCP stdio server
    await registry.add_mcp(command="uvx", args=["mcp-server-filesystem", "/tmp"])
```

Four registration paths (three manual + one profile-driven), one uniform output shape, one dispatcher.

---

## 1. What problem this module solves

By Module 5, the harness can observe, remember, retrieve, and reflect — but it can't *act*. Tools are how an agent turns LLM output into effects on the world: calling a Python function you wrote, invoking a Claude Skill, or going through an MCP server to reach a filesystem / database / API.

Three things make tools harder than they look:

1. **Heterogeneous sources.** A function in your codebase, a markdown-packaged skill, and a remote MCP server are all "tools" to the LLM but behave completely differently at call time.
2. **Context is scarce.** You can't stuff every skill's full instructions into the system prompt. Anthropic Agent Skills solve this with **progressive disclosure**: the LLM sees only `name + description` by default and pulls the body in only when it decides the skill is relevant.
3. **Lifecycle.** MCP servers are subprocesses with streams that must be initialized and closed. If registration is "just append to a list," something will leak.

**Module 6 gives the harness:**

- `ToolRegistry` — one facade, three registration paths, one `execute()` call.
- `Skill` — directory-backed, progressive-disclosure loader.
- `MCPClient` — stdio adapter with explicit lifecycle via `AsyncExitStack`.

---

## 2. Directory map

```
DefenseAgent/tools/                          # 5 files
├── __init__.py                               # re-exports
├── tools.py                                  # ToolRegistry (facade)          ← START HERE
├── types.py                                  # Tool + errors + ToolSource
├── skill.py                                  # Skill (progressive disclosure)
└── mcp.py                                    # MCPClient (stdio adapter)

tests/DefenseAgent/tools/
├── __init__.py
├── test_registry.py                          # 22 tests
├── test_skill.py                             # 20 tests
└── test_mcp.py                               # 5 tests
```

Dependency direction is one-way: `types` → `skill`, `mcp` → `tools` → `__init__`.

---

## 3. The three ideas that shape the design

### 3.1 One `Tool` dataclass unifies every source

```python
@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    source: ToolSource              # "python" | "skill" | "mcp"
    handler: ToolHandler            # async (args) -> str
    metadata: dict[str, Any]
```

A Python-registered function, a loaded skill, and an MCP-discovered tool all flow through `registry._tools: dict[str, Tool]`. The `source` field is informational; the `handler` is what actually runs. The facade never branches on source at dispatch time.

### 3.2 Progressive disclosure for skills is built into the input schema

The LLM doesn't need a second protocol to reach Layer-3 files. The skill's tool takes one optional argument `file`:

- No `file` → returns the SKILL.md body (Layer 2).
- `file="scripts/generate.py"` → returns that file's contents (Layer 3).

The Layer-2 body is free to reference Layer-3 files by their relative path ("see `scripts/generate.py`"); the LLM picks them up by re-invoking the same tool.

### 3.3 Errors become tool results, not crashes

When a tool's handler raises, `execute()` turns the exception into the `content` of the resulting `role="tool"` Message. The LLM sees `"ToolExecutionError: ValueError: bad input"` as a normal tool response and can decide how to react. The event loop keeps going. This is how agents survive misbehaving tools.

---

## 4. File: `types.py` — Tool dataclass + errors

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
    metadata: dict[str, Any] = field(default_factory=dict)
```

Error hierarchy:

```
ToolError                  # base
├── ToolRegistrationError  # name collision
├── ToolNotFoundError      # execute() called on unknown name
├── ToolExecutionError     # handler raised; __cause__ preserved
└── SkillLoadError         # missing SKILL.md, bad frontmatter, path escape
```

Callers catch `ToolError` to handle everything from the module.

---

## 5. File: `skill.py` — Anthropic Agent Skill loader

### 5.1 Eager frontmatter, lazy everything else

```python
class Skill:
    def __init__(self, directory):
        root = Path(directory).resolve()
        if not root.is_dir():
            raise SkillLoadError(f"skill path is not a directory: {root}")
        skill_md = root / "SKILL.md"
        if not skill_md.is_file():
            raise SkillLoadError(f"SKILL.md not found in {root}")
        raw = skill_md.read_text(encoding="utf-8")
        name, description, body = _parse_frontmatter(raw, source=str(skill_md))
        self.root, self.name, self.description, self._body = root, name, description, body
```

Layer 1 (frontmatter) is parsed at construction so `spec()` can emit it immediately. The body is stored but not returned until asked. Layer-3 files are not read until a specific `file` arg requests one.

### 5.2 The `to_tool()` shape

```python
_SKILL_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "file": {
            "type": "string",
            "description": (
                "Optional POSIX-style relative path to an additional file inside the "
                "skill directory (Layer 3). Omit to load the skill's main instructions "
                "(Layer 2: the SKILL.md body)."
            ),
        },
    },
}

def to_tool(self) -> Tool:
    return Tool(
        name=self.name,
        description=self.description,
        input_schema=_SKILL_INPUT_SCHEMA,
        source="skill",
        handler=self._handle,
        metadata={"root": str(self.root)},
    )
```

One schema serves every skill — the LLM learns one optional `file` knob, applies it to any skill tool.

### 5.3 Dispatch on the `file` arg

```python
async def _handle(self, arguments):
    file = arguments.get("file")
    if file is None or file == "":
        return self._body                      # Layer 2
    if not isinstance(file, str):
        raise SkillLoadError(...)
    return self._read_file(file)               # Layer 3
```

### 5.4 Secure Layer-3 reads

```python
def _read_file(self, relative_path):
    if relative_path.startswith("/") or relative_path.startswith("\\"):
        raise SkillLoadError(f"absolute paths are not allowed: {relative_path!r}")
    target = (self.root / relative_path).resolve()
    try:
        target.relative_to(self.root)          # raises if target escaped root
    except ValueError:
        raise SkillLoadError(...)
    if target == self.root / "SKILL.md":
        return self._body                      # serve from cache, not disk
    if not target.is_file():
        raise SkillLoadError(...)
    return target.read_text(encoding="utf-8")
```

Three guards: reject absolute paths, use `Path.resolve()` + `relative_to()` to catch `..` escapes, short-circuit any request for `SKILL.md` back to the cached body.

### 5.5 Frontmatter parsing

Hand-rolled — the YAML block is fenced with `---` lines and pyyaml handles the middle. We fail with a precise `SkillLoadError` at every point where the file could be malformed (missing opening fence, missing closing fence, non-mapping YAML, missing `name`, missing `description`, blank strings).

---

## 6. File: `mcp.py` — MCP stdio adapter

### 6.1 Lifecycle via `AsyncExitStack`

```python
class MCPClient:
    def __init__(self, *, command, args=None, env=None, cwd=None):
        self.command, self.args = command, (list(args) if args else [])
        self.env, self.cwd = env, cwd
        self._stack = AsyncExitStack()
        self._session: ClientSession | None = None

    async def enter(self) -> list[Tool]:
        params = StdioServerParameters(command=..., args=..., env=..., cwd=...)
        transport = await self._stack.enter_async_context(stdio_client(params))
        read, write = transport[0], transport[1]
        session = await self._stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        self._session = session
        listed = await session.list_tools()
        return [_wrap(t) for t in listed.tools]

    async def close(self):
        await self._stack.aclose()
        self._session = None
```

`AsyncExitStack` is the clean way to hold two nested async-context-managed resources (the stdio transport and the session) without writing nested `async with` blocks that have to branch on whether registration succeeded. One `aclose()` tears down both.

### 6.2 Tool wrapping

Each entry in `list_tools()` becomes a `Tool`:

```python
Tool(
    name=t.name,
    description=t.description or "",
    input_schema=_normalize_schema(t.inputSchema),
    source="mcp",
    handler=self._make_handler(t.name),
    metadata={"mcp_command": self.command},
)
```

`_normalize_schema` turns `None` into `{"type": "object", "properties": {}}` — the MCP spec allows null schemas but the LLM providers don't.

### 6.3 Handler forwarding

```python
def _make_handler(self, tool_name):
    async def handler(arguments):
        if self._session is None:
            raise ToolExecutionError(f"MCP session is not open; cannot call tool {tool_name!r}")
        try:
            result = await self._session.call_tool(tool_name, arguments)
        except Exception as e:
            raise ToolExecutionError(...) from e
        if getattr(result, "isError", False):
            raise ToolExecutionError(...)
        return _render_mcp_content(result)
    return handler
```

`_render_mcp_content` concatenates `text` fields from the `CallToolResult.content` blocks into one string. Non-text blocks (images, resources) are skipped for now.

---

## 7. File: `tools.py` — `ToolRegistry` facade

### 7.1 Decorator for Python functions

```python
def tool(self, func=None, *, name=None, description=None):
    def register(f):
        tool_name = name if name is not None else f.__name__
        doc = description if description is not None else (inspect.getdoc(f) or "")
        schema = _schema_from_signature(f)
        handler = _wrap_python_handler(f)
        self.register(Tool(name=tool_name, description=doc, input_schema=schema,
                           source="python", handler=handler))
        return f
    if func is None:
        return register
    return register(func)
```

Supports both `@registry.tool` (bare) and `@registry.tool(name="...", description="...")`. The decorator returns the original callable, so `f(...)` still works from normal Python code.

### 7.2 Schema derivation from signature

```python
_PY_TYPE_TO_JSON = {
    str: "string", int: "integer", float: "number",
    bool: "boolean", list: "array", dict: "object",
}

def _schema_from_signature(func):
    sig = inspect.signature(func)
    properties, required = {}, []
    for name, p in sig.parameters.items():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        annotation = p.annotation if p.annotation is not p.empty else str
        json_type = _PY_TYPE_TO_JSON.get(annotation, "string")
        properties[name] = {"type": json_type}
        if p.default is p.empty:
            required.append(name)
    schema = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema
```

Primitive annotations map to JSON-Schema primitives; anything exotic falls through to `"string"`. Defaults mark a param as optional. Good enough for 90% of real functions; advanced callers can pass a hand-written `input_schema` via `registry.register(Tool(...))`.

### 7.3 Sync-or-async wrapping

```python
def _wrap_python_handler(func):
    async def handler(arguments):
        if inspect.iscoroutinefunction(func):
            result = await func(**arguments)
        else:
            result = await asyncio.to_thread(lambda: func(**arguments))
        return _stringify(result)
    return handler
```

Sync tools run on a worker thread so that blocking I/O doesn't freeze the event loop. Results become strings because that's what the LLM protocol expects.

### 7.4 `spec()` — canonical form

```python
def spec(self):
    return [
        {"name": t.name, "description": t.description, "input_schema": t.input_schema}
        for t in self._tools.values()
    ]
```

This is the Claude/Anthropic tool shape. OpenAI's `{type: "function", function: {...}}` wrapping is the LLM adapter's job, not the tool registry's.

### 7.5 `execute()` — concurrent dispatch, never crashes

```python
async def execute(self, tool_calls):
    if not tool_calls:
        return []
    return await asyncio.gather(*(self._execute_one(tc) for tc in tool_calls))

async def _execute_one(self, tc):
    tool = self._tools.get(tc.name)
    if tool is None:
        return Message(role="tool",
                       content=f"ToolNotFoundError: no tool named {tc.name!r}",
                       tool_call_id=tc.id, name=tc.name)
    try:
        content = await tool.handler(tc.arguments)
    except ToolError as e:
        content = f"{type(e).__name__}: {e}"
    except Exception as e:
        content = f"ToolExecutionError: {type(e).__name__}: {e}"
    return Message(role="tool", content=content,
                   tool_call_id=tc.id, name=tc.name)
```

Two invariants:

- **Concurrency** — multiple tool calls in a single LLM turn run in parallel via `asyncio.gather`.
- **Durability** — every exception becomes a tool-role Message with a readable error string. Dispatch cannot raise to the caller.

### 7.6 `from_profile()` — the profile-driven path

```python
@classmethod
async def from_profile(cls, profile, *, base_dir=None):
    registry = cls()
    if base_dir is None:
        if profile.source_dir is None:
            raise ToolRegistrationError(...)
        base = profile.source_dir
    else:
        base = Path(base_dir).resolve()
    for skill_ref in profile.tools.skills:
        registry.add_skill((base / skill_ref).resolve())
    for mcp_cfg in profile.tools.mcp:
        await registry.add_mcp(command=mcp_cfg.command,
                               args=mcp_cfg.args,
                               env=mcp_cfg.env, cwd=mcp_cfg.cwd)
    return registry
```

Every skill path in `profile.tools.skills` is resolved against the profile's own directory (`profile.source_dir`, set by `AgentProfile.from_yaml`). This is what makes agent bundles isolated — agent A can't accidentally load agent B's skills, because A's profile only reads paths relative to A's own folder. Explicit sharing still works via `../..`-style relative paths.

### 7.7 Lifecycle

```python
async def __aenter__(self):           return self
async def __aexit__(self, *exc_info): await self.close()

async def close(self):
    for client in self._mcp_clients:
        await client.close()
    self._mcp_clients.clear()
```

`async with ToolRegistry()` is the recommended shape when any MCP server is registered; plain Python-function or skill registrations don't require the context manager.

---

## 8. Execution flow: a mixed-source turn

```
LLM returns ToolCall[ add(2,3), skill("rpt", {}), fs("read_file", {"path":"/tmp/x"}) ]
        │
        ▼
registry.execute(calls)
        │
        └── asyncio.gather(
              _execute_one(add):      add.handler({"a":2,"b":3})              → "5"
              _execute_one(rpt):      skill.handler({})                       → SKILL.md body
              _execute_one(fs):       mcp_handler({"path":"/tmp/x"})
                                        └─ session.call_tool("read_file", …)
                                        └─ render content[].text              → file contents
            )
        │
        ▼
[ Message(role="tool", content="5",    tool_call_id="1", name="add"),
  Message(role="tool", content="...",  tool_call_id="2", name="rpt"),
  Message(role="tool", content="...",  tool_call_id="3", name="fs") ]
        │
        ▼
LLM next turn: sees three tool results, continues reasoning
```

The LLM sees three tool results; the agent never needed to know that one came from local Python, one from a markdown skill, and one from a subprocess over stdio.

---

## 9. Test coverage map

| File | Tests | Focus |
|---|---|---|
| `test_skill.py` | 20 | All three layers + every security guard + every frontmatter failure mode |
| `test_registry.py` | 22 | Decorator forms, schema derivation, `spec()`, `execute()` dispatch, error wrapping, concurrency timing, async context manager |
| `test_mcp.py` | 5 | `stdio_client` + `ClientSession` patched with fakes; discovery, argument forwarding, `isError`, post-close behavior, content rendering |
| `test_from_profile.py` | 7 | `ToolRegistry.from_profile` against synthetic bundles + the real Maya bundle; source_dir resolution, parent-relative shared paths, in-memory profile fallback, MCP plumbing |

All 54 tests are fully offline — no network, no subprocess, no real MCP server.

---

## 10. Things worth noticing

- **One `Tool` dataclass unifies everything.** The facade dispatcher never branches on `source`. Adding a new source (e.g., an OpenAPI adapter) is: write a function that yields `Tool(...)` instances, call `registry.register(tool)`.

- **Errors become tool results, not crashes.** A malformed LLM tool call, a missing tool, a raised exception — all return as a readable string in a `role="tool"` Message. The LLM sees the error and decides.

- **Progressive disclosure is in the schema.** The `file` optional argument is the LLM's path from metadata (Layer 1) to body (Layer 2) to assets (Layer 3). No separate protocol, no global read-file tool.

- **Security on Layer-3 reads.** Absolute paths rejected, `Path.resolve()` + `relative_to()` catches every `..` traversal, direct requests for `SKILL.md` are served from the cached body.

- **`AsyncExitStack` owns MCP cleanup.** Two nested async resources (stdio transport + session) with one `aclose()` call. No nested `async with` blocks, no branch-on-registration-success ordering.

- **Sync tools run off the event loop** via `asyncio.to_thread`. Blocking `requests.get` in a user-defined tool won't stall the agent loop.

- **No new hard deps beyond `mcp`.** Everything else is stdlib + already-present (`pyyaml`, `pydantic`).
