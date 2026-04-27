# 模块 6 代码导览 —— 工具

> 配套阅读：[设计规范](../superpowers/specs_ZN/2026-04-24-module-06-tools-design.md)。设计规范记录了**我们决定了什么**以及**为什么**这样决定；本代码导览则说明**如何**在每个文件中落实这些决定。

---

## 核心类：`ToolRegistry`

从这里开始。本模块唯一的公共入口：

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

四条注册路径（三条手动 + 一条由档案驱动），一种统一的输出形态，一个分发器。

---

## 1. 本模块解决的问题

到模块 5 为止，本 harness 已经能够观察、记忆、检索和反思 —— 但它还不能*行动*。工具是智能体把 LLM 的输出转化为对世界产生影响的方式：调用你写的 Python 函数、触发一个 Claude Skill，或者通过 MCP 服务器去访问文件系统 / 数据库 / API。

有三件事让工具比看上去更棘手：

1. **来源异构。** 你代码库里的一个函数、一个用 markdown 打包好的技能、一个远程 MCP 服务器，它们对 LLM 来说都是"工具"，但在调用时行为完全不同。
2. **上下文资源稀缺。** 你不可能把每个技能的完整说明都塞进系统提示。Anthropic Agent Skills 用**渐进式披露**解决这个问题：LLM 默认只看到 `name + description`，只有当它判断技能相关时，才会把正文拉进来。
3. **生命周期。** MCP 服务器是子进程，带有必须被初始化并关闭的数据流。如果注册仅仅是"追加到一个列表里"，总会有东西泄露。

**模块 6 为 harness 提供：**

- `ToolRegistry` —— 一个门面类，三条注册路径，一个 `execute()` 调用。
- `Skill` —— 基于目录、带渐进式披露的加载器。
- `MCPClient` —— 借助 `AsyncExitStack` 显式管理生命周期的 stdio 适配器。

---

## 2. 目录结构图

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

依赖方向是单向的：`types` → `skill`、`mcp` → `tools` → `__init__`。

---

## 3. 塑造本设计的三个核心思想

### 3.1 用一个 `Tool` 数据类统一所有来源

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

一个由 Python 注册的函数、一个已加载的技能、一个由 MCP 发现的工具，全部流经 `registry._tools: dict[str, Tool]`。`source` 字段只是信息性的；真正执行的是 `handler`。门面类在分发时从不根据 source 做分支判断。

### 3.2 技能的渐进式披露被内置在输入 schema 里

LLM 不需要另一套协议来访问第三层的文件。技能的工具接受一个可选参数 `file`：

- 没有 `file` → 返回 SKILL.md 的正文（第二层）。
- `file="scripts/generate.py"` → 返回该文件的内容（第三层）。

第二层的正文可以自由地通过相对路径引用第三层文件（"见 `scripts/generate.py`"）；LLM 通过再次调用同一个工具来获取它们。

### 3.3 错误变成工具的返回结果，而不是崩溃

当某个工具的处理器抛出异常时，`execute()` 会把异常变成结果中那条 `role="tool"` Message 的 `content`。LLM 看到的是 `"ToolExecutionError: ValueError: bad input"`，把它当作一次普通的工具响应，然后自己决定如何应对。事件循环照常继续。这就是智能体在面对行为失常的工具时仍然能存活下来的方法。

---

## 4. 文件：`types.py` —— Tool 数据类 + 错误

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

错误层级：

```
ToolError                  # base
├── ToolRegistrationError  # name collision
├── ToolNotFoundError      # execute() called on unknown name
├── ToolExecutionError     # handler raised; __cause__ preserved
└── SkillLoadError         # missing SKILL.md, bad frontmatter, path escape
```

调用方通过捕获 `ToolError` 即可处理本模块抛出的所有错误。

---

## 5. 文件：`skill.py` —— Anthropic Agent Skill 加载器

### 5.1 frontmatter 先读取，其余一切都惰性加载

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

第一层（frontmatter）在构造时就被解析，这样 `spec()` 可以立刻把它发出去。正文被存起来，但直到被请求时才返回。第三层的文件在具体的 `file` 参数请求它之前都不会被读取。

### 5.2 `to_tool()` 的形态

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

一个 schema 服务所有技能 —— LLM 只需学会一个可选的 `file` 开关，就能把它应用到任何技能工具上。

### 5.3 按 `file` 参数分发

```python
async def _handle(self, arguments):
    file = arguments.get("file")
    if file is None or file == "":
        return self._body                      # Layer 2
    if not isinstance(file, str):
        raise SkillLoadError(...)
    return self._read_file(file)               # Layer 3
```

### 5.4 安全的第三层读取

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

三重防护：拒绝绝对路径；用 `Path.resolve()` + `relative_to()` 捕获 `..` 形式的逃逸；任何对 `SKILL.md` 的请求都直接短路回缓存的正文。

### 5.5 frontmatter 解析

手工实现 —— YAML 块用 `---` 行作为围栏，中间部分交给 pyyaml 处理。在文件可能格式错误的每一个点，我们都用精准的 `SkillLoadError` 报错（缺失起始围栏、缺失结束围栏、YAML 不是映射、缺 `name`、缺 `description`、出现空字符串）。

---

## 6. 文件：`mcp.py` —— MCP stdio 适配器

### 6.1 用 `AsyncExitStack` 管理生命周期

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

`AsyncExitStack` 是持有两个嵌套的异步上下文资源（stdio 传输层和会话）的干净方式，不必写那种还得根据注册是否成功做分支的嵌套 `async with` 块。一次 `aclose()` 就能同时拆除两者。

### 6.2 工具封装

`list_tools()` 的每一项都会被包装成一个 `Tool`：

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

`_normalize_schema` 会把 `None` 转成 `{"type": "object", "properties": {}}` —— MCP 规范允许空 schema，但 LLM 提供商不允许。

### 6.3 处理器的转发

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

`_render_mcp_content` 会把 `CallToolResult.content` 各个块里的 `text` 字段拼接成一个字符串。目前暂且跳过非文本块（图像、资源）。

---

## 7. 文件：`tools.py` —— `ToolRegistry` 门面类

### 7.1 用于 Python 函数的装饰器

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

支持 `@registry.tool`（裸用）和 `@registry.tool(name="...", description="...")` 两种形式。装饰器返回原始的可调用对象，所以普通 Python 代码里调用 `f(...)` 依然可用。

### 7.2 从签名推导 schema

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

原始类型注解会映射到 JSON-Schema 原始类型；任何更奇特的类型都会回落到 `"string"`。默认值会把参数标记为可选。这对 90% 的真实函数已经够用；高级调用方可以通过 `registry.register(Tool(...))` 传入手写的 `input_schema`。

### 7.3 同步或异步的统一封装

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

同步工具在工作线程上运行，这样阻塞式 I/O 就不会冻结事件循环。结果被转成字符串，因为 LLM 协议就是这么要求的。

### 7.4 `spec()` —— 规范形态

```python
def spec(self):
    return [
        {"name": t.name, "description": t.description, "input_schema": t.input_schema}
        for t in self._tools.values()
    ]
```

这就是 Claude/Anthropic 的工具形态。把它包装成 OpenAI 的 `{type: "function", function: {...}}` 是 LLM 适配器的职责，不是工具注册表的职责。

### 7.5 `execute()` —— 并发分发，永不崩溃

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

两个不变式：

- **并发性** —— 在 LLM 一个回合里的多次工具调用通过 `asyncio.gather` 并行执行。
- **持久性** —— 任何异常都会被转成带有可读错误字符串的 tool 角色 Message。分发过程永远不会向调用方抛异常。

### 7.6 `from_profile()` —— 由档案驱动的路径

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

`profile.tools.skills` 里的每一个技能路径都会相对于档案自身的目录（`profile.source_dir`，由 `AgentProfile.from_yaml` 设置）进行解析。这就是智能体 bundle 保持隔离的原因 —— 智能体 A 不可能意外加载到智能体 B 的技能，因为 A 的档案只会读取相对于 A 自己文件夹的路径。需要显式共享时，仍然可以通过 `../..` 这种相对路径实现。

### 7.7 生命周期

```python
async def __aenter__(self):           return self
async def __aexit__(self, *exc_info): await self.close()

async def close(self):
    for client in self._mcp_clients:
        await client.close()
    self._mcp_clients.clear()
```

当注册了任何 MCP 服务器时，推荐使用 `async with ToolRegistry()` 的形态；纯 Python 函数或技能的注册并不需要上下文管理器。

---

## 8. 执行流程：一次混合来源的回合

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

LLM 看到的是三个工具结果；智能体完全不需要知道其中一个来自本地 Python、一个来自 markdown 技能，还有一个来自通过 stdio 通信的子进程。

---

## 9. 测试覆盖图

| 文件 | 测试数 | 关注点 |
|---|---|---|
| `test_skill.py` | 20 | 全部三层 + 所有安全防护 + 所有 frontmatter 失败模式 |
| `test_registry.py` | 22 | 装饰器形式、schema 推导、`spec()`、`execute()` 分发、错误包装、并发时序、异步上下文管理器 |
| `test_mcp.py` | 5 | 用伪实现 patch `stdio_client` + `ClientSession`；发现、参数转发、`isError`、关闭后行为、内容渲染 |
| `test_from_profile.py` | 7 | `ToolRegistry.from_profile` 针对合成 bundle 和真实的 Maya bundle；source_dir 解析、父级相对的共享路径、内存中的 profile 回退、MCP 接入 |

全部 54 个测试完全离线 —— 不联网、不启子进程、不起真实的 MCP 服务器。

---

## 10. 值得留意的细节

- **一个 `Tool` 数据类统一一切。** 门面分发器从不根据 `source` 做分支。增加一个新的来源（比如一个 OpenAPI 适配器）只需要：写一个产出 `Tool(...)` 实例的函数，再调用 `registry.register(tool)`。

- **错误变成工具结果，而不是崩溃。** 畸形的 LLM 工具调用、找不到的工具、被抛出的异常 —— 全部都会作为一条可读的字符串返回到 `role="tool"` Message 里。LLM 看到错误，自己决定怎么办。

- **渐进式披露就在 schema 里。** 可选的 `file` 参数就是 LLM 从元数据（第一层）到正文（第二层）再到素材（第三层）的路径。不需要独立的协议，也不需要一个全局的 read-file 工具。

- **第三层读取上的安全。** 拒绝绝对路径，`Path.resolve()` + `relative_to()` 捕获所有 `..` 遍历，直接请求 `SKILL.md` 会从缓存的正文返回。

- **`AsyncExitStack` 负责 MCP 的清理。** 两个嵌套的异步资源（stdio 传输层 + 会话），只需要一次 `aclose()` 调用。不需要嵌套的 `async with` 块，也不需要根据注册是否成功决定次序。

- **同步工具在事件循环之外运行**，通过 `asyncio.to_thread` 实现。用户自定义工具里阻塞式的 `requests.get` 不会卡住智能体循环。

- **除了 `mcp` 之外没有新增硬依赖。** 其他一切要么是标准库，要么是已经存在的依赖（`pyyaml`、`pydantic`）。
