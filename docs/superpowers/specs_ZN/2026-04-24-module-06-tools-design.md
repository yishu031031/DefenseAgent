# 模块 6 —— Tools（注册表 + 适配器）设计规范

**日期：** 2026-04-24
**状态：** Draft
**模块位置：** N 个模块中的第 6 个。这是首个让 LLM 能够*作用于*外部世界（而不仅仅是输出文本）的模块。

## 目的

为 harness 提供一个统一的注册表，既能向 LLM 呈现一组异构的工具，又能把工具调用分发回正确的处理器。三条注册路径，一个统一接口：

1. **用户自定义的 Python 函数** —— 通过 `@registry.tool` 装饰器，通过反射函数签名构建 JSON schema。
2. **Anthropic 风格的 Agent Skills** —— 目录中包含一个 `SKILL.md`（frontmatter + 正文）以及可选的资源文件，以**渐进式披露**方式提供，使技能内容永远不会膨胀系统提示。
3. **MCP（Model Context Protocol）服务器** —— 通过 stdio 启动的子进程，在连接时发现其工具列表，并通过 MCP 会话转发调用。

这三者最终都产生相同的内存结构（`Tool` 数据类），并走同一条 `execute(tool_calls) → list[Message]` 路径。LLM 模块和反思模块对工具来源一无所知。

第四条、由档案驱动的入口一次性为某个智能体构建整个注册表：

4. **`ToolRegistry.from_profile(profile)`** —— 读取 `profile.tools.skills` 和 `profile.tools.mcp`，将技能路径解析为相对于档案所在目录（`profile.source_dir`）的路径，并返回一个完全填充好的 `ToolRegistry`。这是一个智能体在 YAML 中声明其工具集的方式。

## 范围

**包含：**
- `ToolRegistry` 门面类，提供三条注册路径 + 一个规范化的 `spec()` 发射器 + 一个并发的 `execute()` 分发器。
- 渐进式披露的技能加载（第一层 frontmatter / 第二层正文 / 第三层资源文件）。
- MCP stdio 适配器（通过官方的 `mcp` Python SDK），带有显式的生命周期管理。
- 规范化的 `Tool` 数据类 + 错误层级（`ToolError`、`ToolRegistrationError`、`ToolNotFoundError`、`ToolExecutionError`、`SkillLoadError`）。

**不包含（延后）：**
- MCP 的 **SSE / HTTP** 传输 —— 先做 stdio。
- MCP 的 **prompts / resources** —— 当前只做 `tools` 发现。
- 每个工具的**权限门**（白名单、用户确认钩子）—— 在接入智能体循环时再加。
- **工具结果缓存** / 记忆化。
- **流式**工具结果。

## 三条注册路径

### 1. `@registry.tool` —— 用户自定义的 Python 函数

```python
registry = ToolRegistry()

@registry.tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

# also:  @registry.tool(name="plus", description="Adds.")
```

- **Name** 默认为函数的 `__name__`；可被覆盖。
- **Description** 默认为函数的 docstring；可被覆盖。
- **Input schema** 通过遍历签名推导：注解 → JSON 类型，默认值 → 可选字段，无默认值 → `required`。
- 同时支持同步和异步函数。同步函数通过 `asyncio.to_thread` 被转到工作线程中执行，这样阻塞式 I/O 就不会卡住事件循环。

### 2. `registry.add_skill(directory)` —— 带渐进式披露的 Anthropic Agent Skills

Anthropic Agent Skills 之所以存在，是因为上下文是稀缺资源。一个技能是一个*目录*，不是一个文件。这种设计坚持三层结构：

| 层级 | 内容 | 何时进入上下文 |
|---|---|---|
| **1 — 元数据** | SKILL.md frontmatter 中的 `name` + `description` | 始终；每一轮都出现在 `registry.spec()` 中 |
| **2 — 指令** | SKILL.md 的正文（frontmatter 之后的 markdown） | 仅当 LLM 在不带 `file` 参数的情况下调用该技能工具时 |
| **3 — 资源** | 技能目录中的任何其他文件（脚本、模板、扩展参考资料） | 仅当 LLM 以 `{"file": "rel/path"}` 再次调用该技能工具时 |

注册表把一个技能暴露为**一个工具**，其 input schema 为

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

LLM 学到的是一个统一的协议：“不带参数调用该技能以获取指令；再次调用时带上 `file=...` 以拉取被引用的资源文件。” 第二层正文可以通过相对路径自由引用第三层文件，而这些引用自然可用，因为同一个工具名 + 一个新的 `file` 参数就能把它们取回来。

**安全性：** 第三层的读操作会把给定路径解析为相对于技能根目录的路径，并在任何越界尝试（`..`、绝对路径）时抛出 `SkillLoadError`。对 `SKILL.md` 的请求会路由到已缓存的正文，而不是磁盘上的文件。

### 3. `ToolRegistry.from_profile(profile, *, base_dir=None)` —— 档案驱动

每个智能体都以独立 bundle 的形式存在于 `agents/<agent_id>/` 下，包含 `profile.yaml` + `skills/` + （可选）智能体需要的其他任何文件。档案声明了它的工具集：

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

代码路径：

```python
profile = AgentProfile.from_yaml("agents/maya_rodriguez/profile.yaml")
async with await ToolRegistry.from_profile(profile) as registry:
    ...   # registry is populated with Maya's skills + MCP servers — nothing else
```

技能路径会相对于 `profile.source_dir` 解析，而 `AgentProfile.from_yaml` 会将该字段设置为档案文件的父目录。这让智能体 bundle 具备自包含性和可移植性（把整个 bundle 拷到别处，相对路径依然有效）。对于内存中的档案（没有 `source_dir`），除非显式传入 `base_dir`，否则会抛出 `ToolRegistrationError`。

**隔离属性：** 两个使用 `from_profile` 的智能体不会意外看到对方的技能 —— 每个档案只会读取位于其自身目录下（或通过 `../` 显式上溯）的路径。

### 4. `registry.add_mcp(command=..., args=[...])` —— MCP stdio 服务器

```python
async with ToolRegistry() as registry:
    await registry.add_mcp(command="uvx", args=["mcp-server-filesystem", "/tmp"])
    # all filesystem-server tools are now in registry.spec()
```

- 通过官方 `mcp` SDK 打开 stdio 连接 + `ClientSession`，完成初始化，并调用 `list_tools()`。
- 每个发现到的工具都会被包装为一个 `Tool`，其 handler 会把调用转发给该服务器的 `session.call_tool(name, arguments)`。
- 结果会把 `CallToolResult.content[].text` 扁平化为单个字符串交给 LLM。`isError=True` 会抛出 `ToolExecutionError`。
- 生命周期通过 `MCPClient` 内部的 `AsyncExitStack` 管理 —— `registry.close()`（或异步 `__aexit__`）会拆除注册表打开的每一个 MCP client。

## 数据模型

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

每条注册路径最终都产生一个这样的对象；门面类的分发器对所有三种一视同仁。

### 错误层级

```
ToolError                  # base
├── ToolRegistrationError  # name collision, duplicate registration
├── ToolNotFoundError      # execute() called on unknown tool name
├── ToolExecutionError     # tool handler raised; __cause__ preserved
└── SkillLoadError         # missing SKILL.md, bad frontmatter, path escape
```

## 执行语义

```python
messages = await registry.execute(tool_calls)
```

- `tool_calls: list[ToolCall]`（来自 `DefenseAgent.llm.types` 的 `ToolCall` 数据类，每个 LLM 适配器都已经会发出它）。
- 返回 `list[Message]`，每次调用对应一条，`role="tool"`，保留 `tool_call_id`，`name` 设为工具名，`content` 设为工具的字符串输出。
- **并发性：** `execute()` 用 `asyncio.gather` 收集所有调用，因此相互独立的工具调用会并行执行。
- **错误绝不会让分发崩溃。** 无论是找不到工具、抛出 `ToolError`，还是未预期的异常，都会变成一条结构完整的 `Message` 的 `content`。LLM 把错误当作正常的工具结果看待，可以自行决定如何恢复。

## 文件布局

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

依赖关系：`types` ← `skill`、`mcp` ← `tools`；`tools` 还依赖 `DefenseAgent.config.profile.AgentProfile`（单向依赖，叶子模块）。

## 依赖

**一个新增的运行时依赖：** `mcp>=1.0.0`（Anthropic 官方的 MCP Python SDK，Apache-2.0 许可）。

其余要么来自标准库（`asyncio`、`inspect`、`contextlib.AsyncExitStack`、`pathlib`），要么已经存在（`pyyaml` 用于 frontmatter，`pydantic` 通过 MCP SDK 间接引入）。

## 与更早模块的集成

| 模块 | 如何使用 | 是否需要修改该模块？ |
|---|---|---|
| 模块 1（LLM） | 复用 `DefenseAgent.llm.types` 中的 `Message` 和 `ToolCall`。后续：当 `chat(..., tools=...)` 接通时，LLM 适配器会消费 `registry.spec()`。 | 否 |
| 模块 2（Config） | `AgentProfile.tools`（`ToolsConfig` + `MCPServerConfig`）声明了一个智能体需要的每个技能 + MCP 服务器；`ToolRegistry.from_profile` 会消费它。技能路径相对于 `profile.source_dir` 解析。 | 是 —— 已于 2026-04-24 新增 |
| 模块 3（Ops） | 一个日志智能体可以包装 `registry.execute()` 来发出 `tool.call_started` / `tool.call_finished` 事件。可选。 | 否 |
| 模块 4（Memory） | 不使用。工具结果可以在更高层通过 `memory.remember` 以 `observation` 记录的形式存储。 | 否 |
| 模块 5（Reflection） | 不直接使用。 | 否 |

零回改。

## 测试策略

- **Skill** 测试使用由 `tmp_path` 支撑的技能目录，每个测试都重新构建；三个披露层级和每一项安全防护都会被覆盖。
- **Registry** 测试涵盖装饰器的两种形式（裸 `@registry.tool` 和 `@registry.tool(...)`）、命名冲突、schema 推导、`execute()` 正常路径、错误信息、并发性（一项 `time.monotonic` 断言）以及异步上下文管理器。
- **MCP** 测试永远不启动真实服务器。它们用返回 `asynccontextmanager` 的伪实现，patch 掉 `DefenseAgent.tools.mcp.stdio_client` 和 `DefenseAgent.tools.mcp.ClientSession`，这些伪实现会 yield 一个 `_FakeSession`，记录每一次调用。这样就能在不受子进程脆弱性影响的情况下，验证适配器的线级契约（initialize → list_tools → call_tool）。

所有测试都保持完全离线。

## 开放问题

- **Skill loader 要不要 watch 模式？** 如果 SKILL.md 的正文在磁盘上发生改动，我们不会发现（正文在加载时被缓存）。很可能没问题 —— 技能在文件系统中是版本化的，重新注册成本很低。如果以后确实造成麻烦，再重新讨论。
- **MCP 环境变量传递。** 目前 `env` 已经通过 `StdioServerParameters` 打通，但我们传的是 `None`，以便继承父进程环境。未来可能需要一个“只透传白名单中的环境变量”的策略。
- **复杂注解的 schema 推导。** 我们处理了简单的原始类型；其他类型一律回退到 `"string"`。以后当真实用户触到边界时，可以再补一遍 `Optional`、`Literal` 和 Pydantic 模型的支持。
