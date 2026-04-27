# DefenseAgent

> 中文 · [English README](README.md)

一个用 YAML profile 构建单 Agent LLM 应用的 Python 工具箱。一份 profile 描述 agent,一行 Python 实例化,三种执行策略可选。

```python
from DefenseAgent.agent import AgentConfig, ReActAgent

config = AgentConfig(profile="agents/example_agent/profile.yaml")
agent  = ReActAgent(config)
result = await agent.run("用一句话总结今天的计划。")
```

## 特性

- **单文件 agent 定义。** 身份、LLM 厂商、工具、memory、RAG、system prompt —— 全部写在一份严格校验的 YAML 里(`extra="forbid"`,未知字段会在加载时抛 `ConfigValidationError`)。
- **按字段的配置 fallback。** 每个值都能在 profile 或 `.env` 中设置;profile 优先,`.env` 补缺。切换 LLM 厂商(`openai`、`anthropic`、`deepseek`、`qwen`、`google`、`vllm`)无需改代码。
- **三种 agent 策略。** `SimpleAgent`(单轮)、`ReActAgent`(工具调用循环)、`PlanAndSolveAgent`(规划→执行→综合)。三者从同一份 `AgentConfig` 构造。
- **三种工具来源,统一一个 registry。** 本地 skill 目录(Anthropic 风格的 `SKILL.md` 包)、MCP 服务器(stdio / SSE / WebSocket / streamable-http)、Python 函数(profile 中按文件路径或点分模块引用)。
- **持久化 memory + 内置工具。** mem0 + Qdrant 落盘存储;agent 自动暴露 `memory_recall` 工具给 LLM。`ContextCompressor` 在每次 LLM 调用前裁剪工作上下文。
- **可选 RAG + 内置工具。** 把文档放进目录,设置 `rag.enabled: true`,获得 `rag_search` 工具。Embedder 凭证遵循同样的按字段 profile→env fallback。
- **多模态输入。** `agent.run(task, images=[...])` 发送 OpenAI 风格的 content-block 消息。每张图片接受本地文件路径、`http(s)://` URL、或 `data:` URL。所有 OpenAI 兼容厂商都能直接消费;Anthropic 适配器收到 list 形态 content 时会抛出明确的 `LLMAdapterError`。
- **可依赖注入。** LLM、memory、tools、reflector、compressor、logger 都能通过 `AgentConfig` 替换,方便测试和自定义接线。
- **离线测试套件。** 跑 `pytest` 不需要网络或外部服务。

## 安装

```bash
git clone https://github.com/yishu031031/DefenseAgent.git
cd DefenseAgent
conda create -n agent_lab python=3.12 -y
conda activate agent_lab
pip install -r requirements.txt
```

## 配置

在仓库根目录建立 `.env`,最少这几项:

```bash
AGENT_LAB_LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-…
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1

EMBEDDING_API_KEY=sk-…
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMS=1536

TAVILY_API_KEY=…    # 可选,scripts/react_tools_memory_demo.py 会使用
```

每个字段的解析顺序:profile YAML → 环境变量 → schema 默认值。仅含空白字符的值视为未设置。

### 厂商与凭证

`AGENT_LAB_LLM_PROVIDER` 选择适配器。每个厂商都有自己的 `<PROVIDER>_*` 块(`<PROVIDER>_API_KEY`、`<PROVIDER>_MODEL`、`<PROVIDER>_BASE_URL`)。跨厂商的 `LLM_API_KEY` / `LLM_MODEL_ID` / `LLM_BASE_URL` 会在设置时覆盖 per-provider 那一层。

| Provider | 适配器 | Key 典型格式 | 默认 base URL | 可选 chat 模型示例 |
|---|---|---|---|---|
| `openai` | `OpenAICompatibleAdapter` | `sk-…` 或 `sk-proj-…` | `https://api.openai.com/v1` | `gpt-4o-mini`、`gpt-4o`、`o3-mini` |
| `anthropic` | `AnthropicAdapter` | `sk-ant-…` | `https://api.anthropic.com` | `claude-sonnet-4-6`、`claude-opus-4-7` |
| `deepseek` | `OpenAICompatibleAdapter` | `sk-…` | `https://api.deepseek.com/v1` | `deepseek-chat`、`deepseek-reasoner` |
| `qwen`(DashScope OpenAI 兼容) | `OpenAICompatibleAdapter` | `sk-…` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-plus`、`qwen-vl-max`、`qwen-vl-plus` |
| `google`(OpenAI 兼容端点) | `OpenAICompatibleAdapter` | `sk-…` | `https://generativelanguage.googleapis.com/v1beta/openai` | `gemini-2.0-flash` |
| `vllm`(自部署) | `OpenAICompatibleAdapter` | 任意字符串(常见 `EMPTY` / `token-not-needed`) | 取决于部署,例如 `http://localhost:8000/v1` | 取决于 vLLM 服务挂的什么模型 |

Embedding 单独配 `EMBEDDING_*` 块。常见组合:

| Embedder | `EMBEDDING_BASE_URL` | `EMBEDDING_MODEL` | `EMBEDDING_DIMS` |
|---|---|---|---|
| OpenAI | `https://api.openai.com/v1` | `text-embedding-3-small` | 1536 |
| OpenAI | `https://api.openai.com/v1` | `text-embedding-3-large` | 3072 |
| DashScope | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `text-embedding-v3` | 1024 |
| ModelScope | `https://api-inference.modelscope.cn/v1` | `Qwen/Qwen3-Embedding-0.6B` | 1024 |
| ModelScope | `https://api-inference.modelscope.cn/v1` | `Qwen/Qwen3-Embedding-8B` | 4096 |

`EMBEDDING_DIMS` **必须** 与模型实际输出向量维度一致,否则 Qdrant collection 会拒绝写入 —— 按模型文档列的向量长度填。

## 快速开始

```python
import asyncio
from DefenseAgent.agent import AgentConfig, ReActAgent

config = AgentConfig(profile="agents/example_agent/profile.yaml")

async def main():
    async with ReActAgent(config) as agent:
        result = await agent.run("用一句话总结今天的计划。")
        print(result.final_answer)

asyncio.run(main())
```

端到端 demo(calculator + Tavily web search + memory recall):

```bash
python scripts/react_tools_memory_demo.py
```

## 一步步搭一个 Agent

把 `agents/example_agent/` 复制一份并改 `profile.yaml`。`agent:` 下每一块都可选,身份字段除外。所有字段都被 pydantic 严格校验(`extra="forbid"`)。

### `llm:`

```yaml
llm:
  provider:           # str | null。可选值:openai | anthropic | deepseek | qwen | google | vllm。回退 AGENT_LAB_LLM_PROVIDER。
  model:              # str | null。厂商特定的 model id(见上面的厂商表)。回退 <PROVIDER>_MODEL 或 LLM_MODEL_ID。
  base_url:           # str | null。厂商端点。回退 <PROVIDER>_BASE_URL 或 LLM_BASE_URL。
  api_key:            # str | null。回退 <PROVIDER>_API_KEY。共享 profile 时建议留空。
```

四个字段都是 `str | None`,各自独立 fallback 到 `.env`。仅含空白字符的值视为未设置 —— 半改一半的 YAML 不会盖掉正确的 env。

### 身份(必填)

```yaml
id: "agent_001"     # str, min_length=1。在 mem0 里作为 agent_id,也是日志文件名。
name: "Nova Patel"  # str, min_length=1。{name} 占位符。
age: 27             # int ≥ 0。
traits: "..."       # str, min_length=1。自由格式的特征列表。
backstory: "..."    # str, min_length=1。
initial_plan: "..." # str, min_length=1。
```

每个字段去空白后非空。六个字段都作为 `{id} {name} {age} {traits} {backstory} {initial_plan}` 占位符出现在 prompt 模板中。

### `cognitive:`

```yaml
cognitive:
  max_steps_per_cycle: 10     # int ≥ 1,默认 10。每次 run() 中 ReAct 工具调用循环的上限。
  reflection_threshold: 5     # int ≥ 1,默认 5。触发 Reflector.maybe_reflect() 的未反思记忆数量。
  importance_threshold: 7     # float ∈ [1, 10],默认 7。Reflection 中"重要"记忆的阈值。
  planning_horizon: "1 day"   # str, min_length=1,默认 "1 day"。自由格式;在 prompt 中暴露给 LLM。
```

### `memory:`

```yaml
memory:
  is_retrieve: true                       # bool,默认 true。打开后会注册 memory_recall 工具。
  history_mode: add                       # 'add' | 'overwrite'。'overwrite' 启用 diff/rollback。
  search_limit: 10                        # int ≥ 1,默认 10。memory_recall 单次返回的最大记录数。
  ignore_roles: [tool, system]            # list[str],默认 ['tool', 'system']。这些 role 不会落盘。
  ignore_fields: [reasoning_content]      # list[str],默认 ['reasoning_content']。
  context_limit: 128000                   # int ≥ 1024,默认 128000。ContextCompressor 触发裁剪的 token 阈值。
  prune_protect: 40000                    # int ≥ 0,默认 40000。裁剪时永远不动的 token 数。
  prune_minimum: 20000                    # int ≥ 0,默认 20000。裁剪后保留的最少 token 数。
  reserved_buffer: 20000                  # int ≥ 0,默认 20000。安全余量。
  enable_summary: true                    # bool,默认 true。允许 ContextCompressor 调 LLM 总结老的对话轮。
  storage_path:                           # str | null。默认 <profile_dir>/memory/。
```

mem0 + 本地 Qdrant。注册 `memory_recall` 工具。`ContextCompressor` 每次 LLM 调用前都会运行一次。

### `rag:`

```yaml
rag:
  enabled: false                          # bool,默认 false。改 true 才会接入 LlamaIndexRAG + rag_search。
  documents_dir: rag_corpus               # str | null。相对 profile 目录。第一次 run() 自动建索引。
  storage_dir: rag_index                  # str | null。FAISS 索引的持久化路径。
  embedding_provider: openai              # 'openai' | 'huggingface',默认 'openai'。
  embedding:                              # str | null。→ EMBEDDING_MODEL。
  embedding_api_key:                      # str | null。→ EMBEDDING_API_KEY。
  embedding_base_url:                     # str | null。→ EMBEDDING_BASE_URL。
  embedding_dims:                         # int ≥ 1, null。→ EMBEDDING_DIMS。
  chunk_size: 512                         # int ≥ 1,默认 512。切块时每块的 token 数。
  chunk_overlap: 50                       # int ≥ 0,默认 50。相邻块之间的 token 重叠。
  top_k: 5                                # int ≥ 1,默认 5。rag_search 默认的 top_k。
  score_threshold: 0.0                    # float ∈ [0.0, 1.0],默认 0.0。低于此分数的结果丢弃。
  retrieve_only: true                     # bool,默认 true。改 false 时 RAG 也会综合一个回答。
  use_huggingface: false                  # bool,默认 false。ms-agent 的 HF 下载路径。
```

`enabled: true` 时注册 `rag_search` 工具。Embedder 字段使用与 `llm:` 相同的按字段 profile→env fallback。

### `tools:`

```yaml
tools:
  skills:                                 # list[str]。skill 目录路径,相对 profile 目录。
    - skills/tabular-report
  mcp:                                    # list[MCPServerConfig]。
    - command: uvx                        # str | null。stdio server 必填。
      args: [mcp-server-filesystem, /tmp] # list[str],默认 []。
      env: { TOKEN: "" }                  # dict[str,str] | null。空值会从进程环境变量插值。
      cwd:                                # str | null。可选工作目录。
      include: [read_file]                # list[str]。白名单;与 exclude 互斥。
      exclude: []                         # list[str]。黑名单。
    - transport: sse                      # 'stdio' | 'sse' | 'websocket' | 'streamable_http'。
      url: https://mcp.example.com/sse    # str | null。transport 不是 stdio 时必填。
      headers: { Authorization: "..." }   # dict[str,str] | null。
      timeout: 30                         # float ≥ 0 | null。连接超时(秒)。
      sse_read_timeout: 300               # float ≥ 0 | null。SSE 长轮询超时。
  python:                                 # list[str]。Python entry-point 字符串。
    - python_tools/calc.py:calculator
    - my_pkg.search:web_search
  allow_skill_execution: false            # bool,默认 false。打开后 skill 中的脚本变可执行 Tool。
  skill_execution_timeout: 300            # int ≥ 1,默认 300。子进程超时(秒)。
```

每个 MCP entry 必须只设置 `command:`(stdio)或 `url:`(网络)之一。每个 server 的 `include` 和 `exclude` 互斥。

#### Python 工具文件放在哪里

`tools.python:` 接受两种形式:

**1. 相对文件路径。** 路径相对当前 profile 目录解析,通过 `importlib.util.spec_from_file_location` 加载。无需配置 `sys.path`。

```
agents/example_agent/
├── profile.yaml
├── python_tools/
│   └── calc.py            # def calculator(expression: str) -> str
└── skills/
```

profile 中的写法:`python_tools/calc.py:calculator`。

**2. 点分模块路径。** 模块必须能被运行中的 Python 解释器导入。通过 `importlib.import_module` 解析。

```
my_pkg/
├── __init__.py
└── search.py              # def web_search(query: str) -> str
```

profile 中的写法:`my_pkg.search:web_search`。

两种形式下,函数的类型注解会成为 tool 的输入 schema,docstring 会成为 tool 的描述。

#### 在代码中注册(不写进 profile)

```python
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression."""
    ...

config = AgentConfig(profile="…", tools=[calculator])
```

#### Skill 执行

`allow_skill_execution: true` 时,skill 中每个脚本(`scripts/*.py`、`*.sh`、`*.js`)会被注册为单独的可执行 Tool,命名 `<skill_name>__<script_stem>`。基于子进程,通过 `SkillContainer` 执行,带从上游继承的危险模式守卫。

### `prompt:`

```yaml
prompt:
  path: prompts/system.md         # str | null。相对 profile 目录的文件。
  system:                         # str | null。inline 形式,与 path 互斥。
  extra_instructions:             # str | null。追加在身份块之后。
```

优先级:inline `system:` > `path:` > 自动生成的身份块。模板内可用占位符(通过 `str.format` 渲染):`{id} {name} {age} {traits} {backstory} {initial_plan}`。模板格式坏掉时回退到自动生成的身份块,而不是让 run 崩掉。

## 内置工具

除了你在 `tools:` 下注册的工具,agent 还会自动暴露这些给 LLM:

| 工具 | 注册时机 | 输入 schema | 作用 |
|---|---|---|---|
| `memory_recall` | `memory.is_retrieve: true` 时 | `{query: string, top_k?: int (1–20,默认 5)}` | 在该 agent 的 `(user_id, agent_id, run_id)` 过滤下对 mem0 做语义检索。返回最多 top_k 条记录,渲染为 `- [<memory_type>] <content>` 列表。 |
| `rag_search` | `rag.enabled: true` 时 | `{query: string, top_k?: int}` | 在 RAG 索引上做向量检索。返回分数高于 `score_threshold` 的排序结果。 |
| `<skill>`(每 skill 一个) | 每个 `tools.skills:` 条目一个 | `{file?: string}` | 不传 `file` → 返回 skill 的 SKILL.md 正文。传 `file` → 从 skill 目录返回指定文件,带路径逃逸守卫。 |
| `<skill>__<script>`(每脚本一个) | `allow_skill_execution: true` 时 | `{args?: list[str], stdin?: string, timeout?: int}` | 通过 `SkillContainer` 把脚本作为子进程运行。返回 stdout + stderr + 退出码,渲染给 LLM。 |

## Agent 类

| 类 | 行为 | 适用场景 |
|---|---|---|
| `SimpleAgent` | 每次 `run()` 一次 LLM 调用,无工具循环。 | 纯聊天 agent,不需要工具调用。 |
| `ReActAgent` | 工具调用循环。LLM 返回纯文本或达到 `max_steps` 时停。 | 带工具的 agent 默认选这个。 |
| `PlanAndSolveAgent` | 规划 → 逐步执行 → 综合。 | 长任务,先规划能减少混乱。 |

三种类都接受同一个 `AgentConfig`,共享 `BaseAgent` 的辅助方法。

`agent.run(task, max_steps=None, images=None)`:
- `task: str` —— 用户请求。
- `max_steps: int | None` —— 覆盖 `cognitive.max_steps_per_cycle`(本次调用)。`SimpleAgent` 忽略此参数。
- `images: list[str | Path] | None` —— 见"多模态输入"章节。

返回类型:`AgentResult`。

```python
@dataclass
class AgentResult:
    task: str                      # 原始任务字符串
    final_answer: str              # LLM 给出的最终纯文本回答
    steps: list[AgentStep]         # 完整的 ReAct 轨迹,每个事件一条
    usage: TokenUsage              # 整轮 run 的累计 token 计数
    stopped_reason: Literal["answered", "max_steps"] = "answered"

@dataclass
class AgentStep:
    index: int
    kind: Literal["plan", "tool_call", "tool_result", "answer"]
    content: str = ""              # "answer" / "tool_call" 步:LLM 的文本
    tool_calls: list[ToolCall] = ...    # "tool_call" 步:LLM 请求的调用
    tool_results: list[Message] = ...   # "tool_result" 步:每个调用对应一条 role='tool' Message
    usage: TokenUsage | None = None     # 单次 LLM 调用的 token 计数(tool_result 步为 None)
```

## 多模态输入

三种 agent 的 `run()` 都接受可选的 `images=` 参数:

```python
from pathlib import Path

result = await agent.run(
    "这张图里是什么?跟下面这张 URL 比一下。",
    images=[
        Path("./screenshot.png"),
        "https://example.com/photo.jpg",
    ],
)
```

提供 `images` 时,user 这一轮以 OpenAI content-block 列表形态发出:

```python
[{"type": "text", "text": "<task>"},
 {"type": "image_url", "image_url": {"url": "<resolved-url-1>"}},
 {"type": "image_url", "image_url": {"url": "<resolved-url-2>"}}]
```

每张图片可以是:

| 输入 | 处理方式 |
|---|---|
| `Path` 或本地文件路径字符串 | 读取后 base64 编码,生成 `data:<mime>;base64,…`。MIME 根据扩展名推断,未知扩展名默认 `image/png`。 |
| `http://` 或 `https://` URL | 原样透传。 |
| `data:` URL | 原样透传。 |

厂商兼容性:

- **OpenAI 兼容适配器**(DashScope 上的 Qwen、DeepSeek-VL、GLM、Kimi、托管多模态模型的 vLLM、OpenAI 自身)直接消费这种 list 形态。把 `llm.model:` 设为视觉模型即可。
- **Anthropic 适配器** 收到 list content 时抛 `LLMAdapterError` 并附带明确说明。`Message` 类型本身已经支持 list content,后续要加 Claude 视觉只需在 Anthropic 适配器内部做局部改动。

`ReActAgent` 只在最初的 user turn 携带图片,后续的 tool 结果消息保持纯文本。`PlanAndSolveAgent` 在 Phase 1 的规划消息和 Phase 2 的每一步执行消息都携带相同的图片,这样每个引用原始任务的阶段都能再看图。

## 架构

```
AgentConfig ── profile.yaml + .env
     │
     ▼
build_components_sync ── LLM, Memory, ToolRegistry, Reflector, Compressor, Logger
     │
     ▼
BaseAgent ◀──── ReActAgent | SimpleAgent | PlanAndSolveAgent
     │
     ▼
run(task) ──► AgentResult { final_answer, steps[], usage }
```

`build_components_sync` 同步执行。MCP server 连接和可选的 RAG 索引在第一次 `run()` 时按需构建(它们是 async 的)。

## 模块布局

| 路径 | 内容 |
|---|---|
| `DefenseAgent/config/profile.py` | `AgentProfile`、`LLMConfig`、`MemoryConfig`、`RAGConfig`、`ToolsConfig`、`MCPServerConfig`、`PromptConfig` |
| `DefenseAgent/llm/` | `LLM` 外观,OpenAI 兼容 + Anthropic 适配器 |
| `DefenseAgent/memory/` | mem0 memory + `ContextCompressor` |
| `DefenseAgent/tools/` | `ToolRegistry`、`MCPClient` |
| `DefenseAgent/skills/` | `SkillLoader`、`SkillContainer`、`to_tools()` 适配器 |
| `DefenseAgent/rag/` | `LlamaIndexRAG`、profile 桥接 |
| `DefenseAgent/reflection/` | `Reflector` |
| `DefenseAgent/agent/` | `BaseAgent`、`SimpleAgent`、`ReActAgent`、`PlanAndSolveAgent`、`AgentConfig`、`_builder` |

memory、MCP、skill、RAG 模块均继承自 [ms-agent](https://github.com/modelscope/ms-agent) 的上游类。

## Demo

```bash
python scripts/react_tools_memory_demo.py     # ReAct + calculator + Tavily + memory recall
python scripts/profile_chat_demo.py           # 用 example_agent profile 跑一次单轮对话
python scripts/tools_demo.py                  # 演示 skill 工具的三层
python scripts/memory_demo.py                 # mem0 add / search / dump
```

## 测试

```bash
pytest                       # 全套,离线运行
pytest -k tools              # 只跑某个模块
pytest -x --tb=short         # 第一次失败就停
```

531 个测试,3 个 skip。

## License

MIT.
