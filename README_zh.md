# DefenseAgent

> 中文 · [English README](README.md)

**面向生产场景、可组合的 Python Agent SDK。** 一个 YAML 配置文件 + **两行代码**就能跑起来；切换 LLM 厂商不用改代码;自定义 tools / memory / RAG 后端零侵入；测试时 mock 注入干净。

```python
from DefenseAgent import create_agent

agent  = create_agent("agents/example_agent/profile.yaml")
result = await agent.run("我今天上午做了什么？")
print(result.final_answer)
```

---

## 简介

**DefenseAgent 是一个用于构建"做实际工作"的 LLM Agent 的 Python SDK** —— 调用工具、记住对话、检索知识库、做多步推理。它的目标是让工程师把 agent **部署到生产环境**，而不是停留在 notebook 原型。

### 它解决什么问题

Agent 开发反复撞到同样四堵墙。DefenseAgent 对每堵墙都给出了直接答案：

| 痛点 | DefenseAgent 的方案 |
|---|---|
| **厂商绑定** —— 多数 agent 库与 OpenAI 的 API 形状深度耦合，切到 Claude / DeepSeek / Qwen / 自部署 vLLM 都要大改。 | 统一的 `LLM` 外观。改 `.env` 或调 `LLM.create(provider=..., model=...)` 即可切换；上游代码零改动。 |
| **无状态对话** —— 默认 LLM 在 turn 之间什么都记不住。 | mem0 后端的持久化 memory，把语义检索做成内置工具；外加 reflection 机制，把累积经验提炼成长期洞察。 |
| **脆弱的集成** —— Python tools、MCP 服务器、Skill 包各有一套胶水代码。 | 一个 `ToolRegistry` 同时接受纯 Python 函数、Anthropic 风格 **Skills**、**MCP** 服务器（stdio / SSE / streamable-http）—— 在 LLM 看来三者无差别。 |
| **不可测试** —— 多数框架把 LLM 客户端写死在内部，写真实测试时很痛。 | 每个子系统（LLM / memory / tools / RAG / reflection）都通过 `AgentConfig` 可替换。测试时注入 `ScriptedLLM` + mock memory，生产代码不动。 |

### 设计原则

每一个 API 决策都遵循三条原则：

1. **唯一构造路径。** 每个 agent 都从同一个 `AgentConfig` 对象构造。没有"图方便"的第二条构造路径 —— 重载是清晰度的天敌。
2. **配置而非 fork。** 每个扩展点都是一个 Protocol 或 ABC + 一个注册器。要加自定义 LLM 厂商、Memory 后端、RAG extractor，写一个类注入即可 —— 永远不需要改 SDK 源码。
3. **默认值来自最简 profile，定制走同一组字段。** 两行 `create_agent("profile.yaml")` 用的是高级用户显式配置的同一组 `AgentConfig` 字段。没有第二套隐藏 API。

### 目标用户

- **应用开发者**做领域助手（客服、研究分析、编程 agent），需要开箱即用的 memory + tool 调用 + 多厂商容灾。
- **AI 研究者**做多步推理策略原型，想要一个能给出完整 trace、支持 reflection、不被单一厂商绑死的干净底座。
- **内部工具团队**需要 MCP / Skill 集成、持久化 memory，以及随成本 / 质量需求切换模型的能力。

### 开箱内容

- 三种 agent 策略 —— `SimpleAgent`、`ReActAgent`、`PlanAndSolveAgent`
- 六家 LLM 厂商接好 —— Anthropic + 所有 OpenAI 兼容端点（DeepSeek / Qwen / Google 走代理 / vLLM / OpenAI 自己）
- 持久化 memory（mem0 + qdrant）+ reflection
- 多模态 RAG（LlamaIndex），HTML / PDF 切块时保留嵌入图片和表格
- 三种工具来源归一到同一个 registry：Python 函数、Skill 包、MCP 服务器
- pydantic 校验的 YAML profile schema，拼错字段名立刻报错
- 每个 agent 一份的 JSON-lines logger
- hatchling 构建的 wheel + sdist + `py.typed` marker —— 干净 venv 里 `pip install ".[all]"` 一次到位

---

## 目录

- [简介](#简介)
- [为什么用 DefenseAgent](#为什么用-defenseagent)
- [安装](#安装)
- [5 分钟上手](#5-分钟上手)
- [Part A — 使用 SDK](#part-a--使用-sdk)
  - [A.1  Profile YAML 字段详解](#a1-profile-yaml-字段详解)
  - [A.2  接入自己的 LLM](#a2-接入自己的-llm)
  - [A.3  接入自己的 Tools（Python 函数）](#a3-接入自己的-toolspython-函数)
  - [A.4  写自己的 Prompt](#a4-写自己的-prompt)
  - [A.5  加 Memory](#a5-加-memory)
  - [A.6  加 MCP 服务器和 Skills](#a6-加-mcp-服务器和-skills)
  - [A.7  调用 Agent —— 全部调用形式](#a7-调用-agent--全部调用形式)
- [Part B — 扩展 SDK](#part-b--扩展-sdk)
  - [B.1  自定义 LLM 厂商（含多模态）](#b1-自定义-llm-厂商含多模态)
  - [B.2  自定义 Memory 后端](#b2-自定义-memory-后端)
  - [B.3  自定义 Tool / MCP / Skill 后端](#b3-自定义-tool--mcp--skill-后端)
  - [B.4  自定义 RAG Extractor + Renderer](#b4-自定义-rag-extractor--renderer)
- [三种 Agent 推理策略](#三种-agent-推理策略)
- [模块布局](#模块布局)
- [Demos](#demos)
- [测试](#测试)
- [Roadmap](#roadmap)
- [许可证](#许可证)

---

## 为什么用 DefenseAgent

| 能力 | 说明 |
|---|---|
| **厂商无关的 LLM 抽象** | `LLM` 单一外观封装 Anthropic 和所有 OpenAI 兼容厂商（DeepSeek、Qwen/DashScope、ModelScope、vLLM、Google 走代理）。改 `.env` 或调 `LLM.create(provider=...)` 即切换。 |
| **三种推理策略** | `SimpleAgent`（单次问答）、`ReActAgent`（Yao 2022——交错推理 + 工具调用）、`PlanAndSolveAgent`（Wang 2023——先规划后执行再综合）。三者共用同一个 `AgentConfig`。 |
| **持久化 Memory** | mem0 + qdrant 落盘。跨轮 `memory_recall` 是内置 LLM 工具；outcome / failure / trajectory 都打了 `memory_type` 标签便于过滤和反思。 |
| **多模态 RAG** | `LlamaIndexRAG` + `StructuredDocExtractor`，HTML / PDF 切块时保留嵌入图片和表格。可启用向量 + BM25 混合检索。 |
| **三种工具来源** | 普通 Python 函数（`@registry.tool`）、Anthropic 风格的 **Skill** 包（含 `SKILL.md` 的目录）、stdio / SSE / streamable-http **MCP** 服务器——全部归一到一个 `ToolRegistry`。 |
| **Reflection** | `Reflector` 收集未反思 memory，让 LLM 综合提炼洞察后写回。阈值触发，永不抛异常。 |
| **唯一构造路径** | 每个 Agent 都通过 `Agent(AgentConfig(...))` 构造。测试时通过 `AgentConfig.llm` / `memory` / `tool_registry` / `reflector` 字段注入 mock。 |
| **开放打包** | `pip install -e ".[all]"`、MIT 许可、py.typed 类型标记、hatchling 构建 wheel + sdist。 |

---

## 安装

### 可编辑安装（当前推荐）

```bash
git clone https://github.com/yishu031031/DefenseAgent.git
cd DefenseAgent

python -m venv .venv
.venv/Scripts/activate           # Windows
# source .venv/bin/activate      # macOS / Linux

pip install -e ".[all,dev]"      # core + memory + RAG + MCP + 测试
```

### 选择 extras

核心安装只装 LLM + profile + tools——足够跑 `SimpleAgent`。Memory、RAG、MCP 都是 opt-in，避免不必要时拖入 `torch` / `qdrant-client` / `llama-index`：

```bash
pip install -e .                  # 仅核心 —— LLM + profile + tools
pip install -e ".[memory]"        # + mem0 + qdrant + fastembed
pip install -e ".[rag]"           # + llama-index + pdfplumber + bs4
pip install -e ".[mcp]"           # + MCP 客户端
pip install -e ".[all]"           # 全部用户面向能力
pip install -e ".[all,dev]"       # + pytest
```

> 全栈装下来约 1 GB（包含 `ms-agent`、`mem0ai`、`llama-index-core`、`qdrant-client`、被 modelscope 拖进来的 `torch`）。

PyPI 发布（`pip install defense-agent`）已在路线图最前面——wheel 和 sdist 已经能用 `python -m build` 干净构建。

---

## 5 分钟上手

### 第 1 步——填 `.env`

复制 `.env.example` → `.env`，**至少填这三块**：

```bash
# 1. 选 chat 厂商 —— 决定 agent 用哪个适配器
AGENT_LAB_LLM_PROVIDER=deepseek          # 取值：anthropic | openai | deepseek | qwen | vllm | google

# 2. 对应的密钥块 —— per-provider 的 <PROVIDER>_* 三个变量必填
DEEPSEEK_API_KEY=sk-...                  # 从厂商拿到的 API key
DEEPSEEK_BASE_URL=https://api.deepseek.com   # 厂商的 REST 端点
DEEPSEEK_MODEL=deepseek-chat             # 每次 chat 请求传给厂商的 model id

# 3. Embedding —— mem0 必需，RAG 可选；chat 厂商通常不带 embedding 接口
EMBEDDING_API_KEY=sk-...                 # embedding 厂商的 API key（通常和 chat 是不同账号）
EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1   # OpenAI 兼容端点
EMBEDDING_MODEL=text-embedding-v3        # embedding 模型 id
EMBEDDING_DIMS=1024                      # 该模型输出的向量维度 —— 必须和模型规格匹配
```

**优先级**——`LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL_ID`（跨厂商覆盖）优先级高于 `<PROVIDER>_API_KEY` 等（按厂商）。

### 第 2 步——选一个 profile

仓库里 [`agents/example_agent/profile.yaml`](agents/example_agent/profile.yaml) 是带完整注释的参考 profile，列出了所有可配字段及其默认值。

### 第 3 步——跑起来

```python
import asyncio
from DefenseAgent import create_agent

async def main():
    agent  = create_agent("agents/example_agent/profile.yaml")
    result = await agent.run("现在下午两点。你今天都做了什么？")
    print(result.final_answer)
    await agent.close()

asyncio.run(main())
```

就这样。一个带 memory、tool 支持、RAG 接口预留的 ReAct-loop agent 就建好了。

---

## Part A — 使用 SDK

整个 SDK 的设计目标：**最简单的写法就是正确的写法**。

```python
agent  = create_agent("agents/example_agent/profile.yaml")  # 一行构造
result = await agent.run("Hi")                              # 一行调用
```

下面所有内容都是教你**只动 YAML、不写 Python** 就能调出的旋钮。

### A.1 Profile YAML 字段详解

[`agents/example_agent/profile.yaml`](agents/example_agent/profile.yaml) 既是一个能跑的 profile，**也是文档**。下面是带注释的精简漫游版；完整字段去看那个文件本身。

```yaml
agent:

  # --- LLM（per-agent 覆盖；按字段回退 .env） --------------------------
  # 每个字段都 *可选*。留空时按下列顺序回退到 .env：
  #   provider:  AGENT_LAB_LLM_PROVIDER
  #   model:     LLM_MODEL_ID  >  <PROVIDER>_MODEL
  #   api_key:   LLM_API_KEY   >  <PROVIDER>_API_KEY
  #   base_url:  LLM_BASE_URL  >  <PROVIDER>_BASE_URL
  llm:
    provider:                 # 选择适配器：deepseek | anthropic | openai | qwen | google | vllm
    model:                    # 厂商接受的 model id，如 deepseek-chat / claude-opus-4-7
    base_url:                 # OpenAI 兼容接口的 base URL；anthropic 适配器忽略此字段
    api_key:                  # 共享 profile 里留空 —— 让 .env 里的 <PROVIDER>_API_KEY 注入

  # --- 身份（必填） ----------------------------------------------------
  # 这 6 个字段会被插值进 system prompt：
  # {id} {name} {age} {traits} {backstory} {initial_plan}
  id: "example_agent_001"     # 稳定的字符串 id；同时作为日志文件名 + memory 的 user_id
  name: "Nova Patel"          # 显示名；填到 prompt 的 {name}
  age: 27                     # int；填到 {age}
  traits: "好奇、严谨、坦率"   # 简短的逗号分隔特质 —— 填到 {traits}
  backstory: >                # 多行背景介绍 —— 填到 {backstory}
    一名转去做 AI 研究的现场工程师，最近搬到了 Lakeside。
  initial_plan: >             # agent "今天在做什么" —— 填到 {initial_plan}
    起床后看告警，午饭前做数据分析，下午跟团队站会。

  # --- 认知循环参数（可选；显示默认值） --------------------------------
  cognitive:
    max_steps_per_cycle: 10        # ReAct 单次 run() 工具调用轮次硬上限 —— 防止死循环
    reflection_threshold: 5        # 未反思 memory 累积到 N 条时，run() 后触发 Reflector
    importance_threshold: 7        # 1–10 的重要性分；只有 ≥ 此阈值的 memory 会被保留
    planning_horizon: "1 day"      # 自由文本，注入 prompt 让 LLM 知道在为多长的时间窗规划
                                   # （"1 hour"、"1 day"、"this week" …）。

  # --- Memory（mem0 后端；显示默认值） --------------------------------
  memory:
    is_retrieve: true              # 总开关：启用 mem0 + 注册 memory_recall 工具
    history_mode: add              # 'add' = 每轮追加  |  'overwrite' = mem0 diff / 回滚模式
    search_limit: 10               # 单次 memory_recall 返回记录数上限
    storage_path:                  # qdrant 目录；留空 → <profile_dir>/memory/

  # --- RAG（LlamaIndex 后端；默认关） ---------------------------------
  rag:
    enabled: false                 # 打开则注册 LlamaIndexRAG + rag_search 工具
    documents_dir: "rag_corpus"    # 源文档（HTML/PDF/MD）目录 —— 相对 profile 目录
    storage_dir: "rag_index"       # FAISS / 向量索引落盘目录
    embedding_provider: openai     # 'openai'（或兼容 base_url） | 'huggingface'（本地模型）
    chunk_size: 512                # 切块时每块的 token 数 —— 越小越精细但占用越多
    top_k: 5                       # 单次 rag_search 返回 passage 数

  # --- Tools（skills + MCP 服务器） -----------------------------------
  tools:
    skills:                        # 本地 Skill 目录列表 —— 每个目录必须包含 SKILL.md
      - skills/tabular-report      # 路径相对 profile 目录
    mcp: []                        # MCP 服务器启动配置列表 —— 完整 schema 见 A.6
    allow_skill_execution: false   # opt-in：true 时 Skill 包内每个脚本都变成可调用 Tool
                                   # （在带超时的子进程沙箱里执行）

  # --- 系统 prompt -----------------------------------------------------
  # 优先级：`system`（内联） > `path`（文件） > 自动构造的身份块。
  prompt:
    path: prompts/system.md        # markdown 模板路径（相对 profile 目录） —— 或用 `system: "..."` 内联
    extra_instructions: |          # 追加到解析后的 system prompt 末尾 —— 适合叠语气/格式规则
      简短作答。先讲结论。
```

**校验**：每一块都是 `extra="forbid"`——拼错一个字段名会立即报 `ConfigValidationError` 并指出具体路径。

**一个 profile 是一个目录**——`profile.yaml` 加上可选的兄弟文件：

```
agents/example_agent/
├── profile.yaml          # 你传给 create_agent() 的那个文件
├── prompts/
│   └── system.md         # 外置的系统 prompt 模板
├── skills/
│   └── tabular-report/   # Anthropic 风格的 Skill 包
│       ├── SKILL.md
│       ├── scripts/
│       └── templates/
├── memory/               # 首次运行自动创建；mem0 数据存这
└── logs/                 # 自动创建；每个 profile.id 一个 .log
```

### A.2 接入自己的 LLM

**方案 1——改 `.env`**（推荐；零 Python 改动）：

```bash
AGENT_LAB_LLM_PROVIDER=qwen              # 切换 active 适配器为 qwen
QWEN_API_KEY=sk-...                      # DashScope 的 API key
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1   # OpenAI 兼容端点
QWEN_MODEL=qwen-plus                     # model id（qwen-plus / qwen-max / qwen-turbo / ...）
```

**方案 2——在 YAML 里 per-agent 覆盖：**

```yaml
agent:
  llm:
    provider: anthropic
    model: claude-opus-4-7
    api_key:                       # 留空 —— 让 .env 守密
```

**方案 3——纯 Python（完全跳过 `.env`）：**

```python
from DefenseAgent import AgentConfig, create_agent
from DefenseAgent.llm import LLM

llm = LLM.create(
    provider="anthropic",
    api_key="sk-...",
    model="claude-opus-4-7",
)
agent = create_agent(AgentConfig(
    profile="agents/example_agent/profile.yaml",
    llm=llm,
    load_env=False,
))
```

**开箱即用厂商：** `anthropic`、`openai`、`deepseek`、`qwen`、`vllm`、`google`。任何 OpenAI 兼容接口都能走 `openai` 适配器（设好 `base_url` 即可）。

要新增一个全新的厂商类（比如多模态视觉），见 [B.1 自定义 LLM 厂商](#b1-自定义-llm-厂商含多模态)。

### A.3 接入自己的 Tools（Python 函数）

任何 Python 可调用对象都能当 tool——函数的**名字 + docstring + 签名**会被自动转换成 LLM 看到的 JSON-schema 工具规约：

```python
from DefenseAgent import create_agent

def calculator(expression: str) -> str:
    """计算一个数学表达式并返回数值结果。"""
    return str(eval(expression))           # 仅作演示

def web_search(query: str, top_k: int = 3) -> str:
    """搜索 web，返回前 top_k 条片段拼接成的字符串。"""
    ...

agent = create_agent({
    "profile": "agents/example_agent/profile.yaml",
    "tools": [calculator, web_search],
})
```

LLM 在 ReAct 循环里发出匹配的 `tool_use` 请求时，agent 就会调到这些函数。

> ### 类型注解规范
> 注册器会读签名——参数类型尽量保持简单（`str`、`int`、`float`、`bool`、`list`、`dict`）。带默认值的会被标 `optional`，否则 `required`。

### A.4 写自己的 Prompt

**YAML 内联：**

```yaml
agent:
  prompt:
    system: |
      你是 {name}，一名 {age} 岁、{traits} 的现场工程师。
      {backstory}
      ---
      汇报日志时永远先讲异常。
```

**外置模板文件**——长 prompt 不污染 YAML、版本控制 diff 干净：

```yaml
agent:
  prompt:
    path: prompts/system.md
```

```markdown
# agents/example_agent/prompts/system.md
你是 {name}，一名 {age} 岁、{traits} 的现场工程师。

# 背景
{backstory}

# 今天
{initial_plan}

# 行为准则
- 用第一人称、简洁地回答。
- 当需要回忆之前对话时，调用 `memory_recall`。
- 工具失败时简短承认后继续，不要纠结。
```

**可用占位符：** `{id}`、`{name}`、`{age}`、`{traits}`、`{backstory}`、`{initial_plan}`——都从身份块里插值。

**追加式 extra_instructions**——在共享基础 prompt 上叠语气/格式规则时很有用：

```yaml
agent:
  prompt:
    path: prompts/system.md
    extra_instructions: |
      汇报日志时先讲异常，再补一句上下文，永远不要把重点压在最后。
```

### A.5 加 Memory

`profile.memory.is_retrieve = true`（默认）时 Memory 就是**默认开**的。你不用写 Python——agent 自动：

1. 每次 `run()` 后存 `(问题 → 回答)`（标 `memory_type='outcome'`）
2. 每次工具调用存 `(call → result)`（标 `memory_type='trajectory'`）
3. 暴露一个 `memory_recall` 工具供 LLM 推理时调用

**默认存储路径** `<profile_dir>/memory/`——qdrant collection 落盘在这。

**查看存了什么：**

```bash
python scripts/dump_memory.py agents/example_agent/
```

**完全禁用 Memory**（无状态 agent，不需要 embedding 配置）：

```python
agent = create_agent(AgentConfig(
    profile="agents/example_agent/profile.yaml",
    use_memory=False,
))
```

**程序化配置 mem0 后端**——一行 `.env` 都不想要时：

```python
from DefenseAgent.memory import MemoryBackendConfig

# MemoryBackendConfig 装的是 mem0 自身做"事实抽取 + embedding"用的凭证 ——
# 与 agent 的 chat LLM（前面单独配的那个）相互独立。
backend = MemoryBackendConfig(
    llm_provider="deepseek",                          # mem0 抽事实用哪家厂商
    llm_api_key="...",                                # mem0 LLM 的 API key
    llm_model="deepseek-chat",                        # mem0 LLM 的 model id
    llm_base_url="https://api.deepseek.com",          # mem0 LLM 的 REST 端点
    embedding_api_key="...",                          # embedding 厂商的 API key
    embedding_model="text-embedding-3-small",         # embedding 模型 id
    embedding_base_url="https://api.openai.com/v1",   # OpenAI 兼容 embedding 端点
    embedding_dims=1536,                              # 向量维度，必须和 embedding 模型匹配
)
agent = create_agent(AgentConfig(
    profile="...",
    memory_backend=backend,                           # 让 memory 完全绕过 .env
    load_env=False,                                   # 也不要尝试读 .env
))
```

### A.6 加 MCP 服务器和 Skills

#### Skills

一个 **Skill** 就是一个目录，含 `SKILL.md` 和可选的 scripts/templates——Anthropic 推的可移植包格式。把目录放到 `agents/<name>/skills/` 下，在 YAML 里引用：

```yaml
agent:
  tools:
    skills:                              # Skill 路径列表 —— 相对 profile 目录
      - skills/tabular-report            # 单个 Skill 包
      - skills/                          # 或 父目录 —— 内部所有 SKILL.md 都被发现并注册
    allow_skill_execution: false         # opt-in：true 时，Skill 包内每个脚本都变成可调用 Tool
    skill_execution_timeout: 300         # 子进程超时秒数（脚本在沙箱里跑）
```

最简 `SKILL.md`：

```markdown
---
name: tabular-report                     # LLM 看到的 tool 名字（必须唯一）
description: 把行字典渲染成 Markdown 表格。   # 一行描述，展示给 LLM 决策时用
---
# Tabular Report
当你有一组同结构的字典行需要展示给用户时，用这个 skill...
```

skill 的 metadata 会变成 LLM 可用的工具。当 `allow_skill_execution: true` 时，包里 `scripts/` 下每个 `.py`/`.sh` 自动变成独立 Tool，在带超时的子进程沙箱里运行。

#### MCP 服务器

Model Context Protocol 让 agent 通过 stdio、SSE、websocket 或 streamable-http 与**外部工具服务器**通信。在 YAML 里加：

```yaml
agent:
  tools:
    mcp:
      # ----- stdio 服务器（最常见 —— uvx/npx 启动的本地进程） -----
      - command: uvx                      # 启动命令（如 uvx、npx、python）
        args: [mcp-server-filesystem, /tmp]  # 传给 `command` 的参数列表
        env:                              # 传给子进程的环境变量
          TOKEN:                          # 空值 → 从父进程环境插值
        include: [read_file]              # 工具名白名单；与 `exclude:` 互斥

      # ----- 远程网络服务器（SSE / streamable-http / websocket） -----
      - transport: sse                    # 'sse' | 'streamable_http' | 'websocket'（默认 streamable_http）
        url: https://mcp.example.com/sse  # 端点 URL（网络传输必填）
        headers:                          # HTTP 头；`${VAR}` 从进程环境展开
          Authorization: "Bearer ${MCP_TOKEN}"
        timeout: 30                       # 连接/请求超时秒数
```

每个服务器暴露的每个工具都被并入同一个 `ToolRegistry`——LLM 看不出 Python 函数、Skill、MCP 工具有什么差别。

### A.7 调用 Agent —— 全部调用形式

```python
import asyncio
from DefenseAgent import create_agent, AgentConfig, ReActAgent

# ----- 1. profile 路径 ------------------------------------------------
agent = create_agent("agents/example_agent/profile.yaml")

# ----- 2. dict（自动转给 AgentConfig） --------------------------------
agent = create_agent({
    "profile": "agents/example_agent/profile.yaml",
    "tools":   [calculator],
    "use_rag": True,
})

# ----- 3. AgentConfig（完全控制） -------------------------------------
config = AgentConfig(
    profile="agents/example_agent/profile.yaml",
    tools=[calculator],
    use_rag=True,
    extra_instructions="始终用 JSON 格式回答。",
)
agent = create_agent(config, strategy="plan_and_solve")  # 或 "react" / "simple"

# ----- 4. 显式构造 —— async-context 自动清理 --------------------------
async with ReActAgent(config) as agent:
    result = await agent.run("计算 [1, 2, 3, 4] 的标准差")
    print(result.final_answer)
    for step in result.steps:
        print(f"[{step.kind}]", step.content)
```

`AgentResult.steps` 给出完整轨迹——每次工具调用、每次 LLM 输出、每次错误——便于调试和事后复盘。

---

## Part B — 扩展 SDK

每个子系统都是**一个 Protocol 或 ABC + 一个注册器**。实现接口、注册、就完事——不用 fork。

### B.1 自定义 LLM 厂商（含多模态）

实现 `LLMAdapter`，注册即可：

```python
# my_app/vision_adapter.py
from typing import AsyncIterator
from DefenseAgent.llm.base import LLMAdapter
from DefenseAgent.llm.types import LLMResponse, Message, StreamChunk

class GeminiVisionAdapter(LLMAdapter):
    """多模态适配器 —— 接受 message 里包含 image_url 的多 part。"""

    def __init__(self, *, api_key: str, model: str, base_url: str = ""):
        self.client = ...   # 厂商的 SDK
        self.model  = model

    async def chat(
        self, messages: list[Message], *, tools=None,
        temperature: float = 0.7, max_tokens: int = 1024, system: str | None = None,
    ) -> LLMResponse:
        # 1) 把 `Message`（包括多模态的 image_url part）翻译成厂商请求结构
        # 2) 调用厂商接口，记录 token usage 和 stop_reason
        # 3) 返回标准化的 `LLMResponse(content=..., tool_calls=..., usage=..., stop_reason=...)`
        ...

    async def chat_stream(self, messages, *, tools=None, **kw) -> AsyncIterator[StreamChunk]:
        # 可选 override；基类默认通过 buffer chat() 自动实现
        ...
```

```python
# 接线方式 —— 两种：

# A) 手动构造，通过 AgentConfig.llm 注入
from DefenseAgent import AgentConfig, create_agent
from DefenseAgent.llm import LLM
from my_app.vision_adapter import GeminiVisionAdapter

llm = LLM(GeminiVisionAdapter(api_key="...", model="gemini-2.5-pro-vision"))
agent = create_agent(AgentConfig(profile="...", llm=llm, load_env=False))

# B) 全局注册，让 .env / profile.yaml 按名字选
from DefenseAgent.llm._registry import _ADAPTERS    # 非正式扩展点
_ADAPTERS["gemini-vision"] = GeminiVisionAdapter
# 然后 .env 里：    AGENT_LAB_LLM_PROVIDER=gemini-vision
```

### B.2 自定义 Memory 后端

继承 `Memory` ABC——只需要实现 `run(messages) -> messages`：

```python
from DefenseAgent.memory import Memory
from DefenseAgent.llm.types import Message

class RedisMemory(Memory):
    """玩具示例 —— 把每个 user/assistant 消息存到 Redis 列表里。"""

    def __init__(self, profile, *, redis_url: str):
        super().__init__(profile)
        import redis.asyncio as redis
        self.r   = redis.from_url(redis_url)
        self.key = f"agent:{profile.id}:history"

    async def run(self, messages: list[Message]) -> list[Message]:
        # 1) ingest —— 把新消息尾部持久化
        # 2) retrieve —— 可选地把更早的上下文从 Redis 拉回来注入
        # 3) 返回重写后的消息列表
        for msg in messages[-2:]:
            await self.r.rpush(self.key, msg.model_dump_json())
        return messages
```

```python
agent = create_agent(AgentConfig(
    profile="...",
    memory=RedisMemory(profile, redis_url="redis://localhost:6379"),
))
```

### B.3 自定义 Tool / MCP / Skill 后端

最干净的扩展方式就是写一个 Python 函数——见 [A.3](#a3-接入自己的-toolspython-函数)。其它结构化场景：

**预构 `ToolRegistry`**——多个 agent 共享同一个工具注册表时：

```python
from DefenseAgent.tools import ToolRegistry

registry = ToolRegistry()

@registry.tool
def calculator(expression: str) -> str:
    """计算数学表达式。"""
    return str(eval(expression))

@registry.tool(name="search", description="Web 搜索。")
def google(query: str, top_k: int = 3) -> str: ...

agent = create_agent(AgentConfig(profile="...", tool_registry=registry))
```

**自定义 MCP 传输**——实现 `MCPClient` 的 connect / list_tools / call_tool 接口，通过 registry 的低层 API 注入。代码见 [DefenseAgent/tools/mcp.py](DefenseAgent/tools/mcp.py)。

### B.4 自定义 RAG Extractor + Renderer

DefenseAgent 的 RAG 有**三个可插拔层**：

1. **`StructuredExtractor`** —— 把文件解析成 chunk + 资源（图片、表格……）
2. **`ResourceRenderer`** —— 把已存的资源渲染成 LLM 可读文本
3. **`LlamaIndexRAG`** 本身 —— 注册以上任意一种

#### 自定义 extractor（例：`.docx`）

```python
# scripts/extras/docx_extractor.py —— 现成可复制
from pathlib import Path
from DefenseAgent.rag.extraction import StructuredChunk, StructuredResource

class DocxExtractor:
    def __init__(self, resources_dir: Path):
        self.resources_dir = resources_dir

    def supports(self, source) -> bool:
        return Path(source).suffix.lower() == ".docx"

    def extract(self, source) -> list[StructuredChunk]:
        # 按文档顺序遍历段落 / 内联图片 / 表格；
        # 把图片落到 <resources_dir>/<source_hash>/；
        # 按 Heading-1/2 切成一个个 StructuredChunk。
        ...
```

```python
from DefenseAgent.rag import LlamaIndexRAG, StructuredDocExtractor
from scripts.extras.docx_extractor import DocxExtractor

extractor = StructuredDocExtractor(profile)
extractor.register(DocxExtractor(resources_dir=extractor.resources_dir))
rag = await LlamaIndexRAG.from_profile(profile, extractor=extractor)
```

#### 自定义 renderer（例：`kind="csv"`）

```python
# scripts/extras/csv_renderer.py —— 现成可复制
from DefenseAgent.rag.extraction import StructuredResource

class CsvRenderer:
    kind = "csv"

    async def render(self, resource: StructuredResource) -> str:
        import pandas as pd
        df = pd.read_csv(resource.path)
        return f"csv [{resource.id}]\n\n{df.to_markdown(index=False)}"
```

```python
rag.register_renderer(CsvRenderer())
# LLM 一旦调用 rag_get_resource(rid="...csv...")，CsvRenderer 接管。
```

> 两个现成的扩展样板放在 [`scripts/extras/`](scripts/extras/) —— `csv_renderer.py` 和 `docx_extractor.py`。直接拿来当模板用。

---

## 三种 Agent 推理策略

| 类 | 循环 | 何时用 |
|---|---|---|
| `SimpleAgent` | 一次 LLM 调用，无工具，无循环 | 纯对话、角色扮演、单轮 QA |
| `ReActAgent` | LLM → 工具调用 → 结果 → LLM → ... 直到纯文本回答 | 最通用；推理与工具调用交错 |
| `PlanAndSolveAgent` | 拆成 N 个步骤 → 逐步执行（带工具）→ 综合回答 | 多步任务，先定计划能减少漂移 |

三者都接受同一个 `AgentConfig`，都返回同一个 `AgentResult`（含 `final_answer` 和完整轨迹）。

---

## 模块布局

```
DefenseAgent/
├── llm/         模块 1 —— LLM 外观 + 厂商适配器
├── config/      模块 2 —— pydantic 校验的 YAML profile 加载器
├── ops/         模块 3 —— 每 agent 一份的 JSON-lines 日志
├── memory/      模块 4 —— mem0 后端 memory（继承 ms-agent）
├── reflection/  模块 5 —— 重要性打分 + 洞察综合
├── tools/       模块 6 —— Python 函数 / Skills / MCP 服务器
├── rag/         模块 5+ —— LlamaIndex 后端 RAG，多模态抽取
├── skills/      Skill 加载器 + 子进程沙箱容器
└── agent/       模块 7 —— BaseAgent + Simple / ReAct / PlanAndSolve
```

每个模块在 [docs/superpowers/specs/](docs/superpowers/specs/) 都有设计文档，在 [docs/walkthroughs/](docs/walkthroughs/) 都有用户向 walkthrough。

---

## Demos

所有 demo 都假设 `.env` 已填好。在项目根、激活 venv 后运行。

| Demo | 演示什么 |
|---|---|
| `python scripts/show_profile.py` | profile YAML 加载 + 校验。无 API 调用。 |
| `python scripts/smoke_chat.py` | LLM 外观最小端到端 |
| `python scripts/profile_chat_demo.py` | profile + LLM —— agent 入戏作答 |
| `python scripts/streaming_demo.py` | 流式文本 delta |
| `python scripts/tools_demo.py` | Python 函数 + Skill 包当工具 |
| `python scripts/memory_demo.py` | Memory 写入 + 语义检索 + 反思 |
| `python scripts/reflection_demo.py` | 单独触发 reflection |
| `python scripts/react_tools_memory_demo.py` | **最完整** —— 三轮 ReAct 配 calculator + Tavily 搜索 + 跨轮 memory_recall |
| `python scripts/structured_extraction_demo.py --with-rag` | HTML/PDF → chunk → 多模态 RAG，保留嵌入图片 |
| `python scripts/structured_rag_agent_demo.py` | 端到端 **自定义 renderer + extractor** demo（离线） |
| `python scripts/dump_memory.py` | 查看 agent 的 mem0 目录里存了什么 |
| `python scripts/logger_demo.py` | 结构化 JSON-lines logger 用法 |

先跑 `react_tools_memory_demo.py`——它跑过的代码路径最多，最适合用作整套环境是否接好的烟雾测试。

---

## 测试

```bash
.venv/Scripts/python.exe -m pytest tests/ -v
```

测试**完全离线**——通过 `tests/DefenseAgent/agent/_support.py` 的 `make_test_config(...)` 注入 `ScriptedLLM` + `MagicMock` memory，不打实际 API。

```bash
pytest tests/DefenseAgent/llm/                 # LLM 适配器
pytest tests/DefenseAgent/agent/               # 构造 + 循环
pytest tests/DefenseAgent/memory/              # memory + bridge
pytest tests/DefenseAgent/tools/               # 工具注册器 / skill / MCP
pytest tests/DefenseAgent/rag/                 # 抽取 + RAG 检索
pytest tests/DefenseAgent/test_integration.py  # 跨模块集成
```

---

## Roadmap

仍然待办的 SDK 交付项：

- **PyPI 发布** —— `pip install defense-agent`（wheel 和 sdist 已经能干净构建；缺 PyPI 账号 + GitHub release workflow）
- **CI / pre-commit / ruff 配置** —— 形式化的 lint + 类型检查基线
- **CHANGELOG / CONTRIBUTING / CODE_OF_CONDUCT** —— 开源治理文档
- **多 agent 通信** —— 进程内 `AgentBus` + `AgentSwarm`，让 agent 之间互相当工具调用
- **运维向 RAG API** —— `delete_chunk` / `clear` / `list_resources` / `gc_orphan_resources`
- **LLM facade 的重试 / 超时 / 结构化日志** —— 把横切关注点从各厂商适配器里上提

设计依据见 [docs/superpowers/specs/](docs/superpowers/specs/) 各模块的 spec。

---

## 许可证

[MIT](LICENSE) © 2026 杨颖（Ying Yang）、Zechun Zhao、Yishu Wang。
