# DefenseAgent

> 中文 · [English README](README.md)

一个用 YAML profile 构建单 Agent LLM 应用的 Python 工具箱。一份 profile 描述 agent,一行 Python 实例化,三种执行策略可选。

```python
from DefenseAgent.agent import AgentConfig, ReActAgent
from DefenseAgent.examples import EXAMPLE_PROFILE_PATH

config = AgentConfig(profile=EXAMPLE_PROFILE_PATH)
agent  = ReActAgent(config)
result = await agent.run("用一句话总结今天的计划。")
```

## 目录

- [特性](#特性)
- [安装](#安装)
- [快速开始 —— 从零跑通一个 agent](#快速开始--从零跑通一个-agent)
- [配置](#配置)
  - [厂商与凭证](#厂商与凭证)
- [一步步搭一个 Agent](#一步步搭一个-agent) —— 完整 profile 参考
  - [`llm:`](#llm)
  - [身份](#身份)
  - [`cognitive:`](#cognitive)
  - [`memory:`](#memory)
  - [`rag:`](#rag)
  - [`tools:`](#tools) —— skills / MCP / Python
  - [`prompt:`](#prompt)
- [内置工具](#内置工具)
- [Agent 类](#agent-类)
- [多模态输入](#多模态输入) —— 视觉模型、图片处理、OCR
- [自定义与依赖注入](#自定义与依赖注入)
- [架构](#架构)
- [模块布局](#模块布局)
- [本地开发](#本地开发)
- [License](#license)

## 特性

- **单文件 agent 定义。** 身份、LLM 厂商、工具、memory、RAG、system prompt —— 全部写在一份严格校验的 YAML 里(`extra="forbid"`,未知字段会在加载时抛 `ConfigValidationError`)。
- **按字段的配置 fallback。** 每个值都能在 profile 或 `.env` 中设置;profile 优先,`.env` 补缺。切换 LLM 厂商(`openai`、`anthropic`、`deepseek`、`qwen`、`google`、`vllm`)无需改代码。
- **三种 agent 策略。** `SimpleAgent`(单轮)、`ReActAgent`(工具调用循环)、`PlanAndSolveAgent`(规划→执行→综合)。三者从同一份 `AgentConfig` 构造。
- **三种工具来源,统一一个 registry。** 本地 skill 目录(`SKILL.md` 包)、MCP 服务器(stdio / SSE / WebSocket / streamable-http)、Python 函数(profile 中按文件路径或点分模块引用)。
- **持久化 memory + 内置工具。** mem0 + Qdrant 落盘存储;agent 自动暴露 `memory_recall` 工具给 LLM。`ContextCompressor` 在每次 LLM 调用前裁剪工作上下文。
- **可选 RAG + 内置工具。** 把文档放进目录,设置 `rag.enabled: true`,获得 `rag_search` 工具。Embedder 凭证遵循同样的按字段 profile→env fallback。
- **可选的多模态输入。** 真正需要视觉时,`agent.run(task, images=[...])` 把图片挂到 user turn。默认不启用 —— 详见独立的 [多模态输入](#多模态输入) 章节。
- **可依赖注入。** LLM、memory、tools、reflector、compressor、logger 都能通过 `AgentConfig` 替换,方便测试和自定义接线。

## 安装

**默认安装** —— 第一次用推荐这条:

```bash
pip install 'defense-agent[memory]'
```

这是能让 `agent.run()` 在框架默认配置下(`use_memory=True`)直接跑起来的最小安装。它在核心依赖之上拉入 `mem0ai` + `fastembed`。

如果你**确认**不需要持久化 memory(不用 `memory_recall`、不存对话历史),裸装就够 —— 但你必须在 config 里显式关掉 memory:

```bash
pip install defense-agent
```

```python
config = AgentConfig(profile=..., use_memory=False)   # 裸装必须加这行
```

完整 extras 表:

| Extra | 拉入的依赖 | 何时需要 |
|---|---|---|
| `defense-agent[memory]` | `mem0ai[nlp]`、`fastembed`(`spacy` 间接拉入) | 默认配置可用 + 持久化 memory + `memory_recall` 工具。启动安静(没有 spaCy/fastembed warning)。 |
| `defense-agent[rag]` | `llama-index-core`、`llama-index-embeddings-openai-like`、`llama-index-retrievers-bm25`、`pdfplumber`、`beautifulsoup4`、`Pillow` | profile 里 `rag.enabled: true` + `rag_search` 工具 |
| `defense-agent[mcp]` | `mcp` | 连 MCP 工具服务器(`tools.mcp:` 条目) |
| `defense-agent[all]` | memory + rag + mcp | 一次性全装,所有子系统都能用 |
| `defense-agent[dev]` | `pytest`、`pytest-asyncio` | 跑测试套件 |

要求 Python ≥ 3.10。核心安装会拉入 `openai` + `anthropic` HTTP 客户端,以及 `ms-agent`(`ms-agent` 间接拉入 `torch`)。第一次安装大约 1 GB,留足磁盘和带宽。

### 关于启动 log 和文件落地

从 0.1.4 起,`defense-agent[memory]` 已经把 `mem0ai[nlp]` 和 `fastembed` 都带上,memory 初始化默认就是安静的。

从 0.1.5 起,`import DefenseAgent` 还会顺手抑制掉 ms-agent 默认在当前工作目录里建的 `ms_agent.log`。(上游 `ms_agent.utils.logger` 在被 import 的瞬间无条件给 `<cwd>/ms_agent.log` 挂一个 FileHandler —— DefenseAgent 现在在我们任何子模块碰到它之前就把那个 FileHandler 摘掉。终端上 `[INFO:ms_agent] ...` 那些 log 还会照常打到 stderr。如果你明确想要 ms-agent 的文件日志,自己调 `ms_agent.utils.logger.get_logger(log_file='your-path.log')`,我们的 patch 会让它正常工作。)

如果你(或下游用户)绕过 extras 直接 `pip install mem0ai`,可能看到 `Failed to load spaCy lemma model` / `fastembed not installed — BM25 keyword search disabled` 之类的消息。这些是 mem0 的可选特性探测 —— **可以忽略**,agent 照常工作。装回 `defense-agent[memory]`(或者 `pip install 'mem0ai[nlp]' fastembed`)就清干净。

## 快速开始 —— 从零跑通一个 agent

下面这段把"全新工程"的搭建从头走一遍。

### 1. 建工程目录 + 虚拟环境

```bash
mkdir myagent && cd myagent
python -m venv .venv
source .venv/bin/activate          # Windows 用 .venv\Scripts\activate
pip install --upgrade pip
```

(用 conda 也行:`conda create -n myagent python=3.12 -y && conda activate myagent`)

### 2. 安装

```bash
pip install 'defense-agent[all]'
```

如果不需要 RAG / MCP,可以只装小一点的 extras(例如 `defense-agent[memory]`)—— 见上面那张表。

### 3. 把凭证写进 `.env`

DefenseAgent 在构造时会调一次 `load_dotenv()`(运行环境已经有变量时可以传 `AgentConfig(load_env=False, ...)` 关掉)。在你将运行 Python 的目录创建 `.env`:

```bash
# myagent/.env
AGENT_LAB_LLM_PROVIDER=deepseek                      # 选哪家适配器
DEEPSEEK_API_KEY=sk-…                                # 你的 key
DEEPSEEK_MODEL=deepseek-chat                         # 厂商支持的任意 chat 模型
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1

# 仅在用 memory[memory_recall] 或 rag[rag_search] 时需要:
EMBEDDING_API_KEY=sk-…
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMS=1536
```

完整的厂商列表和 embedding 组合见下面的 [配置](#配置)。

### 4. 跑包内自带的 example agent

wheel 里打包了一份完整的参考 profile。先原样跑一下:

```python
# myagent/run_example.py
import asyncio
from DefenseAgent.agent import AgentConfig, ReActAgent
from DefenseAgent.examples import EXAMPLE_PROFILE_PATH

async def main():
    async with ReActAgent(AgentConfig(profile=EXAMPLE_PROFILE_PATH)) as agent:
        result = await agent.run("用一句话总结今天的计划。")
        print(result.final_answer)

asyncio.run(main())
```

```bash
python run_example.py
```

如果能打印出一句话,说明你的厂商凭证已经接通。

### 5. 拷一份 profile 改成自己的

把 example bundle 从包里拷出来,直接改:

```bash
python -c "
from DefenseAgent.examples import EXAMPLE_AGENT_DIR
import shutil; shutil.copytree(EXAMPLE_AGENT_DIR, './my_profile')
"
```

会得到 `my_profile/` 目录,内含 `profile.yaml`、`prompts/`、`python_tools/`、`skills/`。改 `profile.yaml`(schema 见 [一步步搭一个 Agent](#一步步搭一个-agent)),然后让代码指向它:

```python
from pathlib import Path
config = AgentConfig(profile=Path("./my_profile/profile.yaml"))
```

主流程就这些。剩下的章节都是参考资料。

## 配置

每个字段的解析顺序:profile YAML → 环境变量 → schema 默认值。仅含空白字符的值视为未设置。

### 厂商与凭证

`AGENT_LAB_LLM_PROVIDER` 选择适配器。每个厂商都有自己的 `<PROVIDER>_*` 块(`<PROVIDER>_API_KEY`、`<PROVIDER>_MODEL`、`<PROVIDER>_BASE_URL`)。跨厂商的 `LLM_API_KEY` / `LLM_MODEL_ID` / `LLM_BASE_URL` 会在设置时覆盖 per-provider 那一层。

| Provider | 适配器 | Key 典型格式 | 默认 base URL | 可选 chat 模型示例 |
|---|---|---|---|---|
| `openai` | `OpenAICompatibleAdapter` | `sk-…` 或 `sk-proj-…` | `https://api.openai.com/v1` | `gpt-4o-mini`、`gpt-4o`、`o3-mini` |
| `anthropic` | `AnthropicAdapter` | `sk-ant-…` | `https://api.anthropic.com` | `claude-sonnet-4-6`、`claude-opus-4-7` |
| `deepseek` | `OpenAICompatibleAdapter` | `sk-…` | `https://api.deepseek.com/v1` | `deepseek-chat`、`deepseek-reasoner` |
| `qwen`(DashScope OpenAI 兼容) | `OpenAICompatibleAdapter` | `sk-…` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-plus`、`qwen-max`、`qwen-turbo` |
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

## 一步步搭一个 Agent

一个 profile bundle 就是一个目录:

```
my_profile/
├── profile.yaml          # 必需 —— schema 见下方
├── prompts/              # 可选 —— system prompt 模板
│   └── system.md
├── python_tools/         # 可选 —— 本地 Python 工具入口
│   └── calc.py
├── skills/               # 可选 —— SKILL.md 风格的工具包
│   └── tabular-report/
├── memory/               # memory.is_retrieve=true 时运行时自动创建
└── rag_corpus/           # rag.enabled=true 时被索引的文档目录
```

`AgentConfig(profile=Path("…/my_profile/profile.yaml"))` 会把 profile 里所有相对路径解析为相对 profile 文件所在目录,因此整个 bundle 自包含、可整体迁移。

`agent:` 下每一块都可选,身份字段除外。所有字段都被 pydantic 严格校验(`extra="forbid"`)。

### `llm:`

```yaml
llm:
  provider:           # str | null。可选值:openai | anthropic | deepseek | qwen | google | vllm。回退 AGENT_LAB_LLM_PROVIDER。
  model:              # str | null。厂商特定的 model id(见上面的厂商表)。回退 <PROVIDER>_MODEL 或 LLM_MODEL_ID。
  base_url:           # str | null。厂商端点。回退 <PROVIDER>_BASE_URL 或 LLM_BASE_URL。
  api_key:            # str | null。回退 <PROVIDER>_API_KEY。共享 profile 时建议留空。
```

四个字段都是 `str | None`,各自独立 fallback 到 `.env`。仅含空白字符的值视为未设置 —— 半改一半的 YAML 不会盖掉正确的 env。

#### 按字段的 fallback 实战

每个字段的解析顺序,自上而下(第一个非空的胜出):

1. profile YAML 里的 `llm.<field>:`
2. 跨厂商 env 层 —— `LLM_API_KEY` / `LLM_MODEL_ID` / `LLM_BASE_URL`
3. 单厂商 env 层 —— `<PROVIDER>_API_KEY` / `<PROVIDER>_MODEL` / `<PROVIDER>_BASE_URL`
4. schema 默认值(适用时)

推荐写法:profile 里只放 `llm: { provider: deepseek, model: deepseek-chat }`,其他都在 `.env` —— 模型选择属于 agent 身份,该写在 YAML;凭证属于运维问题,该放 `.env`。

具体例子。给定:

```yaml
# profile.yaml
llm:
  provider: deepseek
  model: deepseek-reasoner             # profile 显式指定
```

```bash
# .env
LLM_API_KEY=sk-shared                  # 跨厂商覆盖,优先级高于单厂商
DEEPSEEK_API_KEY=sk-deepseek           # 单厂商,LLM_API_KEY 不存在时才用
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat           # 被忽略 —— profile 里的 model 胜出
```

最终解析:
- `provider` → `deepseek`(profile)
- `model` → `deepseek-reasoner`(profile 胜过 `DEEPSEEK_MODEL`)
- `base_url` → `https://api.deepseek.com/v1`(profile 留空 → fallback 到 `DEEPSEEK_BASE_URL`)
- `api_key` → `sk-shared`(跨厂商 `LLM_API_KEY` 胜过 `DEEPSEEK_API_KEY`)

#### 切换厂商无需改代码

同一份 agent 代码,三种厂商,只改 `.env`:

```bash
# .env(版本 A —— DeepSeek)
AGENT_LAB_LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-…
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
```

```bash
# .env(版本 B —— DashScope/Qwen)
AGENT_LAB_LLM_PROVIDER=qwen
QWEN_API_KEY=sk-…
QWEN_MODEL=qwen-plus
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

```bash
# .env(版本 C —— 本地 vLLM)
AGENT_LAB_LLM_PROVIDER=vllm
VLLM_API_KEY=EMPTY                     # vLLM 默认不鉴权
VLLM_MODEL=Qwen/Qwen2.5-72B-Instruct   # 跟服务里挂的对齐
VLLM_BASE_URL=http://localhost:8000/v1
```

只要 profile 里 `llm.provider` / `llm.model` 留空(或者整个不写 `llm:` 块),agent 自动跟着 env 走。无需重启,无需改代码。

#### 各厂商的特别说明

| Provider | 注意点 |
|---|---|
| `openai` | `sk-…` 和 `sk-proj-…` 都支持。reasoning 模型(`o3-mini`、`o1`)更贵且请求形态略有不同 —— 适配器透明处理。 |
| `anthropic` | 支持工具调用。Anthropic 的非文本 content 走自己的 wire 格式,跟 OpenAI 不同 —— list-shape `content` 进到适配器会抛 `LLMAdapterError`。视觉模型选择参见 [多模态输入](#多模态输入)。 |
| `deepseek` | `deepseek-reasoner` 在 `reasoning_content` 里返回思考 token —— 适配器把它从 `Message.content` 里剥掉,下游代码不会看到 chain-of-thought。要看的话直接看原始响应。 |
| `google` | 走 Google 的 OpenAI 兼容端点 `generativelanguage.googleapis.com/v1beta/openai`。原生 Gemini SDK 没用。 |
| `vllm` | `VLLM_API_KEY=EMPTY`(字面字符串 EMPTY)是惯例。`VLLM_MODEL` 必须跟服务上挂的模型名一致(看 vLLM 的 `--served-model-name`)。 |

#### 程序化注入 LLM(测试、mock、自定义适配器)

`AgentConfig` 接受预构造的 `LLM` 实例 —— 给了之后,**LLM 这一块的 env 构造路径完全跳过**。适用场景:

```python
from DefenseAgent.llm import LLM
from DefenseAgent.llm.openai_compat import OpenAICompatibleAdapter

# 1. 用 scripted/mocked LLM 做测试
config = AgentConfig(profile="…", llm=ScriptedLLM(responses=[...]))

# 2. 同进程内多 agent 用不同厂商
config_a = AgentConfig(profile=p, llm=LLM(adapter=OpenAICompatibleAdapter(api_key="...", base_url="https://api.openai.com/v1", model="gpt-4o")))
config_b = AgentConfig(profile=p, llm=LLM(adapter=AnthropicAdapter(api_key="...", model="claude-sonnet-4-6")))

# 3. 自定义适配器(继承 LLMAdapter)
config = AgentConfig(profile="…", llm=LLM(adapter=MyCustomAdapter()))
```

同样的注入模式适用于其它每个组件 —— 见下面 [自定义与依赖注入](#自定义与依赖注入)。

### 身份

只有 **`id`** 和 **`name`** 是必填。其它四个(`age`、`traits`、`backstory`、`initial_plan`)只是 persona 调味,有合理默认值 —— 想要最简 agent 就留空,想要丰满 agent 就填上。

```yaml
# 最简 —— 只有 id + name
id: "bot"
name: "Helper"
```

```yaml
# 完整 —— 每个 persona 字段都填
id: "agent_001"     # str, min_length=1。必填。
name: "Nova Patel"  # str, min_length=1。必填。
age: 27             # int ≥ 0 | null。可选,默认 null。
traits: "..."       # str。可选,默认 ""。
backstory: "..."    # str。可选,默认 ""。
initial_plan: "..." # str。可选,默认 ""。
```

六个字段都作为 `{id} {name} {age} {traits} {backstory} {initial_plan}` 占位符出现在 prompt 模板中 —— 见下面的 [`prompt:`](#prompt)。可选字段未设置时渲染为空字符串,所以最简 profile 配上引用 `{traits}` 的模板也不会崩。

#### 各字段的实际作用

| 字段 | 必填? | 用于 |
|---|---|---|
| `id` | **是** | (1) mem0 的 `agent_id` 分区键 —— 记录会被限定到这个 id 之下。(2) 日志文件名:`<log_dir>/<id>.log`。(3) prompt 模板里的 `{id}` 占位符。**选一个稳定的标识符,别随便改** —— 改 `id` 会让现存的 memory 变成孤儿。 |
| `name` | **是** | `{name}` 占位符。自动生成的身份 prompt 以 `You are <name>, ...` 开头。 |
| `age` | 可选(默认 `null`) | `{age}` 占位符。角色扮演型 persona 用得着。未设置时,自动生成的 prompt 直接用 `You are <name>.`(没有 age 子句),用户模板里的 `{age}` 渲染为 `""`。 |
| `traits` | 可选(默认 `""`) | `{traits}` 占位符。自由形式的性格 / 语气 / 风格描述。非空时,自动生成的 prompt 会加一行 `Traits: ...`。 |
| `backstory` | 可选(默认 `""`) | `{backstory}` 占位符。长篇叙事 —— 履历、专业领域、个性特点。让 LLM 锚定在具体 persona 上,这个字段用处最大。 |
| `initial_plan` | 可选(默认 `""`) | `{initial_plan}` 占位符。agent 当前在做什么;给 agent 设定"今天的"语境。 |

#### 可选字段缺失时,自动生成的 prompt 是这样的

字段未设置时,自动生成的身份块**整行跳过**,而不是留空白。最简 profile(`id: "bot"`、`name: "Helper"`),agent 的 system prompt 就是:

```
You are Helper.
```

加上 `traits: "concise, technical"`:

```
You are Helper.
Traits: concise, technical
```

…以此类推。不会出现"You are Helper, a -year-old. Traits: "这种尴尬句子。

#### 校验失败模式

schema 是严格的 —— 错误会在 `AgentProfile.from_yaml()` 阶段就抛 `ConfigValidationError`,不会拖到 `agent.run()`:

| 输入 | 结果 |
|---|---|
| `id: ""` 或 `id: "   "` | `string_too_short`(id 必填且 strip 后非空) |
| `name: ""` | `string_too_short`(name 必填且非空) |
| 缺 `id` 或缺 `name` | `missing` 校验错误 |
| 缺 `age` / `traits` / `backstory` / `initial_plan` | 接受 —— 自动用 `null` / `""` |
| `age: -1` | `greater_than_equal` 失败 |
| `age: 27.5` | `int_type` 失败(必须整数或 null) |
| 多字段 | `extra_forbidden` —— 字段名拼错会立即报错,不会静默 fallback |

### `cognitive:`

```yaml
cognitive:
  max_steps_per_cycle: 10     # int ≥ 1,默认 10。每次 run() 中 ReAct 工具调用循环的上限。
  reflection_threshold: 5     # int ≥ 1,默认 5。触发 Reflector.maybe_reflect() 的未反思记忆数量。
  importance_threshold: 7     # float ∈ [1, 10],默认 7。Reflection 中"重要"记忆的阈值。
  planning_horizon: "1 day"   # str, min_length=1,默认 "1 day"。自由格式;在 prompt 中暴露给 LLM。
```

#### `max_steps_per_cycle` —— ReAct 循环预算

`ReActAgent` 里一个"step"等于一次(tool-call → tool-result)往返。`max_steps_per_cycle: 10` 表示 LLM 最多有 10 次 tool-call 机会,然后循环就会被强制退出。强制退出时:

```python
result = await agent.run("multi-step task")
# result.stopped_reason == "max_steps"   ← 循环触顶
# result.final_answer                    ← LLM 最后给出的部分输出
# result.steps                           ← 完整轨迹(10+ 条 —— call/result 交替)
```

可以单次覆盖 profile 设置:`await agent.run(task, max_steps=20)`。`SimpleAgent` 两个都忽略 —— 按定义就一次 LLM 调用。`PlanAndSolveAgent` 把 `max_steps` 解释为**计划长度上限**(不是每步的子步数上限,那个是 `AgentConfig.max_substeps_per_step`,默认 3)。

按任务复杂度调:

- 简单 Q&A 一次 tool call:`max_steps_per_cycle: 3` 够用
- 多工具研究 ReAct:10–20
- 长 horizon 迭代:谨慎放大 —— 每步都是要花钱的 LLM 调用

#### `reflection_threshold` 和反思周期

每次 `run()` 之后,如果 `reflect_after_run: true`(`AgentConfig` 默认值),agent 会调 `Reflector.maybe_reflect()`。这是个守卫:**当未反思记录数累计到 `reflection_threshold` 才触发反思周期**。低于阈值时是 no-op。

触发后:

1. `_get_unreflected_records()` 拉出 mem0 里所有 `memory_type != 'reflection'` 的记录
2. `InsightSynthesizer.synthesize()` 让 LLM 提炼成 N 条(默认 3 条)bullet 形式的洞察
3. 每条洞察以 `memory_type='reflection'`、importance 8.0 写回 mem0

所以 `reflection_threshold: 5` 大概意思是"每 5 个 run/turn 触发一次反思"(看什么内容入 memory)。调小让自省更频繁;调大让反思更稀疏、更高信号。

反思之后产生的记录后续 `memory_recall` 能看到 —— 让 agent 跨 run 长期建立对自己的理解。

#### 反思什么时候真的有用,什么时候没用

每次反思周期至少多花 2 次 LLM 调用(`ImportanceScorer` + `InsightSynthesizer`),所以它只在"未来真的会有人读这些 reflection 记录"的场景下才划算。先认清自己属于哪种:

| 场景 | 反思有用吗? | 建议 |
|---|---|---|
| **一次性脚本** —— `python my_agent.py` 跑一次就退出 | **没用。** 反思写 3 条记录,进程结束,无人再读。纯浪费。 | `AgentConfig(profile=..., reflect_after_run=False)` |
| **demo / quickstart** —— 第一次试 DefenseAgent | 没用。同上。 | 同上。 |
| **同 `agent_id` 跨多个 session** —— 长期运行的助手、相似任务的反复批处理 | **有用。** 第 N 次 session 的反思会在第 N+1 次 session 通过 `memory_recall` 露面。agent 跑得越久,价值越累积。 | 保持默认(`reflect_after_run=True`)。 |
| **Generative Agents 风格的模拟** —— 多天模拟世界、社交 agent | **有用 —— 这就是设计目标。** `Reflector` 就是为这种场景做的([Park et al. 2023](https://arxiv.org/abs/2304.03442))。 | 保持默认。可以把 `reflection_threshold` 调小让它更频繁。 |
| **高频短任务** —— 客服 agent 处理几百个独立工单 | **看情况。** 关于 agent 失败模式的反思能跨工单留存才有用。 | 先开一阵子,用 `scripts/dump_memory.py` 看看 mem0 里实际写了什么再定。 |

任何"有用"的场景都还有一个前提:**后续 run 里 LLM 得主动调 `memory_recall`**。反思不会自动塞进 prompt —— agent 必须主动去查才能用上。在 system prompt 里明确写"回答前先 `memory_recall` 查相关上下文"能让反思的回报大幅提升;不写就可能整个机制空转。

如果你在做一次性工具,**前期就关掉反思**,跳过那两次 LLM 调用:

```python
config = AgentConfig(
    profile=...,
    reflect_after_run=False,    # 跳过 run 之后的反思周期
)
```

更彻底的做法:`use_reflection=False`,连 `Reflector` 对象都不构造。整个 agent 生命周期都不需要反思时用这个。

#### `importance_threshold`

`ImportanceScorer` 用(LLM 给每条记录打 1–10 分)。反思过程中,低于此阈值的记录在送给 synthesizer 之前会被过滤掉 —— 让 LLM 聚焦实质内容,而不是闲聊。默认 7 偏保守;记录普遍偏低影响时调到 5。

#### `planning_horizon`

自由格式字符串 —— 出现在自动生成的身份 prompt 里,作为 agent 的工作时间窗口。默认 `"1 day"`。常见取值:

- 短窗口运营 agent:`"this hour"`
- 工程类 agent:`"this sprint"`
- 紧 deadline:`"the next 30 minutes"`

LLM 用它判断当前 run 的 scope:什么是要现在做的、什么该 defer。只有自定义 prompt 引用了自动生成的身份块(或自己手动引用)时才可见。

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

需要 `defense-agent[memory]`(`mem0ai`、`fastembed`)。

#### 落盘后实际是什么样

第一次 `run()` 之后磁盘上会出现:

```
my_profile/
└── memory/                              # = storage_path(默认 <profile_dir>/memory/)
    ├── stream.db                        # SQLite —— 完整 block 流(每条 Message 原样保存)
    ├── cache.json                       # ms-agent 用于 dedup 的 block hash
    └── qdrant/                          # 本地 Qdrant —— 这些 block 的向量索引
        └── collection/<agent_id>/
```

两份存储并存:SQLite 按插入顺序保存**完整对话历史**;Qdrant 保存**向量 embedding**,供 `memory_recall` 做语义检索。两份都按三元组 **`(user_id, agent_id, run_id)`** 分区 —— 同一个 agent 跨多个 session 时不会互相污染。

#### `history_mode: add` vs `overwrite`

- **`add`**(默认)—— 每条 Message 都追加。`agent.run("X")` 跑两次会留下两份独立的回答。简单、永远是新增。
- **`overwrite`** —— 用 ms-agent 的 block-hash diff。完全相同的消息不会重复落盘;结构相似的运行会替换之前的 block。靠 hash 链支持回滚。当你想保留"每次 run 的当前最优状态"而不是完整流水账时,选这个。

不论哪种模式,`ignore_roles:` 默认把 `tool` 和 `system` 消息排除在外 —— 工具结果体积大、冗余、能从原始 tool call 复现。如果只想留用户输入,加 `assistant` 进 `ignore_roles:`。

#### `memory_type` 标签体系

每条记录写入时会带一个 `memory_type` 标签(放在 metadata 里)。常见标签:

| 标签 | 来源 | 含义 |
|---|---|---|
| (默认,无标签) | `agent.run()` 的对话轨迹 | 原始消息 |
| `outcome` | `BaseAgent._save_outcome()` | `save_outcome: true` 时,成功 run 的最终回答 |
| `failure` | 同上,但 `AgentError` 时 | 失败 run 的截断错误文本 |
| `reflection` | `Reflector.maybe_reflect()` | LLM 在最近未反思记忆上提炼出的经验 |
| `procedural` | mem0 原生形态 | mem0 的 procedural-memory 通道,我们不直接写入 |

`memory_recall` 返回结果时会带类型前缀:`- [reflection] 工具失败时容易过度解释`。

#### `memory_recall` —— 内置工具

`is_retrieve: true` 时,LLM 会自动拿到 `memory_recall` 工具:

```json
{
  "name": "memory_recall",
  "input_schema": {
    "query": "string",
    "top_k":  "int (1..20,默认 5)"
  }
}
```

它在 Qdrant 上做相似度搜索,过滤当前 run 的 `(user_id, agent_id, run_id)`,最多返回 `top_k` 条记录(被 `search_limit:` 上限钳住)。是否调用由 LLM 自己决定 —— 不会自动注入到每一轮。

#### `ContextCompressor` —— token 预算守卫

跟 memory_recall 是两回事:这个负责保护**每次 LLM 调用**不超 context window。它在每次 LLM 调用**之前**运行,作用对象是工作消息列表(本轮要丢给 `chat()` 的内容)。

四个数字这样配合:

```
工作消息总 token
        │
        │  当  total + reserved_buffer  >  context_limit
        │       触发裁剪
        ▼
裁剪流程:
   ── 保留最近的 prune_protect token 不动(近期消息最重要)
   ── 把更老的内容压缩,使总量 ≥ prune_minimum
   ── enable_summary=true 时,被压缩的老段会变成一条 LLM 生成的摘要 Message
   ── enable_summary=false 时,直接丢弃,不替换
```

举例:`context_limit: 128000` + `reserved_buffer: 20000` 表示"工作消息超过 108K token 就开始裁剪"。`prune_protect: 40000` 表示"最近 40K token 永远不动"。`prune_minimum: 20000` 是地板 —— 哪怕内容已经能塞进 20K,也不再压。这四个数字要一起调;`context_limit` 设得超过模型实际窗口大小只会让 API 直接拒绝,没好处。

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

需要 `defense-agent[rag]`(`llama-index-core`、`llama-index-embeddings-openai-like`、`llama-index-retrievers-bm25`、`pdfplumber`、`beautifulsoup4`、`Pillow`)。

#### Bootstrap 流程 —— 第一次 run 时发生什么

`rag.enabled: true` 状态下第一次 `agent.run()` 触发时:

1. **发现文档** —— 枚举 `documents_dir`(相对 profile 目录,默认 `rag_corpus/`)下的每个文件
2. **抽取结构化 chunk** —— `StructuredDocExtractor` 用注册的 extractor backend(`PyPdfExtractor`、`HtmlExtractor` …)逐文件处理。每个 backend 的 `supports(path)` 按扩展名/内容选用。普通 `.md` / `.txt` 走 LlamaIndex 的默认 loader
3. **token 化 + 切块** —— 抽取出的每个 chunk 按 `chunk_size:` token 再切,相邻块按 `chunk_overlap:` 重叠。chunk 越小召回越细,索引条目越多;chunk 越大条目越少但越粗
4. **embed + 建索引** —— 每个 chunk 过 embedder(`embedding:` 模型),向量落到 `storage_dir`(默认 `rag_index/`)下的持久化 FAISS 索引
5. **持久化** —— 索引落盘,后续 run 完全跳过 1–4 步

最终目录:

```
my_profile/
├── profile.yaml
├── rag_corpus/                            # = documents_dir
│   ├── runbook.pdf
│   ├── architecture.html
│   └── notes.md
└── rag_index/                             # = storage_dir
    ├── default__vector_store.json         # FAISS 向量
    ├── docstore.json                      # 原始 chunk 文本
    └── _resources/                        # 抽取出的图片/表格(被 chunk 引用)
```

文档变了想重新建索引:删 `storage_dir` 后重 run。**没有增量索引** —— 是全有或全无。

#### 文档格式 —— 支持哪些、怎么扩展

| 来源 | Backend | 抽取的内容 |
|---|---|---|
| `.pdf` | `PyPdfExtractor`(基于 `pdfplumber`) | 文字、表格(渲染成 Markdown)、嵌入图像 |
| `.html` | `HtmlExtractor`(基于 `beautifulsoup4`) | 按 section 切的正文、表格、`<img>` 引用 |
| `.md` / `.txt` / `.rst` | LlamaIndex 默认 loader | 纯文本 chunk |
| `.docx` / `.epub` / 其他 | LlamaIndex 默认 loader(尽力支持) | 纯文本 chunk |

Extractor 是可插拔的。继承 `StructuredExtractor` `Protocol`(实现 `supports(source)` 和 `extract(source) -> list[StructuredChunk]`),注册到 extractor 上:

```python
from DefenseAgent.rag.extraction import StructuredDocExtractor

class MyCsvExtractor:
    def supports(self, source): return str(source).endswith(".csv")
    def extract(self, source): return [...]   # list[StructuredChunk]

extractor = StructuredDocExtractor(...)
extractor.register(MyCsvExtractor(), prepend=True)   # 排在内置之前
```

resource renderer(table-to-Markdown、image-to-base64)是同样的形式 —— 见 `DefenseAgent/rag/renderer.py`。

#### Embedding 选择 —— `openai` vs `huggingface`

| `embedding_provider:` | 何时选 | 备注 |
|---|---|---|
| `openai`(默认) | 任何 OpenAI 兼容的 embedding 端点 —— OpenAI 自身、DashScope、ModelScope、vLLM、OpenRouter | 用四个 `embedding_*` 字段(或 `EMBEDDING_*` env 等价)。`openai-like` 适配器全包了。 |
| `huggingface` | 离线、不能联网、想省钱 | 走 ms-agent 的 HF 下载路径,需要 `use_huggingface: true`。`embedding:` 填 Hugging Face model id(例如 `BAAI/bge-large-en-v1.5`)。第一次 run 慢一点(下载模型)。 |

无论哪种 embedder,设置的 `EMBEDDING_DIMS:` 必须跟模型实际输出维度一致 —— `text-embedding-3-small` 是 1536,`text-embedding-3-large` 是 3072,Qwen3-Embedding-8B 是 4096。维度对不上 → FAISS 拒绝写入。

#### `rag_search` 工具 —— LLM 看到的是什么

`enabled: true` 时,registry 里会有:

```json
{
  "name": "rag_search",
  "description": "Vector search over the agent's RAG corpus...",
  "input_schema": {
    "query": "string",
    "top_k": "int (默认 <profile.rag.top_k>)"
  }
}
```

LLM 自己决定何时调用;返回格式取决于 `retrieve_only:`:

- **`retrieve_only: true`**(默认)—— 返回排序后的 top-k chunk,每条带分数前缀:
  ```
  [score=0.84] <chunk text 1>
  [score=0.71] <chunk text 2>
  ...
  ```
  便宜(不需要二次 LLM 调用),agent 拿到原料后可以自己取舍/过滤/重新组织。

- **`retrieve_only: false`** —— 在召回的 chunk 上跑 LlamaIndex 自带的 QA 综合器:再来一次 LLM 调用合成一个最终回答字符串。更贵、灵活性低,但能一次性吐答案。

`score_threshold:` 在返回之前做过滤 —— 低于阈值的 chunk 静默丢弃。比如设 0.4 抑制弱匹配;0.0(默认)把 top_k 召出来的全部返回。

### `tools:`

三种工具来源,合并到同一个 `ToolRegistry`。LLM 看到的是一个扁平命名空间。先看整体形态,下面每节展开一种来源。

```yaml
tools:
  skills:                                 # list[str]。SKILL.md 风格的工具包(默认只读)。
    - skills/tabular-report
  mcp:                                    # list[MCPServerConfig]。外部 MCP 工具服务器。
    - command: uvx
      args: [mcp-server-filesystem, /tmp]
  python:                                 # list[str]。Python entry-point 字符串。
    - python_tools/calc.py:calculator
    - my_pkg.search:web_search
  allow_skill_execution: false            # bool,默认 false。把 skill 里的脚本提升为可执行工具。
  skill_execution_timeout: 300            # int ≥ 1,默认 300。子进程超时(秒)。
```

`run()` 启动时,registry 是三种来源的并集,加上自动注册的 `memory_recall` 和(启用时的)`rag_search`。所有工具的名字必须全局唯一 —— 重名在构造期就会立刻报错,不会等到调用时才崩。

---

#### `tools.skills:` —— 本地 SKILL.md 工具包

一个 skill 就是一个目录,根目录有 `SKILL.md`,位置随便放(profile 里指它就行)。包内自带的参考 bundle [`DefenseAgent/examples/example_agent/skills/tabular-report/`](DefenseAgent/examples/example_agent/skills/tabular-report) 是标准形态:

```
skills/tabular-report/
├── SKILL.md                   # 必填 —— frontmatter + 正文
├── scripts/                   # 可选 —— 可执行脚本
│   └── generate.py
├── references/                # 可选 —— 长篇参考文档
└── templates/                 # 可选 —— 辅助资源文件
    └── header.md
```

`SKILL.md` 顶部是 YAML frontmatter,后面是 LLM 会读到的 Markdown 正文:

```markdown
---
name: tabular-report
description: 把行字典列表渲染成 GitHub 风格 Markdown 表格。
author: kevin                  # 可选,会出现在 tool metadata 里
tags: [reporting, table]       # 可选,会出现在 tool metadata 里
---

# Tabular Report

需要用行字典生成 Markdown 表格时调这个 skill。

## 怎么用

1. 把每行数据收集成一个字典,所有字典 key 一致。
2. 列顺序自己决定 —— skill 不会推断,要显式传列名。
3. 用本工具的 `file=` 参数读取 `scripts/generate.py`,然后在自己的代码里调
   `render_table(rows, columns)`。
```

agent 加载这个 skill 后,registry 里会出现**一个只读工具**,名字就叫 `tabular-report`:

```json
{
  "name": "tabular-report",
  "description": "把行字典列表渲染成 GitHub 风格 Markdown 表格。\n\nBundled files — scripts: generate.py; references: None; resources: header.md.",
  "input_schema": {"file": "string (optional)"}
}
```

description 由 frontmatter 的 `description:` 加上一行 bundled-file 清单组成(让 LLM 知道还能按名字读哪些文件,不需要瞎猜)。

LLM 怎么用:

| 调用 | 返回 |
|---|---|
| `tabular-report({})`(或 `file=""`) | SKILL.md 正文(剥掉 frontmatter)—— 也就是 LLM 拿到 prompt 风格的说明 |
| `tabular-report({"file": "scripts/generate.py"})` | 这个文件的纯文本 |
| `tabular-report({"file": "templates/header.md"})` | 这个文件的纯文本 |
| `tabular-report({"file": "../../etc/passwd"})` | `SkillLoadError("path escapes skill directory ...")`—— 路径逃逸守卫 |

skill 的元数据(skill_id、version、author、tags)挂在 `Tool.metadata` 字典上,后续过滤或审计可以用。

##### 把脚本提升为可执行工具 —— `allow_skill_execution: true`

默认下,skill 里的脚本是**可读但不可跑** —— LLM 拿到源码后只能在自己的推理里照着抄。把 `allow_skill_execution: true` 打开,**每个脚本会变成一个独立的可执行工具**,命名 `<skill>__<stem>`:

```yaml
tools:
  skills:
    - skills/tabular-report
  allow_skill_execution: true
  skill_execution_timeout: 300            # 子进程超时(秒)
```

这样 registry 里同时有 `tabular-report__generate`,输入 schema 是 `{args?: list[str], stdin?: string, timeout?: int}`。每次调用通过 `SkillContainer` 起一个全新的子进程跑(继承了 ms-agent 上游的危险模式守卫,挡 `rm -rf` 这种)。stdout、stderr 和退出码会被合成一个字符串返回给 LLM。

可识别的脚本扩展名:`.py`、`.sh`、`.js`。`scripts/` 下的子目录里的脚本**不会**被递归收 —— 只有顶层脚本会被提升。

---

#### `tools.mcp:` —— 外部 MCP 服务器

[Model Context Protocol](https://modelcontextprotocol.io) 服务器是独立进程,自己持有一份工具目录。DefenseAgent 的 `MCPClient` 继承自 ms-agent 的多服务器客户端,支持四种 transport:

| `transport:` | 何时用 | 必填字段 |
|---|---|---|
| `stdio`(默认) | 本地启动的 server 进程(`uvx`、`npx`、`python` ……) | `command:` |
| `sse` | 长连接 HTTP server-sent-events 端点 | `url:` |
| `websocket` | WebSocket 服务器 | `url:` |
| `streamable_http` | HTTP 流式端点 | `url:` |

每条 entry **必须只设** `command:` 和 `url:` 中的一个,绝不能两个都给。所有 server 的连接是**懒建立** —— 在第一次 `agent.run()` 调用时才 spin up(连接是 async 的,只有真有工具调用才需要)。

##### stdio 例子 —— 本地文件系统服务器

```yaml
tools:
  mcp:
    - command: uvx                        # PATH 上的可执行
      args: [mcp-server-filesystem, /tmp/sandbox]
      env:
        DEBUG: "1"
        GITHUB_TOKEN: ""                  # 空值 → connect() 时从进程环境变量取
      cwd: /workspace                     # 可选工作目录
      include: [read_file, list_dir]      # 白名单 —— 只暴露这几个工具名
      # exclude: [delete_file]            # 备选:黑名单;与 include 互斥
```

行为:

- 服务器声明的每个工具都会变成 registry 里的一个 `Tool`,**名字直接用服务器声明的工具名**(不加前缀)。来源服务器的名字记在 `tool.metadata["server"]` 里,方便溯源。
- 每个 server 的 `include:` / `exclude:` 互斥。某个 server 工具太多时(比如 `mcp-server-filesystem` 有 ~10 个),用 `include: [read_file, list_dir]` 限到只读。
- `env:` 里值为空字符串(例如 `GITHUB_TOKEN: ""`)的字段会在 connect 时从进程环境变量插值 —— 写 `""` 代替硬编码 key。

##### 网络 transport 例子 —— SSE

```yaml
tools:
  mcp:
    - transport: sse
      url: https://mcp.example.com/sse
      headers:
        Authorization: "Bearer ${MCP_API_TOKEN}"  # 不会自动插值,自己展开
      timeout: 30                                  # 连接超时(秒)
      sse_read_timeout: 300                        # 长轮询读取超时
      include: [search]
```

header 里的值是**原样透传** —— DefenseAgent **不会**帮你展开 `${VAR}`。要做环境变量替换,要么在构造 `AgentConfig` 之前自己展开,要么把展开后的值写进 `.env` 里,YAML 中直接 inline。

##### 多 server + 依赖

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

两个 server 的工具进同一个扁平 registry。跨 server 的工具名碰撞会在 registry 构造时直接报错 —— 接多个 server 时命名规范要自己拿捏。

需要装 `defense-agent[mcp]`(官方 `mcp>=1.0` Python SDK)。

---

#### `tools.python:` —— 你自己的 Python 函数

两种形态,都用 entry-point 字符串 `<module-or-file>:<function-name>`:

**1. 相对文件路径**(不需要打成包)。路径相对 profile 目录解析,通过 `importlib.util.spec_from_file_location` 加载。运行的 Python 解释器不需要事先设 `sys.path`。

```
my_profile/
├── profile.yaml              # tools.python: ["python_tools/calc.py:calculator"]
└── python_tools/
    └── calc.py               # def calculator(expression: str) -> str: ...
```

**2. 点分模块路径**(工具放在已安装的包里时用)。通过 `importlib.import_module` 解析。模块必须能被当前解释器 import —— `pip install -e .` 装好或者已经在 `sys.path` 上。

```
my_pkg/
├── __init__.py
└── search.py                 # def web_search(query: str) -> str: ...
```

profile 中的写法:`my_pkg.search:web_search`。

##### 工具 schema 是自动派生的

两种形态下,**函数签名**变成工具的 input schema,**docstring** 变成 description。LLM 看不到你的代码本体,只看到这份合成出来的 metadata:

```python
def calculator(expression: str, precision: int = 4) -> str:
    """Evaluate a Python arithmetic expression and return the result.

    Supports +, -, *, /, **, parentheses, and the math module.
    """
    ...
```

会变成:

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

类型注解支持:`str`、`int`、`float`、`bool`、`list[T]`、`dict`、`Optional[T]`、原生 `Path`。任何无法干净转成 JSON-schema 的复杂类型会在加载时抛 `ToolRegistrationError` —— 命名/签名问题立刻暴露,不会拖到运行时。

##### 在代码里直接注册(不写进 profile)

不想写进 `profile.yaml` 的话,直接以代码注册:

```python
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression."""
    ...

config = AgentConfig(profile="…", tools=[calculator])
```

`tools=` 接受 plain callable —— 自动派生规则一样。临时工具、测试用工具,或者依赖运行时状态(比如闭包持有数据库连接)的工具,适合走这条路径。

### `prompt:`

```yaml
prompt:
  path: prompts/system.md         # str | null。相对 profile 目录的文件。
  system:                         # str | null。inline 形式,与 path 互斥。
  extra_instructions:             # str | null。追加在身份块之后。
```

System prompt 是 LLM 在每次调用时看到的 `system=` 参数 —— agent 的"帽子",跟 user-turn 任务内容是分开的。

#### 三条解析路径

agent 按这个顺序解析 system prompt,**第一个非空的胜出**:

1. **inline `system:` 字段** —— YAML 里直接写一段字面字符串。短的、一次性的 prompt 适用,不值得专门开个文件。
2. **`path:` 指向文件** —— 相对 profile 目录解析。任何非平凡的 prompt 都建议这种方式 —— 版本控制、跨 agent 复用、长占位符模板,都更顺手。
3. **自动生成的身份块** —— 上面两个都空(或渲染失败)时,agent 用身份字段拼一段 fallback prompt。

不论走哪条路径,`extra_instructions:` 都会以空行分隔追加到末尾。同一份基础 prompt 上想叠 agent-instance 维度的微调而不 fork 文件,用这个。

#### 自动生成的身份块长什么样

没有 `system:` 也没有 `path:` 时,agent 会生成大概这样的内容:

```
You are Nova Patel, a 27-year-old field engineer turned AI researcher.

Personality: methodical, asks clarifying questions, prefers concrete examples
over abstractions.

Background: Started in industrial automation, pivoted to applied LLM research.
Currently embedded with the platform team.

Today's plan: shipping the v3 ingestion pipeline by Friday.

Your planning horizon for this run: 1 day.
```

…由 `name`/`age`/`traits`(一行)、`backstory`(段落)、`initial_plan`(段落)、`cognitive.planning_horizon`(末行)组装。这是个最简骨架 —— 任何生产环境的 agent 都建议自己写模板。

#### `prompts/system.md` 具体例子

```markdown
You are {name}, a {age}-year-old {traits} field engineer turned AI researcher.

# Background

{backstory}

# Today

{initial_plan}

# How to behave

- Speak in first person, in natural English. Be concise — sentences, not paragraphs.
- When the answer needs information from earlier conversations or stored facts,
  call `memory_recall` instead of guessing. Don't tell the user you're doing this;
  just do it.
- When the answer needs work done in the world (file lookups, web searches,
  computations), call the appropriate tool.
- If a tool fails or returns nothing useful, acknowledge it briefly and move on.
- Stay in character. You're an engineer, not a chatbot.
```

六个占位符(`{id} {name} {age} {traits} {backstory} {initial_plan}`)通过 Python 的 `str.format` 渲染。其它的 —— `{plan}`、`{date}`、`{user}` —— 都会 `KeyError`。

#### `extra_instructions:` 追加位置

最终 prompt 形态:

```
<resolved-prompt-from-path-or-inline-or-auto-built>
<空行>
<extra_instructions>
```

适用场景:
- 在共享基础 prompt 上加输出格式约束(`Always respond as JSON.`)
- 不动模板,只针对某个 agent 实例收紧语气
- 按环境覆盖("生产环境永远不要暴露 stack trace。")

`AgentConfig.extra_instructions`(Python 端覆盖)优先级高于 `profile.prompt.extra_instructions`,两者都设时取前者 —— 适合做运行时分层。

#### 失败模式与 fallback 行为

| 问题 | 行为 |
|---|---|
| `path:` 指向不存在的文件 | profile load 阶段就抛 `ConfigValidationError` |
| 模板引用了未知占位符(例如 `{date}`) | 渲染报错 → fallback 到自动身份块,run 继续。日志里有 warning。 |
| `system:` 和 `path:` 都设了 | `ConfigValidationError` —— 选其中一个,不能两个都给 |
| 两个都空 + 身份字段不完整 | 自动生成块只有在身份本身校验失败时才出错(那一步在更早就已经失败了) |

fallback 到自动身份块的行为是**故意的**:模板里一个 typo 不该让生产环境的 agent 崩掉。warning 会进日志,可以不重启就修。

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

DefenseAgent 可以把图片挂到 user turn,让 LLM 同时基于文字和视觉内容推理。这是**按需开启** —— 只有真传 `images=` 时才走这条路径。不传时,本 README 其它一切照常生效。

### "多模态"在这里指什么

OpenAI 的 chat-completions 接口允许 user 消息的 `content` 是**一组 content block**而不是一段纯字符串。每个 block 要么是文字、要么是 `image_url`。DefenseAgent 的 `Message` 类型本来就支持这种形态,`agent.run(task, images=[...])` 只是个 ergonomic 帮手,帮你把 list 拼好。

适用场景:

- 视觉问答 —— "这张截图里是什么?" "这张 PNG 里的图表是涨还是跌?"
- OCR —— 从收据、扫描 PDF(逐页)、代码截图里提取文字
- 视觉调试 —— 把 UI 截图丢给 agent,让它建议 CSS 修复
- 图像基础推理 —— 比较两张商品图、找异常、布局审查

**不适用**:图像生成(没接 SDXL 之类)、视频、音频。只接收静态图片送进 LLM。

### 选个支持视觉的模型

[厂商列表](#厂商与凭证) 里的默认 chat 模型都是**纯文本**。要用 `images=`,在同一厂商下换一个支持视觉的 model id —— 通常只改 `<PROVIDER>_MODEL`,其它环境变量不动:

| 厂商 | 支持视觉的模型 | 备注 |
|---|---|---|
| OpenAI | `gpt-4o`、`gpt-4o-mini`、`gpt-4-turbo`(视觉端点) | OCR 类任务最便宜的默认是 `gpt-4o-mini` |
| Qwen(DashScope) | `qwen-vl-max`、`qwen-vl-plus`、`qwen-vl-max-latest` | `-vl-` 前缀代表视觉;非 VL 版的 Qwen 不接受图片 |
| GLM(智谱,OpenAI 兼容) | `glm-4v`、`glm-4v-flash` | 走 GLM 的 OpenAI 兼容端点:`provider: openai` + `OPENAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4` |
| Kimi(Moonshot,OpenAI 兼容) | `moonshot-v1-32k-vision-preview` | 同上,把 `OPENAI_BASE_URL` 指向 Moonshot |
| vLLM(自部署) | 服务上挂的任意视觉模型,例如 `Qwen/Qwen2-VL-7B-Instruct`、`llava-hf/llava-1.5-13b-hf` | vLLM 启动时要加 `--limit-mm-per-prompt image=N` |
| **Anthropic** | **当前不支持** —— 见下面的 "Anthropic 限制" |

设置流程跟其它模型一样,只把 `<PROVIDER>_MODEL` 换成视觉模型 id:

```bash
# .env —— DashScope 上的 Qwen-VL
AGENT_LAB_LLM_PROVIDER=qwen
QWEN_API_KEY=sk-…
QWEN_MODEL=qwen-vl-max
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

### 端到端例子:图像识别

完整可跑的例子。把截图丢进工程目录,让 agent 去看:

```python
import asyncio
from pathlib import Path
from DefenseAgent.agent import AgentConfig, ReActAgent
from DefenseAgent.examples import EXAMPLE_PROFILE_PATH

async def main():
    agent = ReActAgent(AgentConfig(profile=EXAMPLE_PROFILE_PATH))

    result = await agent.run(
        "描述这张图片的内容,包含你能识别的文字。",
        images=[Path("./screenshot.png")],
    )
    print(result.final_answer)

asyncio.run(main())
```

```
$ python recognise.py
图片显示一个终端,里面是 `pytest -v` 的输出。可以看到测试名包括
test_agent_profile_minimal_with_only_id_and_name,底部一行写着
"532 passed, 3 skipped in 4.88s"。背景看起来是 iTerm2 默认的暗色主题。
```

agent 把图片当作 user turn 的一部分 —— LLM 原生看到图,不需要单独走 OCR 流程。识别效果取决于你选的视觉模型:生产用 `qwen-vl-max` 或 `gpt-4o`;小模型在小字、细节上明显差。

### 图片在系统里怎么流转

`agent.run(task, images=[...])` 会逐项遍历 `images=`,每项归一化成一个 URL 字符串,再拼成 OpenAI content-block 消息。三种输入类型都接受:

| 输入 | 处理方式 |
|---|---|
| `Path` / 本地文件路径字符串 | 读取文件 → base64 编码 → 拼成 `data:<mime>;base64,…` URL。MIME 根据扩展名推断(`.png` → `image/png`、`.jpg` → `image/jpeg` …);未知扩展名默认 `image/png`。 |
| `http://` 或 `https://` URL 字符串 | 原样透传。厂商自己去 fetch,DefenseAgent 不下载。 |
| `data:` URL 字符串(已编码) | 原样透传 —— 适合手里已经是 `BytesIO` 自己编完码的场景。 |

归一化后的 URL 进到这个最终请求形态(OpenAI 兼容适配器实际发出去的就是这个):

```python
{
  "role": "user",
  "content": [
    {"type": "text", "text": "<你的任务字符串>"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR..."}},
    {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}},
  ]
}
```

agent **不做任何预处理** —— 不缩放、不压缩、不调画质。你给什么字节,厂商就看什么字节。这有两个实际后果:

1. **Base64 编码会让 payload 膨胀约 33%。** 5 MB 的 PNG 编完是 ~6.7 MB base64。大图会让每次调用明显变慢。模型能用小图就先缩再传。
2. **厂商各有大小限制。** OpenAI 拒绝 ~20 MB 以上的 request body;DashScope 各模型限制不同。超了会被厂商以 4xx 拒,DefenseAgent 不会预先报友好错误。

本地文件用 `Path` 或字符串都行 —— base64 转换在 `_resolve_image_url`(单个 ~10 行的模块帮手)里完成。**图片本来就公开时,优先用 URL** —— 这样跳过 base64 膨胀,厂商还能缓存。

### 约束 + 实践建议

- **一次 turn 多张图:** DefenseAgent 这边没限。但厂商一般有上限(OpenAI 通常最多 10 张,Qwen-VL 类似)。超了 → 请求失败。
- **支持的格式:** 看模型。PNG / JPEG 通用;WebP、GIF(只看第一帧)、BMP 多数厂商都支持;HEIC、AVIF 不稳定。
- **透明通道:** PNG alpha 通道原样透传。视觉模型多半忽略它。
- **OCR 重的任务:** 别狠缩(高分辨率出效果)、选 OCR 强的模型(`qwen-vl-max`、`gpt-4o`)。
- **批量处理:** 多张图想批处理时,**多个 `agent.run()` 并发**比一次塞进同一轮更好 —— 总 token 成本一样,但 wall-clock 更快、错误隔离更容易。

### 多步 agent 里图片怎么传

| Agent | 图片携带方式 |
|---|---|
| `SimpleAgent` | 一轮一调,图片挂在唯一的 user 消息上。 |
| `ReActAgent` | **只挂在最初的 user turn**。后续 tool 结果消息保持纯文本 —— LLM 已经看过图,不需要重复挂。 |
| `PlanAndSolveAgent` | **Phase 1(规划)消息** 和 **Phase 2(每步执行)消息** 都挂同一份图,让每个引用原始任务的阶段都能再看图。Phase 3(综合)是纯文本的 —— 它在每步的文本输出上做总结。 |

也就是说,n 步的 ReAct 看一张图,只有第 1 次调用是带图的,其余 (n-1) 次是纯文本。成本大致是:`1 × (text + image) + (n-1) × text`,不是 n × image。

### Anthropic 限制

Claude 的非文本 content 走 Anthropic 自家的 `{"type": "image", "source": {...}}` block 形态,**不是** OpenAI 的 `{"type": "image_url", ...}`。当前的 `AnthropicAdapter` 不做格式翻译 —— list-shape `content` 进来会抛:

```python
LLMAdapterError: AnthropicAdapter received list-shape content but does not yet
support multimodal translation. Use an OpenAI-compatible vision provider, or
pass plain text content.
```

`Message` 类型本身已经支持 list content,缺的就是 Anthropic 适配器内部的一个 content-block 翻译。欢迎 PR —— 改动局限在 [`DefenseAgent/llm/anthropic.py`](DefenseAgent/llm/anthropic.py)。

现阶段需要视觉的话:从上面的 OpenAI 兼容厂商里选一个。

## 自定义与依赖注入

agent 依赖的每个组件都可以通过 `AgentConfig` 替换。给定预构造组件时,**该组件的 env 构造路径完全跳过**,系统其它部分(其它组件 + 它们的 env fallback)不受影响。这是主要的扩展面 —— 不 fork 框架就能继承、mock、替换任意一层。

### 子系统开关

```python
config = AgentConfig(
    profile="…",
    use_tools=True,         # 默认。False → 不构造 tool registry,LLM 看不到任何工具。
    use_memory=True,        # 默认。False → 跳过 mem0 setup,不注册 memory_recall 工具。
    use_reflection=True,    # 默认。False → 不构造 Reflector,run 后不走反思。
    use_rag=None,           # 默认 → 跟随 profile.rag.enabled。True/False 显式覆盖。
    use_compressor=True,    # 默认。False → ContextCompressor 永不运行(自己管上下文)。
    use_logger=True,        # 默认。False → 不构造 AgentLogger,事件被压制。
)
```

`use_memory` 关掉时,依赖项自动失效:`save_outcome`、`save_trajectory`、`reflect_after_run` 全变 no-op(没 memory 后端 → 没地方写)。不需要自己手动翻。

### 可替换组件

```python
config = AgentConfig(
    profile="…",

    # 任意一项给定后,自动构造的版本被替换掉。
    llm=my_llm,                       # LLM 实例(任意适配器)
    memory=my_mem0_memory,            # Mem0Memory 或鸭子类型兼容
    tool_registry=my_registry,        # 已经填好工具的 ToolRegistry
    logger=my_logger,                 # AgentLogger
    reflector=my_reflector,           # Reflector
    compressor=my_compressor,         # ContextCompressor
    rag=my_rag,                       # LlamaIndexRAG(或任何带 .search(query, top_k) 的对象)

    # mem0 后端控制 —— 仅在 memory=None 且 use_memory=True 时使用。
    # 让你以编程方式配置 mem0 *内部* 的 LLM/embedder(跟 agent 自己的 chat LLM 是
    # 两回事),不需要碰 .env。
    memory_backend=MemoryBackendConfig(
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        embedder_provider="openai",
        embedder_model="text-embedding-3-small",
    ),
)
```

### 内联工具注入(不写进 profile)

除了 `tools.python:` 里写的,直接传 plain callable:

```python
def my_search(query: str) -> str:
    """Web search via my custom backend."""
    ...

config = AgentConfig(profile="…", tools=[my_search])
```

这些跟 `tools.python:` 的条目一起注册到同一个 `ToolRegistry`。自动派生规则一样:签名 → schema、docstring → description。

### 常见模式

**同进程多 LLM。** 两个 config,共享除 `llm` 之外的一切:

```python
shared = dict(profile="…", memory=shared_memory, tool_registry=shared_registry)
config_fast  = AgentConfig(**shared, llm=cheap_llm)
config_smart = AgentConfig(**shared, llm=expensive_llm)
```

**用 scripted response 做测试。** 一个 `ScriptedLLM`,按顺序返回预设的 `LLMResponse` —— 整个测试套件就是这么做的。

```python
config = AgentConfig(profile="…", llm=ScriptedLLM([resp(content="ok")]))
```

**自定义 memory 后端。** 继承 `Mem0Memory`,override `search_records()`:

```python
class CachedMemory(Mem0Memory):
    def search_records(self, query, **kw):
        if query in self._cache:
            return self._cache[query]
        result = super().search_records(query, **kw)
        self._cache[query] = result
        return result

config = AgentConfig(profile="…", memory=CachedMemory(profile=profile))
```

**插一个不同的 RAG 后端。** 任何带 `search(query: str, top_k: int) -> list[dict]` 方法的对象都行:

```python
class ElasticRAG:
    async def search(self, query, top_k=5):
        # 查 Elasticsearch 而不是 FAISS...

config = AgentConfig(profile="…", rag=ElasticRAG(), use_rag=True)
```

agent 的 `rag_search` 工具走你这个对象,跟走 `LlamaIndexRAG` 完全一样。

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
| `DefenseAgent/examples/` | `EXAMPLE_AGENT_DIR` + 包内自带的参考 profile |

memory、MCP、skill、RAG 模块均继承自 [ms-agent](https://github.com/modelscope/ms-agent) 的上游类。

## 本地开发

如果你想改 DefenseAgent 自身(不是只用它),clone 仓库,以可编辑模式装上 dev extras:

```bash
git clone https://github.com/yishu031031/DefenseAgent.git
cd DefenseAgent
python -m venv .venv && source .venv/bin/activate
pip install -e '.[all,dev]'
```

跑测试套件(离线,不需要网络或外部服务):

```bash
pytest                       # 全套
pytest -k tools              # 只跑某个模块
pytest -x --tb=short         # 第一次失败就停
```

531 个测试,3 个 skip。

仓库里 `scripts/` 目录下还有几个独立的 demo 脚本(不在 wheel 里):

```bash
python scripts/react_tools_memory_demo.py     # ReAct + calculator + Tavily + memory recall
python scripts/profile_chat_demo.py           # 用 example profile 跑一次单轮对话
python scripts/tools_demo.py                  # 演示 skill 工具的三层
python scripts/memory_demo.py                 # mem0 add / search / dump
```

## License

MIT.
