# DefenseAgent

> [English README](README.md) · 中文

一个多 LLM 的 agent 框架。配齐了 mem0 后端的长期记忆、llama-index 后端的 RAG、MCP 工具支持、反思机制。从一份 YAML 角色配置出发，三行代码起一个能用的 agent；不改代码就能切换 LLM 厂商；测试时干净地注入 mock。

```python
from DefenseAgent import AgentConfig, ReActAgent

config = AgentConfig(profile="agents/maya_rodriguez/profile.yaml")
agent = ReActAgent(config)
result = await agent.run("我今天上午在干嘛？")
print(result.final_answer)
```

---

## 亮点

- **LLM 厂商无关**。一个 `LLM` facade 屏蔽 Anthropic 和所有 OpenAI 协议兼容厂商（DeepSeek、Qwen / DashScope、ModelScope、vLLM、Google 走代理）。改 `.env` 即可切换；或者代码里 `LLM.from_kwargs(provider="...", api_key="...", model="...")`。
- **三种推理策略**。`SimpleAgent`（单轮）、`ReActAgent`（Yao et al. 2022 —— 推理与工具调用交替）、`PlanAndSolveAgent`（Wang et al. 2023 —— 计划/执行/综合三阶段）。三个共用同一个 `AgentConfig` 入口。
- **持久化语义记忆**。mem0 + qdrant 落盘存储。跨轮 `memory_recall` 作为内置 LLM 工具暴露；outcome / failure / trajectory 用 `memory_type` 分类，便于后续筛选与反思。
- **多模态 RAG**。`LlamaIndexRAG` + `StructuredDocExtractor`，HTML / PDF 切块时**保留嵌入图片和表格**。装了 `fastembed` 之后启用向量 + BM25 混合检索。
- **三种工具来源统一接口**。普通 Python 函数（`@registry.tool`）、Anthropic 风格 Skill 包（一个目录加 `SKILL.md`）、stdio 协议的 MCP 服务器 —— 都通过同一个 `ToolRegistry`。
- **反思机制**。`Reflector` 收集未反思过的记忆，让 LLM 综合成高层洞察，回写到 mem0 并标记 `memory_type="reflection"`。阈值触发，**永远不会向上抛错**。
- **单一构造路径**。每个 agent 都通过一个 `AgentConfig` 对象构建 —— 没有重载构造器、没有"两个门"。测试时通过 `AgentConfig.llm` / `memory` / `tools_registry` / `reflector` / 等字段注入 mock。

---

## 安装

```bash
git clone <repo-url>
cd DefenseAgent

python -m venv .venv3
.venv3/Scripts/activate          # Windows
# source .venv3/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

完整依赖会拉进 `ms-agent`、`mem0ai`、`llama-index-core`、`qdrant-client`、（modelscope 链路上的）`torch` 等等，磁盘占用约 1 GB。把这些拆成 PyPI 可选 extras 在路线图里。

---

## 配置

### .env —— 聊天 LLM + embedding

复制 `.env.example` 到 `.env`，**至少填这三块**：

```bash
# 1. 选一家聊天 LLM 厂商
AGENT_LAB_LLM_PROVIDER=deepseek   # anthropic | openai | deepseek | qwen | vllm | google

# 2. 对应厂商的 key + model + base_url
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat

# 3. embedding（mem0 必须；DeepSeek/Anthropic 自己不出 embedding）
EMBEDDING_API_KEY=sk-...
EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_MODEL=text-embedding-v3
EMBEDDING_DIMS=1024
```

字段优先级：`LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL_ID`（全局覆盖）> `<PROVIDER>_API_KEY` 等（厂商专属）。

### Profile YAML —— agent 的"人设"

每个 agent 一个目录，放在 `agents/<name>/profile.yaml`：

```yaml
agent:
  id: "student_maya_001"
  name: "Maya Rodriguez"
  age: 20
  traits: "好奇、坚持、善于合作"
  backstory: >
    州立大学计算机科学系的二年级学生……
  initial_plan: >
    7:30 起床、复习笔记、9 点上课……

  cognitive:
    max_steps_per_cycle: 8
    reflection_threshold: 4

  memory:
    search_limit: 8
    history_mode: add

  tools:
    skills:
      - skills/tabular-report
    mcp: []

  prompt:
    path: prompts/system.md   # 或者用 `system: "..."` 写内联
```

profile 目录里还能放：
- `prompts/system.md` —— 外置的系统提示
- `skills/<skill-name>/` —— Anthropic 风格 Skill 包（带 `SKILL.md` 和资产）
- `memory/` —— 自动生成；mem0 的 qdrant 集合存这里

可参考 [agents/maya_rodriguez/](agents/maya_rodriguez/) 和 [agents/alice_chen/](agents/alice_chen/)。

---

## 快速开始

### 角色扮演式对话

```python
import asyncio
from DefenseAgent import AgentConfig, ReActAgent

async def main():
    config = AgentConfig(profile="agents/maya_rodriguez/profile.yaml")
    async with ReActAgent(config) as agent:
        result = await agent.run("现在下午 2 点，你今天上午都做了什么？")
        print(result.final_answer)

asyncio.run(main())
```

### 纯代码构造（不依赖 .env）

```python
from DefenseAgent import AgentConfig, ReActAgent
from DefenseAgent.llm import LLM
from DefenseAgent.memory import MemoryBackendConfig

llm = LLM.from_kwargs(
    provider="anthropic", api_key="sk-...", model="claude-opus-4-7",
)
backend = MemoryBackendConfig(
    llm_provider="deepseek", llm_api_key="sk-...", llm_model="deepseek-chat",
    llm_base_url="https://api.deepseek.com",
    embedding_api_key="sk-...", embedding_model="text-embedding-3-small",
    embedding_base_url="https://api.openai.com/v1", embedding_dims=1536,
)
config = AgentConfig(
    profile="agents/maya_rodriguez/profile.yaml",
    llm=llm,
    memory_backend=backend,
    load_env=False,
)
agent = ReActAgent(config)
```

### 注册自定义工具

```python
def calculator(expression: str) -> str:
    """Evaluate a math expression and return the numeric result."""
    return str(eval(expression))   # toy demo only

config = AgentConfig(
    profile="agents/maya_rodriguez/profile.yaml",
    tools=[calculator],
)
agent = ReActAgent(config)
```

函数名 + docstring + 类型签名会自动反推为 LLM 看到的 JSON-schema 工具规范。

---

## 三种 Agent 策略

| 类 | 主循环 | 适用场景 |
|---|---|---|
| `SimpleAgent` | 一次 LLM 调用，无工具循环 | 纯对话、角色扮演、单轮问答 |
| `ReActAgent` | LLM → 工具调用 → 工具结果 → LLM → ... 直到 LLM 给出纯文本答案 | 最通用；推理与工具调用交错 |
| `PlanAndSolveAgent` | 先规划成 N 步 → 每步用工具执行 → 综合答案 | 多步任务，先确定计划能减少漂移 |

三个 agent 都接受同一个 `AgentConfig`，返回同一个 `AgentResult`（包括 `final_answer` 和完整的 step trace）。

---

## 模块概览

```
DefenseAgent/
├── llm/         模块 1 —— LLM 适配层（anthropic + openai 兼容）
├── config/      模块 2 —— pydantic 校验的 YAML profile 加载器
├── ops/         模块 3 —— 每个 agent 一个的 JSON-lines 日志器
├── memory/      模块 4 —— mem0 后端记忆（继承自 ms-agent）
├── reflection/  模块 5 —— 重要性评分 + 洞察综合
├── tools/       模块 6 —— Python 函数 / Skill 包 / MCP 服务器
├── rag/         模块 5+ —— LlamaIndex 知识库，带多模态文档抽取
└── agent/       模块 7 —— BaseAgent + Simple / ReAct / PlanAndSolve
```

每个模块在 `docs/superpowers/specs/` 有设计文档，在 `docs/walkthroughs/` 有用户视角教程。

---

## 项目结构

```
DefenseAgent/
├── DefenseAgent/             # 库本体（包名暂为大写驼峰；改成 PEP 8 风格在路线图里）
├── agents/
│   ├── alice_chen/           # 数据科学家人设，内联 prompt
│   └── maya_rodriguez/       # CS 学生人设，外置 prompt + Skill 包
├── docs/
│   ├── superpowers/specs/    # 设计文档（每模块一份）
│   └── walkthroughs/         # 用户教程（每模块一份）
├── scripts/                  # 可直接运行的 demo（见下表）
├── tests/                    # pytest 套件，397 / 398 通过
├── .env.example
├── pytest.ini
└── requirements.txt
```

---

## Demo 列表

下面所有 demo 都假设 `.env` 已配好。从项目根目录运行，`.venv3` 已激活。

| Demo | 演示内容 |
|---|---|
| `python scripts/show_profile.py` | 加载并校验 profile YAML，无 API 调用 |
| `python scripts/smoke_chat.py` | LLM facade 最小端到端测试 |
| `python scripts/profile_chat_demo.py` | profile + LLM —— Maya 用人设回答 |
| `python scripts/streaming_demo.py` | 流式文本 deltas |
| `python scripts/logger_demo.py` | 结构化 JSON-lines 日志的用法 |
| `python scripts/tools_demo.py` | Python 函数 + Skill 包作为 agent 工具 |
| `python scripts/memory_demo.py` | 记忆写入、语义检索、反思 |
| `python scripts/reflection_demo.py` | 反思机制单独触发 |
| `python scripts/react_tools_memory_demo.py` | **最综合的 demo** —— 三轮对话覆盖 calculator + Tavily 搜索 + 跨轮 memory_recall |
| `python scripts/structured_extraction_demo.py --with-rag` | HTML / PDF → 切块 → 保留图片的多模态 RAG |
| `python scripts/dump_memory.py` | 查看某个 agent 的 mem0 目录里到底存了什么 |

**第一次跑就跑 `react_tools_memory_demo.py`** —— 它覆盖的代码路径最广，是验证整套环境是否健康最好的"冒烟测试"。

---

## 测试

```bash
.venv3/Scripts/python.exe -m pytest tests/ -v
```

当前 **397 / 398 通过**。唯一一个失败 (`test_data_with_path_serializes_via_str`) 是 Windows 路径分隔符问题，跟框架本身无关 —— 它断言 POSIX 风格的 `/tmp/x` 在字符串里，但 Windows 把 `Path("/tmp/x")` 序列化成 `\tmp\x`。

### 跑特定模块的测试

```bash
pytest tests/DefenseAgent/llm/                 # LLM 适配器
pytest tests/DefenseAgent/agent/               # agent 构造与循环
pytest tests/DefenseAgent/memory/              # 内存与 bridge
pytest tests/DefenseAgent/tools/               # 工具注册 / Skill / MCP
pytest tests/DefenseAgent/test_integration.py  # 跨模块集成
```

测试 **完全离线** —— 通过 `make_test_config(...)`（在 `tests/DefenseAgent/agent/_support.py`）注入 `ScriptedLLM` + `MagicMock` memory，没有任何真实 API 调用。

---

## 设计要点

### 单一构造路径
每个 agent 通过 `Agent(config)` 构造，`config: AgentConfig`。v0.1 重构里删掉了老的 `Agent(profile, llm=..., memory=..., ...)` keyword 入口 —— 现在通过 `AgentConfig.llm` / `memory` / `tools_registry` / `reflector` / `compactor` / `rag` 字段做注入。

### 两套 memory 配置入口，同一个对象出口
`MemoryBackendConfig`（纯代码）和 `.env`（默认）最终产生同样的 mem0 连接。SDK 调用方传 `AgentConfig(memory_backend=...)`；本地开发用 `.env`。

### 厂商按需导入
`DefenseAgent.llm._registry._resolve_adapter(provider)` 把厂商 SDK 的 import 放在 if 分支**内部**。一个只用 DeepSeek 的用户，不用承担 import `anthropic` 包的开销。

### 两阶段构造
`AgentConfig` → `build_components_sync(config)`（同步：LLM / memory / tools / reflector / compactor / logger）→ `agent._ensure_async_setup()`（懒加载：MCP 服务器 + RAG，这两者必须 `await`）。

---

## 路线图

SDK 化还要做的几件大事：

- **`pyproject.toml`** —— 让项目能 `pip install`，带 `[memory]`、`[rag]`、`[dev]` 可选 extras
- **顶层 `__init__.py` 改成 lazy 导入** —— 当前导入 agent 模块时会顺带把 torch（通过 modelscope）拖进来
- **PEP 8 包名重命名** —— `DefenseAgent/` → `defense_agent/`
- **`create_agent()` 一行起 agent 工厂** —— 把 `AgentConfig + Agent(config)` 收成一个调用
- **多 agent 通信** —— 进程内 `AgentBus` + `AgentSwarm`，让 agent 互相调用
- **LLM facade 加重试 / 超时 / 日志** —— 把横切关注点从每个 adapter 上移到 facade

各模块的设计原理见 `docs/superpowers/specs/`。

---

## 协议

待定 —— 发布前补一份 LICENSE 文件。
