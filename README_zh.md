# DefenseAgent

> [English](README.md) · **中文**

[![PyPI](https://img.shields.io/pypi/v/defense-agent.svg)](https://pypi.org/project/defense-agent/)
[![Python](https://img.shields.io/pypi/pyversions/defense-agent.svg)](https://pypi.org/project/defense-agent/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**用一份 YAML 文件构建 LLM Agent，一行 Python 就能跑起来。**

---

DefenseAgent 是一个面向单 Agent LLM 应用的 Python 框架。在一份严格校验的 YAML profile 里描述你的 agent —— 身份、LLM 供应商、工具、记忆、RAG、提示词 —— 然后一行代码实例化，在三种执行策略之一上运行任务。

```python
from DefenseAgent.agent import AgentConfig, ReActAgent
from DefenseAgent.examples import EXAMPLE_PROFILE_PATH

agent = ReActAgent(AgentConfig(profile=EXAMPLE_PROFILE_PATH))
result = await agent.run("用一句话总结今天的计划。")
```

## 核心特性

- 🧾 **一份文件定义 Agent** —— 身份、LLM、工具、记忆、RAG、提示词全部写在一份严格校验的 YAML 里。未知字段在加载时直接报错（`extra="forbid"`）。
- 🔌 **供应商无关** —— `openai`、`anthropic`、`deepseek`、`qwen`、`google`、`vllm`。改 `.env` 即可切换，无需改代码。
- 🎯 **三种执行策略** —— `SimpleAgent`（单次）、`ReActAgent`（工具循环）、`PlanAndSolveAgent`（规划→执行→综合）。都基于同一份 `AgentConfig`。
- 🧠 **分层记忆架构（0.2.0）** —— 四个生命周期 tier（Working / Episodic / Semantic / Procedural），混合打分（相似度 × 时效 × 重要性 × 频次），可选后台 consolidation。
- 🛠️ **三种工具来源，一个注册表** —— 本地 skill 包（`SKILL.md`）、MCP 服务器（stdio / SSE / WebSocket / streamable-http）、Python 可调用对象（按文件路径或点分模块）。
- 🖼️ **可选 RAG + 视觉** —— 放文档进目录得到 `rag_search` 工具；传 `images=[…]` 走多模态。**默认关闭** —— 用到才付出代价。

## 安装

```bash
pip install 'defense-agent[memory]'    # 推荐 —— 默认配置需要
```

| Extra | 拉进什么 |
|---|---|
| `defense-agent` | 仅核心（必须传 `use_memory=False`）|
| `defense-agent[memory]` | `mem0ai[nlp]` + `fastembed` —— 持久化记忆 + `memory_recall` 工具 |
| `defense-agent[rag]` | `llama-index` + 抽取器 —— RAG + `rag_search` 工具 |
| `defense-agent[mcp]` | `mcp` —— 连接 MCP 工具服务器 |
| `defense-agent[all]` | memory + rag + mcp |
| `defense-agent[dev]` | `pytest` + `pytest-asyncio` 跑测试 |

要求 **Python ≥ 3.10**。首次安装大约 1 GB（核心依赖 `ms-agent` 间接拉入 `torch`）。

## 快速开始

```bash
mkdir myagent && cd myagent
python -m venv .venv && source .venv/bin/activate
pip install 'defense-agent[all]'
```

把供应商凭据放进 `.env`：

```bash
AGENT_LAB_LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-…
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1

# 仅当用到 memory_recall / rag_search 时需要：
EMBEDDING_API_KEY=sk-…
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMS=1536
```

跑内置的示例 profile：

```python
# run_example.py
import asyncio
from DefenseAgent.agent import AgentConfig, ReActAgent
from DefenseAgent.examples import EXAMPLE_PROFILE_PATH

async def main():
    async with ReActAgent(AgentConfig(profile=EXAMPLE_PROFILE_PATH)) as agent:
        result = await agent.run("用一句话总结今天的计划。")
        print(result.final_answer)

asyncio.run(main())
```

把示例 profile 拷出来，开始编辑你自己的：

```bash
python -c "from DefenseAgent.examples import EXAMPLE_AGENT_DIR; import shutil; shutil.copytree(EXAMPLE_AGENT_DIR, './my_profile')"
```

完整的 profile schema 见 [`docs/configuration.md`](docs/configuration.md)。

## 架构

```
AgentConfig ── profile.yaml + .env
     │
     ▼
build_components_sync ── LLM、Memory、ToolRegistry、Reflector、Compressor、Logger
     │
     ▼
BaseAgent ◀──── ReActAgent | SimpleAgent | PlanAndSolveAgent
     │
     ▼
run(task) ──► AgentResult { final_answer, steps[], usage }
```

`build_components_sync` 是同步构建。MCP 服务器连接和 RAG 索引在首次 `run()` 时懒加载。Memory、MCP、Skills 和 RAG 继承自 [ms-agent](https://github.com/modelscope/ms-agent) 的上游类 —— DefenseAgent 在其上加了 tier-aware 编排器、RAG 抽取器、profile 桥接和统一的 agent 循环。

## 记忆分层（0.2.0 起）

DefenseAgent 的记忆模块是受 Hello-Agents 启发的四层架构，持久化 tier 基于 mem0 + Qdrant。所有写入通过统一的 `MemoryOrchestrator` 入口 —— agent 选 tier；读取默认走跨 tier 混合打分。

```
                  MemoryOrchestrator
                          │
        ┌─────────┬───────┴──────┬──────────────┐
        ▼         ▼              ▼              ▼
    WORKING   EPISODIC       SEMANTIC      PROCEDURAL
   (内存层)   (轨迹/事件)    (反思/经验)   (SOP/攻击模式)
        │         │              │              │
        │         └────────┬─────┴──────────────┘
        │                  ▼
        │            mem0 + Qdrant
        │
        └─►  MemoryConsolidator（可选后台晋升）
```

| Tier | 存储 | 典型内容 |
|------|------|---------|
| **Working** | 内存 deque，TTL + FIFO | 当前会话的临时草稿 |
| **Episodic** | Qdrant (`tier=episodic`) | 原始事件、agent 执行轨迹 |
| **Semantic** | Qdrant (`tier=semantic`) | 提炼的事实、反思、经验教训 |
| **Procedural** | Qdrant (`tier=procedural`) | SOP、攻击模式、工作流 |

LLM 通过自动注册的 `memory_recall` 工具访问记忆。内部细节（混合打分公式、生命周期 consolidation、Working 层淘汰策略）见 [`docs/memory.md`](docs/memory.md)。

## 文档

| 文档 | 内容 |
|------|------|
| [`docs/configuration.md`](docs/configuration.md) | 完整 `profile.yaml` schema —— 身份、cognitive、prompt、RAG 参数、per-field 回退规则、校验失败模式 |
| [`docs/providers.md`](docs/providers.md) | LLM 供应商表、embedding 配对、各家注意点、编程式 LLM 注入、多模态/视觉模型 |
| [`docs/tools.md`](docs/tools.md) | 工具来源 —— 本地 skill 包、MCP 服务器、Python 入口、LLM 永远可见的内置工具 |
| [`docs/memory.md`](docs/memory.md) | Tier-aware 记忆模块 —— `MemoryOrchestrator` API、打分权重、生命周期 consolidation、Working 层语义 |
| [`docs/architecture.md`](docs/architecture.md) | 模块布局、agent 类 + `AgentResult` 结构、`AgentConfig` 全字段、自定义与依赖注入、本地开发 |

文档目前为英文版。

## 本地开发

```bash
git clone https://github.com/yishu031031/DefenseAgent.git
cd DefenseAgent
python -m venv .venv && source .venv/bin/activate
pip install -e '.[all,dev]'
pytest
```

测试套件完全离线（不需要网络或外部服务）。针对真实 Ollama + Qdrant 的后端冒烟测试见 [`scripts/smoke_real_backend.py`](scripts/smoke_real_backend.py)。

## License

MIT。
