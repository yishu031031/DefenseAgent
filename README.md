# DefenseAgent

> English · [中文 README](README_zh.md)

A multi-LLM agent framework with mem0-backed memory, llama-index-backed RAG, MCP tool support, and reflection. Build a working agent from a YAML profile in three lines, swap LLM providers without touching code, and inject mocks cleanly for tests.

```python
from DefenseAgent import AgentConfig, ReActAgent

config = AgentConfig(profile="agents/maya_rodriguez/profile.yaml")
agent = ReActAgent(config)
result = await agent.run("What did I work on this morning?")
print(result.final_answer)
```

---

## Highlights

- **Provider-agnostic LLM layer.** One `LLM` facade in front of Anthropic and any OpenAI-compatible provider (DeepSeek, Qwen/DashScope, ModelScope, vLLM, Google-via-proxy). Swap by editing `.env` or by passing `LLM.create(provider="...", api_key="...", model="...")`.
- **Three reasoning strategies.** `SimpleAgent` (one-shot), `ReActAgent` (Yao et al. 2022 — interleaved reasoning + tool use), `PlanAndSolveAgent` (Wang et al. 2023 — plan / execute / synthesize). All share the same `AgentConfig` constructor.
- **Persistent memory with semantic recall.** mem0 + qdrant on disk. Cross-turn `memory_recall` is exposed as a built-in LLM tool; outcomes / failures / trajectories are tagged via `memory_type` for later filtering and reflection.
- **Multimodal RAG.** `LlamaIndexRAG` with `StructuredDocExtractor` for HTML / PDF chunking that preserves embedded images and tables. Hybrid (vector + BM25) retrieval when `fastembed` is installed.
- **Three tool sources.** Plain Python functions (`@registry.tool`), Anthropic-style Skill bundles (a directory with `SKILL.md` + assets), and stdio MCP servers — all unified under one `ToolRegistry`.
- **Reflection.** `Reflector` gathers unreflected memory records, asks the LLM to synthesize high-level insights, writes them back tagged `memory_type="reflection"`. Threshold-gated; never throws.
- **One construction path.** Every agent is built from a single `AgentConfig` object — no overloaded constructors, no two-doors. Inject mocks for tests via the same `AgentConfig.llm`/`memory`/`tool_registry`/`reflector`/etc. fields.

---

## Installation

```bash
git clone <repo-url>
cd DefenseAgent

python -m venv .venv3
.venv3/Scripts/activate          # Windows
# source .venv3/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

The full stack pulls in `ms-agent`, `mem0ai`, `llama-index-core`, `qdrant-client`, `torch` (via modelscope) and friends — about 1 GB on disk. PyPI packaging that splits this into optional extras is on the roadmap.

---

## Configuration

### .env — chat provider + embedding

Copy `.env.example` to `.env` and fill in **at least these three blocks**:

```bash
# 1. Pick a chat provider
AGENT_LAB_LLM_PROVIDER=deepseek   # anthropic | openai | deepseek | qwen | vllm | google

# 2. The matching block
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat

# 3. Embedding (mem0 needs it; chat providers don't ship embeddings)
EMBEDDING_API_KEY=sk-...
EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_MODEL=text-embedding-v3
EMBEDDING_DIMS=1024
```

Provider precedence: `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL_ID` (global override) wins over `<PROVIDER>_API_KEY` etc. (per-provider).

### Profile YAML — the agent's identity

Each agent lives in `agents/<name>/profile.yaml`:

```yaml
agent:
  id: "student_maya_001"
  name: "Maya Rodriguez"
  age: 20
  traits: "curious, persistent, collaborative"
  backstory: >
    A second-year Computer Science student...
  initial_plan: >
    Wake up at 7:30, review notes, attend the 9 AM lecture...

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
    path: prompts/system.md   # or `system: "..."` inline
```

A profile bundle can include:
- `prompts/system.md` — externalized system prompt
- `skills/<skill-name>/` — Anthropic-style Skill (a directory with `SKILL.md` + assets)
- `memory/` — auto-created on first run; mem0 stores qdrant collections here

See [agents/maya_rodriguez/](agents/maya_rodriguez/) and [agents/alice_chen/](agents/alice_chen/) for working examples.

---

## Quick Start

### Conversation with role-played agent

```python
import asyncio
from DefenseAgent import AgentConfig, ReActAgent

async def main():
    config = AgentConfig(profile="agents/maya_rodriguez/profile.yaml")
    async with ReActAgent(config) as agent:
        result = await agent.run("It's 2 PM. What have you been doing today?")
        print(result.final_answer)

asyncio.run(main())
```

### Pure-code construction (no .env)

```python
from DefenseAgent import AgentConfig, ReActAgent
from DefenseAgent.llm import LLM
from DefenseAgent.memory import MemoryBackendConfig

llm = LLM.create(
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

### Adding tools

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

The function's name + docstring + signature become its JSON-schema tool spec the LLM sees.

---

## Three Agent Strategies

| Class | Loop | Use when |
|---|---|---|
| `SimpleAgent` | One LLM call. No tools, no loop. | Pure conversation, role-play, single-shot QA |
| `ReActAgent` | LLM → tool calls → results → LLM → ... until plain text answer | Most general-purpose; tool use is interleaved with reasoning |
| `PlanAndSolveAgent` | Plan into N steps → execute each (with tools) → synthesize answer | Multi-step tasks where committing to a plan first reduces drift |

All three accept the same `AgentConfig` and emit the same `AgentResult` (a `final_answer` plus a step-by-step trace).

---

## Module Overview

```
DefenseAgent/
├── llm/         Module 1 — LLM adapter (anthropic + openai-compat)
├── config/      Module 2 — pydantic-validated YAML profile loader
├── ops/         Module 3 — per-agent JSON-lines logger
├── memory/      Module 4 — mem0-backed memory (inherits ms-agent)
├── reflection/  Module 5 — importance scorer + insight synthesizer
├── tools/       Module 6 — python functions / skills / MCP servers
├── rag/         Module 5+ — LlamaIndex-backed knowledge base, multimodal extraction
└── agent/       Module 7 — BaseAgent + Simple / ReAct / PlanAndSolve
```

Each module ships with a design doc under `docs/superpowers/specs/` and a user-oriented walkthrough under `docs/walkthroughs/`.

---

## Project Layout

```
DefenseAgent/
├── DefenseAgent/             # the library (uppercase package name; PEP-8 rename pending)
├── agents/
│   ├── alice_chen/           # data scientist persona, inline prompt
│   └── maya_rodriguez/       # CS student persona, externalized prompt + skill bundle
├── docs/
│   ├── superpowers/specs/    # design docs (one per module)
│   └── walkthroughs/         # user guides (one per module)
├── scripts/                  # runnable demos (see below)
├── tests/                    # pytest suite — 397 / 398 passing
├── .env.example
├── pytest.ini
└── requirements.txt
```

---

## Demos

All demos assume `.env` is filled in (see [Configuration](#configuration)). Run from project root with `.venv3` active.

| Demo | What it shows |
|---|---|
| `python scripts/show_profile.py` | Profile YAML loaded + validated. No API call. |
| `python scripts/smoke_chat.py` | Smallest end-to-end test of the LLM facade. |
| `python scripts/profile_chat_demo.py` | Profile + LLM — Maya answers in character. |
| `python scripts/streaming_demo.py` | Streaming text deltas. |
| `python scripts/logger_demo.py` | Structured JSON-lines logger usage. |
| `python scripts/tools_demo.py` | Python function + Skill bundle as agent tools. |
| `python scripts/memory_demo.py` | Memory add + semantic recall + reflection. |
| `python scripts/reflection_demo.py` | Standalone reflection trigger. |
| `python scripts/react_tools_memory_demo.py` | **Most comprehensive** — three-turn ReAct run with calculator + Tavily search + cross-turn memory recall. |
| `python scripts/structured_extraction_demo.py --with-rag` | HTML/PDF → chunks → multimodal RAG with image preservation. |
| `python scripts/dump_memory.py` | Inspect what's stored in an agent's mem0 directory. |

Run `react_tools_memory_demo.py` first — it exercises the most code paths and is the best smoke test that everything is wired up correctly.

---

## Testing

```bash
.venv3/Scripts/python.exe -m pytest tests/ -v
```

Currently **397 / 398 passing**. The single failure (`test_data_with_path_serializes_via_str`) is a Windows path-separator issue unrelated to the harness logic — it asserts a POSIX path against `Path("/tmp/x")` which Windows serializes as `\tmp\x`.

### Targeted test runs

```bash
pytest tests/DefenseAgent/llm/                 # LLM adapter tests
pytest tests/DefenseAgent/agent/               # agent construction + loop tests
pytest tests/DefenseAgent/memory/              # memory + bridge tests
pytest tests/DefenseAgent/tools/               # tool registry / skill / MCP tests
pytest tests/DefenseAgent/test_integration.py  # cross-module integration
```

Tests are **fully offline** — they use `make_test_config(...)` (in `tests/DefenseAgent/agent/_support.py`) to inject `ScriptedLLM` + `MagicMock` memory, no real API calls.

---

## Design Notes

### One construction path
Every agent is built via `Agent(config)` where `config: AgentConfig`. The legacy keyword-argument constructor (`Agent(profile, llm=..., memory=..., ...)`) was removed during the v0.1 refactor — inject components through `AgentConfig.llm`/`memory`/`tool_registry`/`reflector`/`compressor`/`rag` instead.

### Two memory backends, one config object
`MemoryBackendConfig` (pure-code) and `.env` (legacy) both produce the same `mem0` connection. SDK callers pass `AgentConfig(memory_backend=...)`; local development uses `.env`.

### Lazy provider imports
`DefenseAgent.llm._registry._resolve_adapter(provider)` imports the provider SDK *inside the if-branch*. A user who only ever runs DeepSeek doesn't pay the cost of importing the `anthropic` package.

### Build phases
`AgentConfig` → `build_components_sync(config)` (sync; LLM/memory/tools/reflector/compressor/logger) → `agent._ensure_async_setup()` (lazy; MCP servers + RAG, both of which need `await`).

---

## Roadmap

Major SDK-readiness items still open:

- **`pyproject.toml`** — make this `pip install`-able with `[memory]`, `[rag]`, `[dev]` extras
- **Top-level `__init__.py` lazy imports** — currently imports the agent module eagerly, which transitively pulls in torch via modelscope
- **PEP 8 package rename** — `DefenseAgent/` → `defense_agent/`
- **`create_agent()` one-line factory** — collapse `AgentConfig + Agent(config)` into one call
- **Multi-agent communication** — in-process `AgentBus` + `AgentSwarm` so agents can call each other as tools
- **Retry / timeout / logging in the LLM facade** — push the cross-cutting concerns up from each adapter

See `docs/superpowers/specs/` for the per-module design rationale.

---

## License

TBD — add a LICENSE file before publishing.
