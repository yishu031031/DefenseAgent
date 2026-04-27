# DefenseAgent

> English · [中文 README](README_zh.md)

**A composable, production-minded agent SDK for Python.** Build a working LLM agent from a YAML profile in **two lines**, swap LLM providers without touching code, plug in custom tools / memory / RAG backends, and inject mocks cleanly for tests.

```python
from DefenseAgent import create_agent

agent  = create_agent("agents/example_agent/profile.yaml")
result = await agent.run("What did I work on this morning?")
print(result.final_answer)
```

---

## Introduction

**DefenseAgent is a Python SDK for building LLM-powered agents that do real work** — calling tools, remembering conversations, searching knowledge bases, and reasoning over multiple steps. It is designed for engineers who want to **ship an agent in production**, not prototype one in a notebook.

### What it solves

Agent development keeps running into the same four walls. DefenseAgent answers each one directly:

| Pain point | DefenseAgent's answer |
|---|---|
| **Provider lock-in** — most agent libraries are tied to OpenAI's API shape; switching to Claude, DeepSeek, Qwen, or a self-hosted vLLM means rewrites. | One unified `LLM` facade. Swap providers via `.env` or `LLM.create(provider=..., model=...)`; upstream code is unchanged. |
| **Stateless turns** — out-of-the-box LLMs forget everything between calls. | mem0-backed persistent memory with semantic recall as a built-in tool, plus reflection that distills accumulated experience into long-term insights. |
| **Brittle integrations** — Python tools, MCP servers, and Skill bundles each ship their own glue. | One `ToolRegistry` accepts plain functions, Anthropic-style **Skills**, and **MCP** servers (stdio / SSE / streamable-http). The LLM doesn't see a difference. |
| **Untestable agents** — most frameworks bake in their own LLM client, making real tests painful. | Every subsystem (LLM / memory / tools / RAG / reflection) is replaceable through `AgentConfig`. Tests inject `ScriptedLLM` + mock memory; production code stays the same. |

### How it's designed

Three principles guide every API decision:

1. **One construction path.** Every agent is built from a single `AgentConfig` object. There is no second "convenience" constructor — overloads are the enemy of clarity.
2. **Configure, don't fork.** Every extension point is a Protocol or ABC behind a registry. To add a custom LLM provider, Memory backend, or RAG extractor, you write a class and inject it — never edit the SDK source.
3. **Defaults from the simplest profile; customization through the same fields.** The two-line `create_agent("profile.yaml")` call uses the same `AgentConfig` fields a power user would set explicitly. There is no hidden second API.

### Who it's for

- **Application developers** shipping a domain-specific assistant (customer support, research analysis, coding agent) who need memory + tool use + multi-provider failover out of the box.
- **AI researchers** prototyping multi-step reasoning strategies who want a clean substrate that exposes traces, supports reflection, and doesn't tie them to a single provider.
- **Teams building internal tooling** that need MCP / Skill integration, persistent memory, and the ability to swap models as costs or quality requirements shift.

### What's in the box

- Three agent strategies — `SimpleAgent`, `ReActAgent`, `PlanAndSolveAgent`.
- Six LLM providers wired in — Anthropic + every OpenAI-compatible endpoint (DeepSeek / Qwen / Google-via-proxy / vLLM / OpenAI itself).
- Persistent memory (mem0 + qdrant) with reflection.
- Multimodal RAG (LlamaIndex) with HTML / PDF chunking that preserves embedded images and tables.
- Three tool sources unified under one registry: Python functions, Skill bundles, MCP servers.
- A pydantic-validated YAML profile schema with strict typo detection.
- Per-agent JSON-lines logger.
- Hatchling-built wheel + sdist + `py.typed` marker — installs cleanly into a fresh venv with `pip install ".[all]"`.

---

## Table of Contents

- [Introduction](#introduction)
- [Why DefenseAgent?](#why-defenseagent)
- [Installation](#installation)
- [Quick Start (5 minutes)](#quick-start-5-minutes)
- [Part A — Use the SDK](#part-a--use-the-sdk)
  - [A.1  The profile YAML, field by field](#a1-the-profile-yaml-field-by-field)
  - [A.2  Plug in your own LLM](#a2-plug-in-your-own-llm)
  - [A.3  Add tools (Python functions)](#a3-add-tools-python-functions)
  - [A.4  Write your own prompt](#a4-write-your-own-prompt)
  - [A.5  Add memory](#a5-add-memory)
  - [A.6  Add MCP servers and Skills](#a6-add-mcp-servers-and-skills)
  - [A.7  Run the agent — every call shape](#a7-run-the-agent--every-call-shape)
- [Part B — Extend the SDK](#part-b--extend-the-sdk)
  - [B.1  Custom LLM provider (incl. multimodal)](#b1-custom-llm-provider-incl-multimodal)
  - [B.2  Custom Memory backend](#b2-custom-memory-backend)
  - [B.3  Custom Tool / MCP / Skill backend](#b3-custom-tool--mcp--skill-backend)
  - [B.4  Custom RAG extractor + renderer](#b4-custom-rag-extractor--renderer)
- [Three Agent Strategies](#three-agent-strategies)
- [Module Layout](#module-layout)
- [Demos](#demos)
- [Testing](#testing)
- [Roadmap](#roadmap)
- [License](#license)

---

## Why DefenseAgent?

| | What you get |
|---|---|
| **Provider-agnostic LLM** | One `LLM` facade in front of Anthropic and any OpenAI-compatible provider (DeepSeek, Qwen/DashScope, ModelScope, vLLM, Google-via-proxy). Swap by editing `.env` or by passing `LLM.create(provider=...)`. |
| **Three reasoning strategies** | `SimpleAgent` (one-shot), `ReActAgent` (Yao 2022 — interleaved reasoning + tool use), `PlanAndSolveAgent` (Wang 2023 — plan / execute / synthesize). All share the same `AgentConfig`. |
| **Persistent memory** | mem0 + qdrant on disk. Cross-turn `memory_recall` is a built-in LLM tool; outcomes / failures / trajectories are tagged for later filtering and reflection. |
| **Multimodal RAG** | `LlamaIndexRAG` with `StructuredDocExtractor` for HTML / PDF chunking that preserves embedded images and tables. Hybrid (vector + BM25) retrieval supported. |
| **Three tool sources** | Plain Python functions (`@registry.tool`), Anthropic-style **Skill** bundles (a directory with `SKILL.md` + assets), and stdio / SSE / streamable-http **MCP** servers — all unified under one `ToolRegistry`. |
| **Reflection** | `Reflector` collects unreflected memories, asks the LLM to synthesize insights, writes them back tagged. Threshold-gated, never throws. |
| **One construction path** | Every agent is built via `Agent(AgentConfig(...))`. Inject mocks for tests via the same `AgentConfig.llm` / `memory` / `tool_registry` / `reflector` fields. |
| **Open packaging** | `pip install -e ".[all]"`, MIT, py.typed, hatchling-built wheel + sdist. |

---

## Installation

### Editable install (recommended for now)

```bash
git clone https://github.com/yishu031031/DefenseAgent.git
cd DefenseAgent

python -m venv .venv
.venv/Scripts/activate           # Windows
# source .venv/bin/activate      # macOS / Linux

pip install -e ".[all,dev]"      # core + memory + RAG + MCP + tests
```

### Pick your extras

The core install only ships LLM + profile + tools — enough for `SimpleAgent` chat. Memory, RAG, and MCP are opt-in to avoid pulling `torch` / `qdrant-client` / `llama-index` when you don't need them:

```bash
pip install -e .                  # core only — LLM + profile + tools
pip install -e ".[memory]"        # + mem0 + qdrant + fastembed
pip install -e ".[rag]"           # + llama-index + pdfplumber + bs4
pip install -e ".[mcp]"           # + MCP client
pip install -e ".[all]"           # everything user-facing
pip install -e ".[all,dev]"       # + pytest
```

> The full stack pulls in `ms-agent`, `mem0ai`, `llama-index-core`, `qdrant-client`, `torch` (via modelscope) and friends — about 1 GB on disk.

PyPI publication (`pip install defense-agent`) is on the immediate roadmap — the wheel and sdist already build cleanly under `python -m build`.

---

## Quick Start (5 minutes)

### Step 1 — Configure `.env`

Copy `.env.example` → `.env` and fill in **at least these three blocks**:

```bash
# 1. Pick a chat provider — selects which adapter the agent talks to
AGENT_LAB_LLM_PROVIDER=deepseek          # one of: anthropic | openai | deepseek | qwen | vllm | google

# 2. The matching credential block — the per-provider <PROVIDER>_* names are required
DEEPSEEK_API_KEY=sk-...                  # your API key from the provider
DEEPSEEK_BASE_URL=https://api.deepseek.com   # provider's REST endpoint
DEEPSEEK_MODEL=deepseek-chat             # model id passed in every chat request

# 3. Embedding — used by mem0 and (optionally) RAG; chat providers usually don't ship embeddings
EMBEDDING_API_KEY=sk-...                 # API key for the embedding provider (often different from chat)
EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1   # OpenAI-compatible endpoint
EMBEDDING_MODEL=text-embedding-v3        # embedding model id
EMBEDDING_DIMS=1024                      # vector dimension this model emits — must match model's spec
```

**Provider precedence** — `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL_ID` (cross-provider override) wins over `<PROVIDER>_API_KEY` etc. (per-provider).

### Step 2 — Pick a profile

The repo ships with [`agents/example_agent/profile.yaml`](agents/example_agent/profile.yaml) — a fully-commented reference profile that lists every supported field with defaults.

### Step 3 — Run it

```python
import asyncio
from DefenseAgent import create_agent

async def main():
    agent  = create_agent("agents/example_agent/profile.yaml")
    result = await agent.run("It's 2 PM. What have you been doing today?")
    print(result.final_answer)
    await agent.close()

asyncio.run(main())
```

That's it. You now have a ReAct-loop agent with memory, tool support, and RAG hooks ready to be wired up.

---

## Part A — Use the SDK

The SDK is designed so the **simplest** thing is also the **right** thing:

```python
agent  = create_agent("agents/example_agent/profile.yaml")  # one line to build
result = await agent.run("Hi")                              # one line to call
```

Everything below explains the knobs you can turn from the YAML profile alone — no Python edits required for the common cases.

### A.1 The profile YAML, field by field

[`agents/example_agent/profile.yaml`](agents/example_agent/profile.yaml) is a working profile **and** the canonical reference. Below is a condensed annotated walkthrough; open the file itself to see every default.

```yaml
agent:

  # --- LLM (per-agent overrides; per-field fallback to .env) -----------
  # Every field is OPTIONAL. A blank field falls back to .env in this order:
  #   provider:  AGENT_LAB_LLM_PROVIDER
  #   model:     LLM_MODEL_ID  >  <PROVIDER>_MODEL
  #   api_key:   LLM_API_KEY   >  <PROVIDER>_API_KEY
  #   base_url:  LLM_BASE_URL  >  <PROVIDER>_BASE_URL
  llm:
    provider:                 # which adapter to use: deepseek | anthropic | openai | qwen | google | vllm
    model:                    # model id passed to the provider, e.g. deepseek-chat / claude-opus-4-7
    base_url:                 # OpenAI-compatible base URL; ignored by the anthropic adapter
    api_key:                  # leave blank in shared profiles — let .env supply <PROVIDER>_API_KEY

  # --- Identity (required) ---------------------------------------------
  # These six fields are interpolated into the system prompt as
  # {id} {name} {age} {traits} {backstory} {initial_plan}.
  id: "example_agent_001"     # stable string id; used for log filename + memory partition
  name: "Nova Patel"          # display name; appears in {name} prompt placeholder
  age: 27                     # int; appears in {age} prompt placeholder
  traits: "curious, methodical, candid"   # short comma list — feeds {traits}
  backstory: >                # multi-line bio — feeds {backstory}
    A field engineer turned AI researcher who recently moved to Lakeside.
  initial_plan: >             # what the agent is "doing today" — feeds {initial_plan}
    Wake up, review pipeline alerts, work on data analysis until lunch.

  # --- Cognitive loop knobs (optional; defaults shown) -----------------
  cognitive:
    max_steps_per_cycle: 10        # hard cap on ReAct tool turns per run() — prevents runaway loops
    reflection_threshold: 5        # number of unreflected memories that triggers Reflector after run()
    importance_threshold: 7        # 1–10 score; only memories ≥ this score are kept by the importance filter
    planning_horizon: "1 day"      # free-form string injected into prompts so the LLM knows the time frame
                                   # it's planning for ("1 hour", "1 day", "this week", ...).

  # --- Memory (mem0-backed; defaults shown) ----------------------------
  memory:
    is_retrieve: true              # master switch: enable mem0 + register the memory_recall tool
    history_mode: add              # 'add' = append every turn  |  'overwrite' = mem0 diff/rollback mode
    search_limit: 10               # max records returned per memory_recall call
    storage_path:                  # qdrant directory; blank → <profile_dir>/memory/

  # --- RAG (LlamaIndex-backed; off by default) -------------------------
  rag:
    enabled: false                 # flip on to wire LlamaIndexRAG + register the rag_search tool
    documents_dir: "rag_corpus"    # source documents (HTML/PDF/MD) — relative to this profile's directory
    storage_dir: "rag_index"       # where the FAISS / vector index is persisted on disk
    embedding_provider: openai     # 'openai' (or compatible base_url) | 'huggingface' (local model)
    chunk_size: 512                # tokens per chunk during ingestion — smaller = more precise, more storage
    top_k: 5                       # number of passages returned per rag_search call

  # --- Tools (skills + MCP servers) ------------------------------------
  tools:
    skills:                        # local Skill directories — each must contain SKILL.md
      - skills/tabular-report      # path is relative to this profile's directory
    mcp: []                        # list of MCP server launch configs — see A.6 for the full schema
    allow_skill_execution: false   # opt-in: when true, every script in a Skill bundle becomes a callable Tool
                                   # (executed in a sandboxed subprocess)

  # --- System prompt ---------------------------------------------------
  # Precedence: `system` (inline) > `path` (file) > auto-built identity block.
  prompt:
    path: prompts/system.md        # path to a markdown template (relative to profile dir) — OR use `system: "..."` inline
    extra_instructions: |          # appended to the resolved system prompt — useful for tone/format rules
      Be concise. Lead with the anomaly.
```

**Validation:** every block is `extra="forbid"` — typos raise `ConfigValidationError` on load with a precise field path.

**A profile bundle is a directory** — `profile.yaml` plus optional siblings:

```
agents/example_agent/
├── profile.yaml          # the file you pass to create_agent()
├── prompts/
│   └── system.md         # externalized system prompt template
├── skills/
│   └── tabular-report/   # Anthropic-style Skill bundle
│       ├── SKILL.md
│       ├── scripts/
│       └── templates/
├── memory/               # auto-created on first run; mem0 lives here
└── logs/                 # auto-created; one .log per profile.id
```

### A.2 Plug in your own LLM

**Option 1 — Edit `.env`** (recommended; zero Python changes):

```bash
AGENT_LAB_LLM_PROVIDER=qwen              # switch the active adapter to qwen
QWEN_API_KEY=sk-...                      # DashScope key
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1   # OpenAI-compatible endpoint
QWEN_MODEL=qwen-plus                     # model id (qwen-plus / qwen-max / qwen-turbo / ...)
```

**Option 2 — Per-agent override in YAML:**

```yaml
agent:
  llm:
    provider: anthropic
    model: claude-opus-4-7
    api_key:                       # leave blank — let .env keep the secret
```

**Option 3 — Pure Python (skip `.env` entirely):**

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

**Supported providers out of the box:** `anthropic`, `openai`, `deepseek`, `qwen`, `vllm`, `google`. Anything OpenAI-compatible works through the `openai` adapter (set the `base_url`).

To add a brand-new provider class (e.g. multimodal vision), see [B.1 Custom LLM provider](#b1-custom-llm-provider-incl-multimodal).

### A.3 Add tools (Python functions)

Any Python callable can be a tool — the function's **name + docstring + signature** become the JSON-schema spec the LLM sees:

```python
from DefenseAgent import create_agent

def calculator(expression: str) -> str:
    """Evaluate a math expression and return the numeric result."""
    return str(eval(expression))           # toy demo only

def web_search(query: str, top_k: int = 3) -> str:
    """Search the web and return the top-k snippets joined by newlines."""
    ...

agent = create_agent({
    "profile": "agents/example_agent/profile.yaml",
    "tools": [calculator, web_search],
})
```

The agent will call these tools through the ReAct loop whenever the LLM emits a matching `tool_use` request.

> ### Type-hint conventions
> The registry inspects the function signature — keep parameter types simple (`str`, `int`, `float`, `bool`, `list`, `dict`). Default values become `optional`; everything else is `required`.

### A.4 Write your own prompt

**Inline in YAML:**

```yaml
agent:
  prompt:
    system: |
      You are {name}, a {age}-year-old {traits} field engineer.
      {backstory}
      ---
      Always lead with the anomaly when summarizing logs.
```

**Externalized template file** — keeps long prompts out of YAML and version-control diffs clean:

```yaml
agent:
  prompt:
    path: prompts/system.md
```

```markdown
# agents/example_agent/prompts/system.md
You are {name}, a {age}-year-old {traits} field engineer.

# Background
{backstory}

# Today
{initial_plan}

# How to behave
- Speak in first person, in natural English. Be concise.
- When you need information from earlier conversations, call `memory_recall`.
- When a tool fails, acknowledge briefly and move on.
```

**Available placeholders:** `{id}`, `{name}`, `{age}`, `{traits}`, `{backstory}`, `{initial_plan}` — all interpolated from the identity block.

**Append-only extra instructions** — useful for tone / format rules layered on top of a shared base prompt:

```yaml
agent:
  prompt:
    path: prompts/system.md
    extra_instructions: |
      When summarizing logs, lead with the anomaly and follow with one
      sentence of context — never bury the lede.
```

### A.5 Add memory

Memory is **on by default** when `profile.memory.is_retrieve = true` (the default). You don't write any Python — the agent automatically:

1. Persists each `(question → answer)` after `run()` (tagged `memory_type='outcome'`)
2. Persists each tool call's `(call → result)` (tagged `memory_type='trajectory'`)
3. Exposes a `memory_recall` tool the LLM can call mid-reasoning

**Memory storage path** defaults to `<profile_dir>/memory/` — qdrant collections live here.

**Inspect what's stored:**

```bash
python scripts/dump_memory.py agents/example_agent/
```

**Disable memory entirely** (stateless agent, no `.env` for embeddings needed):

```python
agent = create_agent(AgentConfig(
    profile="agents/example_agent/profile.yaml",
    use_memory=False,
))
```

**Programmatic mem0 backend** — when you don't want a `.env` at all:

```python
from DefenseAgent.memory import MemoryBackendConfig

# MemoryBackendConfig holds the credentials mem0 itself uses for fact-extraction
# and embedding — independent of the agent's chat LLM (set separately above).
backend = MemoryBackendConfig(
    llm_provider="deepseek",                          # which provider mem0 uses for fact extraction
    llm_api_key="...",                                # API key for the mem0 LLM
    llm_model="deepseek-chat",                        # model id for the mem0 LLM
    llm_base_url="https://api.deepseek.com",          # REST endpoint for the mem0 LLM
    embedding_api_key="...",                          # API key for the embedding provider
    embedding_model="text-embedding-3-small",         # embedding model id
    embedding_base_url="https://api.openai.com/v1",   # OpenAI-compatible embedding endpoint
    embedding_dims=1536,                              # vector dimension; must match the embedding model
)
agent = create_agent(AgentConfig(
    profile="...",
    memory_backend=backend,                           # bypass .env entirely for memory
    load_env=False,                                   # don't try to load .env at all
))
```

### A.6 Add MCP servers and Skills

#### Skills

A **Skill** is a directory containing `SKILL.md` + optional scripts/templates — Anthropic's portable bundle format. Drop the directory under `agents/<name>/skills/` and reference it in YAML:

```yaml
agent:
  tools:
    skills:                              # list of Skill paths — relative to this profile's directory
      - skills/tabular-report            # one specific Skill bundle
      - skills/                          # OR a parent dir — every SKILL.md found inside gets registered
    allow_skill_execution: false         # opt-in: when true, every script in a Skill bundle becomes a callable Tool
    skill_execution_timeout: 300         # subprocess timeout in seconds (skill scripts run in a sandbox)
```

A minimal `SKILL.md`:

```markdown
---
name: tabular-report                     # tool name the LLM will call (must be unique)
description: Render rows as a Markdown table.   # one-line description shown to the LLM
---
# Tabular Report
Use this skill when you have a list of dict-shaped rows...
```

The skill metadata becomes a tool the LLM can request. When `allow_skill_execution: true`, every `*.py` / `*.sh` in the bundle's `scripts/` becomes its own Tool that runs in a sandboxed subprocess (timeout-capped).

#### MCP servers

The Model Context Protocol lets an agent talk to **external tool servers** over stdio, SSE, websockets, or streamable-http. Add servers in YAML:

```yaml
agent:
  tools:
    mcp:
      # ----- stdio server (most common — uvx/npx-launched local process) -----
      - command: uvx                      # the executable to launch (e.g. uvx, npx, python)
        args: [mcp-server-filesystem, /tmp]  # arguments passed to `command`
        env:                              # environment variables passed to the child process
          TOKEN:                          # empty value → interpolated from the parent process env
        include: [read_file]              # whitelist of tool names to expose; mutually exclusive with `exclude:`

      # ----- remote network server (SSE / streamable-http / websocket) -----
      - transport: sse                    # 'sse' | 'streamable_http' | 'websocket' (defaults to streamable_http)
        url: https://mcp.example.com/sse  # endpoint URL (required for network transports)
        headers:                          # HTTP headers; `${VAR}` is expanded from the process env
          Authorization: "Bearer ${MCP_TOKEN}"
        timeout: 30                       # connect/request timeout in seconds
```

Every tool exposed by every server is folded into the same `ToolRegistry` — the LLM doesn't see a difference between a Python function, a Skill, and an MCP-served tool.

### A.7 Run the agent — every call shape

```python
import asyncio
from DefenseAgent import create_agent, AgentConfig, ReActAgent

# ----- 1. profile path ------------------------------------------------
agent = create_agent("agents/example_agent/profile.yaml")

# ----- 2. dict (forwarded to AgentConfig) -----------------------------
agent = create_agent({
    "profile": "agents/example_agent/profile.yaml",
    "tools":   [calculator],
    "use_rag": True,
})

# ----- 3. AgentConfig (full control) ----------------------------------
config = AgentConfig(
    profile="agents/example_agent/profile.yaml",
    tools=[calculator],
    use_rag=True,
    extra_instructions="Always answer in JSON.",
)
agent = create_agent(config, strategy="plan_and_solve")  # or "react" / "simple"

# ----- 4. Manual class — async-context for clean teardown -------------
async with ReActAgent(config) as agent:
    result = await agent.run("Compute the standard deviation of [1, 2, 3, 4]")
    print(result.final_answer)
    for step in result.steps:
        print(f"[{step.kind}]", step.content)
```

`AgentResult.steps` gives a complete trace — every tool call, every LLM turn, every error — for debugging and post-hoc analysis.

---

## Part B — Extend the SDK

Every subsystem is a **Protocol or ABC** behind one registry. Implement the interface, register it, and you're done — no fork required.

### B.1 Custom LLM provider (incl. multimodal)

Implement `LLMAdapter` and add it to the registry:

```python
# my_app/vision_adapter.py
from typing import AsyncIterator
from DefenseAgent.llm.base import LLMAdapter
from DefenseAgent.llm.types import LLMResponse, Message, StreamChunk

class GeminiVisionAdapter(LLMAdapter):
    """Multimodal adapter — accepts text + image_url parts in messages."""

    def __init__(self, *, api_key: str, model: str, base_url: str = ""):
        self.client = ...   # your provider's SDK
        self.model  = model

    async def chat(
        self, messages: list[Message], *, tools=None,
        temperature: float = 0.7, max_tokens: int = 1024, system: str | None = None,
    ) -> LLMResponse:
        # 1) Translate `Message`s — including multimodal `image_url` parts —
        #    into your provider's request shape.
        # 2) Call the provider, capture token usage and stop_reason.
        # 3) Return a canonical `LLMResponse(content=..., tool_calls=..., usage=..., stop_reason=...)`.
        ...

    async def chat_stream(self, messages, *, tools=None, **kw) -> AsyncIterator[StreamChunk]:
        # Optional override; the base class auto-implements this by buffering chat().
        ...
```

```python
# Wire it up — two ways:

# A) Construct manually and inject via AgentConfig.llm
from DefenseAgent import AgentConfig, create_agent
from DefenseAgent.llm import LLM
from my_app.vision_adapter import GeminiVisionAdapter

llm = LLM(GeminiVisionAdapter(api_key="...", model="gemini-2.5-pro-vision"))
agent = create_agent(AgentConfig(profile="...", llm=llm, load_env=False))

# B) Register globally so .env / profile.yaml can pick it by name
from DefenseAgent.llm._registry import _ADAPTERS    # informal extension point
_ADAPTERS["gemini-vision"] = GeminiVisionAdapter
# Then in .env:    AGENT_LAB_LLM_PROVIDER=gemini-vision
```

### B.2 Custom Memory backend

Subclass the `Memory` ABC — `run(messages) -> messages` is the only method you must implement:

```python
from DefenseAgent.memory import Memory
from DefenseAgent.llm.types import Message

class RedisMemory(Memory):
    """Toy example — store every user/assistant turn in a Redis list."""

    def __init__(self, profile, *, redis_url: str):
        super().__init__(profile)
        import redis.asyncio as redis
        self.r   = redis.from_url(redis_url)
        self.key = f"agent:{profile.id}:history"

    async def run(self, messages: list[Message]) -> list[Message]:
        # 1) ingest — persist the new tail of `messages`
        # 2) retrieve — optionally inject older context from Redis
        # 3) return the rewritten message list
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

### B.3 Custom Tool / MCP / Skill backend

The cleanest extension is just a Python function — see [A.3](#a3-add-tools-python-functions). For more structured cases:

**Pre-built `ToolRegistry`** — when you want to share one registry across multiple agents:

```python
from DefenseAgent.tools import ToolRegistry

registry = ToolRegistry()

@registry.tool
def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

@registry.tool(name="search", description="Web search.")
def google(query: str, top_k: int = 3) -> str: ...

agent = create_agent(AgentConfig(profile="...", tool_registry=registry))
```

**Custom MCP-style transport** — implement `MCPClient`'s connect / list_tools / call_tool surface and pass it via the registry's lower-level API. See [DefenseAgent/tools/mcp.py](DefenseAgent/tools/mcp.py).

### B.4 Custom RAG extractor + renderer

DefenseAgent's RAG has **three pluggable layers**:

1. **`StructuredExtractor`** — parses a file into chunks + per-chunk resources (images, tables, ...)
2. **`ResourceRenderer`** — turns a stored resource into LLM-readable text
3. **`LlamaIndexRAG`** itself — register either of the above

#### Custom extractor (e.g. `.docx`)

```python
# scripts/extras/docx_extractor.py — ready to copy
from pathlib import Path
from DefenseAgent.rag.extraction import StructuredChunk, StructuredResource

class DocxExtractor:
    def __init__(self, resources_dir: Path):
        self.resources_dir = resources_dir

    def supports(self, source) -> bool:
        return Path(source).suffix.lower() == ".docx"

    def extract(self, source) -> list[StructuredChunk]:
        # Walk paragraphs / inline images / tables; persist images under
        # <resources_dir>/<source_hash>/; emit one StructuredChunk per
        # heading-1/2 section.
        ...
```

```python
from DefenseAgent.rag import LlamaIndexRAG, StructuredDocExtractor
from scripts.extras.docx_extractor import DocxExtractor

extractor = StructuredDocExtractor(profile)
extractor.register(DocxExtractor(resources_dir=extractor.resources_dir))
rag = await LlamaIndexRAG.from_profile(profile, extractor=extractor)
```

#### Custom renderer (e.g. `kind="csv"`)

```python
# scripts/extras/csv_renderer.py — ready to copy
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
# When the LLM calls rag_get_resource(rid="...csv...") the CsvRenderer takes over.
```

> Two ready-to-copy backends live in [`scripts/extras/`](scripts/extras/) — `csv_renderer.py` and `docx_extractor.py`. Use them as templates.

---

## Three Agent Strategies

| Class | Loop | Use when |
|---|---|---|
| `SimpleAgent` | One LLM call. No tools, no loop. | Pure conversation, role-play, single-shot QA |
| `ReActAgent` | LLM → tool calls → results → LLM → ... until plain-text answer | Most general-purpose; tool use is interleaved with reasoning |
| `PlanAndSolveAgent` | Plan into N steps → execute each (with tools) → synthesize answer | Multi-step tasks where committing to a plan first reduces drift |

All three accept the same `AgentConfig` and emit the same `AgentResult` (a `final_answer` plus a step-by-step trace).

---

## Module Layout

```
DefenseAgent/
├── llm/         Module 1 — LLM facade + per-provider adapters
├── config/      Module 2 — pydantic-validated YAML profile loader
├── ops/         Module 3 — per-agent JSON-lines logger
├── memory/      Module 4 — mem0-backed memory (inherits ms-agent)
├── reflection/  Module 5 — importance scorer + insight synthesizer
├── tools/       Module 6 — Python functions / Skills / MCP servers
├── rag/         Module 5+ — LlamaIndex-backed RAG, multimodal extraction
├── skills/      Skill loader + sandboxed subprocess container
└── agent/       Module 7 — BaseAgent + Simple / ReAct / PlanAndSolve
```

Each module has a design doc in [docs/superpowers/specs/](docs/superpowers/specs/) and a user walkthrough in [docs/walkthroughs/](docs/walkthroughs/).

---

## Demos

All demos assume `.env` is filled in. Run from the project root with the venv active.

| Demo | What it shows |
|---|---|
| `python scripts/show_profile.py` | Profile YAML loaded + validated. No API call. |
| `python scripts/smoke_chat.py` | Smallest end-to-end test of the LLM facade. |
| `python scripts/profile_chat_demo.py` | Profile + LLM — agent answers in character. |
| `python scripts/streaming_demo.py` | Streaming text deltas. |
| `python scripts/tools_demo.py` | Python function + Skill bundle as agent tools. |
| `python scripts/memory_demo.py` | Memory add + semantic recall + reflection. |
| `python scripts/reflection_demo.py` | Standalone reflection trigger. |
| `python scripts/react_tools_memory_demo.py` | **Most comprehensive** — three-turn ReAct run with calculator + Tavily search + cross-turn memory recall. |
| `python scripts/structured_extraction_demo.py --with-rag` | HTML/PDF → chunks → multimodal RAG with image preservation. |
| `python scripts/structured_rag_agent_demo.py` | End-to-end **custom renderer + extractor** demo (offline). |
| `python scripts/dump_memory.py` | Inspect what's stored in an agent's mem0 directory. |
| `python scripts/logger_demo.py` | Structured JSON-lines logger usage. |

Run `react_tools_memory_demo.py` first — it exercises the most code paths and is the best smoke test that the full stack is wired up correctly.

---

## Testing

```bash
.venv/Scripts/python.exe -m pytest tests/ -v
```

Tests are **fully offline** — they use `make_test_config(...)` (in `tests/DefenseAgent/agent/_support.py`) to inject `ScriptedLLM` + `MagicMock` memory, no real API calls.

```bash
pytest tests/DefenseAgent/llm/                 # LLM adapter tests
pytest tests/DefenseAgent/agent/               # construction + loop tests
pytest tests/DefenseAgent/memory/              # memory + bridge tests
pytest tests/DefenseAgent/tools/               # tool registry / skill / MCP tests
pytest tests/DefenseAgent/rag/                 # extraction + RAG search
pytest tests/DefenseAgent/test_integration.py  # cross-module integration
```

---

## Roadmap

Major SDK-readiness items still open:

- **PyPI publication** — `pip install defense-agent` (wheel and sdist already build cleanly; PyPI account + GitHub release workflow pending)
- **CI / pre-commit / ruff config** — formal lint + type-check baselines
- **CHANGELOG / CONTRIBUTING / CODE_OF_CONDUCT** — open-source governance docs
- **Multi-agent communication** — in-process `AgentBus` + `AgentSwarm` so agents can call each other as tools
- **Operational RAG APIs** — `delete_chunk` / `clear` / `list_resources` / `gc_orphan_resources`
- **Retry / timeout / structured logging in the LLM facade** — push cross-cutting concerns up from each adapter

See [docs/superpowers/specs/](docs/superpowers/specs/) for per-module design rationale.

---

## License

[MIT](LICENSE) © 2026 Ying Yang, Zechun Zhao, Yishu Wang.
