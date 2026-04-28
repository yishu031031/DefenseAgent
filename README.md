# DefenseAgent

> English · [中文 README](README_zh.md)

A Python harness for building single-agent LLM applications. Define an agent in one YAML profile, instantiate it with one line of Python, run tasks against any of three execution strategies.

```python
from DefenseAgent.agent import AgentConfig, ReActAgent
from DefenseAgent.examples import EXAMPLE_PROFILE_PATH

config = AgentConfig(profile=EXAMPLE_PROFILE_PATH)
agent  = ReActAgent(config)
result = await agent.run("Summarise today's plan in one sentence.")
```

## Contents

- [Features](#features)
- [Install](#install)
- [Quickstart — from zero to a running agent](#quickstart--from-zero-to-a-running-agent)
- [Configure](#configure)
  - [Providers and credentials](#providers-and-credentials)
- [Building your own agent](#building-your-own-agent) — full profile reference
  - [`llm:`](#llm)
  - [Identity](#identity)
  - [`cognitive:`](#cognitive)
  - [`memory:`](#memory)
  - [`rag:`](#rag)
  - [`tools:`](#tools) — skills / MCP / Python
  - [`prompt:`](#prompt)
- [Built-in tools](#built-in-tools)
- [Agent classes](#agent-classes)
- [Multimodal input](#multimodal-input) — vision models, image handling, OCR
- [Customization & dependency injection](#customization--dependency-injection)
- [Architecture](#architecture)
- [Module layout](#module-layout)
- [Develop locally](#develop-locally)
- [License](#license)

## Features

- **One-file agent definition.** Identity, LLM provider, tools, memory, RAG, system prompt — all in one strictly-validated YAML (`extra="forbid"`; unknown fields raise `ConfigValidationError` on load).
- **Per-field configuration fallback.** Every value can be set in the profile or in `.env`; profile wins per field, `.env` fills the gaps. Switch LLM providers (`openai`, `anthropic`, `deepseek`, `qwen`, `google`, `vllm`) without code changes.
- **Three agent strategies.** `SimpleAgent` (one-shot), `ReActAgent` (tool-call loop), `PlanAndSolveAgent` (plan → execute → synthesise). All built from the same `AgentConfig`.
- **Three tool sources, one registry.** Local skill directories (`SKILL.md` bundles), MCP servers (stdio / SSE / WebSocket / streamable-http), Python functions (referenced from the profile by file path or dotted module).
- **Persistent memory with a built-in tool.** mem0-backed Qdrant storage; agents automatically expose a `memory_recall` tool to the LLM. `ContextCompressor` keeps the working context within a configured token budget.
- **Optional RAG with a built-in tool.** Drop documents into a directory, set `rag.enabled: true`, get a `rag_search` tool. Embedder credentials follow the same per-field profile→env fallback.
- **Optional multimodal input.** When you do need vision, `agent.run(task, images=[...])` attaches images to the user turn. Disabled by default — see the dedicated [Multimodal input](#multimodal-input) section.
- **Dependency-injectable.** LLM, memory, tools, reflector, compressor and logger are all replaceable in `AgentConfig` for tests and custom wiring.

## Install

**Default install** — recommended for first-time users:

```bash
pip install 'defense-agent[memory]'
```

This is the smallest install that runs `agent.run()` with the framework's default config (`use_memory=True`). It pulls in `mem0ai` + `fastembed` on top of the core deps.

If you only need a stateless agent (no `memory_recall`, no persistence), the bare install is enough — but you must explicitly disable memory in your config:

```bash
pip install defense-agent
```

```python
config = AgentConfig(profile=..., use_memory=False)   # required for bare install
```

The full table of extras:

| Extra | Pulls in | Required for |
|---|---|---|
| `defense-agent[memory]` | `mem0ai[nlp]`, `fastembed` (`spacy` pulled in transitively) | Default config to work; persistent memory + the `memory_recall` tool. Clean startup (no spaCy/fastembed warnings). |
| `defense-agent[rag]` | `llama-index-core`, `llama-index-embeddings-openai-like`, `llama-index-retrievers-bm25`, `pdfplumber`, `beautifulsoup4`, `Pillow` | `rag.enabled: true` profiles + the `rag_search` tool |
| `defense-agent[mcp]` | `mcp` | Connecting to MCP tool servers (entries under `tools.mcp:`) |
| `defense-agent[all]` | memory + rag + mcp | One-shot — every subsystem usable with no further installs |
| `defense-agent[dev]` | `pytest`, `pytest-asyncio` | Running the test suite |

Requires Python ≥ 3.10. The core install pulls in `openai` + `anthropic` HTTP clients and `ms-agent` (which transitively brings in `torch` for its tooling pipeline). Plan for ~1 GB on the first install.

### About startup messages and stray files

Since 0.1.4, `defense-agent[memory]` already pulls `mem0ai[nlp]` and `fastembed`, so memory init is silent out of the box.

Since 0.1.5, `import DefenseAgent` also suppresses ms-agent's default `<cwd>/ms_agent.log` file. (Upstream's `ms_agent.utils.logger` unconditionally creates a `ms_agent.log` in the user's working directory the moment any ms-agent submodule is imported — DefenseAgent now removes that FileHandler before any of our submodules touch it. Terminal `[INFO:ms_agent] ...` log lines still appear on stderr unchanged. If you explicitly want a ms-agent log file, call `ms_agent.utils.logger.get_logger(log_file='your-path.log')` and our patch will leave it alone.)

If you (or a downstream user) ever installed `mem0ai` directly with bare `pip install mem0ai` and see messages like `Failed to load spaCy lemma model` or `fastembed not installed — BM25 keyword search disabled`, those are mem0's optional-feature probes — **safe to ignore**, the agent runs fine without them. Install via `defense-agent[memory]` (or just `pip install 'mem0ai[nlp]' fastembed`) to clean them up.

## Quickstart — from zero to a running agent

This walks through setting up a brand-new project that uses DefenseAgent.

### 1. Create a project directory and virtualenv

```bash
mkdir myagent && cd myagent
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
```

(or, if you prefer conda: `conda create -n myagent python=3.12 -y && conda activate myagent`)

### 2. Install

```bash
pip install 'defense-agent[all]'
```

Pick a smaller extras set (e.g. `defense-agent[memory]`) if you don't need RAG or MCP — see the table above.

### 3. Drop credentials into `.env`

DefenseAgent calls `load_dotenv()` on construction (override with `AgentConfig(load_env=False, ...)` if your env is already populated by your runtime). Create a `.env` next to where you'll run Python:

```bash
# myagent/.env
AGENT_LAB_LLM_PROVIDER=deepseek                      # which provider adapter to load
DEEPSEEK_API_KEY=sk-…                                # your key
DEEPSEEK_MODEL=deepseek-chat                         # any chat model the provider serves
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1

# Only needed if you'll use memory[memory_recall] or rag[rag_search]:
EMBEDDING_API_KEY=sk-…
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMS=1536
```

The full provider list and embedding pairings are in [Configure](#configure) below.

### 4. Run the bundled example agent

The wheel ships a complete reference profile. Start by running it as-is:

```python
# myagent/run_example.py
import asyncio
from DefenseAgent.agent import AgentConfig, ReActAgent
from DefenseAgent.examples import EXAMPLE_PROFILE_PATH

async def main():
    async with ReActAgent(AgentConfig(profile=EXAMPLE_PROFILE_PATH)) as agent:
        result = await agent.run("Summarise today's plan in one sentence.")
        print(result.final_answer)

asyncio.run(main())
```

```bash
python run_example.py
```

If this prints a sentence, your provider credentials are wired correctly.

### 5. Make it your own profile

Copy the example bundle out of the package and edit it:

```bash
python -c "
from DefenseAgent.examples import EXAMPLE_AGENT_DIR
import shutil; shutil.copytree(EXAMPLE_AGENT_DIR, './my_profile')
"
```

You'll get a `my_profile/` directory with `profile.yaml`, `prompts/`, `python_tools/`, `skills/`. Edit `profile.yaml` (the schema is in [Building your own agent](#building-your-own-agent)) and point your code at it:

```python
from pathlib import Path
config = AgentConfig(profile=Path("./my_profile/profile.yaml"))
```

That's the whole loop. The rest of the README is reference material.

## Configure

Resolution order, per field: profile YAML → env var → schema default. Whitespace-only values are treated as unset.

### Providers and credentials

`AGENT_LAB_LLM_PROVIDER` selects the adapter. Each provider has its own block of `<PROVIDER>_*` env vars (`<PROVIDER>_API_KEY`, `<PROVIDER>_MODEL`, `<PROVIDER>_BASE_URL`). The cross-provider `LLM_API_KEY` / `LLM_MODEL_ID` / `LLM_BASE_URL` tier overrides the per-provider tier when set.

| Provider | Adapter | Typical key format | Default base URL | Example chat models |
|---|---|---|---|---|
| `openai` | `OpenAICompatibleAdapter` | `sk-…` or `sk-proj-…` | `https://api.openai.com/v1` | `gpt-4o-mini`, `gpt-4o`, `o3-mini` |
| `anthropic` | `AnthropicAdapter` | `sk-ant-…` | `https://api.anthropic.com` | `claude-sonnet-4-6`, `claude-opus-4-7` |
| `deepseek` | `OpenAICompatibleAdapter` | `sk-…` | `https://api.deepseek.com/v1` | `deepseek-chat`, `deepseek-reasoner` |
| `qwen` (DashScope, OpenAI-compat) | `OpenAICompatibleAdapter` | `sk-…` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-plus`, `qwen-max`, `qwen-turbo` |
| `google` (OpenAI-compat endpoint) | `OpenAICompatibleAdapter` | `sk-…` | `https://generativelanguage.googleapis.com/v1beta/openai` | `gemini-2.0-flash` |
| `vllm` (self-hosted) | `OpenAICompatibleAdapter` | any string (e.g. `EMPTY` / `token-not-needed`) | depends on deployment, e.g. `http://localhost:8000/v1` | whatever the vLLM server is serving |

Embedding: a separate `EMBEDDING_*` block. Common pairings:

| Embedder | `EMBEDDING_BASE_URL` | `EMBEDDING_MODEL` | `EMBEDDING_DIMS` |
|---|---|---|---|
| OpenAI | `https://api.openai.com/v1` | `text-embedding-3-small` | 1536 |
| OpenAI | `https://api.openai.com/v1` | `text-embedding-3-large` | 3072 |
| DashScope | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `text-embedding-v3` | 1024 |
| ModelScope | `https://api-inference.modelscope.cn/v1` | `Qwen/Qwen3-Embedding-0.6B` | 1024 |
| ModelScope | `https://api-inference.modelscope.cn/v1` | `Qwen/Qwen3-Embedding-8B` | 4096 |

`EMBEDDING_DIMS` **must match** what the model emits or the Qdrant collection rejects writes — set it from the model's documented vector size.

## Building your own agent

A profile bundle is a directory:

```
my_profile/
├── profile.yaml          # required — the schema below
├── prompts/              # optional — system-prompt templates
│   └── system.md
├── python_tools/         # optional — local Python tool entry points
│   └── calc.py
├── skills/               # optional — SKILL.md-style tool packs
│   └── tabular-report/
├── memory/               # auto-created at runtime if memory.is_retrieve=true
└── rag_corpus/           # documents indexed when rag.enabled=true
```

`AgentConfig(profile=Path("…/my_profile/profile.yaml"))` resolves every relative path inside the profile against the profile's directory, so the bundle is self-contained and movable.

Each block under `agent:` is independent and optional except identity. All fields are validated by pydantic with `extra="forbid"`.

### `llm:`

```yaml
llm:
  provider:           # str | null. One of: openai | anthropic | deepseek | qwen | google | vllm. Falls back to AGENT_LAB_LLM_PROVIDER.
  model:              # str | null. Provider-specific model id (see Providers table). Falls back to <PROVIDER>_MODEL or LLM_MODEL_ID.
  base_url:           # str | null. Provider endpoint. Falls back to <PROVIDER>_BASE_URL or LLM_BASE_URL.
  api_key:            # str | null. Falls back to <PROVIDER>_API_KEY. Recommend leaving blank in shared profiles.
```

All four fields are `str | None`. Each falls back to `.env` independently. Whitespace-only values count as unset, so a half-edited YAML can't shadow correct env state.

#### Per-field fallback in practice

Resolution order for each field, top to bottom (first non-empty wins):

1. `llm.<field>:` in profile YAML
2. Cross-provider env tier — `LLM_API_KEY` / `LLM_MODEL_ID` / `LLM_BASE_URL`
3. Per-provider env tier — `<PROVIDER>_API_KEY` / `<PROVIDER>_MODEL` / `<PROVIDER>_BASE_URL`
4. Schema default (where applicable)

So a profile with only `llm: { provider: deepseek, model: deepseek-chat }` and the rest in `.env` is the recommended shape — model choice belongs in the YAML (it's part of the agent's identity), credentials belong in `.env` (they're operator concerns).

Concrete example. Given:

```yaml
# profile.yaml
llm:
  provider: deepseek
  model: deepseek-reasoner             # profile sets this explicitly
```

```bash
# .env
LLM_API_KEY=sk-shared                  # cross-provider override, wins over per-provider
DEEPSEEK_API_KEY=sk-deepseek           # per-provider, used if LLM_API_KEY absent
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat           # ignored — profile's model wins
```

Final resolution:
- `provider` → `deepseek` (profile)
- `model` → `deepseek-reasoner` (profile beats `DEEPSEEK_MODEL`)
- `base_url` → `https://api.deepseek.com/v1` (profile empty → falls to `DEEPSEEK_BASE_URL`)
- `api_key` → `sk-shared` (cross-provider `LLM_API_KEY` beats `DEEPSEEK_API_KEY`)

#### Switching providers without code changes

Same agent code, three different providers — only `.env` changes:

```bash
# .env (variant A — DeepSeek)
AGENT_LAB_LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-…
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
```

```bash
# .env (variant B — DashScope/Qwen)
AGENT_LAB_LLM_PROVIDER=qwen
QWEN_API_KEY=sk-…
QWEN_MODEL=qwen-plus
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

```bash
# .env (variant C — local vLLM)
AGENT_LAB_LLM_PROVIDER=vllm
VLLM_API_KEY=EMPTY                     # vLLM doesn't auth by default
VLLM_MODEL=Qwen/Qwen2.5-72B-Instruct   # whatever the server is hosting
VLLM_BASE_URL=http://localhost:8000/v1
```

Provided your profile leaves `llm.provider` / `llm.model` blank (or you don't have an `llm:` block at all), the agent picks up whichever set is active in the env. No reload, no code change.

#### Provider-specific notes

| Provider | Things to know |
|---|---|
| `openai` | Both `sk-…` and `sk-proj-…` keys work. Reasoning models (`o3-mini`, `o1`) cost more and require a slightly different request shape — adapter handles it transparently. |
| `anthropic` | Tool calls supported. The Anthropic wire format for non-text content differs from OpenAI's, so list-shape `content` reaches the adapter as `LLMAdapterError`. See [Multimodal input](#multimodal-input) for vision-capable provider choices. |
| `deepseek` | `deepseek-reasoner` returns thinking tokens in `reasoning_content` — the adapter strips them from `Message.content` so downstream code doesn't see the chain-of-thought. To inspect them, look at the raw response. |
| `google` | Uses Google's OpenAI-compatible endpoint at `generativelanguage.googleapis.com/v1beta/openai`. Native Gemini SDK is not used. |
| `vllm` | `VLLM_API_KEY=EMPTY` (literal string) is the convention. `VLLM_MODEL` must match what's loaded on the server (see vLLM's `--served-model-name`). |

#### Programmatic LLM injection (tests, mocks, custom adapters)

`AgentConfig` accepts a pre-built `LLM` instance — when given, **the env-driven construction path is skipped entirely** for the LLM. Useful for:

```python
from DefenseAgent.llm import LLM
from DefenseAgent.llm.openai_compat import OpenAICompatibleAdapter

# 1. Test with a scripted/mocked LLM
config = AgentConfig(profile="…", llm=ScriptedLLM(responses=[...]))

# 2. Multiple agents with different providers in the same process
config_a = AgentConfig(profile=p, llm=LLM(adapter=OpenAICompatibleAdapter(api_key="...", base_url="https://api.openai.com/v1", model="gpt-4o")))
config_b = AgentConfig(profile=p, llm=LLM(adapter=AnthropicAdapter(api_key="...", model="claude-sonnet-4-6")))

# 3. Custom adapter (subclass LLMAdapter)
config = AgentConfig(profile="…", llm=LLM(adapter=MyCustomAdapter()))
```

The same injection pattern applies to every other component — see [Customization & dependency injection](#customization--dependency-injection) below.

### Identity

Only **`id`** and **`name`** are required. The other four fields (`age`, `traits`, `backstory`, `initial_plan`) flavour the agent's persona and have safe defaults — leave them out for a minimal agent, fill them in for a richer one.

```yaml
# minimal — just id + name
id: "bot"
name: "Helper"
```

```yaml
# full — every persona field populated
id: "agent_001"     # str, min_length=1. Required.
name: "Nova Patel"  # str, min_length=1. Required.
age: 27             # int ≥ 0 | null. Optional, default null.
traits: "..."       # str. Optional, default "".
backstory: "..."    # str. Optional, default "".
initial_plan: "..." # str. Optional, default "".
```

All six are exposed as `{id} {name} {age} {traits} {backstory} {initial_plan}` placeholders in the prompt template — see [`prompt:`](#prompt) below. Optional fields render as empty strings when unset, so a template referencing `{traits}` won't crash on a minimal profile.

#### What each field actually does

| Field | Required? | Used for |
|---|---|---|
| `id` | **yes** | (1) `agent_id` partition key in mem0 — records get scoped to this id. (2) Log file name: `<log_dir>/<id>.log`. (3) Available as `{id}` in the prompt template. **Choose a stable identifier you won't rename casually** — changing `id` orphans existing memory. |
| `name` | **yes** | The `{name}` placeholder. The auto-built identity prompt opens with `You are <name>, ...`. |
| `age` | optional (default `null`) | `{age}` placeholder. Useful for role-play personas. When unset, the auto-built prompt opens with `You are <name>.` (no age clause), and `{age}` in user templates renders as `""`. |
| `traits` | optional (default `""`) | `{traits}` placeholder. Free-form description of personality / tone / approach. When non-empty, the auto-built prompt adds a `Traits: ...` line. |
| `backstory` | optional (default `""`) | `{backstory}` placeholder. Long-form narrative — career, expertise, quirks. The most useful field for grounding the LLM in a specific persona. |
| `initial_plan` | optional (default `""`) | `{initial_plan}` placeholder. What the agent is currently working on; sets up the agent's "today" frame. |

#### Auto-built prompt with optional fields omitted

When fields are unset, the auto-built identity block skips their lines entirely instead of leaving blanks. With a minimal profile (`id: "bot"`, `name: "Helper"`), the agent's system prompt is just:

```
You are Helper.
```

Add `traits: "concise, technical"` and you get:

```
You are Helper.
Traits: concise, technical
```

…and so on. No awkward "You are Helper, a -year-old. Traits: " sentences.

#### Validation failure modes

The schema is strict — bad input fails at `AgentProfile.from_yaml()` with a `ConfigValidationError`, not at `agent.run()`:

| Input | Result |
|---|---|
| `id: ""` or `id: "   "` | `string_too_short` (id is required + non-empty after strip) |
| `name: ""` | `string_too_short` (name is required + non-empty) |
| missing `id` or missing `name` | `missing` validation error |
| missing `age` / `traits` / `backstory` / `initial_plan` | accepted — defaults to `null` / `""` |
| `age: -1` | `greater_than_equal` violation |
| `age: 27.5` | `int_type` violation (must be integer or null) |
| extra field | `extra_forbidden` — typos in field names fail loudly, no silent fallback |

### `cognitive:`

```yaml
cognitive:
  max_steps_per_cycle: 10     # int ≥ 1, default 10. Caps the ReAct tool-call loop per run().
  reflection_threshold: 5     # int ≥ 1, default 5. Unreflected-memory count that triggers Reflector.maybe_reflect().
  importance_threshold: 7     # float in [1, 10], default 7. Floor for "important" memories during reflection.
  planning_horizon: "1 day"   # str, min_length=1, default "1 day". Free-form; surfaced to the LLM in prompts.
```

#### `max_steps_per_cycle` — the ReAct loop budget

A "step" in `ReActAgent` is one (tool-call → tool-result) round-trip. `max_steps_per_cycle: 10` means the LLM gets at most 10 tool-call rounds before the loop force-exits. When that happens:

```python
result = await agent.run("multi-step task")
# result.stopped_reason == "max_steps"   ← loop hit the cap
# result.final_answer                    ← the LLM's last partial output
# result.steps                           ← full trace (10+ entries — call/result interleaved)
```

You can override per-call without editing the profile: `await agent.run(task, max_steps=20)`. `SimpleAgent` ignores both — by definition it makes exactly one LLM call. `PlanAndSolveAgent` interprets `max_steps` as the **plan length cap** (not the per-step substep cap; that's `AgentConfig.max_substeps_per_step`, default 3).

Tune it based on task complexity:

- Simple Q&A with one tool call: `max_steps_per_cycle: 3` is plenty.
- ReAct over multi-tool research: 10–20.
- Long-horizon iteration: raise it cautiously — every step is an LLM call you pay for.

#### `reflection_threshold` and the reflection cycle

After every `run()`, if `reflect_after_run: true` (default in `AgentConfig`), the agent calls `Reflector.maybe_reflect()`. That method is a guard: it only fires the reflection cycle when **at least `reflection_threshold` non-reflection records have accumulated** since the last reflection. Below the threshold, it's a no-op.

When it does fire:

1. `_get_unreflected_records()` pulls every mem0 record where `memory_type != 'reflection'`
2. `InsightSynthesizer.synthesize()` asks the LLM to distill them into N (default 3) bullet-shaped insights
3. Each insight is written back to mem0 tagged `memory_type='reflection'`, importance 8.0

So `reflection_threshold: 5` means "kick off reflection roughly every 5 runs/turns" (depending on what populates memory). Lower it to get more frequent introspection; raise it to keep reflections sparse and high-signal.

Reflections are visible to subsequent `memory_recall` calls — they let the agent build long-running understanding of itself across runs.

#### When reflection actually pays off — and when it doesn't

Reflection costs at least 2 extra LLM calls per cycle (`ImportanceScorer` + `InsightSynthesizer`), so it earns its keep only in scenarios where those reflection records are read back later. Be honest about which one you're in:

| Scenario | Reflection helps? | Recommendation |
|---|---|---|
| **One-off script** — `python my_agent.py` runs once and exits | **No.** Reflection writes 3 records, the process ends, nothing reads them. Pure waste. | `AgentConfig(profile=..., reflect_after_run=False)` |
| **Demo / quickstart** — exploring DefenseAgent for the first time | No. Same as above. | Same as above. |
| **Same `agent_id` across many sessions** — long-running assistant, recurring batch processing of similar tasks | **Yes.** Reflections from session N surface in session N+1 via `memory_recall`. The longer the agent lives, the more value compounds. | Keep default (`reflect_after_run=True`). |
| **Generative-Agents-style simulations** — multi-day simulated worlds, social agents | **Yes — by design.** This is the use case `Reflector` was built for ([Park et al. 2023](https://arxiv.org/abs/2304.03442)). | Keep default. Maybe lower `reflection_threshold` to fire more often. |
| **High-volume short tasks** — a customer-service agent handling hundreds of independent tickets | **Maybe.** Helpful only if reflections about agent failure modes survive across tickets. | Run with reflection on for a while, inspect mem0 records via `scripts/dump_memory.py`, decide. |

There's also a precondition for any of the "yes" cases: **the LLM in subsequent runs has to actually call `memory_recall`**. Reflections aren't auto-injected into the prompt — they only surface when the agent actively looks them up. A system prompt that explicitly tells the agent "before answering, call `memory_recall` for relevant prior context" makes reflection much more useful; a prompt that doesn't may waste the entire mechanism.

If you're building a one-shot tool, **disable reflection up front** to skip those LLM calls entirely:

```python
config = AgentConfig(
    profile=...,
    reflect_after_run=False,    # skip the post-run reflection cycle
)
```

You can also disable the underlying subsystem completely with `use_reflection=False` — that skips constructing the `Reflector` object at all. Use this when you have no `Reflector` need across the entire agent's lifetime.

#### `importance_threshold`

Used by `ImportanceScorer` (LLM-based 1–10 rating per record). During reflection, records below this threshold are filtered out before being fed to the synthesizer — keeps the LLM focused on substantive content rather than chitchat. Default 7 is conservative; lower to 5 if your records skew lower-impact.

#### `planning_horizon`

Free-form string — surfaces in the auto-built identity prompt as the agent's working time-horizon. Defaults to `"1 day"`. Examples that make sense:

- `"this hour"` for short-window operational agents
- `"this sprint"` for engineering agents
- `"the next 30 minutes"` for tight-deadline agents

The LLM uses it to decide what's in scope for the current run vs. what should be deferred. Visible only if your prompt template includes the auto-built identity block (or you reference it manually).

### `memory:`

```yaml
memory:
  is_retrieve: true                       # bool, default true. Wires up the memory_recall tool.
  history_mode: add                       # 'add' | 'overwrite'. 'overwrite' enables diff/rollback.
  search_limit: 10                        # int ≥ 1, default 10. Max records returned per memory_recall call.
  ignore_roles: [tool, system]            # list[str], default ['tool', 'system']. Roles excluded from persistence.
  ignore_fields: [reasoning_content]      # list[str], default ['reasoning_content'].
  context_limit: 128000                   # int ≥ 1024, default 128000. Token budget before ContextCompressor prunes.
  prune_protect: 40000                    # int ≥ 0, default 40000. Tokens never touched during prune.
  prune_minimum: 20000                    # int ≥ 0, default 20000. Min tokens kept after prune.
  reserved_buffer: 20000                  # int ≥ 0, default 20000. Safety margin.
  enable_summary: true                    # bool, default true. Allow ContextCompressor to LLM-summarise old turns.
  storage_path:                           # str | null. Default: <profile_dir>/memory/.
```

Requires `defense-agent[memory]` (`mem0ai`, `fastembed`).

#### How it actually stores

After the first `run()`, you'll see this on disk:

```
my_profile/
└── memory/                              # = storage_path (default <profile_dir>/memory/)
    ├── stream.db                        # SQLite — full block stream (every Message kept verbatim)
    ├── cache.json                       # block hashes for ms-agent's dedup
    └── qdrant/                          # local Qdrant — vector index over those blocks
        └── collection/<agent_id>/
```

Two stores side by side: SQLite keeps the **full conversation history** in insertion order; Qdrant keeps the **vector embeddings** that `memory_recall` semantic-searches over. Both are partitioned by the triple **`(user_id, agent_id, run_id)`** — a single agent across multiple sessions stays cleanly separated.

#### `history_mode: add` vs `overwrite`

- **`add`** (default) — every Message is appended. Re-running `agent.run("X")` twice creates two separate stored copies of the response. Simple and additive.
- **`overwrite`** — uses ms-agent's block-hash diff. Identical messages don't get re-stored; structurally similar runs replace the prior block. Enables rollback via the cached hash chain. Pick this when you want a "current best state" per run, not a permanent transcript.

Either way, `ignore_roles:` keeps `tool` and `system` messages out of persistence by default — the rationale is that tool results are large, redundant, and replayable from the original tool call. Add `assistant` to `ignore_roles:` if you only want to retain user-facing input.

#### `memory_type` taxonomy

When records are written, they're tagged with a `memory_type` (stored under metadata). Built-in tags you'll see:

| Tag | Source | Meaning |
|---|---|---|
| (default, untagged) | `agent.run()` trajectories | Raw conversation messages |
| `outcome` | `BaseAgent._save_outcome()` | The final answer from a successful run, when `save_outcome: true` |
| `failure` | Same path on `AgentError` | Truncated error text from a failed run |
| `reflection` | `Reflector.maybe_reflect()` | LLM-distilled lessons drawn over recent unreflected memories |
| `procedural` | mem0's native shape | mem0's procedural-memory channel; we don't write to this directly |

`memory_recall` returns records with their type prefix: `- [reflection] you tend to over-explain on tool failures`.

#### `memory_recall` — the built-in tool

When `is_retrieve: true`, the LLM gets a `memory_recall` tool registered automatically:

```json
{
  "name": "memory_recall",
  "input_schema": {
    "query": "string",
    "top_k":  "int (1..20, default 5)"
  }
}
```

It runs a Qdrant similarity search filtered by this run's `(user_id, agent_id, run_id)` and returns up to `top_k` records (capped by `search_limit:`). The agent decides when to call it — it's not auto-injected into every turn.

#### `ContextCompressor` — token-budget guard

Independent from memory_recall: this is what protects each LLM call from overflowing the context window. It runs **before** every LLM call and operates on the working messages (what you'd send to `chat()` this turn).

The four numbers interlock like this:

```
total tokens in working messages
        │
        │  if  total + reserved_buffer  >  context_limit
        │      then prune
        ▼
prune phase:
   ── keep most recent prune_protect tokens untouched (recent turns matter most)
   ── compress older turns down so total ≥ prune_minimum
   ── if enable_summary=true, the older block becomes a single LLM-generated summary turn
   ── if false, older turns are dropped without replacement
```

So `context_limit: 128000` + `reserved_buffer: 20000` means "start pruning when working messages cross 108K tokens." `prune_protect: 40000` says "never touch the most recent 40K tokens." `prune_minimum: 20000` is the floor — even if everything fits in 20K, don't compress further. Tune the four together; raising `context_limit` past your model's actual window causes API rejections with no upside.

### `rag:`

```yaml
rag:
  enabled: false                          # bool, default false. Flip to true to wire LlamaIndexRAG + rag_search.
  documents_dir: rag_corpus               # str | null. Relative to profile dir. Auto-indexed on first run().
  storage_dir: rag_index                  # str | null. Where the FAISS index is persisted.
  embedding_provider: openai              # 'openai' | 'huggingface', default 'openai'.
  embedding:                              # str | null. → EMBEDDING_MODEL.
  embedding_api_key:                      # str | null. → EMBEDDING_API_KEY.
  embedding_base_url:                     # str | null. → EMBEDDING_BASE_URL.
  embedding_dims:                         # int ≥ 1, null. → EMBEDDING_DIMS.
  chunk_size: 512                         # int ≥ 1, default 512. Tokens per chunk during ingestion.
  chunk_overlap: 50                       # int ≥ 0, default 50. Token overlap between adjacent chunks.
  top_k: 5                                # int ≥ 1, default 5. Default rag_search top_k.
  score_threshold: 0.0                    # float in [0.0, 1.0], default 0.0. Min score to return.
  retrieve_only: true                     # bool, default true. When false, RAG also synthesises an answer.
  use_huggingface: false                  # bool, default false. ms-agent's HF download path.
```

Requires `defense-agent[rag]` (`llama-index-core`, `llama-index-embeddings-openai-like`, `llama-index-retrievers-bm25`, `pdfplumber`, `beautifulsoup4`, `Pillow`).

#### Bootstrap flow — what happens on first run

The first time `agent.run()` fires under `rag.enabled: true`:

1. **Discover documents** — every file under `documents_dir` (relative to profile dir, default `rag_corpus/`) is enumerated.
2. **Extract structured chunks** — a `StructuredDocExtractor` walks each file with the registered extractor backends (`PyPdfExtractor`, `HtmlExtractor`, …). Each backend's `supports(path)` chooses by file extension/content. Plain `.md` / `.txt` go through LlamaIndex's default loader.
3. **Tokenise + chunk** — each extracted chunk is sub-split using `chunk_size:` tokens with `chunk_overlap:` overlap. Smaller chunks → finer recall but more index entries; larger chunks → fewer but coarser hits.
4. **Embed + index** — every chunk goes through the embedder (`embedding:` model), and the resulting vectors land in a persistent FAISS index under `storage_dir` (default `rag_index/`).
5. **Persist** — the index is dumped to disk so subsequent runs skip steps 1–4 entirely.

End-state directory:

```
my_profile/
├── profile.yaml
├── rag_corpus/                            # = documents_dir
│   ├── runbook.pdf
│   ├── architecture.html
│   └── notes.md
└── rag_index/                             # = storage_dir
    ├── default__vector_store.json         # FAISS vectors
    ├── docstore.json                      # original chunk text
    └── _resources/                        # extracted images/tables (referenced by chunks)
```

To re-index after document changes: delete `storage_dir` and run again. There's no incremental indexing — the index is whole-or-nothing.

#### Document formats — what's supported and how to extend

| Source | Backend | What gets extracted |
|---|---|---|
| `.pdf` | `PyPdfExtractor` (via `pdfplumber`) | Text, tables (rendered as Markdown), embedded images |
| `.html` | `HtmlExtractor` (via `beautifulsoup4`) | Body text segmented by section, tables, `<img>` references |
| `.md` / `.txt` / `.rst` | LlamaIndex default loader | Plain-text chunks |
| `.docx` / `.epub` / others | LlamaIndex default loader (best-effort) | Plain-text chunks |

Extractors are pluggable. Subclass the `StructuredExtractor` `Protocol` (must implement `supports(source)` and `extract(source) -> list[StructuredChunk]`), then register it on the extractor:

```python
from DefenseAgent.rag.extraction import StructuredDocExtractor

class MyCsvExtractor:
    def supports(self, source): return str(source).endswith(".csv")
    def extract(self, source): return [...]   # list[StructuredChunk]

extractor = StructuredDocExtractor(...)
extractor.register(MyCsvExtractor(), prepend=True)   # tried before built-ins
```

Same shape for resource renderers (table-to-Markdown, image-to-base64) — see `DefenseAgent/rag/renderer.py`.

#### Embedding choice — `openai` vs `huggingface`

| `embedding_provider:` | When to pick | Notes |
|---|---|---|
| `openai` (default) | Any OpenAI-compatible embedding endpoint — OpenAI itself, DashScope, ModelScope, vLLM, OpenRouter | Pulls the four `embedding_*` fields (or `EMBEDDING_*` env equivalents). The `openai-like` adapter handles all of these. |
| `huggingface` | Local-only, no API access (running offline / cost-sensitive) | Triggers ms-agent's HF download path via `use_huggingface: true`. Requires Hugging Face model id in `embedding:` (e.g. `BAAI/bge-large-en-v1.5`). Slower first run (model download). |

Whatever embedder you pick must match the `EMBEDDING_DIMS:` you set — `text-embedding-3-small` emits 1536, `text-embedding-3-large` emits 3072, Qwen3-Embedding-8B emits 4096. Mismatched dims → FAISS rejects writes.

#### `rag_search` tool — what the LLM sees

When `enabled: true`, the registry gets:

```json
{
  "name": "rag_search",
  "description": "Vector search over the agent's RAG corpus...",
  "input_schema": {
    "query": "string",
    "top_k": "int (default <profile.rag.top_k>)"
  }
}
```

The agent decides when to call it; the result format depends on `retrieve_only:`:

- **`retrieve_only: true`** (default) — returns the top-k chunks ranked, each prefixed with its score:
  ```
  [score=0.84] <chunk text 1>
  [score=0.71] <chunk text 2>
  ...
  ```
  Cheaper (no second LLM call), and gives the agent freedom to ignore/filter/rephrase.

- **`retrieve_only: false`** — runs LlamaIndex's built-in QA synthesizer on top of the retrieved chunks: a second LLM call composes a single answer string. More expensive, less flexible, but a one-shot answer comes out.

`score_threshold:` filters before returning — chunks below the threshold are dropped silently. Set to e.g. 0.4 to suppress weak matches; 0.0 (default) returns everything top_k surfaces.

### `tools:`

Three tool sources, all merged into a single `ToolRegistry` that the LLM sees as a flat namespace. Skim this YAML for the shape; each subsection below explains one source.

```yaml
tools:
  skills:                                 # list[str]. SKILL.md-style bundles (read-only by default).
    - skills/tabular-report
  mcp:                                    # list[MCPServerConfig]. External MCP tool servers.
    - command: uvx
      args: [mcp-server-filesystem, /tmp]
  python:                                 # list[str]. Python entry-point strings.
    - python_tools/calc.py:calculator
    - my_pkg.search:web_search
  allow_skill_execution: false            # bool, default false. Promote skill scripts to executable tools.
  skill_execution_timeout: 300            # int ≥ 1, default 300. Subprocess timeout (seconds).
```

When a `run()` starts, the registry is the union of all three sources plus the auto-registered `memory_recall` and `rag_search` (when enabled). Each tool name must be globally unique — collisions fail loud at construction.

---

#### `tools.skills:` — local SKILL.md bundles

A skill is a directory anywhere under (or pointed at by) the profile, with `SKILL.md` at its root. The reference bundle [`DefenseAgent/examples/example_agent/skills/tabular-report/`](DefenseAgent/examples/example_agent/skills/tabular-report) is the canonical shape:

```
skills/tabular-report/
├── SKILL.md                   # required — frontmatter + body
├── scripts/                   # optional — runnable scripts
│   └── generate.py
├── references/                # optional — long reference docs
└── templates/                 # optional — supporting resource files
    └── header.md
```

`SKILL.md` opens with YAML frontmatter, then a free-form Markdown body the LLM reads:

```markdown
---
name: tabular-report
description: Render a list of row dictionaries as a GitHub-flavored Markdown table.
author: kevin                  # optional, surfaces in tool metadata
tags: [reporting, table]       # optional, surfaces in tool metadata
---

# Tabular Report

Use this skill when you have row dicts and need a Markdown table.

## How to use it

1. Collect rows as a list of dicts with the same keys.
2. Pass column names explicitly — the skill won't infer them.
3. Read `scripts/generate.py` via this tool's `file=` argument, then call
   `render_table(rows, columns)` from your own code.
```

When the agent loads this skill, **one read-only tool** appears in the registry, named after the skill (`tabular-report`):

```json
{
  "name": "tabular-report",
  "description": "Render a list of row dictionaries as a GitHub-flavored Markdown table.\n\nBundled files — scripts: generate.py; references: None; resources: header.md.",
  "input_schema": {"file": "string (optional)"}
}
```

The description is the frontmatter `description:` plus a one-line inventory of bundled files (so the LLM can ask for them by name without guessing).

How the LLM uses it:

| Call | Returns |
|---|---|
| `tabular-report({})` (or `file=""`) | The SKILL.md body, frontmatter stripped — i.e. the LLM gets the prompt-style docs |
| `tabular-report({"file": "scripts/generate.py"})` | Raw text of that file |
| `tabular-report({"file": "templates/header.md"})` | Raw text of that file |
| `tabular-report({"file": "../../etc/passwd"})` | `SkillLoadError("path escapes skill directory ...")` — path-escape-guarded |

Skill metadata (skill_id, version, author, tags) rides along on the `Tool` object's `metadata` dict for downstream filtering or audit.

##### Promoting scripts to executable tools — `allow_skill_execution: true`

By default, scripts are *readable* but not *runnable* — the LLM has to paste their contents into its own reasoning. Flip `allow_skill_execution: true` and **each script becomes a separate executable tool** named `<skill>__<stem>`:

```yaml
tools:
  skills:
    - skills/tabular-report
  allow_skill_execution: true
  skill_execution_timeout: 300            # subprocess timeout (seconds)
```

Now the registry also exposes `tabular-report__generate` with input schema `{args?: list[str], stdin?: string, timeout?: int}`. Each call runs the script as a fresh subprocess via `SkillContainer` (inheriting ms-agent's dangerous-pattern guard against `rm -rf`-style payloads). Stdout, stderr and exit code are returned to the LLM as a single string.

Recognised script extensions: `.py`, `.sh`, `.js`. Scripts in subdirectories of `scripts/` are NOT recursively included — only top-level scripts get promoted.

---

#### `tools.mcp:` — external MCP servers

[Model Context Protocol](https://modelcontextprotocol.io) servers are external processes that expose their own tool catalogues. DefenseAgent's `MCPClient` extends ms-agent's multi-server client and supports four transports:

| `transport:` | When to use | Required field |
|---|---|---|
| `stdio` (default) | Locally-launched server processes (`uvx`, `npx`, `python`, ...) | `command:` |
| `sse` | Long-lived HTTP server-sent-events endpoints | `url:` |
| `websocket` | WS-based servers | `url:` |
| `streamable_http` | HTTP streaming-style endpoints | `url:` |

Each entry **must set exactly one** of `command:` or `url:` — never both. Servers are connected lazily on the first `agent.run()` call (the connection is async and only spun up when a tool actually fires).

##### stdio example — local filesystem server

```yaml
tools:
  mcp:
    - command: uvx                        # binary on PATH
      args: [mcp-server-filesystem, /tmp/sandbox]
      env:
        DEBUG: "1"
        GITHUB_TOKEN: ""                  # empty value → looked up in process env at connect()
      cwd: /workspace                     # optional working directory
      include: [read_file, list_dir]      # whitelist — only these tool names exposed
      # exclude: [delete_file]            # alternative: blacklist; mutually exclusive with include
```

Behaviour:

- Each tool the server advertises becomes a `Tool` in the registry, **named after the server's tool name** (no prefix). The originating server name is recorded in `tool.metadata["server"]` for traceability.
- `include:` / `exclude:` are mutually exclusive per server. Use them to scope down a chatty server (e.g. `mcp-server-filesystem` exposes ~10 tools — restrict to read-only with `include: [read_file, list_dir]`).
- Empty `env:` values (e.g. `GITHUB_TOKEN: ""`) are interpolated from the process environment at connect time — write `""` instead of hardcoding the key.

##### Network transport example — SSE

```yaml
tools:
  mcp:
    - transport: sse
      url: https://mcp.example.com/sse
      headers:
        Authorization: "Bearer ${MCP_API_TOKEN}"  # not auto-interpolated; expand yourself
      timeout: 30                                  # connection timeout in seconds
      sse_read_timeout: 300                        # long-poll read timeout
      include: [search]
```

Header values are passed verbatim — DefenseAgent does **not** expand `${VAR}` for you. If you want env-var substitution, do it programmatically before constructing `AgentConfig`, or store the resolved value in `.env` and inline it.

##### Multiple servers + dependency

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

Both servers' tools end up in the same flat registry. Tool-name collisions across servers fail at registry build, so name discipline matters when you compose many servers.

Install with `defense-agent[mcp]` (the official `mcp>=1.0` Python SDK).

---

#### `tools.python:` — your own Python functions

Two forms, both pointed at by an entry-point string `<module-or-file>:<function-name>`:

**1. Relative file path** (no packaging needed). Resolved against the profile's directory and loaded via `importlib.util.spec_from_file_location`. The interpreter doesn't need `sys.path` set up.

```
my_profile/
├── profile.yaml              # tools.python: ["python_tools/calc.py:calculator"]
└── python_tools/
    └── calc.py               # def calculator(expression: str) -> str: ...
```

**2. Dotted module path** (when your tool lives in an installed package). Resolved via `importlib.import_module`. The module must be importable from the running interpreter — installed via `pip install -e .` or already on `sys.path`.

```
my_pkg/
├── __init__.py
└── search.py                 # def web_search(query: str) -> str: ...
```

Profile entry: `my_pkg.search:web_search`.

##### Tool schema is auto-derived

For both forms the **function signature** becomes the tool's input schema and the **docstring** becomes the description. The LLM never sees your code body — only this synthesised metadata:

```python
def calculator(expression: str, precision: int = 4) -> str:
    """Evaluate a Python arithmetic expression and return the result.

    Supports +, -, *, /, **, parentheses, and the math module.
    """
    ...
```

Becomes:

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

Type-hint coverage: `str`, `int`, `float`, `bool`, `list[T]`, `dict`, `Optional[T]`, plain `Path`. Any complex type without a clean JSON-schema fallback raises `ToolRegistrationError` at load — name issues surface immediately, not on first call.

##### Custom tool in code (no profile entry)

If you don't want to put the tool in `profile.yaml`, register it programmatically:

```python
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression."""
    ...

config = AgentConfig(profile="…", tools=[calculator])
```

The `tools=` kwarg accepts plain callables — same auto-derivation applies. Use this for one-off tools, tests, or tools whose definition only makes sense at runtime (closures over an open DB connection, etc.).

### `prompt:`

```yaml
prompt:
  path: prompts/system.md         # str | null. File relative to profile dir.
  system:                         # str | null. Inline alternative to `path:`.
  extra_instructions:             # str | null. Appended after the resolved identity.
```

The system prompt is what the LLM sees as its `system=` argument on every call — the agent's "hat", separate from the user-turn task content.

#### Three resolution paths

The agent resolves the system prompt in this order, **first non-empty wins**:

1. **Inline `system:` field** — a literal string in the YAML. Use this for ad-hoc agents whose prompt is short and not worth a separate file.
2. **`path:` to a file** — resolved relative to the profile's directory. Use this for any non-trivial prompt — version control, reuse across agents, larger placeholders all become easier when the prompt is its own file.
3. **Auto-built identity block** — if both fields above are empty (or fail to render), the agent falls back to a generated prompt that fills out the persona using the identity fields.

In all three paths, `extra_instructions:` is appended at the end with a blank-line separator. Use it to layer agent-instance-specific tweaks on top of a shared base prompt without forking the file.

#### What the auto-built identity block looks like

When you have no `system:` and no `path:`, the agent generates something like:

```
You are Nova Patel, a 27-year-old field engineer turned AI researcher.

Personality: methodical, asks clarifying questions, prefers concrete examples
over abstractions.

Background: Started in industrial automation, pivoted to applied LLM research.
Currently embedded with the platform team.

Today's plan: shipping the v3 ingestion pipeline by Friday.

Your planning horizon for this run: 1 day.
```

…assembled from `name`/`age`/`traits` (one-liner), `backstory` (paragraph), `initial_plan` (paragraph), `cognitive.planning_horizon` (last line). It's a minimal scaffold — for any production agent, write your own template.

#### Concrete `prompts/system.md` example

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

The six placeholders (`{id} {name} {age} {traits} {backstory} {initial_plan}`) are rendered via Python's `str.format`. Anything else — `{plan}`, `{date}`, `{user}` — would `KeyError`.

#### `extra_instructions:` placement

Final prompt looks like:

```
<resolved-prompt-from-path-or-inline-or-auto-built>
<blank line>
<extra_instructions>
```

Use it for:
- Adding output-format constraints to a shared base prompt (`Always respond as JSON.`)
- Tightening tone for one specific agent instance without touching the template
- Per-environment overrides ("In production, never reveal stack traces.")

`AgentConfig.extra_instructions` (Python-side override) takes precedence over `profile.prompt.extra_instructions` if both are set — useful for runtime layering.

#### Failure modes and fallback behaviour

| Problem | Behaviour |
|---|---|
| `path:` points at a non-existent file | `ConfigValidationError` at profile load |
| Template references an unknown placeholder (e.g. `{date}`) | Renders error → falls back to auto-built identity block; the run continues. A warning is logged. |
| Both `system:` and `path:` set | `ConfigValidationError` — pick one, not both |
| Both empty + identity fields incomplete | Auto-build raises only if identity itself is invalid (which would already have failed earlier) |

The fall-back-to-auto-built behaviour is deliberate: a template typo shouldn't crash an agent in production. You'll see the warning in logs and can fix it without redeploying.

## Built-in tools

In addition to anything you register under `tools:`, the agent automatically exposes these to the LLM:

| Tool | When registered | Input schema | What it does |
|---|---|---|---|
| `memory_recall` | When `memory.is_retrieve: true` | `{query: string, top_k?: int (1–20, default 5)}` | Semantic search over mem0 records under this agent's `(user_id, agent_id, run_id)` filter. Returns up to top_k records as a `- [<memory_type>] <content>` bullet list. |
| `rag_search` | When `rag.enabled: true` | `{query: string, top_k?: int}` | Vector search over the RAG index. Returns ranked chunks above `score_threshold`. |
| `<skill>` (one per skill) | One per `tools.skills:` entry | `{file?: string}` | No `file` → returns the skill's SKILL.md body. With `file` → returns the named file from the skill directory. Path-escape-guarded. |
| `<skill>__<script>` (one per script) | When `allow_skill_execution: true` | `{args?: list[str], stdin?: string, timeout?: int}` | Runs the script as a subprocess via `SkillContainer`. Returns stdout + stderr + exit code rendered for the LLM. |

## Agent classes

| Class | Behaviour | When to use |
|---|---|---|
| `SimpleAgent` | One LLM call per `run()`. No tool loop. | Chat-shaped agents, zero tool use. |
| `ReActAgent` | Tool-call loop. Stops when the LLM returns plain text or `max_steps` is hit. | Default for tool-using agents. |
| `PlanAndSolveAgent` | Plan → execute each step → synthesise. | Long-horizon tasks where up-front planning helps. |

All three are constructed from the same `AgentConfig` and share `BaseAgent`'s helpers.

`agent.run(task, max_steps=None, images=None)`:
- `task: str` — user request.
- `max_steps: int | None` — overrides `cognitive.max_steps_per_cycle` for this call. Ignored by `SimpleAgent`.
- `images: list[str | Path] | None` — see Multimodal input.

Return type: `AgentResult`.

```python
@dataclass
class AgentResult:
    task: str                      # the original task string
    final_answer: str              # the LLM's final plain-text answer
    steps: list[AgentStep]         # full ReAct trace; one entry per event
    usage: TokenUsage              # aggregate token counts across the run
    stopped_reason: Literal["answered", "max_steps"] = "answered"

@dataclass
class AgentStep:
    index: int
    kind: Literal["plan", "tool_call", "tool_result", "answer"]
    content: str = ""              # for "answer" / "tool_call" steps: the LLM's text
    tool_calls: list[ToolCall] = ...    # for "tool_call": the requested calls
    tool_results: list[Message] = ...   # for "tool_result": one role='tool' Message per call
    usage: TokenUsage | None = None     # per-LLM-call token counts (None for tool_result steps)
```

## Multimodal input

DefenseAgent can attach images to the user turn so the LLM reasons about visual content alongside text. This is opt-in — you only pay the multimodal cost when you actually pass `images=`. Everything in the rest of this README applies unchanged when you don't.

### What "multimodal" means here

The OpenAI chat-completions API allows the `content` field of a user message to be a **list of content blocks** instead of a plain string. Each block is either text or an `image_url`. DefenseAgent's `Message` type already supports this shape, and `agent.run(task, images=[...])` is just an ergonomic helper that builds the list for you.

Useful for:

- Visual Q&A — "what's in this screenshot?", "is the chart in this PNG showing growth or decline?"
- OCR — extracting text from receipts, scanned PDFs (one page at a time), screenshots of code
- Visual debugging — passing a UI screenshot to an agent that suggests CSS fixes
- Image-grounded reasoning — comparing two product photos, identifying anomalies, layout review

It is **not** for: image generation (no SDXL etc. wired in), video, audio. Just static images going into the model.

### Pick a vision-capable model

The default chat models in the [Providers](#providers-and-credentials) table are text-only. To use `images=`, switch to a vision-capable model from the same provider — usually a different `<PROVIDER>_MODEL` value, no other env changes:

| Provider | Vision-capable models | Notes |
|---|---|---|
| OpenAI | `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo` (vision endpoint) | `gpt-4o-mini` is the cheap default for OCR-style tasks |
| Qwen (DashScope) | `qwen-vl-max`, `qwen-vl-plus`, `qwen-vl-max-latest` | The `-vl-` prefix signals visual; non-VL Qwen models won't accept images |
| GLM (智谱, OpenAI-compat) | `glm-4v`, `glm-4v-flash` | Hit GLM's OpenAI-compatible endpoint via `provider: openai` + `OPENAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4` |
| Kimi (Moonshot, OpenAI-compat) | `moonshot-v1-32k-vision-preview` | Same pattern — point `OPENAI_BASE_URL` at Moonshot |
| vLLM (self-hosted) | Anything visual you serve, e.g. `Qwen/Qwen2-VL-7B-Instruct`, `llava-hf/llava-1.5-13b-hf` | The vLLM server must be launched with `--limit-mm-per-prompt image=N` |
| **Anthropic** | **Not supported** in this version — see "Anthropic limitation" below |

Setup is the same as any other model — just point `<PROVIDER>_MODEL` at a vision-capable id:

```bash
# .env — Qwen-VL via DashScope
AGENT_LAB_LLM_PROVIDER=qwen
QWEN_API_KEY=sk-…
QWEN_MODEL=qwen-vl-max
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

### End-to-end: image recognition example

Concrete working example. Drop a screenshot into your project, point the agent at it:

```python
import asyncio
from pathlib import Path
from DefenseAgent.agent import AgentConfig, ReActAgent
from DefenseAgent.examples import EXAMPLE_PROFILE_PATH

async def main():
    agent = ReActAgent(AgentConfig(profile=EXAMPLE_PROFILE_PATH))

    result = await agent.run(
        "Describe what's in this image, including any text you can read.",
        images=[Path("./screenshot.png")],
    )
    print(result.final_answer)

asyncio.run(main())
```

```
$ python recognise.py
The image shows a terminal with the output of `pytest -v`. Visible test names
include test_agent_profile_minimal_with_only_id_and_name. The footer reads
"532 passed, 3 skipped in 4.88s". Background appears to be the iTerm2 default
dark theme.
```

The agent treats the image as part of the user turn — the LLM sees it natively, no separate OCR pass. Quality of the recognition is bounded by the vision model you picked: `qwen-vl-max` or `gpt-4o` for production work; smaller models are noticeably worse at small text or fine detail.

### How images flow through the system

`agent.run(task, images=[...])` walks each entry in `images=`, normalises it into a single URL string, and builds the OpenAI content-block message. Three input types are accepted:

| Input | What happens to it |
|---|---|
| `Path` / local file path string | The file is read, base64-encoded, and turned into a `data:<mime>;base64,…` URL. MIME is inferred from the file extension (`.png` → `image/png`, `.jpg` → `image/jpeg`, …); unknown extensions default to `image/png`. |
| `http://` or `https://` URL string | Passed through unchanged. The provider fetches the URL itself; DefenseAgent never downloads it. |
| `data:` URL string (already encoded) | Passed through unchanged — useful when you have an in-memory `BytesIO` you've already encoded. |

The resolved URLs end up in this exact request shape (this is what the OpenAI-compatible adapter sends):

```python
{
  "role": "user",
  "content": [
    {"type": "text", "text": "<your task string>"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR..."}},
    {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}},
  ]
}
```

The agent does **no preprocessing** — no resizing, no compression, no quality normalisation. Whatever bytes you point it at, the provider sees. This matters for two reasons:

1. **Base64 encoding inflates payload size by ~33%.** A 5 MB PNG becomes ~6.7 MB of base64. Large images add real latency to every call. Resize before passing if your model can work with smaller dimensions.
2. **Provider-specific size limits apply.** OpenAI rejects request bodies above ~20 MB; DashScope's limits vary by model. Hit the limit and you'll see a 4xx from the provider, not a friendly DefenseAgent error.

For local files, use a `Path` or string — both work. The base64 conversion happens in `_resolve_image_url` (a single ~10-line module helper). For URLs, **prefer them over local files when the image is already public** — passing a URL skips the base64 inflation and lets the provider cache it.

### Constraints and good practice

- **One turn, multiple images:** the list is unbounded on DefenseAgent's side, but most providers cap the number of images per request (OpenAI: typically up to 10; Qwen-VL: similar). Hit the cap → request fails.
- **Supported formats:** whatever the model supports. PNG and JPEG are universal; WebP, GIF (first frame), BMP work on most providers; HEIC and AVIF are spotty.
- **Transparency:** PNG alpha channels are passed through verbatim. Vision models tend to ignore them.
- **OCR-heavy use:** prefer high resolution (don't resize aggressively), pick a model marketed for OCR (`qwen-vl-max`, `gpt-4o`).
- **Batch processing:** for many images, fire many `agent.run()` calls in parallel rather than stuffing them all into one turn — same total token cost but faster wall-clock and easier error isolation.

### Where images get carried across multi-step agents

| Agent | Image-carrying behaviour |
|---|---|
| `SimpleAgent` | One turn, one call. Images attached to that single user message. |
| `ReActAgent` | Images attached **only to the initial user turn**. Subsequent tool-result messages stay text — the LLM has already seen the images, doesn't need them re-attached. |
| `PlanAndSolveAgent` | Images attached to **Phase 1 (plan) message** AND **every Phase 2 (execute-step) message**, so each phase that re-references the original task can re-inspect the visual content. Phase 3 (synthesis) is text-only — it summarises the per-step text outputs. |

This means an n-step ReAct over an image makes one image-carrying call and (n-1) text-only follow-ups. Cost is roughly: `1 × (text + image) + (n-1) × text`. Not n × image.

### Anthropic limitation

Claude's wire format for non-text content uses Anthropic's own `{"type": "image", "source": {...}}` block shape, **not** OpenAI's `{"type": "image_url", ...}` form. The `AnthropicAdapter` does not currently translate between them — passing list-shape `content` to it raises:

```python
LLMAdapterError: AnthropicAdapter received list-shape content but does not yet
support multimodal translation. Use an OpenAI-compatible vision provider, or
pass plain text content.
```

The `Message` type itself already accepts list content, so the missing piece is just a content-block translator inside the Anthropic adapter. PRs welcome — the change is localised to [`DefenseAgent/llm/anthropic.py`](DefenseAgent/llm/anthropic.py).

For now, if you need vision: pick any of the OpenAI-compatible providers above.

## Customization & dependency injection

Every component the agent depends on is replaceable via `AgentConfig`. When a pre-built component is given, **the env-driven construction path is skipped entirely for that component** — the rest of the system (other components + their env fallback) is unaffected. This is the primary extensibility surface: subclass, mock, or substitute any layer without forking the harness.

### Subsystem on/off switches

```python
config = AgentConfig(
    profile="…",
    use_tools=True,         # default. False → no tool registry built; LLM gets no tools.
    use_memory=True,        # default. False → skips mem0 setup, no memory_recall tool.
    use_reflection=True,    # default. False → no Reflector built, no post-run reflection cycle.
    use_rag=None,           # default → follows profile.rag.enabled. True/False overrides it.
    use_compressor=True,    # default. False → ContextCompressor never runs (you handle context yourself).
    use_logger=True,        # default. False → no AgentLogger; events suppressed.
)
```

When you toggle off `use_memory`, dependent toggles auto-disable too: `save_outcome`, `save_trajectory`, `reflect_after_run` all become no-ops (no memory backing → nowhere to write). No need to flip them yourself.

### Replaceable components

```python
config = AgentConfig(
    profile="…",

    # Each of these, when given, replaces the auto-built version.
    llm=my_llm,                       # LLM instance (any adapter)
    memory=my_mem0_memory,            # Mem0Memory or compatible duck-type
    tool_registry=my_registry,        # ToolRegistry already populated
    logger=my_logger,                 # AgentLogger
    reflector=my_reflector,           # Reflector
    compressor=my_compressor,         # ContextCompressor
    rag=my_rag,                       # LlamaIndexRAG (or any object with .search(query, top_k))

    # mem0 backend control — only used when memory=None and use_memory=True.
    # Lets you configure mem0's *internal* LLM/embedder programmatically, separate
    # from the agent's chat LLM, without ever touching .env.
    memory_backend=MemoryBackendConfig(
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        embedder_provider="openai",
        embedder_model="text-embedding-3-small",
    ),
)
```

### Inline tool injection (no profile entry)

In addition to anything in `tools.python:`, pass plain callables:

```python
def my_search(query: str) -> str:
    """Web search via my custom backend."""
    ...

config = AgentConfig(profile="…", tools=[my_search])
```

These get registered alongside `tools.python:` entries in the same `ToolRegistry`. Same auto-derivation rules: signature → schema, docstring → description.

### Common patterns

**Multi-LLM in one process.** Build two configs that share everything except `llm`:

```python
shared = dict(profile="…", memory=shared_memory, tool_registry=shared_registry)
config_fast  = AgentConfig(**shared, llm=cheap_llm)
config_smart = AgentConfig(**shared, llm=expensive_llm)
```

**Test with scripted responses.** A `ScriptedLLM` that returns canned `LLMResponse` objects in order — the entire test suite uses this.

```python
config = AgentConfig(profile="…", llm=ScriptedLLM([resp(content="ok")]))
```

**Custom memory backing.** Subclass `Mem0Memory`, override `search_records()`:

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

**Plug a different RAG backend.** Anything with a `search(query: str, top_k: int) -> list[dict]` method works:

```python
class ElasticRAG:
    async def search(self, query, top_k=5):
        # query Elasticsearch instead of FAISS...

config = AgentConfig(profile="…", rag=ElasticRAG(), use_rag=True)
```

The agent's `rag_search` tool will route through your object exactly the same way it routes through `LlamaIndexRAG`.

## Architecture

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

`build_components_sync` runs synchronously. MCP server connections and the optional RAG index are built lazily on the first `run()` call (they are async).

## Module layout

| Path | Contents |
|---|---|
| `DefenseAgent/config/profile.py` | `AgentProfile`, `LLMConfig`, `MemoryConfig`, `RAGConfig`, `ToolsConfig`, `MCPServerConfig`, `PromptConfig` |
| `DefenseAgent/llm/` | `LLM` facade, OpenAI-compatible + Anthropic adapters |
| `DefenseAgent/memory/` | mem0 memory + `ContextCompressor` |
| `DefenseAgent/tools/` | `ToolRegistry`, `MCPClient` |
| `DefenseAgent/skills/` | `SkillLoader`, `SkillContainer`, `to_tools()` adapter |
| `DefenseAgent/rag/` | `LlamaIndexRAG`, profile bridge |
| `DefenseAgent/reflection/` | `Reflector` |
| `DefenseAgent/agent/` | `BaseAgent`, `SimpleAgent`, `ReActAgent`, `PlanAndSolveAgent`, `AgentConfig`, `_builder` |
| `DefenseAgent/examples/` | `EXAMPLE_AGENT_DIR` + the bundled reference profile |

The memory, MCP, skill and RAG components are subclasses of [ms-agent](https://github.com/modelscope/ms-agent)'s upstream classes.

## Develop locally

If you want to modify DefenseAgent itself (vs. just consume it), clone the repo and install in editable mode with the dev extras:

```bash
git clone https://github.com/yishu031031/DefenseAgent.git
cd DefenseAgent
python -m venv .venv && source .venv/bin/activate
pip install -e '.[all,dev]'
```

Run the test suite (offline, no network or external services):

```bash
pytest                       # full suite
pytest -k tools              # one module
pytest -x --tb=short         # stop on first failure
```

531 tests, 3 skipped.

The repo also ships standalone demo scripts under `scripts/` (not part of the wheel):

```bash
python scripts/react_tools_memory_demo.py     # ReAct + calculator + Tavily + memory recall
python scripts/profile_chat_demo.py           # one-turn chat with the example profile
python scripts/tools_demo.py                  # walk the skill tool layers
python scripts/memory_demo.py                 # mem0 add / search / dump
```

## License

MIT.
