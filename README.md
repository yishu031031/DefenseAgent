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

## Features

- **One-file agent definition.** Identity, LLM provider, tools, memory, RAG, system prompt — all in one strictly-validated YAML (`extra="forbid"`; unknown fields raise `ConfigValidationError` on load).
- **Per-field configuration fallback.** Every value can be set in the profile or in `.env`; profile wins per field, `.env` fills the gaps. Switch LLM providers (`openai`, `anthropic`, `deepseek`, `qwen`, `google`, `vllm`) without code changes.
- **Three agent strategies.** `SimpleAgent` (one-shot), `ReActAgent` (tool-call loop), `PlanAndSolveAgent` (plan → execute → synthesise). All built from the same `AgentConfig`.
- **Three tool sources, one registry.** Local skill directories (Anthropic-style `SKILL.md` bundles), MCP servers (stdio / SSE / WebSocket / streamable-http), Python functions (referenced from the profile by file path or dotted module).
- **Persistent memory with a built-in tool.** mem0-backed Qdrant storage; agents automatically expose a `memory_recall` tool to the LLM. `ContextCompressor` keeps the working context within a configured token budget.
- **Optional RAG with a built-in tool.** Drop documents into a directory, set `rag.enabled: true`, get a `rag_search` tool. Embedder credentials follow the same per-field profile→env fallback.
- **Multimodal input.** `agent.run(task, images=[...])` sends an OpenAI-style content-block message. Each image accepts a local file path, an `http(s)://` URL, or a `data:` URL. Supported on every OpenAI-compatible provider; the Anthropic adapter raises a clear `LLMAdapterError` if list content reaches it.
- **Dependency-injectable.** LLM, memory, tools, reflector, compressor and logger are all replaceable in `AgentConfig` for tests and custom wiring.
- **Offline test suite.** No network or external services required to run `pytest`.

## Install

```bash
git clone https://github.com/yishu031031/DefenseAgent.git
cd DefenseAgent
conda create -n agent_lab python=3.12 -y
conda activate agent_lab
pip install -r requirements.txt
```

## Configure

Create `.env` in the repo root. Minimum:

```bash
AGENT_LAB_LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-…
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1

EMBEDDING_API_KEY=sk-…
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMS=1536

TAVILY_API_KEY=…    # optional, used by scripts/react_tools_memory_demo.py
```

Resolution order, per field: profile YAML → env var → schema default. Whitespace-only values are treated as unset.

### Providers and credentials

`AGENT_LAB_LLM_PROVIDER` selects the adapter. Each provider has its own block of `<PROVIDER>_*` env vars (`<PROVIDER>_API_KEY`, `<PROVIDER>_MODEL`, `<PROVIDER>_BASE_URL`). The cross-provider `LLM_API_KEY` / `LLM_MODEL_ID` / `LLM_BASE_URL` tier overrides the per-provider tier when set.

| Provider | Adapter | Typical key format | Default base URL | Example chat models |
|---|---|---|---|---|
| `openai` | `OpenAICompatibleAdapter` | `sk-…` or `sk-proj-…` | `https://api.openai.com/v1` | `gpt-4o-mini`, `gpt-4o`, `o3-mini` |
| `anthropic` | `AnthropicAdapter` | `sk-ant-…` | `https://api.anthropic.com` | `claude-sonnet-4-6`, `claude-opus-4-7` |
| `deepseek` | `OpenAICompatibleAdapter` | `sk-…` | `https://api.deepseek.com/v1` | `deepseek-chat`, `deepseek-reasoner` |
| `qwen` (DashScope, OpenAI-compat) | `OpenAICompatibleAdapter` | `sk-…` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-plus`, `qwen-vl-max`, `qwen-vl-plus` |
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

## Quickstart

```python
import asyncio
from DefenseAgent.agent import AgentConfig, ReActAgent
from DefenseAgent.examples import EXAMPLE_PROFILE_PATH

config = AgentConfig(profile=EXAMPLE_PROFILE_PATH)

async def main():
    async with ReActAgent(config) as agent:
        result = await agent.run("Summarise today's plan in one sentence.")
        print(result.final_answer)

asyncio.run(main())
```

End-to-end demo (calculator + Tavily web search + memory recall):

```bash
python scripts/react_tools_memory_demo.py
```

## Building your own agent

Copy `DefenseAgent/examples/example_agent/` (also available at runtime as `EXAMPLE_AGENT_DIR` in `DefenseAgent.examples`) to a new directory and edit `profile.yaml`. Each block under `agent:` is independent and optional except identity. All fields are validated by pydantic with `extra="forbid"`.

### `llm:`

```yaml
llm:
  provider:           # str | null. One of: openai | anthropic | deepseek | qwen | google | vllm. Falls back to AGENT_LAB_LLM_PROVIDER.
  model:              # str | null. Provider-specific model id (see Providers table). Falls back to <PROVIDER>_MODEL or LLM_MODEL_ID.
  base_url:           # str | null. Provider endpoint. Falls back to <PROVIDER>_BASE_URL or LLM_BASE_URL.
  api_key:            # str | null. Falls back to <PROVIDER>_API_KEY. Recommend leaving blank in shared profiles.
```

All four fields are `str | None`. Each falls back to `.env` independently. Whitespace-only values count as unset, so a half-edited YAML can't shadow correct env state.

### Identity (required)

```yaml
id: "agent_001"     # str, min_length=1. Used as agent_id in mem0 + as the log file name.
name: "Nova Patel"  # str, min_length=1. The {name} placeholder.
age: 27             # int ≥ 0.
traits: "..."       # str, min_length=1. Free-form trait list.
backstory: "..."    # str, min_length=1.
initial_plan: "..." # str, min_length=1.
```

Every field is non-empty after stripping. All six are exposed as `{id} {name} {age} {traits} {backstory} {initial_plan}` placeholders in the prompt template.

### `cognitive:`

```yaml
cognitive:
  max_steps_per_cycle: 10     # int ≥ 1, default 10. Caps the ReAct tool-call loop per run().
  reflection_threshold: 5     # int ≥ 1, default 5. Unreflected-memory count that triggers Reflector.maybe_reflect().
  importance_threshold: 7     # float in [1, 10], default 7. Floor for "important" memories during reflection.
  planning_horizon: "1 day"   # str, min_length=1, default "1 day". Free-form; surfaced to the LLM in prompts.
```

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

mem0 + Qdrant on disk. Registers a `memory_recall` tool. `ContextCompressor` runs before each LLM turn.

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

When `enabled: true`, registers a `rag_search` tool. Embedder fields use the same per-field profile→env fallback as `llm:`.

### `tools:`

```yaml
tools:
  skills:                                 # list[str]. Skill directory paths, relative to profile dir.
    - skills/tabular-report
  mcp:                                    # list[MCPServerConfig].
    - command: uvx                        # str | null. Required for stdio servers.
      args: [mcp-server-filesystem, /tmp] # list[str], default [].
      env: { TOKEN: "" }                  # dict[str,str] | null. Empty values interpolated from process env.
      cwd:                                # str | null. Optional working dir.
      include: [read_file]                # list[str]. Whitelist; mutually exclusive with `exclude`.
      exclude: []                         # list[str]. Blacklist.
    - transport: sse                      # 'stdio' | 'sse' | 'websocket' | 'streamable_http'.
      url: https://mcp.example.com/sse    # str | null. Required when transport != 'stdio'.
      headers: { Authorization: "..." }   # dict[str,str] | null.
      timeout: 30                         # float ≥ 0 | null. Connection timeout (seconds).
      sse_read_timeout: 300               # float ≥ 0 | null. SSE long-poll timeout.
  python:                                 # list[str]. Python entry-point strings.
    - python_tools/calc.py:calculator
    - my_pkg.search:web_search
  allow_skill_execution: false            # bool, default false. Opt-in to script execution.
  skill_execution_timeout: 300            # int ≥ 1, default 300. Subprocess timeout (seconds).
```

Each MCP entry must specify exactly one of `command:` (stdio) or `url:` (network). `include` and `exclude` are mutually exclusive per server.

#### Where to place a Python tool file

`tools.python:` accepts two forms:

**1. Relative file path.** Resolved against the profile's directory and loaded via `importlib.util.spec_from_file_location`. No `sys.path` setup needed.

```
DefenseAgent/examples/example_agent/
├── profile.yaml
├── python_tools/
│   └── calc.py            # def calculator(expression: str) -> str
└── skills/
```

Profile entry: `python_tools/calc.py:calculator`.

**2. Dotted module path.** The module must be importable from the running interpreter. Resolved via `importlib.import_module`.

```
my_pkg/
├── __init__.py
└── search.py              # def web_search(query: str) -> str
```

Profile entry: `my_pkg.search:web_search`.

For both forms, the function's type hints become the tool's input schema and its docstring becomes the tool description.

#### Custom tool in code (no profile entry)

```python
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression."""
    ...

config = AgentConfig(profile="…", tools=[calculator])
```

#### Skill execution

`allow_skill_execution: true` registers each script bundled in a skill (`scripts/*.py`, `*.sh`, `*.js`) as a separate executable Tool, named `<skill_name>__<script_stem>`. Subprocess-based via `SkillContainer` with the inherited dangerous-pattern guard.

### `prompt:`

```yaml
prompt:
  path: prompts/system.md         # str | null. File relative to profile dir.
  system:                         # str | null. Inline alternative to `path:`.
  extra_instructions:             # str | null. Appended after the resolved identity.
```

Precedence: inline `system:` > `path:` > auto-built identity block. Available placeholders inside the template (rendered via `str.format`): `{id} {name} {age} {traits} {backstory} {initial_plan}`. A broken template falls back to the auto-built identity rather than crashing the run.

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

All three agents accept an optional `images=` argument on `run()`:

```python
from pathlib import Path

result = await agent.run(
    "What's in this image, and how does it compare to this URL?",
    images=[
        Path("./screenshot.png"),
        "https://example.com/photo.jpg",
    ],
)
```

When `images` is provided, the user turn is sent as an OpenAI content-block list:

```python
[{"type": "text", "text": "<task>"},
 {"type": "image_url", "image_url": {"url": "<resolved-url-1>"}},
 {"type": "image_url", "image_url": {"url": "<resolved-url-2>"}}]
```

Each image entry can be:

| Input | Behaviour |
|---|---|
| `Path` or local file path string | Read, base64-encoded, emitted as `data:<mime>;base64,…`. MIME inferred from extension; defaults to `image/png`. |
| `http://` or `https://` URL string | Passed through unchanged. |
| `data:` URL string | Passed through unchanged. |

Provider compatibility:

- **OpenAI-compatible adapters** (Qwen via DashScope, DeepSeek-VL, GLM, Kimi, vLLM serving multimodal models, OpenAI itself) consume the list-shape directly. Set `llm.model:` to a vision-capable model.
- **Anthropic adapter** raises `LLMAdapterError` with an explicit message if list content arrives. The `Message` type already supports list content, so adding Claude vision later is a localised adapter change.

For `ReActAgent`, only the initial user turn carries images — subsequent tool-result messages stay text. For `PlanAndSolveAgent`, the Phase 1 plan message and every Phase 2 execute-step message carry the same images, so each phase can re-inspect the visual content.

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

The memory, MCP, skill and RAG components are subclasses of [ms-agent](https://github.com/modelscope/ms-agent)'s upstream classes.

## Demos

```bash
python scripts/react_tools_memory_demo.py     # ReAct + calculator + Tavily + memory recall
python scripts/profile_chat_demo.py           # one-turn chat with the example profile
python scripts/tools_demo.py                  # walk the skill tool layers
python scripts/memory_demo.py                 # mem0 add / search / dump
```

## Tests

```bash
pytest                       # full suite, offline
pytest -k tools              # one module
pytest -x --tb=short         # stop on first failure
```

531 tests, 3 skipped.

## License

MIT.
