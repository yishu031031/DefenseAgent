# Architecture

Module layout, agent classes, the `AgentResult` shape, dependency-injection surface, and how to run / develop the codebase locally.

## High-level flow

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

`build_components_sync` runs synchronously. MCP server connections and the optional RAG index are built lazily on the first `run()` call (they're async).

## Module layout

| Path | Contents |
|---|---|
| `DefenseAgent/config/profile.py` | `AgentProfile`, `LLMConfig`, `MemoryConfig`, `ScoringWeights`, `TierLimits`, `ConsolidationConfig`, `RAGConfig`, `ToolsConfig`, `MCPServerConfig`, `PromptConfig` |
| `DefenseAgent/llm/` | `LLM` facade, OpenAI-compatible + Anthropic adapters |
| `DefenseAgent/memory/` | `MemoryOrchestrator`, `Mem0Memory`, `WorkingMemory`, `MemoryConsolidator`, `MemoryItem`/`MemoryTier`, `scoring`, `ContextCompressor` |
| `DefenseAgent/tools/` | `ToolRegistry`, `MCPClient` |
| `DefenseAgent/skills/` | `SkillLoader`, `SkillContainer`, `to_tools()` adapter |
| `DefenseAgent/rag/` | `LlamaIndexRAG`, profile bridge, document extractors |
| `DefenseAgent/reflection/` | `Reflector`, `ImportanceScorer`, `InsightSynthesizer` |
| `DefenseAgent/agent/` | `BaseAgent`, `SimpleAgent`, `ReActAgent`, `PlanAndSolveAgent`, `AgentConfig`, `_builder` |
| `DefenseAgent/examples/` | `EXAMPLE_AGENT_DIR` + the bundled reference profile |

The memory, MCP, skill and RAG components subclass [ms-agent](https://github.com/modelscope/ms-agent)'s upstream classes — DefenseAgent adds the tier-aware orchestrator, profile bridge, dependency-injection surface, and the unified agent loop on top.

## Agent classes

| Class | Behaviour | When to use |
|---|---|---|
| `SimpleAgent` | One LLM call per `run()`. No tool loop. | Chat-shaped agents, zero tool use. |
| `ReActAgent` | Tool-call loop. Stops when the LLM returns plain text or `max_steps` is hit. | Default for tool-using agents. |
| `PlanAndSolveAgent` | Plan → execute each step → synthesise. | Long-horizon tasks where up-front planning helps. |

All three are constructed from the same `AgentConfig` and share `BaseAgent`'s helpers.

```python
async def run(
    self,
    task: str,
    max_steps: int | None = None,
    images: list[str | Path] | None = None,
) -> AgentResult: ...
```

- `task: str` — user request.
- `max_steps: int | None` — overrides `cognitive.max_steps_per_cycle` for this call. Ignored by `SimpleAgent`.
- `images: list[str | Path] | None` — see [`providers.md` → Multimodal](providers.md#multimodal-input).

### `AgentResult` / `AgentStep`

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
    content: str = ""              # for "answer" / "tool_call": LLM text
    tool_calls: list[ToolCall] = ...    # for "tool_call": requested calls
    tool_results: list[Message] = ...   # for "tool_result": one role='tool' Message per call
    usage: TokenUsage | None = None     # per-LLM-call token counts (None for tool_result steps)
```

## `AgentConfig` — single-argument agent setup

All three agent strategies (`SimpleAgent`, `ReActAgent`, `PlanAndSolveAgent`) accept the **same single argument**: an `AgentConfig`. It bundles the agent's identity (a YAML profile or pre-built `AgentProfile`) with on/off switches for every optional subsystem (tools, memory, reflection, RAG, context compressor, logger) plus per-strategy knobs.

```python
from DefenseAgent.agent import AgentConfig, ReActAgent

config = AgentConfig(
    profile="agents/my_agent/profile.yaml",
    tools=[calculator, web_search],   # plain Python functions
    use_memory=True,
    use_reflection=True,
    use_rag=True,
)

async with ReActAgent(config) as agent:        # also: agent = ReActAgent(config); await agent.close()
    result = await agent.run("Hello")
```

### All `AgentConfig` fields

#### Identity (required)

| Field | Type | Default | Meaning |
|---|---|---|---|
| `profile` | `AgentProfile \| str \| Path` | — | A pre-loaded `AgentProfile` or a path to a profile YAML file. |

#### Environment loading

| Field | Type | Default | Meaning |
|---|---|---|---|
| `dotenv_path` | `str \| None` | `None` | Path to a `.env` file. `None` means "use the project-default `.env`". |
| `load_env` | `bool` | `True` | Read `.env` into `os.environ`. Set `False` if env vars are already present. |

#### Subsystem toggles

| Field | Type | Default | Meaning |
|---|---|---|---|
| `use_tools` | `bool` | `True` | Register user tools (`tools=`, `profile.tools.skills`, `profile.tools.mcp`). |
| `use_memory` | `bool` | `True` | Build the tier-aware `MemoryOrchestrator`, register `memory_recall`, persist outcomes/trajectories. |
| `use_reflection` | `bool` | `True` | Build a `Reflector` and run it after every `run()` (still gated by its own threshold). Needs memory. |
| `use_rag` | `bool \| None` | `None` | `True` forces RAG on, `False` forces off, `None` follows `profile.rag.enabled`. |
| `use_compressor` | `bool` | `True` | Build a `ContextCompressor` and chain it after memory in the per-step condense pipeline. |
| `use_logger` | `bool` | `True` | Build an `AgentLogger` writing to `<log_dir>/<profile.id>.log`. |

#### Tool wiring

| Field | Type | Default | Meaning |
|---|---|---|---|
| `tools` | `list[Callable]` | `[]` | Extra Python callables. Signature + docstring become the JSON schema. |
| `log_dir` | `str \| Path \| None` | `None` | Where to put the agent's log file. Defaults to `<profile.source_dir>/logs` when the profile was loaded from disk. |

#### Behaviour knobs

| Field | Type | Default | Meaning |
|---|---|---|---|
| `memory_recall_top_k` | `int` | `5` | Default `top_k` for `memory_recall` when the LLM omits one. `0` suppresses recall. |
| `save_outcome` | `bool` | `True` | After each `run()`, write `(Q → A)` to memory tagged `outcome` (or `failure` on errors). |
| `save_trajectory` | `bool` | `True` | Per ReAct tool turn, write a one-line summary tagged `trajectory`. ReAct only. |
| `reflect_after_run` | `bool` | `True` | Call `Reflector.maybe_reflect` after each `run()`. |
| `extra_instructions` | `str \| None` | `None` | Free-form text appended to the system prompt. Overrides `profile.prompt.extra_instructions`. |
| `max_substeps_per_step` | `int` | `3` | `PlanAndSolveAgent` only — per-plan-step tool-call budget. |
| `max_steps` | `int \| None` | `None` | Default `max_steps` for `agent.run(task)`. `None` → falls back to `profile.cognitive.max_steps_per_cycle`. |

### Auto-disable rules

Some toggles depend on each other. The builder silently disables dependents when the parent is off — you don't need to flip every flag manually:

- `use_memory=False` → `save_outcome`, `save_trajectory`, `reflect_after_run` and the `memory_recall` tool are all forced off.
- `use_reflection=False` → `reflect_after_run` is forced off.

### Sync vs. async setup

`agent = ReActAgent(config)` is **synchronous** — builds LLM, memory, Python-function tools, skills, reflector, compressor and logger immediately.

Two pieces need `await` and are wired lazily on the first `run()` call:

- MCP servers from `profile.tools.mcp` (each spawns a subprocess).
- `LlamaIndexRAG` indexing (when RAG is on).

You don't need to do anything — the first `await agent.run(...)` finishes the wiring before executing. To force eager setup, call `await agent._ensure_async_setup()` yourself.

### Worked examples

**Minimal — chat-only agent, no memory, no tools:**

```python
config = AgentConfig(
    profile=profile,
    use_tools=False,
    use_memory=False,
    use_reflection=False,
    use_compressor=False,
    use_logger=False,
)
agent = SimpleAgent(config)
```

**ReAct with calculator + Tavily, memory on, RAG off:**

```python
config = AgentConfig(
    profile="agents/my_agent/profile.yaml",
    tools=[calculator, web_search],
    use_memory=True,
    use_rag=False,
)
agent = ReActAgent(config)
```

**PlanAndSolve with everything on, custom log dir:**

```python
config = AgentConfig(
    profile=profile,
    tools=[calculator],
    log_dir="/tmp/agent-logs",
    max_substeps_per_step=5,
    max_steps=20,
)
agent = PlanAndSolveAgent(config)
```

## Customization & dependency injection

Every component the agent depends on is replaceable via `AgentConfig`. When a pre-built component is given, **the env-driven construction path is skipped entirely for that component** — the rest of the system (other components + their env fallback) is unaffected. This is the primary extensibility surface: subclass, mock, or substitute any layer without forking the harness.

### Subsystem on/off switches

```python
config = AgentConfig(
    profile="…",
    use_tools=True,         # default. False → no tool registry; LLM gets no tools.
    use_memory=True,        # default. False → skips mem0 setup, no memory_recall tool.
    use_reflection=True,    # default. False → no Reflector built, no post-run reflection cycle.
    use_rag=None,           # default → follows profile.rag.enabled. True/False overrides it.
    use_compressor=True,    # default. False → ContextCompressor never runs.
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
    memory=my_mem0_memory,            # Mem0Memory OR MemoryOrchestrator
    tool_registry=my_registry,        # ToolRegistry already populated
    logger=my_logger,                 # AgentLogger
    reflector=my_reflector,           # Reflector
    compressor=my_compressor,         # ContextCompressor
    rag=my_rag,                       # LlamaIndexRAG (or any object with .search)

    # mem0 backend control — only used when memory=None and use_memory=True.
    # Configures mem0's *internal* LLM/embedder programmatically, without .env.
    memory_backend=MemoryBackendConfig(
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        embedder_provider="openai",
        embedder_model="text-embedding-3-small",
    ),
)
```

The builder auto-wraps an injected bare `Mem0Memory` in a `MemoryOrchestrator` so downstream code always sees the tier-aware facade.

### Inline tool injection (no profile entry)

In addition to anything in `tools.python:`, pass plain callables:

```python
def my_search(query: str) -> str:
    """Web search via my custom backend."""
    ...

config = AgentConfig(profile="…", tools=[my_search])
```

These get registered alongside `tools.python:` entries in the same `ToolRegistry`. Same auto-derivation rules: signature → schema, docstring → description.

## Common patterns

### Multi-LLM in one process

Two configs that share everything except `llm`:

```python
shared = dict(profile="…", memory=shared_memory, tool_registry=shared_registry)
config_fast  = AgentConfig(**shared, llm=cheap_llm)
config_smart = AgentConfig(**shared, llm=expensive_llm)
```

### Test with scripted responses

A `ScriptedLLM` that returns canned `LLMResponse` objects in order — the entire test suite uses this:

```python
config = AgentConfig(profile="…", llm=ScriptedLLM([resp(content="ok")]))
```

### Custom memory backing

Subclass `Mem0Memory`, override `search_records()`:

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

### Plug a different RAG backend

Anything with a `search(query: str, top_k: int) -> list[dict]` method works:

```python
class ElasticRAG:
    async def search(self, query, top_k=5):
        # query Elasticsearch instead of FAISS...

config = AgentConfig(profile="…", rag=ElasticRAG(), use_rag=True)
```

The agent's `rag_search` tool routes through your object exactly the same way it routes through `LlamaIndexRAG`.

## Develop locally

```bash
git clone https://github.com/yishu031031/DefenseAgent.git
cd DefenseAgent
python -m venv .venv && source .venv/bin/activate
pip install -e '.[all,dev]'
```

### Test suite

Fully offline — no network or external services:

```bash
pytest                       # full suite
pytest -k tools              # one module
pytest -x --tb=short         # stop on first failure
```

### Standalone demo scripts

Ship under `scripts/` (not part of the wheel):

```bash
python scripts/react_tools_memory_demo.py   # ReAct + calculator + Tavily + memory recall
python scripts/profile_chat_demo.py         # one-turn chat with the example profile
python scripts/tools_demo.py                # walk the skill tool layers
python scripts/memory_demo.py               # mem0 add / search / dump
python scripts/smoke_new_memory.py          # tier-aware memory smoke (no external deps)
python scripts/smoke_real_backend.py        # real Ollama + Qdrant smoke (needs running services)
```

### Build & publish

```bash
python -m build                                  # produces dist/defense_agent-X.Y.Z-*.whl
python -m twine upload dist/defense_agent-X.Y.Z*
```

Restore editable mode after publishing experiments:

```bash
pip install -e . --force-reinstall --no-deps
```
