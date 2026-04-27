# `AgentConfig` — single-argument agent setup

DefenseAgent ships three agent strategies — `SimpleAgent`, `ReActAgent`,
`PlanAndSolveAgent` — that all accept the **same single argument**: an
`AgentConfig` instance. The config bundles the agent's identity (a YAML
profile or a pre-built `AgentProfile`) with on/off switches for every
optional subsystem (tools, memory, reflection, RAG, context compressor,
logger) plus the per-strategy knobs.

```python
from DefenseAgent import AgentConfig, ReActAgent

config = AgentConfig(
    profile="agents/maya_rodriguez/profile.yaml",
    tools=[calculator, web_search],   # plain Python functions
    use_memory=True,
    use_reflection=True,
    use_rag=True,
)

agent = ReActAgent(config)
result = await agent.run("Hello")
await agent.close()
```

`SimpleAgent(config)` and `PlanAndSolveAgent(config)` work identically.
The `async with` form also works:

```python
async with ReActAgent(config) as agent:
    result = await agent.run("Hello")
```

---

## All `AgentConfig` fields

### Identity (required)

| Field     | Type                              | Default | Meaning                                                                 |
| --------- | --------------------------------- | ------- | ----------------------------------------------------------------------- |
| `profile` | `AgentProfile \| str \| Path`     | —       | Either a pre-loaded `AgentProfile` or a path to a profile YAML file.    |

### Environment loading

| Field          | Type           | Default | Meaning                                                                |
| -------------- | -------------- | ------- | ---------------------------------------------------------------------- |
| `dotenv_path`  | `str \| None`  | `None`  | Path to a `.env` file. `None` means "use the project-default `.env`".  |
| `load_env`     | `bool`         | `True`  | Read `.env` into `os.environ`. Set `False` if env vars already in process. |

### Subsystem toggles

| Field             | Type             | Default | Meaning                                                                                                    |
| ----------------- | ---------------- | ------- | ---------------------------------------------------------------------------------------------------------- |
| `use_tools`       | `bool`           | `True`  | Register user tools (`tools=`, `profile.tools.skills`, `profile.tools.mcp`).                               |
| `use_memory`      | `bool`           | `True`  | Build mem0-backed `Mem0Memory`, register `memory_recall`, persist outcomes/trajectories.                   |
| `use_reflection`  | `bool`           | `True`  | Build a `Reflector` and run it after every `run()` (still gated by its own threshold). Needs memory.       |
| `use_rag`         | `bool \| None`   | `None`  | `True` forces RAG on, `False` forces it off, `None` follows `profile.rag.enabled`.                         |
| `use_compressor`   | `bool`           | `True`  | Build a `ContextCompressor` and chain it after memory in the per-step condense pipeline.                   |
| `use_logger`      | `bool`           | `True`  | Build an `AgentLogger` writing to `<log_dir>/<profile.id>.log`.                                            |

### Tool wiring

| Field      | Type                          | Default | Meaning                                                                                                  |
| ---------- | ----------------------------- | ------- | -------------------------------------------------------------------------------------------------------- |
| `tools`    | `list[Callable]`              | `[]`    | Extra Python callables to register on the agent's `ToolRegistry`. Function signature + docstring become the JSON schema. |
| `log_dir`  | `str \| Path \| None`         | `None`  | Where to put the agent's log file. Defaults to `<profile.source_dir>/logs` when the profile was loaded from disk. |

### Behavior knobs

| Field                    | Type           | Default | Meaning                                                                                                     |
| ------------------------ | -------------- | ------- | ----------------------------------------------------------------------------------------------------------- |
| `memory_recall_top_k`    | `int`          | `5`     | Default `top_k` for the agent-owned `memory_recall` tool when the LLM omits one. `0` suppresses recall.     |
| `save_outcome`        | `bool`         | `True`  | After each `run()`, write a `(Q → A)` pair to memory tagged `outcome` (or `failure` on errors).             |
| `save_trajectory`     | `bool`         | `True`  | Per ReAct tool turn, write a one-line summary tagged `trajectory`. ReAct only.                              |
| `reflect_after_run`      | `bool`         | `True`  | Call `Reflector.maybe_reflect` after each `run()`.                                                          |
| `extra_instructions`     | `str \| None`  | `None`  | Free-form text appended to the system prompt — tone, output format, hard rules.                             |
| `max_substeps_per_step`  | `int`          | `3`     | `PlanAndSolveAgent` only — per-plan-step tool-call budget.                                                  |
| `max_steps`              | `int \| None`  | `None`  | Default `max_steps` for `agent.run(task)`. `None` falls back to `profile.cognitive.max_steps_per_cycle`.    |

---

## Subsystem auto-disable rules

Some toggles depend on each other. The agent silently disables dependents
when the parent is off — you don't need to flip every flag manually:

* `use_memory=False` → `save_outcome`, `save_trajectory`,
  `reflect_after_run`, and the `memory_recall` built-in tool are all forced off.
* `use_reflection=False` → `reflect_after_run` is forced off.

---

## Sync vs. async setup

`agent = ReActAgent(config)` is **synchronous**. It builds the LLM, memory,
Python-function tools, skills, reflector, compressor and logger immediately.

Two pieces need `await` and are wired lazily on the first `run()` call:

* MCP servers from `profile.tools.mcp` (each spawns a subprocess).
* `LlamaIndexRAG` indexing (when `use_rag` is on).

You don't need to do anything — the first `await agent.run(...)` finishes
the wiring before executing the task. If you'd rather force it eagerly,
call `await agent._ensure_async_setup()` yourself.

---

## Worked examples

### Minimal — chat-only agent, no memory, no tools

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

### ReAct with calculator + Tavily, memory on, RAG off

```python
config = AgentConfig(
    profile="agents/maya_rodriguez/profile.yaml",
    tools=[calculator, web_search],
    use_memory=True,
    use_rag=False,
)
agent = ReActAgent(config)
```

### PlanAndSolve with everything on, custom log dir

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

---

## Legacy keyword constructor (test fixtures)

For tests and callers that want to inject custom adapters / mocks, every
agent still accepts the original keyword shape:

```python
agent = ReActAgent(
    profile,             # AgentProfile, not AgentConfig
    llm=my_llm,
    memory=my_memory,
    tools=my_registry,
    reflector=my_reflector,
    save_outcome=False,
)
```

The two paths are interchangeable — pick whichever fits the calling site.
