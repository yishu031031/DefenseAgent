# Module 7 — Agent (Orchestration) Design

**Date:** 2026-04-25
**Status:** Draft
**Module position:** 7 of N. The top-level composition layer — every earlier module feeds in; nothing imports from this one.

> **Amendment (2026-04-25 #2): Memory contract migrated to ms-agent + mem0.**
> Module 4's `MemoryRecord` / `MemoryKind` / `ScoredMemory` / `memory.recall()` / `memory.remember()` are gone (see Module 4's amendment for the full picture). Agent-side knock-on changes:
>
> - `Agent.from_profile` now constructs `DefaultMemory.from_profile(profile)` instead of the old `Memory.from_env` / `Memory.from_profile`. The `persist_memory` kwarg is gone; mem0 always persists.
> - `Agent._recall_memories(query, top_k)` returns `list[dict]` (mem0 records) instead of `list[ScoredMemory]`. `_memory_block` formats `[memory_type] content` from the dict, not `[kind] content` from a typed dataclass.
> - `Agent._persist_outcome(task, answer, *, memory_type="outcome")` replaces the old `importance=` kwarg. Failures pass `memory_type="failure"` instead of `importance=6.0`.
> - `_handle_memory_recall` (the agent-owned tool) calls `memory.search_records(query, limit, memory_type)` — DefenseAgent's added wrapper that returns dict records with the `memory_type` filter applied. ms-agent's inherited `search()` returns `list[str]` and is reserved for ms-agent's own `run()` path.
> - `ReActAgent._persist_trajectory` writes one consolidated entry per step tagged `memory_type="trajectory"`; the old `importance` / `metadata={"trajectory": True}` shape is replaced by mem0's free-form metadata + the memory_type tag.
> - `Agent.close()` no longer calls `memory.close()` (mem0 manages its own resources).
>
> The four prior fixes (memory-as-tool, per-step trajectory consolidation, reflection in `finally`, failure outcomes) all still apply — the underlying memory system changed but the Agent's *behavioral contracts* didn't. See the next block for the original 2026-04-25 amendment.

> **Amendment (2026-04-25 #1): Memory-as-tool, trajectory persistence, reflection on every exit path, failure outcomes.**
> The original spec had three structural gaps that post-review made load-bearing:
> 1. **Memory recall was a one-shot upfront prime**, frozen at step 0. Fixed by exposing `memory_recall` as an Agent-owned tool. The LLM now sees it in `registry.spec()` every turn and can query memory mid-loop with any refined query. The upfront prime stays (primes the context); the tool gives step-5 the option to search for something the step-0 query would never have retrieved.
> 2. **Only the final answer was persisted**, so past reasoning trajectories were lost. ReAct now writes **one consolidated observation per step** (not per tool call) summarizing all `(call → result)` pairs at `importance=5.0` with `metadata={"trajectory": True, "task": ..., "tool_names": [...], "step": ...}`. Per-step (not per-call) means one embedding+write per turn regardless of how many tools the model invoked concurrently.
> 3. **Reflection only fired on success**, discarding the runs most worth reflecting on. Reflection now runs in a `finally` block on both the success path and the `AgentStepLimitError` path; reflection failures are caught and logged via `agent.reflect_failed` so they can't mask the original run outcome.
> 4. **Failure outcomes are now persisted** too. When ReAct raises `AgentStepLimitError`, it writes `"Q: <task>\nA: FAILED: exceeded max_steps=N"` at `importance=6.0` before re-raising; Plan-and-Solve does the same for its `AgentError` (bad plan) path. This gives the subsequent reflection pass a high-signal marker to reflect against.

## Infrastructure added on the base class

These helpers live on `Agent` so both `ReActAgent` and `PlanAndSolveAgent` share them:

- `_combined_tool_specs()` — returns `tools.spec()` + `[memory_recall spec]`; the LLM sees user tools first, agent built-ins after.
- `_dispatch_tool_calls(tool_calls)` — routes agent-owned calls (currently just `memory_recall`) to built-in handlers, forwards everything else to `ToolRegistry.execute`. Preserves input order via index tracking so mixed user+agent calls in one turn come back in the caller's order.
- `_handle_memory_recall(arguments)` — validates query (non-empty, string), clamps `top_k` to `[1, 20]`, renders hits as a bullet list or returns a diagnostic on empty/no-match.
- `_persist_outcome(task, answer, *, importance=5.0)` — now takes an importance kwarg so failures can bump to 6.0.
- `_run_reflection_safely()` — wraps `_maybe_reflect()` in try/except; logs via `self._log("warn", "agent.reflect_failed", ...)` instead of raising.

## Trajectory schema (metadata on observation records)

```python
{
    "trajectory": True,
    "task": "<first 120 chars of the original task>",
    "tool_names": ["search_web", "memory_recall"],   # list of tool names in this step
    "step": 0,                                        # 0-indexed step within the run
}
```

One observation per step, `kind="observation"`, `importance=5.0`. Content shape:

```
Trajectory step 0: search_web({"q": "foo"}) → <result snippet, 100 chars>; memory_recall({"query": "bar"}) → <...>
```

Args serialized via `json.dumps(default=str, sort_keys=True)` truncated to 80 chars; each result truncated to 100 chars.

**Why not promote `trajectory` to a dedicated `MemoryKind`?** Extending the `MemoryKind` Literal touches 4+ files (types, retriever's no-decay set, tests, docs). Metadata-marker approach is zero-schema-change and retrieval semantics are identical: trajectory records decay like observations (they *are* observations), importance-weighted the same, BM25-indexed the same. If trajectory becomes a first-class concept later, promotion is mechanical.

## Rationale for `trajectory_importance = 5.0` (not lower)

My first pass had `trajectory_importance = 3.0` on the theory that trajectory steps should rank below organic observations so they don't dominate recall. That defeats the whole point of persisting them — when the LLM calls `memory_recall("previous attempts at X")`, trajectory records compete against regular observations (importance 5–7) and lose on importance-weighted scoring. At 5.0 they're on equal footing and actually retrievable.

## Failure-outcome importance (6.0)

One notch above normal successful outcomes. Rationale: a failure is high-signal for reflection — a run that exhausted its step budget tells the reflector something about task difficulty or tool misuse. Marking it at 6.0 (vs 5.0 for success) means the failure shows up near the top of any reflector's subsequent memory recall without saturating.

## Purpose

Tie Modules 1–6 into a **runnable agent** with one `.run(task)` call, and package the two canonical loop shapes — **ReAct** (Yao et al. 2022) and **Plan-and-Solve** (Wang et al. 2023) — as drop-in classes.

The design constraint was explicit: *don't reinvent what the earlier modules already do*. Memory, tools, reflection, logging, and the LLM adapter are already complete. This module is glue — it should be small, clearly structured, and hold as little logic as possible.

## Scope

**In:**
- `Agent` abstract base class composing all six earlier modules (profile, llm, memory, reflector, tools, logger) with an async context-manager lifecycle and shared helpers (identity prompt rendering, memory-block formatting, outcome persistence, reflection trigger, logging wrapper, max_steps resolution).
- `ReActAgent` — interleaved reasoning + tool calls via the provider function-calling API. Terminates on a plain-text answer (no `tool_calls` in the response) or `max_steps`.
- `PlanAndSolveAgent` — three-phase loop: plan the task into numbered steps, execute each with a bounded sub-loop that may call tools, synthesize the final answer.
- `Agent.from_profile(profile, **kwargs)` — one classmethod that builds all the wiring from env + profile. Because it's defined on `Agent`, both concrete subclasses inherit it (`ReActAgent.from_profile(...)`, `PlanAndSolveAgent.from_profile(...)`).
- Shared `AgentResult` / `AgentStep` data shapes so every agent type emits the same trace format.

**Out (deferred):**
- **Streaming runs** — `agent.run_stream()` yielding step-by-step events. Straightforward to add later; not needed by the current demo surface.
- **Multi-agent orchestration** — two agents talking, or a "director" agent dispatching to others. A future Module 8.
- **Checkpointing mid-run** — saving the conversation between steps so a crashed run can resume. Memory already persists observations; resuming a partial step is a bigger design.
- **Tool-call permission gates** — manual-confirm hooks, allow-lists per agent. Trivial to bolt onto `ToolRegistry.execute` when needed.
- **LLM-parsed "thought" steps** — emitting a `thought` kind for the text portion of a tool-call response. Currently we put that text in the `content` field of the `tool_call` step; upgrading to a distinct step is a one-line change.

## Design

### Pattern: abstract base + concrete subclasses (no separate Strategy object)

Earlier I sketched a Strategy pattern with an `AgentContext` bundle. Removed because it was redundant: the Agent itself already holds `self.llm`, `self.memory`, etc. — there's no second object to inject. Flat hierarchy is cleaner:

```
Agent (ABC)
├── from_profile (classmethod, inherited)
├── shared helpers: _identity_prompt, _memory_block, _recall_memories,
│                   _persist_outcome, _maybe_reflect, _log, _resolve_max_steps,
│                   close, __aenter__, __aexit__
├── @abstractmethod async def run(task, *, max_steps=None) -> AgentResult
│
├── ReActAgent             # implements run() with a tool-call loop
└── PlanAndSolveAgent      # implements run() with plan / execute / synthesize
```

Extensibility path: subclass `Agent`, implement `run`. That's it.

### `AgentStep` / `AgentResult`

One shared step/result shape across both agents so callers can treat `ReActAgent.run` and `PlanAndSolveAgent.run` uniformly:

```python
StepKind = Literal["plan", "tool_call", "tool_result", "answer"]

@dataclass
class AgentStep:
    index: int
    kind: StepKind
    content: str
    tool_calls: list[ToolCall]      # populated on kind="tool_call"
    tool_results: list[Message]     # populated on kind="tool_result"
    usage: TokenUsage | None        # populated on LLM-call steps

@dataclass
class AgentResult:
    task: str
    final_answer: str
    steps: list[AgentStep]
    usage: TokenUsage               # aggregate across every LLM call
    stopped_reason: Literal["answered", "max_steps"]
```

Keeping `plan` as a step kind lets Plan-and-Solve emit its plan as step 0 in the same trace format.

### ReAct loop

```
for i in range(max_steps):
    resp = await llm.chat(messages, system=system_prompt, tools=registry.spec())
    if resp.tool_calls:
        messages += [assistant_msg(resp.tool_calls), *tool_results]
        continue
    # resp is a plain-text answer → finalize
    await persist_outcome(task, resp.content)
    await maybe_reflect()
    return AgentResult(...)
raise AgentStepLimitError(...)
```

Uses the provider's native function-calling interface — no ad-hoc `Thought:/Action:/Observation:` text parsing. Relies on the fact that both `AnthropicAdapter` and `OpenAICompatibleAdapter` already plumb `tools=` through and parse `tool_calls` back out of the response.

### Plan-and-Solve loop

```
# Phase 1 — plan
plan_resp = await llm.chat([prompt_build_plan(task)], system=identity+memories)
plan = parse_plan(plan_resp.content)               # ["step 1 ...", "step 2 ..."]
if not plan: raise AgentError
plan = plan[:max_steps]                             # cap by max_steps

# Phase 2 — execute each planned step
for step in plan:
    sub_result = await react_sub_loop(
        user="execute ONLY this step",
        max_substeps=max_substeps_per_step,
    )

# Phase 3 — synthesize
answer = await llm.chat([prompt_synthesize(task, plan, sub_results)], system=identity+memories)
await persist_outcome(task, answer)
await maybe_reflect()
return AgentResult(...)
```

`parse_plan` accepts both `"1. xxx"` and `"1) xxx"`, tolerates leading whitespace, ignores non-numbered lines.

The per-step sub-loop is a short ReAct cycle bounded by `max_substeps_per_step` (default 3). When a step exhausts its substep budget without a plain-text return, the synthesis phase receives a `"(step N incomplete: tool loop exceeded)"` marker — the agent doesn't hang on pathological tool-calling.

### Shared helpers on `Agent`

Both strategies reach into the same helpers so nothing is duplicated:

| Helper | What it does |
|---|---|
| `_identity_prompt()` | Render name/age/traits/backstory/initial_plan into the system-prompt identity block |
| `_memory_block(memories)` | Render a ScoredMemory list as `"- [kind] content"` bullets; empty string when empty |
| `_recall_memories(query, top_k)` | Call `self.memory.recall(query, top_k=top_k)`; returns `[]` when `top_k<=0` |
| `_persist_outcome(task, answer)` | Append `"Q: ...\nA: ..."` as an `observation` record |
| `_maybe_reflect()` | Trigger `reflector.check_and_reflect()`; no-op when no reflector is wired |
| `_log(level, event_type, message, **data)` | Emit a structured log event at the named level; no-op when no logger is wired |
| `_resolve_max_steps(override)` | Use caller's value, else `profile.cognitive.max_steps_per_cycle` |

### `from_profile`

One classmethod inherited by both concrete agents:

```python
@classmethod
async def from_profile(cls, profile, *, persist_memory=True, log_dir=None, **kwargs):
    llm      = LLM.from_env()
    memory   = Memory.from_profile(profile, persist=persist_memory)   # ← or from_env if no source_dir
    tools    = await ToolRegistry.from_profile(profile)
    reflector = Reflector(memory, llm)
    logger    = AgentLogger.from_profile(profile, log_file=...)       # at <log_dir>/<id>.log
    return cls(profile, llm=llm, memory=memory, tools=tools, reflector=reflector, logger=logger, **kwargs)
```

`**kwargs` forwards strategy-specific knobs (`memory_recall_top_k`, `persist_outcome`, `max_substeps_per_step`, etc.) to the subclass `__init__`. No per-type from_profile methods needed.

## File layout

```
DefenseAgent/agent/                    # 4 files, one concern per file
├── __init__.py                          # re-exports
├── agent.py                             # Agent (ABC) + AgentResult/AgentStep + errors + from_profile + helpers
├── react.py                             # ReActAgent
└── plan_and_solve.py                    # PlanAndSolveAgent + plan parser

tests/DefenseAgent/agent/
├── __init__.py
├── _support.py                          # ScriptedLLM, ZeroEmbedder, profile factory, resp() builder
├── test_agent.py                        # base-class contract + from_profile + lifecycle
├── test_react.py                        # 9 ReAct tests
└── test_plan_and_solve.py               # 9 Plan-and-Solve tests (incl. plan-parser)
```

Dependency graph (one-way): `agent` → everything in `config`, `llm`, `memory`, `reflection`, `tools`, `ops`. No other module imports from `agent`.

## Dependencies

**No new runtime deps.** Everything is already present.

## Integration with earlier modules

| Module | How Agent uses it | Do we modify that module? |
|---|---|---|
| Module 1 (LLM) | `self.llm.chat(messages, system=..., tools=registry.spec())` is the core primitive | No |
| Module 2 (Config) | Reads `profile.cognitive.max_steps_per_cycle`, `profile.source_dir` (for memory dir + logs dir), forwards `profile.tools` via `ToolRegistry.from_profile` | No |
| Module 3 (Ops) | Every agent step emits a `logger.info("agent.tool_call" / "agent.answer" / "agent.run.start", ...)` event | No |
| Module 4 (Memory) | `memory.recall(task, top_k=...)` at start; `memory.remember(Q+A, kind="observation")` at end | No |
| Module 5 (Reflection) | `reflector.check_and_reflect()` after each run (threshold-gated — no-op below threshold) | No |
| Module 6 (Tools) | `registry.spec()` goes to the LLM; `registry.execute(tool_calls)` produces the tool-result messages | No |

Zero retrofits. Every module's public API was already complete.

## Testing strategy

All tests stay offline via a scripted LLM stub (`ScriptedLLM` in `_support.py`). Each test constructs:

- A minimal `AgentProfile` (via `make_profile()`).
- A `ScriptedLLM` pre-loaded with the exact sequence of `LLMResponse` objects the loop should see.
- A `Memory` wired to `ZeroEmbedder` (returns `[0.0]` for every text — enough for `memory.remember` / `memory.recall` to work without network).
- A `ToolRegistry` with whatever `@registry.tool` functions the test needs.

Coverage highlights:

- **ReAct**: direct-answer path, tool-then-answer path, multi-tool-call-per-response, max_steps exhaustion (raises `AgentStepLimitError`), outcome persistence, memory recall into system prompt, `tools=None` when registry is empty, context-manager lifecycle.
- **Plan-and-Solve**: plan parser (`.`/`)` styles, whitespace, empty input), full plan → execute → synthesize flow, step count capped by `max_steps`, empty plan raises `AgentError`, substep cap produces a `"(step N incomplete)"` marker, system-prompt separation between plan/exec/synthesis phases, outcome persistence.
- **Base class**: can't instantiate `Agent` directly (abstract), `_resolve_max_steps` precedence, `close()` idempotence, `from_profile` wires everything against Maya's real agent bundle.

**25 tests total** across 3 files. All offline, no network, no subprocess.

## Execution flow

### ReAct run

```
agent.run("How's Maya's homework going?")
│
├─ system = _identity_prompt() + _memory_block(recall(task)) + REACT_INSTRUCTIONS
├─ messages = [user_task]
│
├─ loop: for i in range(max_steps)
│      │
│      ├─ resp = llm.chat(messages, system=system, tools=registry.spec())
│      │
│      ├─ if resp.tool_calls:
│      │     messages += assistant_msg(tool_calls), *tools.execute(tool_calls)
│      │     log("agent.tool_call", ...)
│      │     continue
│      │
│      └─ else (plain-text answer):
│            log("agent.answer", ...)
│            memory.remember("Q: ...\nA: ...")
│            reflector.check_and_reflect()
│            return AgentResult(final_answer=..., steps=[...], usage=...)
│
└─ if loop exits: raise AgentStepLimitError
```

### Plan-and-Solve run

```
agent.run("How's Maya's homework going?")
│
├─ Phase 1 ──────────────────────────────────────
│    plan_resp  = llm.chat([plan_prompt(task)], system=identity+memories)
│    plan_steps = parse_plan(plan_resp.content)
│    plan_steps = plan_steps[:max_steps]
│    steps.append(AgentStep(kind="plan", ...))
│
├─ Phase 2 ──────────────────────────────────────
│    for plan_step in plan_steps:
│        sub_loop (bounded by max_substeps_per_step):
│            resp = llm.chat([exec_prompt], system=identity+memories+EXEC_INSTRUCTIONS, tools=...)
│            if resp.tool_calls: execute, continue
│            else: step_answer = resp.content, break
│        step_outputs.append(step_answer or "(step N incomplete)")
│
└─ Phase 3 ──────────────────────────────────────
     synthesis = llm.chat([synthesize_prompt(task, plan+outputs)], system=identity+memories)
     memory.remember("Q: ...\nA: ..."); reflector.check_and_reflect()
     return AgentResult(final_answer=synthesis.content, steps=[...], usage=...)
```

## Open questions

- **Should `AgentResult.steps` grow during a long run, or be emitted incrementally?** Current shape collects them in memory and returns all at once. Fine at harness scale (tens of steps max). If a run ever grows to thousands of steps, move to a streaming iterator.
- **Should `persist_outcome` record the full conversation or just Q/A?** Currently Q/A only. Full-conversation persistence is useful for replay but bloats memory fast. Defer until an actual use case appears.
- **Should the Plan-and-Solve sub-loop share a single `messages` list across steps?** Right now each plan step gets a fresh conversation to keep context bounded. An optional "cumulative context" mode is a future knob if step-N needs results from step-(N-1) beyond what the synthesis prompt carries.
