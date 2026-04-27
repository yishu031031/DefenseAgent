# Module 7 Walkthrough — Agent (Orchestration)

> Companion to the [design spec](../superpowers/specs/2026-04-25-module-07-agent-design.md). The spec records **what** we decided and **why**; this walkthrough explains **how** each file composes the six earlier modules into a runnable agent.

> **Revision (2026-04-25):** The file-by-file walk below still reflects the initial build. Three post-review upgrades — memory-as-tool, per-step trajectory consolidation, reflection/failure-outcome on every exit path — are covered in **§9 Post-review upgrades** at the bottom, with deltas to the earlier sections called out inline.

---

## CORE CLASS: `Agent` (abstract), `ReActAgent`, `PlanAndSolveAgent`

Start here. The module's public surface is three classes: one base + two concrete agents:

```python
from DefenseAgent.config import AgentProfile
from DefenseAgent.agent import ReActAgent, PlanAndSolveAgent

profile = AgentProfile.from_yaml("agents/maya_rodriguez/profile.yaml")

# Classic ReAct
async with await ReActAgent.from_profile(profile) as agent:
    result = await agent.run("How's Maya's homework going?")
    print(result.final_answer)

# Plan-and-Solve (same profile, same from_profile, different loop)
async with await PlanAndSolveAgent.from_profile(profile) as agent:
    result = await agent.run("Draft Maya's weekly study plan.")
    for step in result.steps:
        print(step.kind, step.content[:60])
```

Both agents share the same `.run()` contract, the same `AgentResult`/`AgentStep` trace format, and the same `from_profile` classmethod. The difference is **only** in the loop shape inside `run()`.

---

## 1. What problem this module solves

By Module 6 we have: an LLM that can call tools, a memory layer that persists observations, a reflector that synthesizes insights, a tool registry with skills + MCP, a profile system with agent bundles. What's missing is the **control flow** that turns these into an agent you can hand a task to.

Two things specifically:

1. **One `run(task)` call**. Users shouldn't have to manually assemble a message list, read the LLM's response, dispatch tool calls, append results, recall memories, and decide when to stop. That's what Agent does.

2. **Interchangeable loop shapes**. The canonical ReAct loop and the canonical Plan-and-Solve loop are well-known patterns. Agents in a real project often switch between them depending on task complexity. Packaging both as drop-in classes means you pick via `ReActAgent` vs `PlanAndSolveAgent` at the import line.

The design constraint was: **don't rebuild what the earlier modules already do**. Memory recall, tool execution, reflection triggers, profile config — all of it lives upstream. Module 7 just wires.

---

## 2. Directory map

```
DefenseAgent/agent/                 # 4 files
├── __init__.py                      # re-exports
├── agent.py                         # Agent (ABC) + AgentResult + AgentStep + errors + from_profile + helpers   ← START HERE
├── react.py                         # ReActAgent — the tool-call loop
└── plan_and_solve.py                # PlanAndSolveAgent — plan / execute / synthesize

tests/DefenseAgent/agent/
├── __init__.py
├── _support.py                      # ScriptedLLM, ZeroEmbedder, profile factory, resp() builder
├── test_agent.py                    # base-class contract + from_profile + lifecycle (7 tests)
├── test_react.py                    # 9 ReAct tests
└── test_plan_and_solve.py           # 9 Plan-and-Solve tests (incl. plan parser)
```

Dependency direction is one-way: `agent` imports from `config`, `llm`, `memory`, `reflection`, `tools`, `ops`. Nothing imports from `agent`.

---

## 3. The three ideas that shape the design

### 3.1 Abstract base + concrete subclasses (no Strategy object)

I initially built a Strategy pattern with an `AgentContext` bundle and an `AgentStrategy` ABC. Removed it. Redundant: the Agent already holds `self.llm`, `self.memory`, etc. — there's no second object worth injecting. Flat hierarchy is cleaner:

```
Agent (ABC)                          # composes the 6 modules; declares abstract run()
├── ReActAgent                       # implements run() as a tool-call loop
└── PlanAndSolveAgent                # implements run() as plan/execute/synthesize
```

Extensibility path: subclass `Agent`, implement `run`, done. `from_profile` is inherited.

### 3.2 One shared step/result shape

Both agents return the same `AgentResult`. The step trace uses the same `AgentStep` with one of four kinds: `plan`, `tool_call`, `tool_result`, `answer`. That lets a caller write `for step in result.steps` uniformly — ReAct emits `tool_call` / `tool_result` / `answer`, Plan-and-Solve emits `plan` + same — without branching on agent type.

### 3.3 Shared helpers, not duplicated prose

Every agent needs to render the identity block, recall memories, format a memory list, persist the outcome, trigger reflection, and log a step. Those seven helpers live once on the base class. The two subclasses call into them — neither reimplements prompt assembly or memory handling.

---

## 4. File: `agent.py` — base class + data shapes

### 4.1 `AgentStep` and `AgentResult`

```python
StepKind = Literal["plan", "tool_call", "tool_result", "answer"]

@dataclass
class AgentStep:
    index: int
    kind: StepKind
    content: str = ""
    tool_calls: list[ToolCall]   = field(default_factory=list)   # populated on kind="tool_call"
    tool_results: list[Message]  = field(default_factory=list)   # populated on kind="tool_result"
    usage: TokenUsage | None     = None                          # set on LLM-call steps

@dataclass
class AgentResult:
    task: str
    final_answer: str
    steps: list[AgentStep]
    usage: TokenUsage            # aggregate across every LLM call
    stopped_reason: Literal["answered", "max_steps"] = "answered"
```

Both strategies populate the same dataclass. The `usage` on each step is the per-call usage; the `usage` on `AgentResult` is the sum, kept via the module-level `add_usage(a, b)` helper.

### 4.2 `Agent.__init__`

```python
class Agent(ABC):
    def __init__(self, profile, *, llm, memory, tools, reflector=None, logger=None):
        self.profile = profile
        self.llm = llm
        self.memory = memory
        self.tools = tools
        self.reflector = reflector
        self.logger = logger
```

Minimal — the subclasses add their own strategy knobs (`memory_recall_top_k`, `save_outcome`, etc.) via `super().__init__(...)`.

### 4.3 `Agent.from_profile` — one classmethod, inherited by both concrete agents

```python
@classmethod
async def from_profile(cls, profile, *, persist_memory=True, log_dir=None,
                       dotenv_path=None, load_env=True, **kwargs):
    llm = LLM.from_env(dotenv_path=dotenv_path, load_env=load_env)
    if persist_memory and profile.source_dir is not None:
        memory = Memory.from_profile(profile, ...)           # SQLite at agents/<id>/memory/stream.db
    else:
        memory = Memory.from_env(profile, ...)               # RAM-only
    tools    = await ToolRegistry.from_profile(profile)      # loads skills + MCP from profile.tools
    reflector = Reflector(memory, llm)
    logger   = _build_logger(profile, log_dir)               # writes to agents/<id>/logs/<id>.log
    return cls(profile, llm=llm, memory=memory, tools=tools,
               reflector=reflector, logger=logger, **kwargs)
```

Because it's defined on `Agent`, both `ReActAgent.from_profile(...)` and `PlanAndSolveAgent.from_profile(...)` work — `cls` dispatches to the right subclass. `**kwargs` forwards strategy knobs.

### 4.4 Shared helpers

```python
def _identity_prompt(self):
    return f"You are {p.name}, a {p.age}-year-old.\nTraits: ...\nBackstory: ...\nToday's plan: ..."

def _memory_block(self, memories):
    return "" if not memories else "Relevant memories:\n" + "\n".join(
        f"- [{m.record.kind}] {m.record.content}" for m in memories
    )

async def _recall_memories(self, query, top_k):
    return [] if top_k <= 0 else await self.memory.recall(query, top_k=top_k)

async def _save_outcome(self, task, answer):
    await self.memory.remember(f"Q: {task}\nA: {answer}", kind="observation")

async def _maybe_reflect(self):
    if self.reflector is not None:
        await self.reflector.maybe_reflect()

def _log(self, level, event_type, message, **data):
    if self.logger is None: return
    getattr(self.logger, level)(event_type, message, **data)

def _resolve_max_steps(self, override):
    return override if override is not None else self.profile.cognitive.max_steps_per_cycle
```

Each helper encapsulates one reusable operation. Neither subclass duplicates prompt assembly or memory plumbing.

### 4.5 Lifecycle

```python
async def close(self):
    await self.tools.close()         # closes MCP clients
    self.memory.close()              # closes SQLite if persistent

async def __aenter__(self): return self
async def __aexit__(self, *exc): await self.close()
```

`async with await ReActAgent.from_profile(profile) as agent:` is the recommended shape — guarantees MCP subprocesses and the SQLite handle close on exit.

---

## 5. File: `react.py` — ReActAgent

### 5.1 The loop

```python
async def run(self, task, *, max_steps=None):
    cap = self._resolve_max_steps(max_steps)
    system_prompt = await self._build_system_prompt(task)    # identity + memories + ReAct instructions
    messages = [Message(role="user", content=task)]
    steps, total = [], TokenUsage(0, 0, 0)
    tool_specs = self.tools.spec() or None

    for i in range(cap):
        response = await self.llm.chat(messages, system=system_prompt, tools=tool_specs)
        total = add_usage(total, response.usage)

        if response.tool_calls:
            messages.append(assistant_msg(response))
            steps.append(AgentStep(kind="tool_call", ...))
            tool_results = await self.tools.execute(response.tool_calls)
            messages.extend(tool_results)
            steps.append(AgentStep(kind="tool_result", tool_results=tool_results))
            continue

        # plain-text answer → finalize
        steps.append(AgentStep(kind="answer", content=response.content, usage=response.usage))
        if self.save_outcome:   await self._save_outcome(task, response.content)
        if self.reflect_after_run: await self._maybe_reflect()
        return AgentResult(task, response.content, steps, total)

    raise AgentStepLimitError(f"ReAct exceeded max_steps={cap}")
```

Uses the provider's native function-calling — no `Thought:/Action:/Observation:` text parsing. Works against both `AnthropicAdapter` and `OpenAICompatibleAdapter` because both plumb `tools=` through and parse `tool_calls` back from the response.

### 5.2 System prompt assembly

```python
async def _build_system_prompt(self, task):
    memories = await self._recall_memories(task, self.memory_recall_top_k)
    parts = [self._identity_prompt()]
    memory_block = self._memory_block(memories)
    if memory_block:
        parts.append(memory_block)
    parts.append(_REACT_INSTRUCTIONS)
    if self.extra_instructions:
        parts.append(self.extra_instructions)
    return "\n\n".join(parts)
```

Four blocks joined with blank lines: identity → relevant memories (if any) → "call tools, answer in plain text when done" → user extras. Memory recall happens once per run, not once per step.

### 5.3 Knobs

Constructor args the subclass adds on top of the base wiring:

| Knob | Default | Purpose |
|---|---|---|
| `memory_recall_top_k` | 5 | How many memories to inject into the system prompt. 0 disables recall. |
| `save_outcome` | True | Whether to `memory.remember("Q: ...\nA: ...")` after a successful run. |
| `reflect_after_run` | True | Whether to call `reflector.maybe_reflect()` after a successful run. |
| `extra_instructions` | None | Appended to the system prompt — useful for task-specific framing without subclassing. |

---

## 6. File: `plan_and_solve.py` — PlanAndSolveAgent

### 6.1 Three phases

```python
async def run(self, task, *, max_steps=None):
    cap = self._resolve_max_steps(max_steps)
    memories = await self._recall_memories(task, self.memory_recall_top_k)
    memory_block = self._memory_block(memories)
    identity = self._identity_prompt()

    # Phase 1 — plan
    plan_system = _join_blocks(identity, memory_block)
    plan_resp = await self.llm.chat([user(PLAN_PROMPT.format(task=task))], system=plan_system)
    plan = _parse_plan(plan_resp.content)
    if not plan: raise AgentError(...)
    plan = plan[:cap]
    steps.append(AgentStep(kind="plan", content="\n".join(plan)))

    # Phase 2 — execute each step in a short sub-loop
    exec_system = _join_blocks(identity, memory_block, EXEC_INSTRUCTIONS)
    step_outputs = []
    for plan_step in plan:
        step_answer, step_usage = await self._execute_plan_step(
            plan_step, task, exec_system, tool_specs, ..., all_steps=steps,
        )
        step_outputs.append(step_answer)

    # Phase 3 — synthesize
    synthesis = await self.llm.chat(
        [user(SYNTHESIS_PROMPT.format(task=task, plan_with_results=...))],
        system=plan_system,
    )
    steps.append(AgentStep(kind="answer", content=synthesis.content))

    if self.save_outcome:   await self._save_outcome(task, synthesis.content)
    if self.reflect_after_run: await self._maybe_reflect()
    return AgentResult(task, synthesis.content, steps, total)
```

### 6.2 Plan parser

```python
_STEP_LINE_RE = re.compile(r"^\s*\d+[\.)]\s*(.+?)\s*$")

def _parse_plan(text):
    out = []
    for line in text.splitlines():
        m = _STEP_LINE_RE.match(line)
        if m and m.group(1).strip():
            out.append(m.group(1).strip())
    return out
```

Accepts both `"1. step"` and `"1) step"`, tolerates leading whitespace, ignores non-numbered lines. Empty plan → the agent raises `AgentError` rather than guessing.

### 6.3 Per-step sub-loop

```python
async def _execute_plan_step(self, *, plan_step, original_task, exec_system, tool_specs,
                             step_index, all_steps):
    messages = [user(f"Original task: {original_task}\nExecute ONLY this step: {plan_step}")]
    sub_total = TokenUsage(0, 0, 0)

    for _ in range(self.max_substeps_per_step):       # default 3
        response = await self.llm.chat(messages, system=exec_system, tools=tool_specs)
        sub_total = add_usage(sub_total, response.usage)
        if response.tool_calls:
            messages.append(assistant_msg(response))
            all_steps.append(AgentStep(kind="tool_call", ...))
            messages.extend(await self.tools.execute(response.tool_calls))
            all_steps.append(AgentStep(kind="tool_result", ...))
            continue
        return response.content, sub_total

    # Budget exhausted — feed synthesis a diagnostic rather than hanging.
    return f"(step {step_index} incomplete: tool loop exceeded)", sub_total
```

The sub-loop is essentially a tiny ReAct cycle. Each planned step gets a **fresh conversation** — this keeps the context window bounded as the plan grows. If step-N needs results from step-(N-1), the synthesis prompt carries the whole plan-with-outputs table.

### 6.4 Knobs

| Knob | Default | Purpose |
|---|---|---|
| `memory_recall_top_k` | 5 | How many memories to inject at plan + synthesis time. |
| `max_substeps_per_step` | 3 | LLM-call budget per planned step. |
| `save_outcome` | True | Same as ReAct. |
| `reflect_after_run` | True | Same as ReAct. |

---

## 7. Test coverage map

| File | Tests | Focus |
|---|---|---|
| `test_agent.py` | 7 | `Agent` is abstract, `_resolve_max_steps` precedence, `close()` idempotence, async-context-manager semantics, `from_profile` wires everything for both subclass types against Maya's real bundle |
| `test_react.py` | 9 | Direct-answer path, tool-then-answer, multi-tool-per-response, max_steps exhaustion, outcome persistence, memory recall → system prompt, `tools=None` when empty, context manager, system prompt contents |
| `test_plan_and_solve.py` | 9 | Plan parser (3 tests: dot/paren styles, ignore non-numbered, empty input), end-to-end plan → exec → synthesize, plan capped by max_steps, empty plan raises, substep cap yields incomplete marker, prompt separation across phases, outcome persistence |

**25 tests. All offline.** The `ScriptedLLM` stub in `_support.py` is the secret — a per-test list of `LLMResponse` objects that the loop consumes one at a time, letting every branch be hand-scripted without touching a real LLM.

---

## 8. Things worth noticing

- **The loops are small.** ReAct's `run()` is ~50 lines of real logic. Plan-and-Solve's is ~80. The modules they compose do everything heavier.

- **Shared helpers prevent drift.** Both agents build the identity block, render memory, persist the outcome, and trigger reflection through the same five base-class methods. Change the memory-block format once, both agents update.

- **`from_profile` is one method.** Because it's defined on `Agent`, not on each subclass, new agent types inherit it automatically. Just subclass, implement `run`, and `from_profile` already works.

- **The trace format is uniform.** Both agents emit `AgentStep` objects with the same four kinds (`plan`, `tool_call`, `tool_result`, `answer`). A UI or logger can render either agent's trace with one code path.

- **Memory/tools/reflection/logger are all optional.** `Agent.__init__` only requires `profile`, `llm`, `memory`, `tools`. Reflector and logger default to `None` and every helper that uses them no-ops on `None`. Tests build agents with a reflector-less, logger-less setup to keep each case tight.

- **No streaming yet.** `run()` returns after the full trace completes. Streaming is a straightforward follow-up — add an `async def run_stream()` yielding `AgentStep` objects as they happen. Not needed for the current surface.

---

## 9. Post-review upgrades (2026-04-25)

Three behavioral gaps surfaced in review and are now fixed. Each maps to a concrete change in the code you'd read.

### 9.1 Memory-as-tool — live access, not one-shot prime

**Before:** `memory.recall(task)` ran once at the start; the recalled bullet list went into the system prompt and never changed. Step-5 searching for hotels couldn't find "user prefers Marriott in 国贸" because the query that retrieved memories was step-0's "plan a trip."

**Now:** The Agent exposes a built-in tool named `memory_recall` alongside the user's `ToolRegistry`. The LLM sees it in every `registry.specs()` payload and calls it with whatever query the current step needs.

Three helpers on `Agent`:

```python
def _combined_tool_specs(self):
    user_specs    = self.tools.spec()
    builtin_specs = [_MEMORY_RECALL_TOOL_SPEC]      # always included
    return (user_specs + builtin_specs) or None

async def _dispatch_tool_calls(self, tool_calls):
    # Route agent-owned calls to handlers; forward everything else to self.tools.execute.
    # Preserves input order via index tracking.
    ...

async def _handle_memory_recall(self, arguments):
    # Validate query (str, non-empty), clamp top_k to [1, 20], render hits as bullets.
    # Returns "(no memories matched query=...)" on miss.
    ...
```

The upfront prime stays — `_build_system_prompt` still recalls `memory_recall_top_k` records and injects them into the system prompt as before. It's just no longer the only shot. Step-N can go ask memory something step-0 didn't know to ask.

### 9.2 Per-step trajectory persistence (not per call)

**Before:** `save_outcome=True` recorded only the final answer. Intermediate tool calls + their results were lost, so future runs couldn't retrieve past approaches.

**First cut (per-call):** one memory write per `(tool_call, tool_result)` pair. Rejected — a single LLM turn with 3 concurrent tool calls triggered 3 embedding-API calls and 3 DB writes.

**Now (per-step):** ONE observation per step summarizing every call in that step:

```python
async def _save_trajectory(self, *, task, step_index, tool_calls, tool_results):
    pair_parts = []
    for tc, tr in zip(tool_calls, tool_results):
        pair_parts.append(
            f"{tc.name}({_preview_json(tc.arguments)}) → {truncate(tr.content or '', 100)}"
        )
    calls_summary = "; ".join(pair_parts)

    await self.memory.remember(
        f"Trajectory step {step_index}: {calls_summary}",
        kind="observation",
        importance=self.trajectory_importance,          # default 5.0
        metadata={
            "trajectory": True,
            "task": truncate(task, 120),
            "tool_names": [tc.name for tc in tool_calls],   # list, not singular
            "step": step_index,
        },
    )
```

Content shape (one line per step):

```
Trajectory step 0: search_web({"q": "hotels"}) → <200-char snippet>; memory_recall({"query": "Marriott"}) → <100-char snippet>
```

One embedding + one DB write per step regardless of tool-call fan-out. Metadata carries a list of tool names so later retrieval can filter. Args are serialized compactly via `json.dumps(default=str, sort_keys=True)` then truncated to 80 chars; each result truncated to 100 chars.

**Importance bump 3.0 → 5.0.** The first pass used 3.0 on the theory trajectory shouldn't dominate. That defeats the purpose — when the LLM asks `memory_recall("previous attempts at X")`, trajectory records were losing on importance-weighted scoring against organic observations (5–7). At 5.0 they're on equal footing and actually retrievable.

### 9.3 Reflection on every exit path + failure outcome persistence

**Before:** `_maybe_reflect()` sat inside the success branch. A run that exhausted `max_steps` raised `AgentStepLimitError` and reflection never fired — exactly the runs most worth reflecting on.

**Now:** Reflection lives in a `finally` block. Both subclasses follow the same structure:

```python
async def run(self, task, *, max_steps=None):
    ...
    try:
        # main loop / phases
        ...
        # success path:
        if self.save_outcome:
            await self._save_outcome(task, answer)    # importance=5.0 default
        return AgentResult(...)
    except AgentStepLimitError:                          # ReAct
        if self.save_outcome:
            await self._save_outcome(
                task,
                f"FAILED: exceeded max_steps={cap}",
                importance=6.0,
            )
        raise
    except AgentError as e:                              # Plan-and-Solve
        if self.save_outcome:
            await self._save_outcome(
                task,
                f"FAILED: {truncate(str(e), 200)}",
                importance=6.0,
            )
        raise
    finally:
        if self.reflect_after_run:
            await self._run_reflection_safely()          # never raises — just logs
```

Two specific guarantees:

- **Reflection runs on both success and failure.** `finally` covers every exit path, including propagated exceptions.
- **Reflection failures don't mask the run outcome.** `_run_reflection_safely` catches anything the reflector throws and emits `agent.reflect_failed` via the logger. A good run stays good even if reflection crashed; a failed run still raises the original `AgentStepLimitError`, not the reflector's `RuntimeError`.

**Failure outcome importance is 6.0 (not 5.0).** One notch above successful outcomes. A failure is high-signal for reflection — the reflector's subsequent memory recall surfaces it prominently without saturating.

### 9.4 Tests for the three upgrades

`tests/DefenseAgent/agent/` gained two new files covering the deltas:

| File | Tests | Focus |
|---|---|---|
| `test_memory_recall_tool.py` | 7 | Tool appears in combined spec, user tools first then built-ins, end-to-end invocation with hits, empty-match diagnostic, empty-query handling, top_k clamping, order-preservation across mixed user+agent tool calls |
| `test_react_trajectory_and_reflection.py` | 10 | One-record-per-step, multi-call consolidation, `tool_names` metadata list, trajectory_importance defaults to 5.0, failure outcome persisted on `AgentStepLimitError` with `FAILED:` prefix + importance 6.0, save_outcome=False disables both outcome writes, reflection fires on success, on max_steps, failure survives a raising reflector on both paths, reflect_after_run=False skips reflection everywhere |

Plus `test_plan_and_solve.py` gained two tests for the PS-side failure-outcome path (bad plan → FAILED outcome + importance 6.0, and `save_outcome=False` suppresses it).

**All offline via the ScriptedLLM stub.** Full suite: **443 passed**.
