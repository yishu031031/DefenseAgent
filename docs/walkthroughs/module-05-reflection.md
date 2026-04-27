# Module 5 Walkthrough — Reflection

> Companion to the [design spec](../superpowers/specs/2026-04-22-module-05-reflection-design.md). Explains the code line-by-line and traces the execution of `scripts/reflection_demo.py`.

---

## CORE CLASS: `Reflector`

Start here. The module is one class in one file:

```python
from DefenseAgent.reflection import Reflector

reflector = Reflector(memory, llm, num_insights=3, reflection_importance=8.0)

# Rate how poignant an observation is (Park §3.2.1)
score = await reflector.score_importance("Got stuck on problem 3 for an hour.")

# Synthesize higher-level insights (Park §3.2.2)
insights = await reflector.maybe_reflect()  # no-op if below threshold
forced   = await reflector.reflect_now()        # force it, returns records
```

`Reflector` reads from `Memory` and writes back `kind="reflection"` records. It also exposes an LLM-based importance scorer so callers can write `await memory.remember(content, importance=await reflector.score_importance(content))`.

This is the **first module that touches both Module 1 (LLM) and Module 4 (Memory)**. It's the feedback loop: memory stores, reflector thinks above it, the thoughts go back into memory.

---

## 1. What problem this module solves

Module 4 is a passive memory layer. It records and retrieves but never *thinks*. An agent that only records observations would, after a week of use, have a long flat list of events with no sense of pattern — "I had coffee," "I had coffee again," "I had coffee with Chloe," "I had coffee alone."

The insight from Park et al. 2023 is that **memory quality compounds when the agent periodically reflects over its recent experiences** and writes those reflections back. Reflections are higher-importance, they're recalled as context for future decisions, and they let the agent notice things about itself that no single observation reveals.

**Module 5 gives the harness:**

1. **LLM-based importance scoring** — rate how poignant an observation is on a 1–10 scale. Caller uses the score when calling `memory.remember(importance=score)`.
2. **Reflection synthesis** — read recent non-reflection memories, prompt the LLM for `N` high-level insights, store each as a new `kind="reflection"` record with configurable importance.
3. **Count-based trigger logic** — `maybe_reflect()` is a no-op until unreflected observations reach `profile.cognitive.reflection_threshold`.

---

## 2. Directory map

```
DefenseAgent/reflection/
├── __init__.py              # CORE CLASS header + Reflector re-export
└── reflection.py            # everything: Reflector + prompts + parsers (~180 lines)

tests/DefenseAgent/reflection/
├── __init__.py
└── test_reflection.py       # 33 tests, 5 behavior groups

scripts/
└── reflection_demo.py       # Maya observes 6 things, scores them, reflects, retrieves
```

One file. The user's directive was "strive for conciseness: aim for a single core class file, and only create additional files if you need to implement extra functionality that requires a significant amount of code" — and at ~180 lines, the whole module fits comfortably.

---

## 3. Anatomy of a reflection cycle

```
                    stream of observations
                           │
                           ▼
               ┌───────────────────────────┐
               │  reflector.check_and_     │
               │  reflect()                │
               └───────────────────────────┘
                           │
          unreflected_count < threshold?
               ┌───────────┴───────────┐
           yes │                       │ no
               ▼                       ▼
          return []             reflect_now():
                                ┌──────────────────────────┐
                                │ 1. gather recent non-    │
                                │    reflection records    │
                                │ 2. format prompt         │
                                │ 3. call LLM              │
                                │ 4. parse N insights      │
                                │ 5. memory.remember(      │
                                │      insight,            │
                                │      kind="reflection",  │
                                │      importance=8.0)     │
                                │ 6. advance cutoff        │
                                └──────────────────────────┘
                                           │
                                           ▼
                                    new memories in stream,
                                    retrievable like any other kind
```

Two knobs on the `Reflector` constructor control the behavior:

- `num_insights` — how many reflections to request from the LLM per cycle. Default 3 (Park's value).
- `reflection_importance` — the importance value assigned to each new reflection record. Default 8.0 (reflections are higher-importance than raw observations by design).

---

## 4. Code walk-through: `reflection.py`

### 4.1 Prompt templates

Two module-level constants, kept readable by pulling them out of the class:

```python
_IMPORTANCE_PROMPT = """\
On a scale of 1 to 10, where 1 is purely mundane (e.g., brushing teeth,
making the bed) and 10 is extremely poignant (e.g., a breakup, a college
acceptance), rate the likely poignancy of the following memory.

Memory: {content}

Respond with a single integer between 1 and 10. No explanation."""
```

This is almost verbatim from the Park paper, with our explicit "integer between 1 and 10, no explanation" suffix to make parsing trivial.

```python
_REFLECTION_PROMPT = """\
Recent memories (chronological):
{memory_list}

Given the memories above, produce exactly {n} high-level insights about
patterns, lessons, or deeper observations. Each insight should be a
single clear sentence. Return one per line. No numbering, no bullets,
no empty lines."""
```

Single-step (not Park's two-step question-then-answer). Simpler, one API call instead of two. If reflection quality suffers in real use, we can upgrade to two-step later.

### 4.2 Parsers — intentionally tolerant

Both parsers are defensive about LLM output. LLMs don't always follow instructions to the letter.

```python
_INT_RE = re.compile(r"\d+")
_BULLET_PREFIX_RE = re.compile(r"^\s*(?:\d+[\.)]\s*|[-*•]\s*)")


def _parse_importance_response(text: str) -> float:
    m = _INT_RE.search(text or "")
    if m is None:
        return 5.0                  # default on unparseable
    value = int(m.group())
    return float(max(1, min(10, value)))    # clip to [1, 10]
```

**Three things to notice:**
- Finds the *first* integer anywhere in the response (`"I'd rate this an 8."` → 8).
- Clips to `[1, 10]` — if the model returns `42` it becomes `10`, not a range violation.
- Returns `5.0` (the middle) on parse failure — never raises. A broken importance score shouldn't crash a recording pipeline that has hundreds of things to process.

```python
def _parse_reflection_response(text: str, n: int) -> list[str]:
    if not text:
        return []
    insights = []
    for raw_line in text.splitlines():
        line = _BULLET_PREFIX_RE.sub("", raw_line).strip()
        if line:
            insights.append(line)
    return insights[:n]
```

Handles the three common formats LLMs produce even when asked for plain lines:
- `"1. First insight.\n2. Second insight."` — numbered
- `"- alpha\n* beta\n• gamma"` — bulleted (three bullet styles)
- `"Insight one.\n\nInsight two."` — clean lines with accidental blanks

All reduce to `["clean text 1", "clean text 2", ...]`. Take the first `n`, drop the rest.

### 4.3 `Reflector` — the core class

#### Constructor

```python
def __init__(
    self,
    memory: Memory,
    llm: LLM,
    *,
    num_insights: int = 3,
    reflection_importance: float = 8.0,
    clock: Callable[[], datetime] | None = None,
) -> None:
    self.memory = memory
    self.llm = llm
    self._num_insights = num_insights
    self._reflection_importance = reflection_importance
    self._clock = clock or _default_clock
    self._last_reflection_time: datetime | None = None
```

The only state `Reflector` owns is `_last_reflection_time`. Everything else is pass-through to Memory or LLM. **Bring-your-own-memory, bring-your-own-LLM** — Reflector doesn't care how either was built.

#### `score_importance()` — Park §3.2.1

```python
async def score_importance(self, content: str) -> float:
    resp = await self.llm.chat(
        [Message(role="user", content=_IMPORTANCE_PROMPT.format(content=content))],
        temperature=0.0,        # deterministic — we want the model's best guess
        max_tokens=16,          # one integer doesn't need more
    )
    return _parse_importance_response(resp.content)
```

**Why `temperature=0.0`:** consistency matters more than creativity here. If the same content gets scored twice, we'd prefer the same answer.

**Why `max_tokens=16`:** the prompt asks for a single integer. 16 tokens is generous — if the model is going to drift into explanation, it'll waste tokens without adding value. Keep the budget tight.

#### `unreflected_count` — the trigger's state

```python
@property
def unreflected_count(self) -> int:
    return len(self._recent_unreflected())

def _recent_unreflected(self) -> list[MemoryRecord]:
    cutoff = self._last_reflection_time
    return [
        r for r in self.memory.stream.get_all()
        if r.kind != "reflection"
        and (cutoff is None or r.timestamp > cutoff)
    ]
```

Two filters:

1. **Exclude reflections.** Prevents infinite regress — the act of storing a reflection shouldn't bump the counter enough to trigger another.
2. **Only count records after the cutoff.** `_last_reflection_time` starts as `None` (no reflection yet), so everything counts. After a reflection, the cutoff advances to "now"; only records with strictly later timestamps count.

#### `maybe_reflect()` — the soft trigger

```python
async def maybe_reflect(self) -> list[MemoryRecord]:
    threshold = self.memory.profile.cognitive.reflection_threshold
    if self.unreflected_count < threshold:
        return []
    return await self.reflect_now()
```

Three lines. Reads the threshold from the profile (`CognitiveConfig.reflection_threshold`, default 5). Returns `[]` silently when there's nothing to do — the future `Agent` class can call this after every `observe()` without worrying.

#### `reflect_now()` — Park §3.2.2

```python
async def reflect_now(self) -> list[MemoryRecord]:
    recent = self._recent_unreflected()
    if not recent:
        return []

    prompt = _REFLECTION_PROMPT.format(
        memory_list=self._format_memories_for_prompt(recent),
        n=self._num_insights,
    )
    resp = await self.llm.chat(
        [Message(role="user", content=prompt)],
        temperature=0.5,        # some variability is good for synthesis
        max_tokens=512,
    )
    insights = _parse_reflection_response(resp.content, self._num_insights)

    stored: list[MemoryRecord] = []
    for insight in insights:
        record = await self.memory.remember(
            insight,
            kind="reflection",
            importance=self._reflection_importance,
        )
        stored.append(record)

    self._last_reflection_time = self._clock()     # advance cutoff
    return stored
```

**Two subtle design points:**

1. **`self._last_reflection_time` advances *after* the `remember()` calls, not before.** The new reflection records get timestamps before the cutoff advances, which means the cutoff update correctly excludes them from the NEXT cycle (their timestamps will be `<= cutoff`, the filter requires `> cutoff`).

2. **Cutoff advances even on empty output.** If the LLM returns garbage and parsing yields 0 insights, `stored` is `[]` but `_last_reflection_time` still advances. Otherwise, every subsequent `maybe_reflect()` call would re-attempt the same batch, re-call the LLM, re-get garbage, forever. Silent failure is acceptable; a retry storm isn't.

#### `_format_memories_for_prompt()`

```python
@staticmethod
def _format_memories_for_prompt(records: list[MemoryRecord]) -> str:
    chronological = sorted(records, key=lambda r: r.timestamp)
    return "\n".join(
        f"- [{r.kind}, imp={r.importance:.1f}] {r.content}"
        for r in chronological
    )
```

Chronological order so the LLM sees the flow of time. Each memory tagged with `kind` and `importance` so the model has context beyond raw text.

---

## 5. Temperature + token budget — why the two calls differ

| Call | Temperature | Max tokens | Rationale |
|---|---|---|---|
| `score_importance` | 0.0 | 16 | Deterministic integer; no creativity needed |
| `reflect_now` (synthesis) | 0.5 | 512 | Some variability helps; insights need room to breathe |

The reflection call uses **moderate temperature, not zero** — different runs over similar observations should be able to surface different patterns. But not high-temperature: we want faithful reflection, not creative writing.

---

## 6. Execution flow: `scripts/reflection_demo.py`

Maya's day:

```
$ python scripts/reflection_demo.py

┌─ main() (async)
│
├─ Step 1: AgentProfile.from_yaml("agents/maya_rodriguez/profile.yaml")
│          profile.cognitive.reflection_threshold = 4
│
├─ Step 2: AgentLogger.from_profile(...)   # records every step
│
├─ Step 3: LLM.from_env()                  # real DeepSeek adapter
│          Memory.from_env(profile)        # real Qwen embedder
│
├─ Step 4: Reflector(memory, llm, num_insights=3, reflection_importance=8.5)
│
├─ Step 5: for each of 6 observations:
│            score     = await reflector.score_importance(obs)   ← LLM call
│            memory.remember(obs, kind="observation", importance=score,
│                            timestamp=back-dated)
│            logger.info("memory.observed", importance=score, ...)
│
│          Example output:
│            imp= 6.0  (−9h)  Attended the 9 AM data structures lecture...
│            imp= 3.0  (−10h) Grabbed coffee with my roommate before class.
│            imp= 7.0  (−4h)  Spent two hours in the library ...
│            imp= 8.0  (−2h)  Got stuck on problem 3 for an hour...
│            imp= 7.0  (−1h)  Met the study group at 5 PM and walked them through problem 3.
│            imp= 9.0  (−0h)  Realized I learn faster when I struggle alone first.
│
├─ Step 6: unreflected_count = 6 >= threshold (4), so:
│          new = await reflector.reflect_now()    ← one more LLM call
│
│          Prints each reflection:
│            • Maya learns algorithms deepest by first struggling independently
│              before seeking help from peers or TAs.
│            • The library environment, for Maya, operates as a productivity
│              ritual that pairs with fellow students into study accountability.
│            • Teaching peers (problem 3 walkthrough) consolidates Maya's own
│              understanding — she benefits twice from one struggle.
│
├─ Step 7: query = "what patterns are emerging in how Maya studies?"
│          results = await memory.recall(query, top_k=5)
│          Prints results with a ★ marker on reflection-kind hits.
│          Reflections now surface at the TOP of retrieval because:
│            • importance 8.5 → importance_score 0.85
│            • recency ~ 1.0 (just-added)
│            • relevance high (semantically aligned with "patterns" query)
│
└─ Report log file line count.
```

### What could fail and where

| Situation | Surface |
|---|---|
| `EMBEDDING_API_KEY` blank | Script exits with code 2 and a clear message |
| DeepSeek rate limit during `score_importance` | `LLMProviderError` propagates; script exits 1 |
| DeepSeek returns malformed integer | `_parse_importance_response` returns 5.0 silently |
| DeepSeek returns fewer than 3 insights | `_parse_reflection_response` returns what it got; `reflect_now` stores just that many; cutoff still advances |
| DeepSeek returns zero insights | `reflect_now` returns `[]`; cutoff advances; no infinite retry |

---

## 7. Test coverage map

| Section | Tests | Covers |
|---|---|---|
| `_parse_importance_response` | 7 | plain int, int-in-sentence, upper/lower clip, unparseable, empty, returns float |
| `_parse_reflection_response` | 7 | clean lines, numbered prefix, bulleted prefix (3 styles), empty-line drop, `n` cap, empty input, whitespace-only |
| `score_importance` | 4 | LLM integration, default on garbage, `LLMProviderError` propagation, uses temperature=0 |
| `unreflected_count` | 4 | empty stream, ignores reflections, resets after a reflection, timestamp cutoff honored |
| `reflect_now` | 6 | parses + stores N insights, tolerates bullets, empty-on-no-recent, advances cutoff on empty response, doesn't double-count own reflections, configured importance honored |
| `maybe_reflect` | 4 | below threshold no-op, at-threshold triggers, above-threshold triggers, noop until fresh observations |
| Integration | 1 | Reflections retrievable via `memory.recall()` alongside observations |

**33 tests total.** All offline. Stub `_StubLLMAdapter` returns canned responses from a queue; stub `_StubEmbedder` assigns deterministic vectors per content.

---

## 8. Things worth noticing

- **The trigger state is one `datetime`.** No counters, no flags, no "dirty" markers. Just `_last_reflection_time`. The unreflected count is computed on demand from `stream.get_all()` — always correct, no stale state.

- **Reflections are stored exactly like any other memory.** `kind="reflection"`, `importance=8.5`, through the same `memory.remember()`. That means they get embedded, dedup-checked, BM25-indexed, and retrieval-ranked by the same pipeline as observations. The retriever treats them identically (same recency-decay rule as observations — unlike facts/preferences).

- **No new error class.** The only way reflection can "fail" in production is `LLMProviderError` propagating from the LLM call. Parsing failures are silent + recoverable. Adding a `ReflectionError` hierarchy would be dead abstraction.

- **The cutoff-advancing-on-empty-response rule prevents retry storms.** This is the one non-obvious invariant: a reflection that produces zero insights *still* advances the clock. Without this, every subsequent `maybe_reflect()` would find the same unreflected batch, call the LLM, get garbage, and repeat — a bill-builder in production. The test `test_reflect_now_advances_cutoff_even_on_empty_response` locks this in.

- **Bring-your-own-Memory, bring-your-own-LLM.** `Reflector.__init__` takes `memory: Memory` and `llm: LLM` with no opinion about how either was constructed. This makes the class trivially testable and trivially composable — the future `Agent` class just builds one and injects.

- **Single file, 180 lines.** This was deliberately scoped — the user said "aim for a single core class file." Everything the reflection module owns fits. If we later add a two-step question-then-answer prompt, a cross-encoder reranker, or background scheduling, they'll fit in the same file until they don't.
