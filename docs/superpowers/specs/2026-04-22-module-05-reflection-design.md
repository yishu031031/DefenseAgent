# Module 5 — Reflection Design

**Date:** 2026-04-22
**Status:** Draft, ready to implement
**Module position:** 5 of N. The first module that brings Module 1 (LLM) and Module 4 (Memory) together — it reads memories, asks the LLM to think above them, and writes new memories back.

## Purpose

Module 4 gave the harness a passive memory layer — it records, retrieves, but never *thinks*. Module 5 adds the missing feedback loop from the Park et al. 2023 Generative Agents paper:

1. As the agent accumulates observations, importance piles up unexamined.
2. When enough importance accumulates, prompt the LLM with those memories and ask: *"what high-level insights emerge?"*
3. Store each insight as a new `reflection`-kind memory in the same stream, with higher importance than raw observations.

Reflections now participate in retrieval like any other kind-of-record, so future queries can surface synthesized insights alongside raw events.

**A secondary capability lives here too**: LLM-based **importance scoring** — a thin wrapper around `LLM.chat()` that asks the model to rate how poignant an observation is on a 1–10 scale. We deferred this from Module 4 because it needed the LLM; this module is where both memory-touching LLM wrappers belong.

## Scope

### In scope
- One core class: `Reflector`.
- Two LLM-backed capabilities:
  - `score_importance(content) -> float` — rate a single observation 1–10.
  - `reflect_now()` / `check_and_reflect()` — synthesize insights over recent memories; store each as a new record with `kind="reflection"`.
- Trigger logic: count-based. When the number of non-reflection records added since the last reflection crosses `profile.cognitive.reflection_threshold`, `check_and_reflect()` runs a reflection; otherwise it's a no-op.
- Response parsing for both LLM calls (tolerant, never crashes).
- Full test coverage (offline, stub LLM + stub embedder).
- One demo script that hits a real LLM.

### Out of scope
- Automatic triggering on every `memory.remember()` — the `Agent` class (future) will call `check_and_reflect()`; reflector itself is manual.
- Multi-step reflection (Park's "generate questions, then answer them") — single-step for v1; upgrade if quality suffers.
- Background / async scheduling — caller decides cadence.
- Feedback reflections-of-reflections — reflections don't count toward the threshold; avoids infinite regress.
- Persistence — same as Module 4 (in-memory only).

## Design

### Core class

```python
# DefenseAgent/reflection/reflection.py

class Reflector:
    """Reads from a Memory, asks an LLM to think about it, writes back.

    Bring-your-own-Memory and bring-your-own-LLM: Reflector has no opinion
    about how either was constructed. It just needs the two front-door
    objects.
    """

    def __init__(
        self,
        memory: Memory,
        llm: LLM,
        *,
        num_insights: int = 3,
        reflection_importance: float = 8.0,
    ) -> None: ...

    # ---- LLM-based importance scoring (Park §3.2.1) ----

    async def score_importance(self, content: str) -> float:
        """Rate how poignant `content` is on a 1–10 scale (via LLM).

        Returns 5.0 (the middle) if parsing fails — never raises.
        """

    # ---- Reflection (Park §3.2.2) ----

    async def check_and_reflect(self) -> list[MemoryRecord]:
        """If the unreflected count >= threshold, reflect now. Else no-op.
        Returns the new reflection records ([] if nothing happened).
        """

    async def reflect_now(self) -> list[MemoryRecord]:
        """Force a reflection regardless of threshold."""

    @property
    def unreflected_count(self) -> int:
        """Non-reflection records added since the last reflection."""
```

### Trigger: count-based, reading from the shared profile

`profile.cognitive.reflection_threshold` (already in Module 2's schema, defaults to `5`, comment: *"trigger reflection after N new memories"*) is the knob. This matches what the schema advertises, and keeps the API simple — no new config fields.

Reflector tracks a single piece of state: `self._last_reflection_time: datetime | None`. `unreflected_count` returns `len([r for r in memory.stream.get_all() if r.timestamp > self._last_reflection_time and r.kind != "reflection"])`. After a successful reflection, set `_last_reflection_time = self._clock()`.

**Why not importance-sum triggering (as in Park)?** Our profile schema specifies count, not accumulated importance. Count is simpler to reason about and matches the documentation. If users later want importance-sum, they can either score observations high and keep the threshold low, or we add that as an option.

**Why are reflections excluded from the count?** To avoid infinite regress — a reflection is itself stored as a new memory with kind=`reflection`; if it counted, the very act of reflecting would tip the counter again.

### Importance-scoring prompt

Park-style, adapted for our `LLM.chat()`:

```
On a scale of 1 to 10, where 1 is purely mundane (e.g., brushing teeth,
making the bed) and 10 is extremely poignant (e.g., a breakup, a college
acceptance), rate the likely poignancy of the following memory.

Memory: {content}

Respond with a single integer between 1 and 10. No explanation.
```

Parsing: find the first integer in the response; clip to `[1, 10]`; if nothing parseable, return `5.0`. Never raises.

### Reflection prompt

```
Recent memories (chronological):
{chronological_list_of_recent_memories}

Given the memories above, produce exactly {N} high-level insights about
patterns, lessons, or deeper observations. Each insight should be a
single clear sentence. Return one per line. No numbering, no bullets,
no empty lines.
```

Parsing: split on `\n`, strip each line of whitespace + leading bullet/number prefixes (`-`, `*`, `1.`, `1)`), drop empties, take the first `N` non-empty lines. If fewer than N survive parsing, return whatever did; if the model returned nothing usable, return an empty list (no reflection records are created; the trigger state is still advanced so we don't keep retrying the same batch on every call).

### What "recent memories" means in the prompt

All records in `memory.stream.get_all()` whose timestamp is `> self._last_reflection_time`. This is the set of new observations since the last reflection — exactly the scope of what's being synthesized.

Each memory is rendered as `"- [kind, imp={importance}] {content}"` for the prompt, chronological order.

### Integration table

| Module | How Reflection uses it | Changes to that module |
|---|---|---|
| Module 1 (LLM) | `LLM.chat()` for both importance scoring and reflection synthesis. Read-only. | None |
| Module 2 (config) | Reads `profile.cognitive.reflection_threshold` via `memory.profile` | None |
| Module 3 (ops) | No coupling — caller can wrap `check_and_reflect()` with `logger.info()` | None |
| Module 4 (memory) | Reads `memory.stream.get_all()`; writes via `memory.remember(kind="reflection", …)` | None |

Zero retrofits. `Reflector` is strictly additive.

### Errors

The module raises nothing of its own. Two implicit contracts:

- `LLMProviderError` from `llm.chat()` propagates to the caller (network failures, rate limits, etc.).
- Malformed LLM responses (unparseable integer for importance; empty or garbage for reflection) are swallowed and return sensible defaults — never crash the agent.

No `ReflectionError` class. If `LLMProviderError` is the only error, we don't need a new hierarchy.

### File layout — one file per the user's directive

```
DefenseAgent/reflection/
├── __init__.py           # re-exports Reflector
└── reflection.py         # everything: Reflector class, prompts, parsers (~150 lines)

tests/DefenseAgent/reflection/
├── __init__.py
└── test_reflection.py

scripts/
└── reflection_demo.py    # seed Maya with observations, force a reflection, retrieve
```

### Dependencies

No new deps. Uses `LLM` (Module 1), `Memory` and `MemoryRecord` (Module 4), `AgentProfile` (Module 2) — all already in the harness.

## Testing strategy

Offline throughout. Stub `LLM` and stub `EmbeddingAdapter` let us pin exact responses and assert the full flow.

Test outline:

1. **Importance scoring — parsing variants** (6 tests):
   - Plain integer (`"7"`) → 7.0.
   - Integer inside a sentence (`"I'd rate this a 8."`) → 8.0.
   - Out-of-range clipped to [1, 10] (`"42"` → 10, `"-3"` → 1… actually regex only matches non-negative, so `-3` → 3; check).
   - Unparseable (`"hmm"`) → 5.0 (default).
   - Returns a float, not an int.
   - LLM exception propagates (don't swallow those).

2. **Unreflected count** (4 tests):
   - Empty stream → 0.
   - Counts observations, ignores reflections.
   - Resets after a reflection runs.
   - Respects timestamp cutoff (records older than the last reflection are excluded).

3. **`reflect_now`** (6 tests):
   - Parses exactly N insights from a clean response.
   - Tolerates bullets/numbering.
   - Reflections are stored with `kind="reflection"` + configured importance.
   - Reflections show up in a subsequent `memory.recall()`.
   - Empty-ish LLM response → returns `[]`, still advances `_last_reflection_time`.
   - Doesn't double-count its own reflections on the next cycle.

4. **`check_and_reflect` trigger logic** (3 tests):
   - Below threshold → no-op, returns `[]`, no LLM call.
   - At threshold → triggers, returns records.
   - Above threshold, after manual reset → no-op until new records arrive.

5. **Integration** (1 test): end-to-end with `AgentProfile.from_yaml(maya_rodriguez.yaml)` + stub LLM + stub embedder + real `Memory`; verify reflections are retrievable by query.

## Execution flow (what happens on `reflect_now()`)

```
reflector.reflect_now()
│
├─ 1. recent = [r for r in memory.stream.get_all()
│               if r.timestamp > self._last_reflection_time
│               and r.kind != "reflection"]
│
├─ 2. if not recent:           # nothing new to reflect on
│         return []
│
├─ 3. prompt = build_reflection_prompt(recent, self._num_insights)
│
├─ 4. resp = await self._llm.chat([Message(role="user", content=prompt)])
│
├─ 5. insights = _parse_reflection_response(resp.content, self._num_insights)
│
├─ 6. records: list[MemoryRecord] = []
│     for insight in insights:
│         r = await memory.remember(
│                 insight,
│                 kind="reflection",
│                 importance=self._reflection_importance,
│             )
│         records.append(r)
│
├─ 7. self._last_reflection_time = self._clock()
│
└─ return records
```

`check_and_reflect()` adds one early check: if `unreflected_count < threshold`, return `[]` before step 3.

## Open questions

None. Design decisions confirmed inline:
1. **Count-based trigger** matches the existing `profile.cognitive.reflection_threshold` schema.
2. **Single-step prompt** (not Park's question-then-answer two-step) — upgrade later if quality is poor.
3. **Manual trigger** — caller controls cadence; Reflector does not hook into `memory.remember()` automatically.
4. **One file** (`reflection.py`) — fits comfortably in ~150 lines.
5. **No new error class** — `LLMProviderError` from Module 1 is enough.
