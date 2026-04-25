# Module 3 Walkthrough — Structured Logger

> Companion to the [design spec](../superpowers/specs/2026-04-22-module-03-logger-design.md). Explains the code line-by-line and traces the execution of `scripts/logger_demo.py`.

---

## CORE CLASS: `AgentLogger`

Start here. The module is one class in one file:

```python
from DefenseAgent.ops import AgentLogger

logger = AgentLogger.from_profile(profile, log_file="logs/maya.log")
logger.info("llm.request", "Calling model", model="deepseek-chat")
```

No facade needed — `AgentLogger` was already the single unified interface. The walkthrough below dissects that one class: record shape, sinks, level filtering, and the I/O-failure tolerance contract.

---

## 1. What problem this module solves

Agents make many small decisions per second — call the LLM, retrieve memories, invoke tools, handle errors. If the harness only used `print()` statements, debugging would mean re-reading terminal scrollback hoping to spot the issue. If it used Python's default `logging`, every team member would configure it slightly differently, and log aggregators (Splunk, Datadog, `jq` on disk) would receive unstructured text.

**Module 3 gives the harness one logger that:**
- Emits exactly one **JSON object per line** (the "JSON-lines" format). An aggregator can index every event without a parser.
- Attaches every line to an `agent_id`, so a single log file can hold events from multiple agents without ambiguity.
- **Never raises on I/O failure** — a full disk shouldn't crash the agent.
- Plugs in with one line: `logger = AgentLogger.from_profile(profile)`.

---

## 2. Directory map

```
DefenseAgent/ops/                      # all "operational" concerns
├── __init__.py                         # re-exports AgentLogger
└── logger.py                           # the whole module — one class, one file

tests/DefenseAgent/ops/
├── __init__.py
└── test_logger.py                      # 36 tests covering every behavior

tests/DefenseAgent/integration/
└── test_logger_integration.py          # 2 tests: profile + LLM + logger compose

scripts/
└── logger_demo.py                      # comprehensive demonstration
```

The module is deliberately one small file. There's no `errors.py` — the logger has exactly one user-facing exception (`ValueError` on reserved-key misuse) and stdlib already provides that class.

---

## 3. Anatomy of a log record

Every call produces this exact shape, one JSON object, one line, newline-terminated:

```json
{
  "timestamp":  "2026-04-22T10:15:30.480Z",
  "agent_id":   "student_maya_001",
  "level":      "INFO",
  "event_type": "llm.request",
  "message":    "Calling DeepSeek",
  "data":       {"model": "deepseek-chat", "max_tokens": 200}
}
```

| Key | Shape | Who sets it |
|---|---|---|
| `timestamp` | ISO-8601 UTC, millisecond precision, `Z` suffix | the logger via its clock |
| `agent_id` | string | the logger (from `__init__` or `from_profile`) |
| `level` | one of `DEBUG/INFO/WARNING/ERROR/CRITICAL` | the level method called |
| `event_type` | caller-supplied dotted id (`llm.request`, `tool.timeout`, …) | the caller |
| `message` | caller-supplied human-readable summary | the caller |
| `data` | object with caller's kwargs; `{}` if none | the caller |

**Why millisecond precision, not microsecond?** Most log aggregators index to ms; full microseconds in the timestamp waste 3 bytes per record. Python's `datetime.isoformat()` gives microseconds by default, so the logger has its own `_format_timestamp` that truncates.

---

## 4. Code walk-through: `DefenseAgent/ops/logger.py`

### Module-level constants

```python
_RESERVED_KEYS = frozenset(
    {"agent_id", "level", "timestamp", "event_type", "message", "data"}
)
```

The six top-level record keys. A caller cannot pass kwargs with these names — we'd silently overwrite the real value. Using a `frozenset` means membership checks are O(1) and the set is immutable (can't be mutated at import time).

```python
def _default_clock() -> datetime:
    return datetime.now(timezone.utc)
```

The default time source. Tests inject a fixed-time lambda instead, so assertions on `timestamp` are deterministic.

```python
def _format_timestamp(dt: datetime) -> str:
    ms = dt.microsecond // 1000
    return dt.strftime("%Y-%m-%dT%H:%M:%S") + f".{ms:03d}Z"
```

Millisecond precision + explicit `Z` suffix. This format is what `datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%fZ")` expects, so tests can round-trip it.

### Constructor

```python
def __init__(
    self,
    agent_id: str,
    *,
    log_file: str | Path | None = None,
    stream: TextIO | None = sys.stdout,
    level: int = logging.INFO,
    clock: Callable[[], datetime] | None = None,
) -> None:
    self.agent_id = agent_id
    self._level = level
    self._stream = stream
    self._file = Path(log_file) if log_file is not None else None
    self._clock = clock or _default_clock
    self._lock = threading.Lock()
    if self._file is not None:
        try:
            self._file.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
```

Four parameters are keyword-only (`*`). This means you can't write `AgentLogger("a", "path/x.log")` by accident — paths go in `log_file=`, levels go in `level=`, etc. Explicit call sites, fewer ambiguities.

**Parent directory creation** is attempted up front but wrapped in `try/except OSError`. If the parent can't be created (permissions, full disk), we swallow the error — the actual write will fail later and be swallowed there too. The logger's contract is "never crash the agent", and that starts here.

### `from_profile` — the Module-2 handshake

```python
@classmethod
def from_profile(cls, profile: "AgentProfile", **kwargs) -> "AgentLogger":
    return cls(agent_id=profile.id, **kwargs)
```

Three lines. All remaining kwargs pass through unchanged. The `AgentProfile` type is imported only under `TYPE_CHECKING` so the runtime import graph is `ops → nothing`, no circularity risk.

### Level methods

```python
def debug   (self, event_type, message, /, **data): self.log(logging.DEBUG,    event_type, message, **data)
def info    (self, event_type, message, /, **data): self.log(logging.INFO,     event_type, message, **data)
def warning (self, event_type, message, /, **data): self.log(logging.WARNING,  event_type, message, **data)
def error   (self, event_type, message, /, **data): self.log(logging.ERROR,    event_type, message, **data)
def critical(self, event_type, message, /, **data): self.log(logging.CRITICAL, event_type, message, **data)
```

**The `/` matters.** It makes `event_type` and `message` positional-only. Without it, calling
```python
logger.info("e", "m", event_type="oops")
```
would raise a raw `TypeError: got multiple values for argument 'event_type'`. With `/`, the kwarg `event_type="oops"` is routed into `**data`, and our explicit `_RESERVED_KEYS` check raises a clean `ValueError` with an informative message. Same behavior for all five method names.

### The core dispatch: `log()`

```python
def log(self, level, event_type, message, /, **data):
    if level < self._level:
        return                                            # (1) fast early-out

    collision = _RESERVED_KEYS.intersection(data.keys())
    if collision:
        raise ValueError(
            f"data kwargs may not reuse reserved record keys: {sorted(collision)}."
        )                                                 # (2) hard-fail on programmer error

    record = {
        "timestamp":  _format_timestamp(self._clock()),
        "agent_id":   self.agent_id,
        "level":      logging.getLevelName(level),
        "event_type": event_type,
        "message":    message,
        "data":       dict(data),
    }                                                     # (3) build the record
    line = json.dumps(record, ensure_ascii=False, default=str) + "\n"

    with self._lock:                                      # (4) atomic write
        self._write_stream(line)
        self._write_file(line)
```

Four moments to notice:

**(1) Level filter first, everything else later.** If you sprinkle `logger.debug(...)` calls across the codebase, they cost nothing in production — one integer comparison, return. No dict building, no JSON, no lock, no I/O.

**(2) Reserved-key check raises.** This is the *only* place the logger raises (outside of the I/O paths, which are swallowed). The error is deterministic, actionable, and points the user to the fix.

**(3) Record is a plain dict.** `json.dumps(..., default=str)` handles non-JSON-safe types by calling `str()` on them. Passing a `Path`, a `datetime`, or even a custom object won't crash — you'll just see its `str()` form in the log. That's usually what you want from diagnostic logs.

**(4) One lock, both sinks.** The alternative (two locks, or no lock) risks interleaving bytes when two threads log concurrently. One lock means either the stream line is fully written before the next starts, or the file line is — but never two half-lines in either sink.

### The sink helpers

```python
def _write_stream(self, line):
    if self._stream is None:
        return
    try:
        self._stream.write(line)
        self._stream.flush()
    except Exception:
        pass

def _write_file(self, line):
    if self._file is None:
        return
    try:
        with open(self._file, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass
```

Two identical shapes: `if sink is None, bail; else try/except: pass`. The `try/except: pass` is deliberate and is where the "never crash the agent" contract is realized in practice. If your log disk fills mid-run, you lose log lines; the agent keeps running.

The file is **opened and closed per call**, not kept open. Three consequences:
- Lines are flushed to disk immediately after each `logger.x()` call (no process-crash losing buffered lines).
- File rotation by external tools (`logrotate`) works: we can't hold a stale file handle.
- Tiny perf hit of `open()` per call — not a concern at human-readable log volumes.

---

## 5. Errors & tolerance matrix

| Situation | What happens |
|---|---|
| Reserved kwarg collision | `ValueError` (programmer bug, fail loud) |
| Stream write raises (broken pipe, closed stdout) | Swallowed silently |
| File parent dir can't be created | Swallowed silently; later writes will also fail silently |
| File open/write fails (disk full, no perms) | Swallowed silently |
| JSON serialization hits a weird type (datetime, Path, object) | Converted to `str()` via `default=str` |
| Log level below threshold | Early-returned, no record built |

The only way to get an exception out of the logger is the ValueError for reserved-key misuse. Everything else is either silent or a logged line.

---

## 6. How Module 3 integrates with Modules 1 & 2

**Module 2 (config):** explicit, one line.
```python
logger = AgentLogger.from_profile(profile)
```
Reads `profile.id`. That's the only coupling.

**Module 1 (LLM): no direct coupling.** Adapters stay pure — they don't import the logger. Instead, **callers wrap adapter calls**:
```python
logger.info("llm.request", "Sending", model=model, max_tokens=200)
try:
    resp = await adapter.chat(messages)
except LLMProviderError as e:
    logger.error("llm.error", "Provider failed", provider=e.provider, status_code=e.status_code)
    raise
logger.info("llm.response", "OK", stop_reason=resp.stop_reason, total_tokens=resp.usage.total_tokens)
```

Why not log inside the adapter itself? Because:
1. The adapter would then need a logger injected (extra parameter, extra lifecycle concern).
2. Different callers want different verbosity around the same call (a cognitive-loop step logs more context than a smoke test).
3. It would make `OpenAICompatibleAdapter` know about `AgentLogger`, tightly coupling Module 1 to Module 3 just for observability.

The wrap-at-caller pattern is what the future cognitive loop module will do systematically. The demo script shows it by hand.

---

## 7. Execution flow: `scripts/logger_demo.py`

The demo runs five labeled sections that cover every logger capability:

```
$ python scripts/logger_demo.py

┌─ main() (async)
│
├─ Step 1: load_profile(agents/maya_rodriguez/profile.yaml)     [Module 2]
├─ Step 2: log_file = <repo>/logs/<profile.id>.log
├─ Step 3: logger = AgentLogger.from_profile(profile, log_file=log_file)
│
├─ [section 1] demo_level_filtering(logger)
│     • debug()    ← dropped (below INFO threshold)
│     • info()     ← emitted
│     • warning()  ← emitted
│     • error()    ← emitted
│     • critical() ← emitted
│     4 lines appear on stdout AND in the log file
│
├─ [section 2] demo_structured_data(logger)
│     • info("demo.data", "...", request_id="...", retry_count=0, nested={"k":"v","list":[1,2,3]})
│     • `data` object in the emitted line carries those kwargs verbatim
│
├─ [section 3] demo_reserved_key_rejection(logger)
│     • info("...", "...", agent_id="other")  ← raises ValueError
│     • caught; printed to stdout
│     • no log line emitted (ValueError precedes record construction)
│
├─ [section 4] await demo_successful_llm_call(logger, question, system)    [Module 1]
│     • adapter = make_adapter_from_env()
│     • logger.info("llm.request", ..., model="deepseek-chat", max_tokens=160)
│     • await adapter.chat(...)   ← real HTTPS call to DeepSeek
│     • logger.info("llm.response", ..., stop_reason="end_turn", total_tokens=127)
│     • prints Maya's in-character reply
│
├─ [section 5] demo_error_path_logging(logger)
│     • raise LLMProviderError(provider="demo-stub", status_code=503, ...)
│     • caught; logger.error("llm.error", ..., provider="demo-stub", status_code=503)
│     • nothing propagates
│
└─ Tail summary
      • count total JSON lines in the file
      • print the last 3 lines so the user sees what's on disk
```

### Real-run output (abbreviated)

```
[section 4] Wrapping a real LLM call with log events
------------------------------------------------------------
{"timestamp": "...", "agent_id": "student_maya_001", "level": "INFO", "event_type": "llm.request", "message": "Sending chat request", "data": {"adapter": "OpenAICompatibleAdapter", "model": "deepseek-chat", "messages_count": 1, "max_tokens": 160}}
{"timestamp": "...", "agent_id": "student_maya_001", "level": "INFO", "event_type": "llm.response", "message": "Received chat response", "data": {"stop_reason": "end_turn", "prompt_tokens": 101, "completion_tokens": 26, "total_tokens": 127, ...}}
[demo] assistant reply: I've got Data Structures and Algorithms in 20 minutes, and I'm actually pretty excited because we're covering graph traversal today.
```

That's **all three modules cooperating** in seven lines:
1. Module 2 loaded Maya's profile.
2. Module 3 bound a logger to her id and opened the log file.
3. Module 1 talked to DeepSeek in her voice.
4. Module 3 recorded the before/after.

---

## 8. Test coverage map

| File | Tests | Covers |
|---|---|---|
| `tests/DefenseAgent/ops/test_logger.py` | 36 | Every field, every level, both sinks, clock injection, reserved-key rejection, JSON safety, I/O failure tolerance, `from_profile` |
| `tests/DefenseAgent/integration/test_logger_integration.py` | 2 | Happy-path wrap (StubAdapter) + error-path wrap (StubErrorAdapter) with profile + logger |

Highlights worth reading:
- `test_reserved_kwargs_raise_value_error[level|event_type|message]` — these exist specifically because they're the kwargs that would collide with the method's positional-only parameters, and prove the `/` marker actually does its job.
- `test_default_clock_produces_iso_utc_ms_with_z` — verifies the timestamp is parseable back into a `datetime` with the exact format string the spec documents.
- `test_file_write_failure_is_swallowed` — uses `monkeypatch` to replace `builtins.open` with a raising function, then asserts the logger call returns normally.
- `test_logger_records_provider_error_without_crashing` — in the integration file, this is the single most important test: it proves logger + adapter-exception-wrapping works even when the adapter fails.

All tests are fully offline (`io.StringIO`, `tmp_path`, `StubAdapter`). No sleeps, no flakiness, 0.07s total.

---

## 9. Things worth noticing

- **One file, no errors module.** The logger needs a grand total of one custom exception: `ValueError` from stdlib. Adding an `errors.py` with `LoggerError(Exception)` would be scope-drift.
- **Keyword-only constructor args.** The `*` in `__init__` forces explicit keywords for everything except `agent_id`. Saves a class of "oh I thought the second arg was the path" bugs forever.
- **Positional-only method params.** The `/` in `debug/info/…/log` is what turns "dangerous kwarg collision" into "clean ValueError". This is a Python 3.8+ feature worth knowing about — it's the cleanest fix for this exact class of API problem.
- **Per-call file open.** Simpler, safer, plays nicely with `logrotate`. Costs one `open()` syscall per log line, which is fine for human-readable throughput.
- **`default=str` in `json.dumps`.** This is why `logger.info("...", "...", where=Path("/tmp"))` doesn't crash. Any type with a meaningful `__str__` just works.
- **Adapter module stays log-agnostic.** Module 1 has not changed. The logger integrates *at the boundary* — in the cognitive loop (future) or in the demo script — not inside adapter internals.
