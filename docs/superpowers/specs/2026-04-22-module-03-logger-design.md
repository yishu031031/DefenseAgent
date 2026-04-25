# Module 3 — Structured Logger Design

**Date:** 2026-04-22
**Status:** Approved, ready for implementation
**Module position:** 3 of N. Consumed by every subsequent module that emits events (cognitive loop, tool executor, memory, MCP client). No runtime coupling back into Module 1 (LLM) or Module 2 (config) — those modules remain log-agnostic; callers wrap them.

## Purpose

Give every module in the harness **one** structured logging facility so that:
1. Each event is an **atomic JSON line** an external aggregator can index.
2. Every entry carries the **agent_id**, letting one log file hold events from several agents unambiguously.
3. Logging **never raises** — a logger failure must not crash the agent.
4. A logger **binds to an `AgentProfile`** with one call, so downstream modules don't re-derive agent_id from ad-hoc places.

Running this module produces JSON like:

```json
{"timestamp": "2026-04-22T14:15:32.481Z", "agent_id": "student_maya_001", "level": "INFO", "event_type": "llm.request", "message": "Calling DeepSeek", "data": {"model": "deepseek-chat", "max_tokens": 200}}
```

## Scope

### In scope
- `AgentLogger` class with five level methods (`debug`, `info`, `warning`, `error`, `critical`).
- JSON-lines output to both a file (optional) and a text stream (default stdout).
- `AgentLogger.from_profile(profile, ...)` classmethod for Module 2 integration.
- Level filtering using stdlib `logging` level constants (`logging.DEBUG`, `logging.INFO`, …).
- Custom clock injection for deterministic timestamps in tests.
- Thread-safe write path (lock around the two sink writes).
- No external dependencies — stdlib only (`json`, `logging`, `threading`, `pathlib`, `datetime`).
- Unit tests for every behavior, plus a `scripts/logger_demo.py` that integrates logger + profile + LLM.

### Out of scope (deferred)
- File rotation / size limits (let the OS or `logrotate` handle it).
- Async / queue-based handler (current API is sync; acceptable for a learning-scale harness).
- EventBus integration — EventBus doesn't exist yet (Module 4 or later). Adding an auto-log subscriber is a one-file change when the event bus lands.
- Network sinks (Datadog, Cloud Logging, etc.). Callers can subclass or write their own stream wrapper.
- Pretty console output / coloring. JSON lines are machine-readable first; an operator can pipe through `jq` if they want pretty output.
- Modifying Module 1 adapters to emit logs. Adapters stay pure; callers wrap `adapter.chat(...)` in log calls.

## Design

### Record shape

Every call produces exactly this JSON object (one line, `\n`-terminated):

| Key          | Type   | Source                                           |
|--------------|--------|--------------------------------------------------|
| `timestamp`  | string | ISO-8601 UTC, millisecond precision, `Z` suffix  |
| `agent_id`   | string | Value from the `AgentLogger` instance            |
| `level`      | string | One of `DEBUG/INFO/WARNING/ERROR/CRITICAL`       |
| `event_type` | string | Caller-supplied dotted identifier (e.g. `llm.request`) |
| `message`    | string | Caller-supplied human-readable summary           |
| `data`       | object | Caller-supplied kwargs; `{}` if none             |

Timestamps are always UTC, `2026-04-22T14:15:32.481Z`. The `Z` suffix is ISO-compliant and cheaper to filter than parsing a timezone offset.

### API

```python
class AgentLogger:
    def __init__(
        self,
        agent_id: str,
        *,
        log_file: str | Path | None = None,
        stream: TextIO | None = sys.stdout,
        level: int = logging.INFO,
        clock: Callable[[], datetime] | None = None,
    ): ...

    @classmethod
    def from_profile(
        cls,
        profile: AgentProfile,
        **kwargs,
    ) -> "AgentLogger": ...

    def debug   (self, event_type: str, message: str, **data) -> None: ...
    def info    (self, event_type: str, message: str, **data) -> None: ...
    def warning (self, event_type: str, message: str, **data) -> None: ...
    def error   (self, event_type: str, message: str, **data) -> None: ...
    def critical(self, event_type: str, message: str, **data) -> None: ...

    def log(self, level: int, event_type: str, message: str, **data) -> None:
        """Underlying dispatch; the five level methods delegate here."""
```

### Design decisions

**D1. Instance-per-agent, not a module-level singleton.**
Each `AgentLogger` carries `agent_id`. When multiple agents run in one process (future scope), each has its own logger. Downstream modules receive the logger via dependency injection — they never reach into a global.

**D2. `kwargs` become `data`; reserved keys rejected.**
`logger.info("chat.request", "sending prompt", model="gpt-4o", tokens=200)` puts `{model: "gpt-4o", tokens: 200}` under `data`. But these keys collide with the top-level record shape and must raise `ValueError` if passed:
- `agent_id`, `level`, `timestamp`, `event_type`, `message`, `data`

Rationale: ambiguity is worse than strictness. If you really need to log `"message": "foo"` as payload, nest it: `logger.info("...", "...", payload={"message": "foo"})`.

**D3. Level filtering happens BEFORE record construction.**
If `level < self._level`, the method returns immediately — no dict, no JSON, no lock acquisition, no I/O. Keeps `logger.debug(...)` cheap in production.

**D4. Sinks: stream + optional file.**
- `stream` defaults to `sys.stdout`. Set to `None` to silence stdout.
- `log_file` is optional. When set, every record also appends to that file (file is opened/closed per write — simple, crash-safe).
- If neither sink is configured, the logger is a no-op (still useful for silencing tests).

**D5. Logger never raises on I/O failure.**
If `stream.write` or file open/write fails, the exception is swallowed silently. The alternative — crashing the agent because the log disk is full — is far worse than losing one log line. (Future enhancement: expose a `on_error=` callback so callers can opt into handling, but default stays silent.)

**D6. The only raise is `ValueError` for reserved-key misuse.**
That's a programmer error (bug in calling code), not a runtime environmental failure. It should surface loud during development.

**D7. Clock injection for deterministic tests.**
`clock` is an optional `Callable[[], datetime]`. Tests pass a fixed-time lambda so the `timestamp` field is predictable. Production uses `datetime.now(timezone.utc)`.

**D8. Thread safety via one lock around the two writes.**
A single `threading.Lock` protects the stream-write-then-file-write pair. Without the lock, two threads calling `info()` simultaneously could interleave bytes in stdout.

**D9. `from_profile` factory keeps Module 2 integration minimal.**
Just one line of integration:
```python
@classmethod
def from_profile(cls, profile: AgentProfile, **kwargs) -> "AgentLogger":
    return cls(agent_id=profile.id, **kwargs)
```
No circular import: `ops/logger.py` imports `AgentProfile` only for the type annotation; we can use `TYPE_CHECKING` to avoid even that at runtime.

**D10. No adapter modifications.**
Module 1's `LLMAdapter` subclasses are NOT touched by this module. The cognitive loop (future) will log around adapter calls. The logger demo script in `scripts/` shows the pattern.

### File layout

```
DefenseAgent/ops/
├── __init__.py          # re-exports AgentLogger
└── logger.py            # AgentLogger + helpers

tests/DefenseAgent/ops/
├── __init__.py
└── test_logger.py       # all logger tests (stdlib-only; uses io.StringIO + tmp_path)

scripts/
└── logger_demo.py       # profile + logger + LLM demo
```

### Dependencies

Stdlib only. No updates to `requirements.txt`.

## Testing strategy

All tests are fully in-memory or on `tmp_path`. No network, no sleep, no time-dependent assertions (clock is injected).

Coverage outline:

**Construction**
- `AgentLogger(agent_id=...)` stores agent_id.
- Default level is `logging.INFO`.
- Default stream is `sys.stdout`.

**Record shape**
- Every emitted line is valid JSON.
- Exactly the six top-level keys: `timestamp`, `agent_id`, `level`, `event_type`, `message`, `data`.
- `data` is `{}` when no kwargs provided.
- `data` carries kwargs verbatim.

**Level filtering**
- `info()` on a DEBUG-level logger emits.
- `debug()` on an INFO-level logger does NOT emit.
- Every level method maps to the correct `logging.*` constant.

**Sinks**
- Stream-only: writes to provided stream, not to file.
- File-only: `stream=None`, `log_file=path` — writes to file.
- Both: writes the same line to both.
- Neither (`stream=None, log_file=None`): no-op.
- File is appended, not truncated, on repeated calls.
- Parent directory created if missing.

**Clock injection**
- `clock=lambda: datetime(2026, 4, 22, 10, 15, 30, 480000, tzinfo=timezone.utc)` → `timestamp` is `"2026-04-22T10:15:30.480Z"`.
- Default clock produces a timezone-aware timestamp ending in `Z`.

**Reserved-key rejection**
- `logger.info("e", "m", agent_id="other")` → `ValueError`.
- Same for `level`, `timestamp`, `event_type`, `message`, `data`.

**Non-JSON-safe data**
- Passing a `Path` in data does NOT crash — serializer uses `default=str`.
- Passing a `datetime` in data does NOT crash — serialized as its `str()` form.

**I/O failure tolerance**
- A stream whose `write` raises does not propagate.
- A path pointing to an un-writable directory does not propagate.
- (These tests use `monkeypatch` to inject a raising write function.)

**`from_profile`**
- `AgentLogger.from_profile(profile)` returns a logger with `agent_id == profile.id`.
- Additional kwargs pass through (`level`, `log_file`, etc.).

**Integration sanity (in `tests/DefenseAgent/integration/`):**
- One test exercises: load profile → `from_profile` → StubAdapter.chat → log request, log response → assert log file contains both events with the right `event_type` and `agent_id`.

## Execution flow (what happens on one `logger.info(...)` call)

```
logger.info("llm.request", "Calling model", model="deepseek-chat")
│
├─ log(logging.INFO, "llm.request", "Calling model", model="...")
│
├─ if logging.INFO < self._level: return          ← early exit
│
├─ check kwargs for reserved keys → raise ValueError if any
│
├─ record = {
│     "timestamp": iso8601(self._clock()),
│     "agent_id":  self.agent_id,
│     "level":     "INFO",
│     "event_type":"llm.request",
│     "message":   "Calling model",
│     "data":      {"model": "deepseek-chat"},
│  }
│
├─ line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
│
└─ with self._lock:
      try: self._stream.write(line); self._stream.flush()
      except: pass
      if self._file:
          try: append line to file
          except: pass
```

## Integration with earlier modules

- **Module 2 (config):** `AgentLogger.from_profile(profile)` reads `profile.id`. That's the only contact point.
- **Module 1 (LLM):** untouched. The demo script shows the recommended wrapping pattern around `adapter.chat(...)`.

## Future extensions

When the event bus arrives (Module 4 or so), add:
- `AgentLogger.subscribe_to(bus)` — auto-emit a log line for each event on the bus.
- Level mapping via a small event-type→log-level table.

When a real server arrives, add:
- `NetworkSink` subclass supporting HTTP or syslog output.
- A `QueueHandler`-style asyncio wrapper so writes don't block the event loop.

Both are opt-in; the core API doesn't change.

## Open questions

None at spec-approval time. Design decisions D1–D10 above settle every choice.
