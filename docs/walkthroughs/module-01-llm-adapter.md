# Module 1 Walkthrough — LLM Adapter

> Companion to the [design spec](../superpowers/specs/2026-04-22-module-01-llm-adapter-design.md). The spec records **what** we decided and **why**; this walkthrough explains **how** the code implements those decisions and **what happens** when you run the scripts.

---

## CORE CLASS: `LLM`

Start here. The `LLM` class in [`DefenseAgent/llm/llm.py`](../../DefenseAgent/llm/llm.py) is the canonical entry point:

```python
from DefenseAgent.llm import LLM, Message

llm  = LLM.from_env()                              # reads AGENT_LAB_LLM_PROVIDER from .env
resp = await llm.chat([Message(role="user", content="hi")])
```

`LLM` wraps an `LLMAdapter` (one of the concrete provider adapters discussed below) and exposes it as `llm.adapter`. Everything else in this module is either:
- machinery the facade uses internally (adapters, factory, canonical types),
- or error classes the facade may raise.

The walkthrough below follows the data flow from `.env` through the adapter stack so you understand what `LLM.from_env()` and `llm.chat(...)` actually do.

---

## 1. What problem this module solves

If downstream modules (cognitive loop, memory, context manager) called vendor SDKs directly, switching providers would ripple through the codebase. The harness would also have to know about five different message formats, five different tool-call shapes, five different response structures.

**Module 1 gives the rest of the harness exactly one interface:**

```python
resp = await adapter.chat(messages, tools=..., system=...)
```

Whether the adapter routes to Claude, OpenAI, DeepSeek, Qwen, Google, or a local vLLM server is invisible to the caller. The caller always sends canonical `Message` objects and always gets back a canonical `LLMResponse`.

---

## 2. Directory map

```
DefenseAgent/llm/
├── __init__.py                     # public API re-exports
├── types.py                        # canonical data shapes (Message, ToolCall, LLMResponse, TokenUsage)
├── errors.py                       # exception hierarchy (LLMError + 3 subclasses)
├── llm_adapter.py                  # abstract base class (LLMAdapter)
├── anthropic_adapter.py            # concrete: Claude (Anthropic SDK)
├── openai_compatible_adapter.py    # concrete: OpenAI / DeepSeek / Qwen / Google / vLLM (OpenAI SDK)
└── factory.py                      # make_adapter_from_env() — reads .env, returns the right adapter

scripts/
└── smoke_chat.py                   # runnable demo, hits whichever provider .env configures
```

Every file is ≤200 lines. The module has zero external runtime dependencies beyond `anthropic`, `openai`, and `python-dotenv`.

---

## 3. Canonical types (`types.py`)

These dataclasses are **the contract between the harness and the LLM world**. Every provider gets translated to and from these shapes.

### `Message`
```python
@dataclass
class Message:
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)  # assistant-only
    tool_call_id: str | None = None                            # tool-result-only
    name: str | None = None                                    # tool name on tool-result
```

- `role` is the same four-value vocabulary OpenAI uses. Anthropic's Claude uses `user/assistant/system` but we normalize to this list.
- `tool_calls` is populated on assistant messages that requested tool use. Empty for plain assistant text.
- `tool_call_id` + `name` are populated on `role="tool"` messages (the results we send back in the next turn).

### `ToolCall`
```python
@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]   # parsed dict, NOT a JSON string
```

**Key decision:** `arguments` is always a `dict`. OpenAI returns `arguments` as a JSON string; the adapter parses it before handing it back. Callers never have to remember "is this one a string or a dict" — it's always a dict.

### `TokenUsage`
```python
@dataclass
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
```

Filled on every response. If the provider doesn't return `total_tokens`, the adapter computes it as `prompt + completion`.

### `LLMResponse`
```python
@dataclass
class LLMResponse:
    content: str                # assistant text; "" if only tool calls
    tool_calls: list[ToolCall]  # [] if no tools requested
    usage: TokenUsage
    stop_reason: str | None     # normalized vocabulary
    raw: dict[str, Any]         # original provider dict, for debugging
```

**Normalized `stop_reason` vocabulary** (same strings regardless of provider):
- `"end_turn"` — the model finished on its own.
- `"tool_use"` — the model wants a tool invoked.
- `"max_tokens"` — hit the token cap.
- `"stop_sequence"` — hit a user-supplied stop string (Anthropic only today).
- `"other"` — anything else (content filter, function_call legacy, etc.).

---

## 4. Abstract interface (`llm_adapter.py`)

```python
class LLMAdapter(ABC):
    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        *,
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        system: str | None = None,
    ) -> LLMResponse: ...
```

Three things worth noticing:

1. **`ABC` + `@abstractmethod`.** You cannot instantiate `LLMAdapter` directly — Python raises `TypeError`. You also can't subclass it without implementing `chat()`. This is why `test_llm_adapter.py` has `test_cannot_instantiate_abstract_adapter` and `test_subclass_missing_chat_cannot_instantiate`.

2. **`async def`.** Every call to an LLM provider crosses the network. Async lets the rest of the harness (event bus, tool executor, cognitive loop) do other work while waiting.

3. **`system` as kwarg.** Some providers (Anthropic) have a dedicated `system` parameter; others (OpenAI-compatible) just take a system message at the top of the list. The abstract layer accepts it either way; concrete adapters translate.

---

## 5. `OpenAICompatibleAdapter` — step by step

This one handler serves OpenAI, DeepSeek, Qwen, Google (via OpenRouter), and vLLM. They all speak the OpenAI `chat/completions` wire protocol.

### Constructor (client injection)

```python
def __init__(self, *, api_key, base_url, model, client=None):
    self._model = model
    self._client = client or AsyncOpenAI(api_key=api_key or None, base_url=base_url or None)
```

The optional `client` parameter is a test seam. Production code passes nothing and an `AsyncOpenAI` is created. Tests pass a `MagicMock` so no network calls happen. Without this seam, tests would need complex module-level patching.

### `chat()` — five stages

**Stage 1: System-message conflict check.**
```python
has_system_in_messages = any(m.role == "system" for m in messages)
if system is not None and has_system_in_messages:
    raise LLMAdapterError(...)
```
Disallows ambiguity: caller must pick one way or the other.

**Stage 2: Translate canonical → OpenAI wire messages.** Done by `_message_to_wire(m)`:
- Plain user/assistant/system → `{"role": ..., "content": ...}`.
- Assistant with `tool_calls` → include a `tool_calls` array, each entry `{"id", "type": "function", "function": {"name", "arguments"}}` where `arguments` is re-serialized to a JSON string (OpenAI wants the string form on the wire).
- `role="tool"` → `{"role": "tool", "tool_call_id", "name", "content"}`.

**Stage 3: Translate tool schema.** Callers pass JSON-Schema dicts; OpenAI wants each wrapped as `{"type": "function", "function": {...}}`.

**Stage 4: API call with error wrapping.**
```python
try:
    response = await self._client.chat.completions.create(**kwargs)
except Exception as e:
    raise LLMProviderError(
        provider="openai-compatible",
        status_code=getattr(e, "status_code", None),
        message=str(e),
    ) from e
```
The `raise ... from e` preserves the original exception as `__cause__` so callers can still inspect it.

**Stage 5: Parse response.** Done by `_parse_response(response)`:
- Pull `choices[0].message.content` → `content` (empty string if `None`).
- For each `tool_call` on the message: parse `arguments` JSON back to a dict, build a `ToolCall`.
- Map `finish_reason` via `_FINISH_REASON_MAP`: `"stop"→"end_turn"`, `"tool_calls"→"tool_use"`, `"length"→"max_tokens"`, everything else → `"other"`.
- Build `TokenUsage`, compute `total_tokens` if missing.
- Serialize the provider object via `_to_dict_safe` for the `raw` field (best-effort — tries `model_dump()` then `to_dict()`, falls back to `{"repr": repr(obj)}`).

---

## 6. `AnthropicAdapter` — what's different

Same five-stage shape as above, but the wire protocol differs:

| Concern | OpenAI-compatible | Anthropic |
|---|---|---|
| System prompt | Top message with `role="system"` | Separate `system=` kwarg on API call |
| Multiple system messages | Illegal | Our adapter joins them with `\n` and passes as one `system` string |
| Assistant with tools | `content` string + `tool_calls` array | `content` as a **list of blocks**, mixing `text` and `tool_use` blocks |
| Tool result | `role="tool"` message | `role="user"` message with a `tool_result` content block |
| Tool schema field | `parameters` | `input_schema` |
| Response content | `choices[0].message.content` string | `response.content` list of typed blocks; we collect `text` blocks (concatenated) + `tool_use` blocks |
| Stop reason | `finish_reason` (stop/tool_calls/length/…) | `stop_reason` (end_turn/tool_use/max_tokens/stop_sequence) — passes through mostly unchanged |

**Why one adapter can't handle both:** the content shape is fundamentally different (strings vs. block lists), and Claude's tool flow relies on matching `tool_use` IDs across block-structured user messages. Forcing both through one code path would cost more than keeping two ~150-line adapters.

---

## 7. Factory (`factory.py`)

The factory reads environment variables and returns the right concrete adapter. It never makes a network call — constructing the SDK client is lazy.

### Env var structure (per the design spec)

- **Selector:** `AGENT_LAB_LLM_PROVIDER` — one of `openai | anthropic | google | deepseek | qwen | vllm`.
- **Per-provider blocks:** `{PROVIDER}_API_KEY`, `{PROVIDER}_BASE_URL`, `{PROVIDER}_MODEL`.
- **Override tier (per field):** `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL_ID`. Each independently wins over the provider-specific one if non-empty.

### Flow inside `make_adapter_from_env()`

```
1. (optional) load_dotenv(dotenv_path)                       ← reads .env into os.environ
2. _resolve_provider()                                       ← read AGENT_LAB_LLM_PROVIDER, validate
3. _resolve_fields(provider)                                 ← per-field override resolution
4. _validate(provider, api_key, base_url, model)             ← raise LLMConfigError if anything missing
5. if provider == "vllm" and not api_key: api_key = "token-not-needed"
6. if provider == "anthropic":
        return AnthropicAdapter(...)
   else:
        return OpenAICompatibleAdapter(...)
```

`_pick(override, fallback)` is the three-line helper at the heart of the override tier:

```python
def _pick(override, fallback):
    if override:
        return override
    return fallback or ""
```

Treats empty string as "not set" so `LLM_API_KEY=""` correctly falls back to the provider's block.

### Test control

Tests pass `load_env=False` to skip `load_dotenv()`, which would otherwise repopulate env vars the test had deliberately cleared. The `clear_llm_env` fixture in `tests/DefenseAgent/llm/test_factory.py` also monkeypatches `load_dotenv` to a no-op for defense in depth.

---

## 8. Errors (`errors.py`)

```
LLMError (base)
├── LLMConfigError        — .env / configuration problem, raised by the factory
├── LLMAdapterError       — caller misused the adapter (e.g. both system sources supplied)
└── LLMProviderError      — provider API returned an error; wraps original via __cause__
```

Callers can catch `LLMError` broadly, or branch on the subclass:
```python
try:
    resp = await adapter.chat(...)
except LLMConfigError:
    ... # fix env
except LLMProviderError as e:
    ... # maybe retry; inspect e.status_code
```

---

## 9. Execution flow: `scripts/smoke_chat.py`

When you run `python scripts/smoke_chat.py`:

```
┌─ main() (async)
│
├─ 1. make_adapter_from_env()
│     • python-dotenv loads .env into os.environ
│     • factory reads AGENT_LAB_LLM_PROVIDER
│     • factory resolves {PROVIDER}_API_KEY / BASE_URL / MODEL
│                (with LLM_* overrides applied per-field)
│     • factory instantiates AsyncOpenAI (or AsyncAnthropic) lazily
│     • factory returns the configured adapter
│
├─ 2. Print diagnostic info (provider, model, adapter class)
│
├─ 3. adapter.chat([Message(role="user", content="Say hello in 5 words or fewer.")])
│     │
│     ├─ OpenAICompatibleAdapter._chat
│     │   ├─ no system conflict
│     │   ├─ wire_messages = [{"role": "user", "content": "Say hello ..."}]
│     │   ├─ kwargs = {model, messages, temperature=0.2, max_tokens=64}
│     │   ├─ await self._client.chat.completions.create(**kwargs)
│     │   │    └─ HTTPS POST to https://api.deepseek.com/v1/chat/completions
│     │   └─ _parse_response → LLMResponse
│     │         • content extracted from choices[0].message.content
│     │         • finish_reason="stop" normalized to "end_turn"
│     │         • usage filled, total_tokens computed if missing
│     │
│     └─ returns LLMResponse
│
└─ 4. Print content, stop_reason, usage
```

**What could fail and where:**

| Error | Stage | Surface |
|---|---|---|
| `AGENT_LAB_LLM_PROVIDER` unset | Stage 1 | `LLMConfigError` |
| Unknown provider value | Stage 1 | `LLMConfigError` |
| Missing model / base_url / api_key | Stage 1 | `LLMConfigError` |
| Network failure, bad auth, rate limit | Stage 3 | `LLMProviderError` (original exception as `__cause__`) |

The script catches `LLMError` at both stages and returns non-zero exit codes: `2` for config errors, `1` for runtime errors.

---

## 10. Things worth noticing

- **Zero coupling to the harness.** The `llm` module knows nothing about agent profiles, memory, tools-as-classes, or the cognitive loop. Other modules import it; it imports none of them.
- **Test seam via client injection.** `client=fake` in adapter constructors means tests can assert exactly what request was built without any module-level patching. 60/60 tests passed before any real network call happened.
- **Stop-reason vocabulary centralization.** The `_FINISH_REASON_MAP` dict in `openai_compatible_adapter.py` and the `_PASSTHROUGH_STOP_REASONS` set in `anthropic_adapter.py` are the only places the rest of the harness ever needs to understand. Adding a new stop reason is a one-line change in each adapter.
- **`_to_dict_safe` appears in both adapters.** It's duplicated intentionally — the adapters don't import each other. If the duplication grows to three adapters, hoist it to a `_common.py`; until then, YAGNI.
