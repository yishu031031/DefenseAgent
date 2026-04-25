# Module 1 — LLM Adapter Design

**Date:** 2026-04-22
**Status:** Approved, ready for implementation plan
**Module position:** 1 of N. First module of the agent harness. The cognitive loop and memory modules will depend on this.

## Purpose

Give the harness one stable, provider-agnostic interface for LLM chat. Downstream modules (cognitive loop, context manager, memory retriever) must never know which vendor is serving requests. Swapping providers should be a config change, not a code change.

## Scope

### In scope (this module)
- Abstract `LLMAdapter` base class with a single method: `async chat(...)`.
- Two concrete adapters:
  - `AnthropicAdapter` — Anthropic messages API (native or proxied).
  - `OpenAICompatibleAdapter` — OpenAI-compatible APIs, covering OpenAI, Google/Gemini (via proxy), Qwen, DeepSeek, and vLLM (local). Differentiated at runtime by `base_url`, `api_key`, `model`.
- Canonical internal types (`Message`, `ToolCall`, `LLMResponse`, `TokenUsage`) that the rest of the harness will use.
- `.env` loading via `python-dotenv` and a `make_adapter_from_env()` factory with per-provider blocks and a per-field override tier.
- Unit tests with mocked HTTP — no real API calls in the test suite.

### Out of scope (deferred to later modules)
- `embed()` — no consumer yet; will be added to the abstract interface when the memory module lands. Defers the embedding-model-per-provider choice.
- `score_importance()` — a thin wrapper over `chat()`; lives in the memory module when that module is built.
- Streaming responses. Non-streaming only for now; add streaming when the UI or event bus needs it.
- Retry logic, rate limiting, circuit breakers. Raw errors propagate for now; the executor/operator layers will add policy later.
- Cost/token accounting in a persistent store. `LLMResponse.usage` is returned but not aggregated yet (that's the `ops/metrics.py` module's job).

## Design

### Provider shape

| Provider  | Wire protocol          | How we reach it                                    |
|-----------|------------------------|----------------------------------------------------|
| OpenAI    | OpenAI chat/completions| `openai` SDK (base_url defaults to `api.openai.com`, or user-supplied proxy like OpenRouter) |
| Anthropic | Anthropic messages     | `anthropic` SDK (base_url defaults; may be overridden for compatible proxies) |
| Google    | OpenAI chat/completions| `openai` SDK via an OpenAI-compatible proxy (e.g. OpenRouter); base_url required |
| Qwen      | OpenAI chat/completions| `openai` SDK, DashScope or OpenAI-compatible proxy; base_url required |
| DeepSeek  | OpenAI chat/completions| `openai` SDK, `base_url=https://api.deepseek.com`  |
| vLLM      | OpenAI chat/completions| `openai` SDK, user-supplied local `base_url`       |

Five of the six providers share the OpenAI wire protocol. This motivates a shared adapter class.

### Canonical internal types

Defined in `DefenseAgent/llm/types.py`. These are the **only** shapes the rest of the harness sees.

```python
from dataclasses import dataclass, field
from typing import Literal, Any

Role = Literal["system", "user", "assistant", "tool"]

@dataclass
class Message:
    role: Role
    content: str
    # Present on assistant messages that requested tools:
    tool_calls: list["ToolCall"] = field(default_factory=list)
    # Present on tool-result messages (role="tool"):
    tool_call_id: str | None = None
    name: str | None = None  # tool name, when role="tool"

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]  # parsed JSON, not a string

@dataclass
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int  # = prompt + completion; adapter fills if provider omits

@dataclass
class LLMResponse:
    content: str               # assistant text (may be empty if only tool calls)
    tool_calls: list[ToolCall] # empty if no tools requested
    usage: TokenUsage
    stop_reason: str | None    # "end_turn", "tool_use", "max_tokens", etc. — normalized
    raw: dict[str, Any]        # raw provider response for debugging
```

**Normalization decisions:**
- `arguments` is always a parsed dict, never a JSON string. The OpenAI adapter parses the JSON string returned by the API; the Anthropic adapter passes its `input` dict through directly.
- `stop_reason` is normalized to a small vocabulary: `end_turn`, `tool_use`, `max_tokens`, `stop_sequence`, `other`. Both adapters map their native values into this vocabulary.
- `total_tokens` is filled even if the provider returns it separately or not at all. Adapter does the arithmetic.

### Abstract interface

```python
# DefenseAgent/llm/llm_adapter.py
from abc import ABC, abstractmethod

class LLMAdapter(ABC):
    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        *,
        tools: list[dict] | None = None,   # JSON Schema list; adapter translates to vendor format
        temperature: float = 0.7,
        max_tokens: int = 1024,
        system: str | None = None,         # optional; some providers accept a dedicated system param
    ) -> LLMResponse:
        """Send a chat request and return a canonical LLMResponse."""
```

**Signature notes:**
- `messages` is canonical. If a caller includes a `Message(role="system", ...)` at the top of the list AND also passes `system=...`, the adapter raises `LLMAdapterError` — it's ambiguous. Callers should pick one.
- `tools` uses JSON Schema (the OpenAI shape). The Anthropic adapter translates to Anthropic's `tools` format.

### `AnthropicAdapter`

```python
# DefenseAgent/llm/anthropic_adapter.py
class AnthropicAdapter(LLMAdapter):
    def __init__(self, api_key: str, model: str, base_url: str | None = None): ...
    async def chat(self, messages, *, tools=None, ...): ...
```

**Responsibilities:**
1. Strip any `role="system"` messages from the list and merge them into a single system string (concatenated with newlines). Use either that OR the `system=` kwarg — error if both.
2. Translate `messages` → Anthropic's message format:
   - `tool_calls` on assistant messages → `content` blocks of type `tool_use`.
   - `role="tool"` messages → `content` blocks of type `tool_result` inside a `user` message.
3. Translate `tools` (JSON Schema) → Anthropic's `tools` list (same shape, slightly different keys).
4. Call `anthropic.AsyncAnthropic(api_key=...).messages.create(...)`.
5. Translate response → `LLMResponse`:
   - Concatenate all `text` blocks → `content`.
   - Collect `tool_use` blocks → `tool_calls` (passing `input` dict directly into `arguments`).
   - Map `stop_reason`: `end_turn`→`end_turn`, `tool_use`→`tool_use`, `max_tokens`→`max_tokens`, `stop_sequence`→`stop_sequence`, anything else→`other`.
   - Build `TokenUsage` from `response.usage`.

### `OpenAICompatibleAdapter`

```python
# DefenseAgent/llm/openai_compatible_adapter.py
class OpenAICompatibleAdapter(LLMAdapter):
    def __init__(self, api_key: str, base_url: str, model: str): ...
    async def chat(self, messages, *, tools=None, ...): ...
```

**Responsibilities:**
1. If `system=` kwarg provided, prepend a `{"role": "system", "content": system}` message; error if a system role is already in `messages`.
2. Translate canonical `Message` list → OpenAI message dicts:
   - Assistant messages with `tool_calls` → include `tool_calls` field (parsed dict → JSON string for `arguments`).
   - `role="tool"` → `{"role": "tool", "tool_call_id": ..., "name": ..., "content": ...}`.
3. Translate `tools` (JSON Schema) → OpenAI's `[{"type": "function", "function": {...}}]` shape.
4. Call `openai.AsyncOpenAI(api_key=..., base_url=...).chat.completions.create(...)`.
5. Translate response → `LLMResponse`:
   - `choices[0].message.content` → `content` (may be `None`/empty when only tools called).
   - `choices[0].message.tool_calls` → `tool_calls` (parse each `function.arguments` JSON string into a dict).
   - Map `finish_reason`: `stop`→`end_turn`, `tool_calls`→`tool_use`, `length`→`max_tokens`, else→`other`.
   - Build `TokenUsage` from `response.usage`.

### Factory

```python
# DefenseAgent/llm/factory.py
def make_adapter_from_env(
    dotenv_path: str | None = None,
    *,
    load_env: bool = True,
) -> LLMAdapter:
    """Read env (optionally loading .env first) and return the configured adapter."""
```

**Env var structure:**

- **Selector:** `AGENT_LAB_LLM_PROVIDER` — one of `openai | anthropic | google | deepseek | qwen | vllm` (case-insensitive, trimmed).
- **Per-provider blocks:** `{PROVIDER}_API_KEY`, `{PROVIDER}_BASE_URL`, `{PROVIDER}_MODEL` where `{PROVIDER}` is the selector value uppercased (e.g. `OPENAI_API_KEY`, `ANTHROPIC_MODEL`, `DEEPSEEK_BASE_URL`).
- **Override tier (per field):** `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL_ID`. Each, if non-empty, overrides the corresponding provider-specific value. Evaluated independently — a user can override just the model without touching the key or base_url.

**Logic:**
1. If `load_env=True`, call `dotenv.load_dotenv(dotenv_path, override=False)`. Tests pass `load_env=False` to work from a clean controlled env.
2. Resolve `provider` from `AGENT_LAB_LLM_PROVIDER` (strip, lowercase). Empty or unknown → `LLMConfigError`.
3. For each field (api_key, base_url, model): take the `LLM_*` override if non-empty, otherwise the `{PROVIDER}_*` value. Empty string is treated as "not set".
4. Validate:
   - `model` empty → `LLMConfigError`.
   - `base_url` empty AND `provider in {google, qwen, deepseek, vllm}` → `LLMConfigError`.
   - `api_key` empty AND `provider != vllm` → `LLMConfigError`. (vLLM tolerates an empty key — defaults to `"token-not-needed"`, which OpenAI-compatible servers accept.)
5. Build:
   - `provider == anthropic` → `AnthropicAdapter(api_key=..., model=..., base_url=base_url or None)`.
   - Otherwise → `OpenAICompatibleAdapter(api_key=..., base_url=..., model=...)`. For `openai`, `base_url` may still be empty (SDK default).

**Extra env vars present but not wired into adapters yet** (reserved for later modules):
`AGENT_LAB_LLM_TEMPERATURE`, `LLM_MAX_TOKENS`, `LLM_TIMEOUT`, `LLM_MAX_RETRIES`, `TAVILY_API_KEY`, `AGENT_LAB_DISABLE_BACKGROUND_RUNTIME`, `AGENT_LAB_RUNTIME_AUTONOMOUS_A2A`, `AGENT_LAB_RUNTIME_LLM_DECISIONS`. They live in `.env.example` so users know the shape, but this module does not consume them.

### Errors

Defined in `DefenseAgent/llm/errors.py`:

```python
class LLMError(Exception): ...                  # base
class LLMConfigError(LLMError): ...             # bad/missing .env config
class LLMAdapterError(LLMError): ...            # adapter-layer bug or misuse
class LLMProviderError(LLMError):               # provider API returned an error
    def __init__(self, provider: str, status_code: int | None, message: str): ...
```

Adapters wrap provider exceptions in `LLMProviderError`, preserving original cause via `raise ... from e`.

### `.env` and `.env.example`

See the committed [`.env.example`](../../../.env.example) for the full template. Summary of sections:

- **Selector:** `AGENT_LAB_LLM_PROVIDER`.
- **Override tier:** `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL_ID`.
- **Provider blocks:** one per supported provider (`OPENAI_*`, `ANTHROPIC_*`, `GOOGLE_*`, `DEEPSEEK_*`, `QWEN_*`, `VLLM_*`).
- **Model-parameter defaults (not wired yet):** `AGENT_LAB_LLM_TEMPERATURE`, `LLM_MAX_TOKENS`, `LLM_TIMEOUT`, `LLM_MAX_RETRIES`.
- **Future-module placeholders:** `TAVILY_API_KEY`, `AGENT_LAB_DISABLE_BACKGROUND_RUNTIME`, `AGENT_LAB_RUNTIME_AUTONOMOUS_A2A`, `AGENT_LAB_RUNTIME_LLM_DECISIONS`.

`.env` is the same shape but git-ignored and populated with user secrets.

### File layout

```
agent_lab/
├── .env                           # gitignored, blank initially
├── .env.example                   # committed template
├── .gitignore                     # .env, __pycache__, .pytest_cache
├── requirements.txt               # anthropic, openai, python-dotenv, pytest, pytest-asyncio
├── docs/superpowers/specs/
│   └── 2026-04-22-module-01-llm-adapter-design.md   # this document
├── DefenseAgent/                 # all harness code lives here
│   ├── __init__.py
│   └── llm/
│       ├── __init__.py            # re-exports LLMAdapter, types, factory
│       ├── types.py
│       ├── errors.py
│       ├── llm_adapter.py         # abstract base
│       ├── anthropic_adapter.py
│       ├── openai_compatible_adapter.py
│       └── factory.py
└── tests/
    └── DefenseAgent/
        ├── __init__.py
        └── llm/
            ├── __init__.py
            ├── conftest.py        # shared fixtures
            ├── test_types.py
            ├── test_errors.py
            ├── test_llm_adapter.py
            ├── test_anthropic_adapter.py
            ├── test_openai_compatible_adapter.py
            └── test_factory.py
```

## Testing strategy

No real API calls in tests. Strategies:

- **`AnthropicAdapter` tests**: patch `anthropic.AsyncAnthropic` to return a stub object whose `messages.create` is an `AsyncMock` returning a hand-built response object. Assert the request built by the adapter (captured via the mock's call args) and the translated `LLMResponse`.
- **`OpenAICompatibleAdapter` tests**: same approach with `openai.AsyncOpenAI`.
- **Factory tests**: use `monkeypatch.setenv` to set `PROVIDER`/`API_KEY`/`BASE_URL`/`MODEL`, assert the right adapter type and its wired config. Also test every error path.
- **Type tests**: cheap sanity checks that the dataclasses round-trip and defaults are right.

Coverage goals for this module:
- Every public method of every adapter has at least one happy-path test.
- Message-format translation tested in both directions (input translation AND response translation).
- Tool-call path tested end-to-end for both adapters.
- Every `LLMConfigError` branch in the factory is tested.

## Dependencies this module adds

Added to `requirements.txt`:
```
anthropic>=0.40.0
openai>=1.50.0
python-dotenv>=1.0.0
pytest>=8.0.0
pytest-asyncio>=0.23.0
```

## Future extensions (not in this module)

When future modules need them, this interface grows:
- `async embed(text: str | list[str]) -> list[list[float]]` — added with the memory module.
- `async chat_stream(...) -> AsyncIterator[Chunk]` — added when UI or event-based consumers exist.
- `async score_importance(observation: str) -> float` — lives in the memory module as a wrapper over `chat()`; not an adapter method.

The canonical types are designed so these additions don't break callers.

## Open questions

None at spec-approval time. Design choices confirmed with the user on 2026-04-22:
1. One shared `OpenAICompatibleAdapter` class for OpenAI/Google/Qwen/DeepSeek/vLLM — confirmed.
2. `embed()` and `score_importance()` deferred — confirmed.
3. Env var structure: per-provider blocks plus a per-field `LLM_*` override tier (not all-or-nothing) — adopted after user supplied their preferred `.env` format.
4. Supported providers: `openai, anthropic, google, deepseek, qwen, vllm`. `claude` renamed to `anthropic` to match SDK name.

## Revision history

- **2026-04-22**: Switched factory from flat env vars (`PROVIDER/API_KEY/BASE_URL/MODEL`) to namespaced per-provider blocks with an override tier (`AGENT_LAB_LLM_PROVIDER`, `{PROVIDER}_*`, `LLM_*`). Added `google` provider. Renamed `claude` to `anthropic`. No change to adapter classes or canonical types.
