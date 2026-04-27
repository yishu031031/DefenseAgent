"""Tests for DefenseAgent.llm.llm — canonical types, errors, abstract base, LLM facade.

Per-provider adapter tests live in test_anthropic.py and test_openai_compat.py.
"""
import pytest

from DefenseAgent.llm import (
    LLM,
    LLMConfigError,
    LLMError,
    LLMProviderError,
    LLMResponse,
    Message,
    StreamChunk,
    StreamEnd,
    TextDelta,
    TokenUsage,
    ToolCall,
)
from DefenseAgent.llm.anthropic import AnthropicAdapter
from DefenseAgent.llm.base import LLMAdapter
from DefenseAgent.llm.errors import LLMAdapterError
from DefenseAgent.llm.openai_compat import OpenAICompatibleAdapter


# ============================================================
# Canonical types
# ============================================================


def test_message_minimal_construction():
    msg = Message(role="user", content="hello")
    assert msg.role == "user"
    assert msg.content == "hello"
    assert msg.tool_calls == []
    assert msg.tool_call_id is None
    assert msg.name is None


def test_message_with_tool_calls():
    call = ToolCall(id="call_1", name="get_weather", arguments={"city": "SF"})
    msg = Message(role="assistant", content="", tool_calls=[call])
    assert msg.tool_calls[0].id == "call_1"
    assert msg.tool_calls[0].arguments == {"city": "SF"}


def test_message_tool_role():
    msg = Message(
        role="tool",
        content='{"temp": 72}',
        tool_call_id="call_1",
        name="get_weather",
    )
    assert msg.role == "tool"
    assert msg.tool_call_id == "call_1"
    assert msg.name == "get_weather"


def test_tool_call_arguments_is_dict_not_string():
    call = ToolCall(id="x", name="f", arguments={"k": "v"})
    assert isinstance(call.arguments, dict)


def test_token_usage_fields():
    usage = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 5
    assert usage.total_tokens == 15


def test_llm_response_minimal():
    resp = LLMResponse(
        content="hi", tool_calls=[],
        usage=TokenUsage(1, 1, 2),
        stop_reason="end_turn", raw={},
    )
    assert resp.content == "hi"
    assert resp.tool_calls == []
    assert resp.stop_reason == "end_turn"
    assert resp.usage.total_tokens == 2
    assert resp.raw == {}


def test_llm_response_tool_calls_preserved():
    call = ToolCall(id="c1", name="lookup", arguments={"q": "x"})
    resp = LLMResponse(
        content="", tool_calls=[call],
        usage=TokenUsage(1, 1, 2),
        stop_reason="tool_use", raw={"some": "payload"},
    )
    assert resp.tool_calls[0].name == "lookup"
    assert resp.stop_reason == "tool_use"


# ---- stream types ----


def test_text_delta_carries_text():
    assert TextDelta(text="hello").text == "hello"


def test_stream_end_carries_stop_and_usage():
    e = StreamEnd(
        stop_reason="end_turn",
        usage=TokenUsage(1, 2, 3),
        raw={"k": "v"},
    )
    assert e.stop_reason == "end_turn"
    assert e.usage.total_tokens == 3
    assert e.raw == {"k": "v"}


def test_stream_chunk_union_admits_both():
    chunks: list[StreamChunk] = [
        TextDelta(text="hi"),
        StreamEnd(stop_reason="end_turn", usage=TokenUsage(1, 1, 2), raw={}),
    ]
    assert isinstance(chunks[0], TextDelta)
    assert isinstance(chunks[1], StreamEnd)


# ============================================================
# Error hierarchy
# ============================================================


def test_all_errors_inherit_from_llm_error():
    assert issubclass(LLMConfigError, LLMError)
    assert issubclass(LLMAdapterError, LLMError)
    assert issubclass(LLMProviderError, LLMError)


def test_config_error_is_plain_exception():
    assert str(LLMConfigError("missing PROVIDER in .env")) == "missing PROVIDER in .env"


def test_adapter_error_is_plain_exception():
    assert "both" in str(LLMAdapterError("both messages system and system kwarg set"))


def test_provider_error_carries_context():
    err = LLMProviderError(provider="openai", status_code=429, message="rate limited")
    assert err.provider == "openai"
    assert err.status_code == 429
    assert "rate limited" in str(err)
    assert "openai" in str(err)


def test_provider_error_allows_none_status():
    err = LLMProviderError(provider="claude", status_code=None, message="connection refused")
    assert err.status_code is None


def test_provider_error_can_be_raised_from_cause():
    original = RuntimeError("network blew up")
    with pytest.raises(LLMProviderError) as excinfo:
        try:
            raise original
        except RuntimeError as e:
            raise LLMProviderError(
                provider="openai", status_code=None, message="wrapped",
            ) from e
    assert excinfo.value.__cause__ is original


# ============================================================
# Abstract LLMAdapter base
# ============================================================


def test_cannot_instantiate_abstract_adapter():
    with pytest.raises(TypeError):
        LLMAdapter()  # type: ignore[abstract]


async def test_subclass_implementing_chat_works():
    class StubAdapter(LLMAdapter):
        async def chat(self, messages, *, tools=None, temperature=0.7,
                       max_tokens=1024, system=None):
            return LLMResponse(
                content="ok", tool_calls=[],
                usage=TokenUsage(1, 1, 2),
                stop_reason="end_turn", raw={},
            )

    resp = await StubAdapter().chat([Message(role="user", content="hi")])
    assert resp.content == "ok"
    assert resp.stop_reason == "end_turn"


def test_subclass_missing_chat_cannot_instantiate():
    class BrokenAdapter(LLMAdapter):
        pass

    with pytest.raises(TypeError):
        BrokenAdapter()  # type: ignore[abstract]


# ============================================================
# chat_stream default fallback (only chat() implemented)
# ============================================================


class _NoStreamAdapter(LLMAdapter):
    """Implements chat() only — exercises the default chat_stream fallback."""

    async def chat(self, messages, *, tools=None, temperature=0.7,
                   max_tokens=1024, system=None):
        return LLMResponse(
            content="hello world", tool_calls=[],
            usage=TokenUsage(10, 3, 13),
            stop_reason="end_turn", raw={"source": "stub"},
        )


async def test_default_chat_stream_yields_one_delta_then_end():
    chunks = [c async for c in _NoStreamAdapter().chat_stream(
        [Message(role="user", content="hi")],
    )]
    assert len(chunks) == 2
    assert isinstance(chunks[0], TextDelta)
    assert chunks[0].text == "hello world"
    assert isinstance(chunks[1], StreamEnd)
    assert chunks[1].stop_reason == "end_turn"
    assert chunks[1].usage.total_tokens == 13


async def test_default_chat_stream_skips_empty_delta():
    class _EmptyAdapter(LLMAdapter):
        async def chat(self, messages, **kwargs):
            return LLMResponse(
                content="", tool_calls=[],
                usage=TokenUsage(1, 0, 1),
                stop_reason="end_turn", raw={},
            )

    chunks = [c async for c in _EmptyAdapter().chat_stream(
        [Message(role="user", content="x")],
    )]
    assert len(chunks) == 1
    assert isinstance(chunks[0], StreamEnd)


# ============================================================
# LLM facade — construction & chat() delegation
# ============================================================


class _StubAdapter(LLMAdapter):
    def __init__(self, canned: str = "hi"):
        self.canned = canned
        self.calls: list[dict] = []

    async def chat(self, messages, *, tools=None, temperature=0.7,
                   max_tokens=1024, system=None):
        self.calls.append(
            {
                "messages": messages, "tools": tools,
                "temperature": temperature, "max_tokens": max_tokens,
                "system": system,
            }
        )
        return LLMResponse(
            content=self.canned, tool_calls=[],
            usage=TokenUsage(5, 2, 7),
            stop_reason="end_turn", raw={},
        )


def test_construct_from_adapter():
    adapter = _StubAdapter()
    llm = LLM(adapter)
    assert llm.adapter is adapter


async def test_chat_delegates():
    adapter = _StubAdapter(canned="hello")
    llm = LLM(adapter)
    resp = await llm.chat([Message(role="user", content="hi")])
    assert resp.content == "hello"
    call = adapter.calls[0]
    assert call["messages"][0].content == "hi"
    assert call["temperature"] == 0.7
    assert call["max_tokens"] == 1024


async def test_chat_forwards_all_kwargs():
    adapter = _StubAdapter()
    llm = LLM(adapter)
    await llm.chat(
        [Message(role="user", content="x")],
        system="Be brief.", temperature=0.2, max_tokens=500,
        tools=[{"name": "t", "description": "", "parameters": {}}],
    )
    call = adapter.calls[0]
    assert call["system"] == "Be brief."
    assert call["temperature"] == 0.2
    assert call["max_tokens"] == 500
    assert len(call["tools"]) == 1


# ============================================================
# LLM.from_env — env resolution
# ============================================================


_ENV_VARS = [
    "AGENT_LAB_LLM_PROVIDER",
    "LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL_ID",
    "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL",
    "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "ANTHROPIC_MODEL",
    "GOOGLE_API_KEY", "GOOGLE_BASE_URL", "GOOGLE_MODEL",
    "DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL",
    "QWEN_API_KEY", "QWEN_BASE_URL", "QWEN_MODEL",
    "VLLM_API_KEY", "VLLM_BASE_URL", "VLLM_MODEL",
]


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    for v in _ENV_VARS:
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr("DefenseAgent.llm.llm.load_dotenv", lambda *a, **kw: None)
    yield


def _set(monkeypatch, **kv):
    for k, v in kv.items():
        monkeypatch.setenv(k, v)


# ---- happy paths per provider ----


def test_from_env_openai(monkeypatch):
    _set(monkeypatch,
         AGENT_LAB_LLM_PROVIDER="openai",
         OPENAI_API_KEY="sk-oai", OPENAI_MODEL="gpt-4o-mini")
    llm = LLM.from_env()
    assert isinstance(llm.adapter, OpenAICompatibleAdapter)
    assert llm.adapter.model == "gpt-4o-mini"


def test_from_env_anthropic(monkeypatch):
    _set(monkeypatch,
         AGENT_LAB_LLM_PROVIDER="anthropic",
         ANTHROPIC_API_KEY="sk-ant", ANTHROPIC_MODEL="claude-sonnet-4-6")
    llm = LLM.from_env()
    assert isinstance(llm.adapter, AnthropicAdapter)
    assert llm.adapter.model == "claude-sonnet-4-6"


def test_from_env_google(monkeypatch):
    _set(monkeypatch,
         AGENT_LAB_LLM_PROVIDER="google",
         GOOGLE_API_KEY="sk-g",
         GOOGLE_BASE_URL="https://openrouter.ai/api/v1",
         GOOGLE_MODEL="google/gemini-2.5-pro")
    assert isinstance(LLM.from_env().adapter, OpenAICompatibleAdapter)


def test_from_env_deepseek(monkeypatch):
    _set(monkeypatch,
         AGENT_LAB_LLM_PROVIDER="deepseek",
         DEEPSEEK_API_KEY="sk-ds",
         DEEPSEEK_BASE_URL="https://api.deepseek.com",
         DEEPSEEK_MODEL="deepseek-chat")
    assert isinstance(LLM.from_env().adapter, OpenAICompatibleAdapter)


def test_from_env_qwen(monkeypatch):
    _set(monkeypatch,
         AGENT_LAB_LLM_PROVIDER="qwen",
         QWEN_API_KEY="sk-q",
         QWEN_BASE_URL="https://dashscope.example.com/v1",
         QWEN_MODEL="qwen-max")
    assert isinstance(LLM.from_env().adapter, OpenAICompatibleAdapter)


def test_from_env_vllm_with_blank_api_key(monkeypatch):
    _set(monkeypatch,
         AGENT_LAB_LLM_PROVIDER="vllm",
         VLLM_BASE_URL="http://localhost:8000/v1",
         VLLM_MODEL="Qwen/Qwen2.5-72B-Instruct")
    assert isinstance(LLM.from_env().adapter, OpenAICompatibleAdapter)


def test_from_env_provider_case_insensitive(monkeypatch):
    _set(monkeypatch,
         AGENT_LAB_LLM_PROVIDER="  Anthropic  ",
         ANTHROPIC_API_KEY="k", ANTHROPIC_MODEL="m")
    assert isinstance(LLM.from_env().adapter, AnthropicAdapter)


# ---- per-field override tier ----


def test_llm_api_key_override_wins(monkeypatch):
    _set(monkeypatch,
         AGENT_LAB_LLM_PROVIDER="openai",
         OPENAI_API_KEY="provider-key", OPENAI_MODEL="m",
         LLM_API_KEY="override-wins")
    assert LLM.from_env().adapter._client.api_key == "override-wins"


def test_llm_model_override_wins(monkeypatch):
    _set(monkeypatch,
         AGENT_LAB_LLM_PROVIDER="openai",
         OPENAI_API_KEY="k", OPENAI_MODEL="gpt-4o-mini",
         LLM_MODEL_ID="gpt-4o")
    assert LLM.from_env().adapter.model == "gpt-4o"


def test_empty_override_is_ignored(monkeypatch):
    _set(monkeypatch,
         AGENT_LAB_LLM_PROVIDER="openai",
         OPENAI_API_KEY="provider-key", OPENAI_MODEL="provider-model",
         LLM_API_KEY="", LLM_MODEL_ID="")
    llm = LLM.from_env()
    assert llm.adapter._client.api_key == "provider-key"
    assert llm.adapter.model == "provider-model"


def test_per_field_override_is_independent(monkeypatch):
    _set(monkeypatch,
         AGENT_LAB_LLM_PROVIDER="openai",
         OPENAI_API_KEY="provider-key", OPENAI_MODEL="provider-model",
         LLM_MODEL_ID="override-model")
    llm = LLM.from_env()
    assert llm.adapter._client.api_key == "provider-key"
    assert llm.adapter.model == "override-model"


# ---- error branches ----


def test_from_env_empty_provider_raises():
    with pytest.raises(LLMConfigError) as e:
        LLM.from_env()
    assert "AGENT_LAB_LLM_PROVIDER" in str(e.value)


def test_from_env_unknown_provider_lists_supported(monkeypatch):
    _set(monkeypatch, AGENT_LAB_LLM_PROVIDER="gemini")
    with pytest.raises(LLMConfigError) as e:
        LLM.from_env()
    for p in ("openai", "anthropic", "google", "deepseek", "qwen", "vllm"):
        assert p in str(e.value)


def test_from_env_missing_model_raises(monkeypatch):
    _set(monkeypatch, AGENT_LAB_LLM_PROVIDER="openai", OPENAI_API_KEY="k")
    with pytest.raises(LLMConfigError) as e:
        LLM.from_env()
    assert "MODEL" in str(e.value).upper()


def test_from_env_missing_api_key_for_non_vllm_raises(monkeypatch):
    _set(monkeypatch, AGENT_LAB_LLM_PROVIDER="openai", OPENAI_MODEL="m")
    with pytest.raises(LLMConfigError) as e:
        LLM.from_env()
    assert "API_KEY" in str(e.value).upper()


@pytest.mark.parametrize("provider,prefix", [
    ("google", "GOOGLE"),
    ("deepseek", "DEEPSEEK"),
    ("qwen", "QWEN"),
    ("vllm", "VLLM"),
])
def test_from_env_missing_base_url_raises(monkeypatch, provider, prefix):
    _set(monkeypatch,
         AGENT_LAB_LLM_PROVIDER=provider,
         **{f"{prefix}_API_KEY": "k", f"{prefix}_MODEL": "m"})
    with pytest.raises(LLMConfigError) as e:
        LLM.from_env()
    assert "BASE_URL" in str(e.value).upper()


def test_openai_allows_empty_base_url(monkeypatch):
    _set(monkeypatch,
         AGENT_LAB_LLM_PROVIDER="openai",
         OPENAI_API_KEY="k", OPENAI_MODEL="m")
    assert isinstance(LLM.from_env().adapter, OpenAICompatibleAdapter)


# ============================================================
# LLM.from_profile — profile-then-env per-field resolution
# ============================================================


def _make_profile(**llm_kwargs) -> "object":
    """Build an in-memory AgentProfile whose `llm` block carries `llm_kwargs`. Imported lazily so this test file doesn't pull config types in unless we hit this section."""
    from DefenseAgent.config import AgentProfile, LLMConfig
    return AgentProfile(
        id="test", name="T", age=20, traits="t",
        backstory="b", initial_plan="p",
        llm=LLMConfig(**llm_kwargs),
    )


def test_from_profile_uses_profile_when_fully_populated(monkeypatch):
    """Empty env, fully-populated profile → all fields come from the profile."""
    profile = _make_profile(
        provider="openai",
        api_key="sk-from-profile",
        base_url="https://from-profile.example/v1",
        model="gpt-from-profile",
    )
    llm = LLM.from_profile(profile)
    assert isinstance(llm.adapter, OpenAICompatibleAdapter)
    assert llm.adapter.model == "gpt-from-profile"


def test_from_profile_falls_back_to_env_when_profile_is_empty(monkeypatch):
    """Profile with no llm block → behaves identically to from_env."""
    _set(
        monkeypatch,
        AGENT_LAB_LLM_PROVIDER="openai",
        OPENAI_API_KEY="sk-from-env",
        OPENAI_MODEL="gpt-from-env",
    )
    profile = _make_profile()
    llm = LLM.from_profile(profile)
    assert isinstance(llm.adapter, OpenAICompatibleAdapter)
    assert llm.adapter.model == "gpt-from-env"


def test_from_profile_partial_overrides_env_per_field(monkeypatch):
    """Profile sets `provider` and `model`; env supplies `api_key` and `base_url` for that provider."""
    _set(
        monkeypatch,
        AGENT_LAB_LLM_PROVIDER="anthropic",  # overridden by profile
        ANTHROPIC_API_KEY="sk-anthropic-from-env",
        ANTHROPIC_BASE_URL="https://anthropic.from-env",
        ANTHROPIC_MODEL="claude-from-env",  # overridden by profile
        OPENAI_API_KEY="sk-openai-from-env",
        OPENAI_BASE_URL="https://openai.from-env",
        OPENAI_MODEL="gpt-from-env",
    )
    profile = _make_profile(provider="openai", model="gpt-from-profile")
    llm = LLM.from_profile(profile)
    assert isinstance(llm.adapter, OpenAICompatibleAdapter)
    assert llm.adapter.model == "gpt-from-profile"
    # api_key and base_url come from env (under the OPENAI_* block, not ANTHROPIC_*).
    assert llm.adapter._client.api_key == "sk-openai-from-env"


def test_from_profile_provider_winner_drives_env_lookup_block(monkeypatch):
    """When profile picks a different provider than env, the per-provider env keys for the *profile's* provider are used."""
    _set(
        monkeypatch,
        AGENT_LAB_LLM_PROVIDER="anthropic",
        ANTHROPIC_API_KEY="sk-ant-env",
        ANTHROPIC_MODEL="claude-env",
        DEEPSEEK_API_KEY="sk-ds-env",
        DEEPSEEK_BASE_URL="https://api.deepseek.com/v1",
        DEEPSEEK_MODEL="deepseek-env",
    )
    profile = _make_profile(provider="deepseek")
    llm = LLM.from_profile(profile)
    assert isinstance(llm.adapter, OpenAICompatibleAdapter)
    assert llm.adapter.model == "deepseek-env"


def test_from_profile_none_is_equivalent_to_from_env(monkeypatch):
    _set(
        monkeypatch,
        AGENT_LAB_LLM_PROVIDER="openai",
        OPENAI_API_KEY="sk-x", OPENAI_MODEL="gpt-x",
    )
    via_profile = LLM.from_profile(None)
    via_env = LLM.from_env()
    assert type(via_profile.adapter) is type(via_env.adapter)
    assert via_profile.adapter.model == via_env.adapter.model


def test_from_profile_raises_when_neither_profile_nor_env_has_provider(monkeypatch):
    profile = _make_profile()
    with pytest.raises(LLMConfigError):
        LLM.from_profile(profile)


def test_from_profile_rejects_invalid_provider(monkeypatch):
    profile = _make_profile(provider="not-a-real-provider", api_key="x", model="y")
    with pytest.raises(LLMConfigError):
        LLM.from_profile(profile)


def test_from_profile_treats_blank_strings_as_unset(monkeypatch):
    """Profile fields are stripped; whitespace-only values fall back to env (so a half-edited YAML doesn't override good env state)."""
    _set(
        monkeypatch,
        AGENT_LAB_LLM_PROVIDER="openai",
        OPENAI_API_KEY="sk-from-env",
        OPENAI_MODEL="gpt-from-env",
    )
    profile = _make_profile(provider="   ", model="   ")
    llm = LLM.from_profile(profile)
    assert isinstance(llm.adapter, OpenAICompatibleAdapter)
    assert llm.adapter.model == "gpt-from-env"


# ============================================================
# LLM facade — chat_stream delegation
# ============================================================


async def test_llm_chat_stream_delegates():
    """LLM.chat_stream must forward to adapter.chat_stream (uses the default fallback)."""
    llm = LLM(_NoStreamAdapter())
    texts = []
    final = None
    async for c in llm.chat_stream([Message(role="user", content="hi")]):
        if isinstance(c, TextDelta):
            texts.append(c.text)
        elif isinstance(c, StreamEnd):
            final = c
    assert "".join(texts) == "hello world"
    assert final is not None
    assert final.stop_reason == "end_turn"
