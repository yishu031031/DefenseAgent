import os
from typing import AsyncIterator

from dotenv import load_dotenv

from DefenseAgent.llm.base import LLMAdapter
from DefenseAgent.llm.errors import LLMConfigError
from DefenseAgent.llm.types import LLMResponse, Message, StreamChunk


_SUPPORTED_PROVIDERS = ("openai", "anthropic", "google", "deepseek", "qwen", "vllm")
_BASE_URL_REQUIRED = {"google", "deepseek", "qwen", "vllm"}
_API_KEY_OPTIONAL = {"vllm"}
_VLLM_DEFAULT_KEY = "token-not-needed"


class LLM:
    """Module 1's unified facade; wraps one concrete LLMAdapter and exposes chat() / chat_stream()."""

    def __init__(self, adapter: LLMAdapter) -> None:
        """Bind an already-constructed LLMAdapter for delegation."""
        self.adapter = adapter

    @classmethod
    def from_env(
        cls,
        dotenv_path: str | None = None,
        *,
        load_env: bool = True,
    ) -> "LLM":
        """Build an LLM by resolving AGENT_LAB_LLM_PROVIDER + per-provider env block from .env."""
        if load_env:
            load_dotenv(dotenv_path, override=False)

        provider = _resolve_provider()
        api_key, base_url, model = _resolve_fields(provider)
        _validate_fields(provider, api_key, base_url, model)

        if provider == "vllm" and not api_key:
            api_key = _VLLM_DEFAULT_KEY

        if provider == "anthropic":
            from DefenseAgent.llm.anthropic import AnthropicAdapter
            adapter: LLMAdapter = AnthropicAdapter(
                api_key=api_key,
                model=model,
                base_url=base_url if base_url else None,
            )
        else:
            from DefenseAgent.llm.openai_compat import OpenAICompatibleAdapter
            adapter = OpenAICompatibleAdapter(
                api_key=api_key,
                base_url=base_url,
                model=model,
            )
        return cls(adapter=adapter)

    async def chat(
        self,
        messages: list[Message],
        *,
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        system: str | None = None,
    ) -> LLMResponse:
        """Delegate to the wrapped adapter's chat()."""
        return await self.adapter.chat(
            messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            system=system,
        )

    def chat_stream(
        self,
        messages: list[Message],
        *,
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        system: str | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Delegate to the wrapped adapter's chat_stream()."""
        return self.adapter.chat_stream(
            messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            system=system,
        )


def _resolve_provider() -> str:
    """Read AGENT_LAB_LLM_PROVIDER and validate it against the supported set."""
    raw = os.environ.get("AGENT_LAB_LLM_PROVIDER", "")
    provider = raw.strip().lower()
    if not provider:
        raise LLMConfigError(
            "AGENT_LAB_LLM_PROVIDER is not set. "
            f"Supported values: {', '.join(_SUPPORTED_PROVIDERS)}."
        )
    if provider not in _SUPPORTED_PROVIDERS:
        raise LLMConfigError(
            f"AGENT_LAB_LLM_PROVIDER={raw!r} is not supported. "
            f"Supported values: {', '.join(_SUPPORTED_PROVIDERS)}."
        )
    return provider


def _resolve_fields(provider: str) -> tuple[str, str, str]:
    """Pick api_key / base_url / model using the LLM_* override tier then the {PROVIDER}_* fallback."""
    prefix = provider.upper()
    api_key = _pick_override(
        os.environ.get("LLM_API_KEY"),
        os.environ.get(f"{prefix}_API_KEY"),
    )
    base_url = _pick_override(
        os.environ.get("LLM_BASE_URL"),
        os.environ.get(f"{prefix}_BASE_URL"),
    )
    model = _pick_override(
        os.environ.get("LLM_MODEL_ID"),
        os.environ.get(f"{prefix}_MODEL"),
    )
    return api_key, base_url, model


def _pick_override(override: str | None, fallback: str | None) -> str:
    """Return `override` when non-empty, else `fallback` when set, else an empty string."""
    if override:
        return override
    if fallback:
        return fallback
    return ""


def _validate_fields(
    provider: str,
    api_key: str,
    base_url: str,
    model: str,
) -> None:
    """Raise LLMConfigError when any provider-required field is missing."""
    prefix = provider.upper()
    if not model:
        raise LLMConfigError(
            f"MODEL is not set for provider '{provider}'. "
            f"Set {prefix}_MODEL or LLM_MODEL_ID."
        )
    if provider in _BASE_URL_REQUIRED and not base_url:
        raise LLMConfigError(
            f"BASE_URL is required for provider '{provider}'. "
            f"Set {prefix}_BASE_URL or LLM_BASE_URL."
        )
    if provider not in _API_KEY_OPTIONAL and not api_key:
        raise LLMConfigError(
            f"API_KEY is not set for provider '{provider}'. "
            f"Set {prefix}_API_KEY or LLM_API_KEY."
        )
