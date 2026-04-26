import os
from typing import AsyncIterator

from dotenv import load_dotenv

from DefenseAgent.llm._registry import (
    _resolve_adapter,
    _validate_fields,
    _validate_provider,
    _VLLM_DEFAULT_KEY,
)
from DefenseAgent.llm.base import LLMAdapter
from DefenseAgent.llm.errors import LLMConfigError
from DefenseAgent.llm.types import LLMResponse, Message, StreamChunk


class LLM:
    """Module 1's unified facade; wraps one concrete LLMAdapter and exposes chat() / chat_stream()."""

    def __init__(self, adapter: LLMAdapter) -> None:
        """Bind an already-constructed LLMAdapter for delegation."""
        self.adapter = adapter

    @classmethod
    def from_kwargs(
        cls,
        *,
        provider: str,
        model: str,
        api_key: str = "",
        base_url: str | None = None,
    ) -> "LLM":
        """Build an LLM from explicit arguments — the canonical instantiation path.

        `from_env` itself delegates here after parsing the .env file, so this is
        the single source of truth for "how to construct an LLM". SDK callers,
        tests, multi-LLM apps, and anyone who needs to bypass global env state
        should call this directly.
        """
        provider = (provider or "").strip().lower()
        _validate_provider(provider)
        _validate_fields(
            provider,
            api_key=api_key,
            base_url=base_url or "",
            model=model,
        )

        if provider == "vllm" and not api_key:
            api_key = _VLLM_DEFAULT_KEY

        adapter_cls = _resolve_adapter(provider)
        if provider == "anthropic":
            adapter: LLMAdapter = adapter_cls(
                api_key=api_key,
                model=model,
                base_url=base_url if base_url else None,
            )
        else:
            adapter = adapter_cls(
                api_key=api_key,
                base_url=base_url or "",
                model=model,
            )
        return cls(adapter=adapter)

    @classmethod
    def from_env(
        cls,
        dotenv_path: str | None = None,
        *,
        load_env: bool = True,
    ) -> "LLM":
        """Build an LLM by resolving AGENT_LAB_LLM_PROVIDER + per-provider env block from .env.

        Convenience wrapper: parses environment variables, then defers all
        actual instantiation to `from_kwargs`.
        """
        if load_env:
            load_dotenv(dotenv_path, override=False)

        provider = _resolve_provider_from_env()
        api_key, base_url, model = _resolve_fields_from_env(provider)
        return cls.from_kwargs(
            provider=provider,
            api_key=api_key,
            base_url=base_url or None,
            model=model,
        )

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


def _resolve_provider_from_env() -> str:
    """Read AGENT_LAB_LLM_PROVIDER from the environment, normalize, and return.

    Raises LLMConfigError when unset; full provider-list validation is deferred
    to `_validate_provider` inside `from_kwargs` so the supported-list message
    has one canonical home.
    """
    raw = os.environ.get("AGENT_LAB_LLM_PROVIDER", "")
    provider = raw.strip().lower()
    if not provider:
        raise LLMConfigError(
            "AGENT_LAB_LLM_PROVIDER is not set. "
            "Set it in your .env, or use LLM.from_kwargs(provider=...) directly."
        )
    return provider


def _resolve_fields_from_env(provider: str) -> tuple[str, str, str]:
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
    """Return `override` when non-empty, else `fallback`, else an empty string."""
    return override or fallback or ""
