import os
from typing import TYPE_CHECKING, AsyncIterator

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


if TYPE_CHECKING:
    from DefenseAgent.config.profile import AgentProfile


class LLM:
    """Module 1's unified facade; wraps one concrete LLMAdapter and exposes chat() / chat_stream()."""

    def __init__(self, adapter: LLMAdapter) -> None:
        """Bind an already-constructed LLMAdapter for delegation."""
        self.adapter = adapter

    @classmethod
    def create(
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
        actual instantiation to `create`.
        """
        return cls.from_profile(profile=None, dotenv_path=dotenv_path, load_env=load_env)

    @classmethod
    def from_profile(
        cls,
        profile: "AgentProfile | None",
        *,
        dotenv_path: str | None = None,
        load_env: bool = True,
    ) -> "LLM":
        """Build an LLM with profile values taking precedence over .env, per field. When `profile.llm` populates a field that field is used; missing fields fall back to env (`AGENT_LAB_LLM_PROVIDER`, `<PROVIDER>_API_KEY`, `<PROVIDER>_BASE_URL`, `<PROVIDER>_MODEL`, with the cross-provider `LLM_*` tier as the second-tier env fallback). Pass `profile=None` for pure env-driven construction (equivalent to `from_env`)."""
        if load_env:
            load_dotenv(dotenv_path, override=False)

        llm_cfg = profile.llm if profile is not None else None
        provider = _profile_or_env(
            llm_cfg.provider if llm_cfg is not None else None,
            os.environ.get("AGENT_LAB_LLM_PROVIDER"),
        )
        if not provider:
            raise LLMConfigError(
                "AGENT_LAB_LLM_PROVIDER is not set and profile.llm.provider is empty. "
                "Set one in your .env or in the agent profile."
            )
        provider = provider.strip().lower()

        env_api_key, env_base_url, env_model = _resolve_fields_from_env(provider)
        api_key = _profile_or_env(
            llm_cfg.api_key if llm_cfg is not None else None,
            env_api_key,
        )
        base_url = _profile_or_env(
            llm_cfg.base_url if llm_cfg is not None else None,
            env_base_url,
        )
        model = _profile_or_env(
            llm_cfg.model if llm_cfg is not None else None,
            env_model,
        )
        return cls.create(
            provider=provider,
            api_key=api_key or "",
            base_url=base_url or None,
            model=model or "",
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


def _profile_or_env(profile_value: str | None, env_value: str | None) -> str | None:
    """Profile value wins when set + non-empty; else fall back to env. Returns None when both are empty so callers can decide whether the missing field is fatal."""
    if profile_value:
        stripped = profile_value.strip()
        if stripped:
            return stripped
    if env_value:
        stripped = env_value.strip()
        if stripped:
            return stripped
    return None


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
