"""Internal: provider-name → adapter-class registry plus field validation.

Private module (leading underscore) — users construct LLMs via
`LLM.create(provider=...)` or `LLM.from_env()`, never by importing
from here.

Adapter classes are imported lazily inside `_resolve_adapter`: only the
provider actually used pays the import cost (and pulls in its SDK package).
"""
from typing import TYPE_CHECKING

from DefenseAgent.llm.errors import LLMConfigError

if TYPE_CHECKING:
    from DefenseAgent.llm.base import LLMAdapter


_SUPPORTED_PROVIDERS = ("openai", "anthropic", "google", "deepseek", "qwen", "vllm")
_BASE_URL_REQUIRED = {"google", "deepseek", "qwen", "vllm"}
_API_KEY_OPTIONAL = {"vllm"}
_VLLM_DEFAULT_KEY = "token-not-needed"


def _validate_provider(provider: str) -> None:
    """Raise LLMConfigError when `provider` is empty or not in the supported list."""
    if not provider:
        raise LLMConfigError(
            "provider is empty. "
            f"Supported values: {', '.join(_SUPPORTED_PROVIDERS)}."
        )
    if provider not in _SUPPORTED_PROVIDERS:
        raise LLMConfigError(
            f"provider={provider!r} is not supported. "
            f"Supported values: {', '.join(_SUPPORTED_PROVIDERS)}."
        )


def _validate_fields(
    provider: str,
    *,
    api_key: str,
    base_url: str,
    model: str,
) -> None:
    """Raise LLMConfigError when any provider-required field is missing."""
    if not model:
        raise LLMConfigError(
            f"model is required for provider {provider!r}."
        )
    if provider in _BASE_URL_REQUIRED and not base_url:
        raise LLMConfigError(
            f"base_url is required for provider {provider!r}."
        )
    if provider not in _API_KEY_OPTIONAL and not api_key:
        raise LLMConfigError(
            f"api_key is required for provider {provider!r}."
        )


def _resolve_adapter(provider: str) -> type["LLMAdapter"]:
    """Return the LLMAdapter subclass for `provider`, importing it lazily.

    Each branch's `from .xxx import ...` lives inside the if-block so users
    who only need one provider don't pay the import cost — or the pip-install
    cost — of the others.
    """
    if provider == "anthropic":
        from DefenseAgent.llm.anthropic import AnthropicAdapter
        return AnthropicAdapter
    if provider in ("openai", "google", "deepseek", "qwen", "vllm"):
        from DefenseAgent.llm.openai_compat import OpenAICompatibleAdapter
        return OpenAICompatibleAdapter
    raise LLMConfigError(
        f"no adapter registered for provider {provider!r}; "
        f"supported: {', '.join(_SUPPORTED_PROVIDERS)}."
    )
