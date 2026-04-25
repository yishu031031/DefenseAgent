class LLMError(Exception):
    """Base class for every error raised from the LLM module."""


class LLMConfigError(LLMError):
    """Raised when .env or adapter configuration is missing or invalid."""


class LLMAdapterError(LLMError):
    """Raised when the adapter is misused (e.g., conflicting system arguments)."""


class LLMProviderError(LLMError):
    """Raised when the provider API returned an error; original is chained via __cause__."""

    def __init__(self, provider: str, status_code: int | None, message: str) -> None:
        """Bind provider, optional HTTP status, and message into a readable string."""
        self.provider = provider
        self.status_code = status_code
        self.message = message
        status_part = f" (status {status_code})" if status_code is not None else ""
        super().__init__(f"[{provider}]{status_part} {message}")
