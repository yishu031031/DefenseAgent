from typing import Any

from dotenv import load_dotenv

from ms_agent.memory.condenser.context_compressor import (
    ContextCompressor as MsContextCompressor,
)

from DefenseAgent.config.profile import AgentProfile
from DefenseAgent.llm.types import Message
from DefenseAgent.memory._bridge import (
    MemoryBackendConfig,
    messages_ours_to_theirs,
    messages_theirs_to_ours,
    profile_to_dictconfig,
)


class ContextCompressor(MsContextCompressor):
    """Inherits ms-agent's ContextCompressor; takes our AgentProfile and converts at the Message boundary on `run()`."""

    def __init__(
        self,
        profile: AgentProfile,
        *,
        load_env: bool = True,
        dotenv_path: str | None = None,
        storage_path: str | None = None,
        backend: MemoryBackendConfig | None = None,
    ) -> None:
        """Build the ms-agent DictConfig from `profile` (+ optional backend kwargs or .env), then defer to ms_agent.ContextCompressor's `__init__`. `storage_path` is forwarded to `profile_to_dictconfig` so in-memory profiles (no `source_dir`) can opt in by passing it explicitly. When `backend=` is given, env vars are skipped."""
        if load_env and backend is None:
            load_dotenv(dotenv_path, override=False)
        config = profile_to_dictconfig(
            profile, backend=backend, storage_path=storage_path,
        )
        super().__init__(config)
        self.profile = profile

    async def run(self, messages: list[Message], **kwargs: Any) -> list[Message]:
        """Convert our Messages → ms-agent's, defer to super().run() (which compacts), convert the result back."""
        ms_messages = messages_ours_to_theirs(messages)
        ms_result = await super().run(ms_messages, **kwargs)
        return messages_theirs_to_ours(ms_result)

    @classmethod
    def from_profile(
        cls,
        profile: AgentProfile,
        **kwargs: Any,
    ) -> "ContextCompressor":
        """Convenience constructor matching DefenseAgent's `from_profile` pattern across modules."""
        return cls(profile, **kwargs)
