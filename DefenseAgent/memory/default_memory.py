from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from ms_agent.memory.default_memory import DefaultMemory as MsDefaultMemory

from DefenseAgent.config.profile import AgentProfile
from DefenseAgent.llm.types import Message
from DefenseAgent.memory._bridge import (
    messages_ours_to_theirs,
    messages_theirs_to_ours,
    profile_to_dictconfig,
    record_memory_type,
)


class DefaultMemory(MsDefaultMemory):
    """Inherits ms-agent's mem0-backed DefaultMemory; takes our AgentProfile and converts at the Message boundary on `run()` / `add()`."""

    def __init__(
        self,
        profile: AgentProfile,
        *,
        user_id: str = "default_user",
        agent_id: str | None = None,
        run_id: str = "default_run",
        storage_path: str | Path | None = None,
        load_env: bool = True,
        dotenv_path: str | None = None,
    ) -> None:
        """Build the ms-agent DictConfig from `profile` + .env, ensure the storage dir exists, then defer to ms_agent.DefaultMemory's `__init__`."""
        if load_env:
            load_dotenv(dotenv_path, override=False)
        config = profile_to_dictconfig(
            profile,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            storage_path=storage_path,
        )
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        super().__init__(config)
        self.profile = profile

    async def run(self, messages: list[Message], **kwargs: Any) -> list[Message]:
        """Convert our Messages → ms-agent Messages, defer to super().run(), convert the result back."""
        ms_messages = messages_ours_to_theirs(messages)
        ms_result = await super().run(ms_messages, **kwargs)
        return messages_theirs_to_ours(ms_result)

    async def add(
        self,
        messages: list[Message],
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        memory_type: str | None = None,
    ) -> None:
        """Convert our Messages → ms-agent's at the boundary, then defer to super().add() for ingestion."""
        ms_messages = messages_ours_to_theirs(messages)
        await super().add(
            ms_messages,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            memory_type=memory_type,
        )

    def search_records(
        self,
        query: str,
        *,
        limit: int | None = None,
        memory_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return mem0 records as dicts (the ms-agent `search()` collapses to str — this preserves the full record for DefenseAgent callers)."""
        if not query:
            return []
        response = self.memory.search(
            query,
            user_id=self.user_id,
            agent_id=self.agent_id,
            run_id=self.run_id,
            limit=limit if limit is not None else self.search_limit,
        )
        results = response.get("results", []) if isinstance(response, dict) else []
        if memory_type is not None:
            results = [r for r in results if record_memory_type(r) == memory_type]
        return results

    def get_all(
        self,
        *,
        memory_type: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Override ms-agent's `get_all` to add an optional `memory_type` filter; otherwise defers to super()."""
        results = super().get_all(
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
        )
        if memory_type is not None:
            results = [r for r in results if record_memory_type(r) == memory_type]
        return results

    @classmethod
    def from_profile(
        cls,
        profile: AgentProfile,
        **kwargs: Any,
    ) -> "DefaultMemory":
        """Convenience constructor matching the rest of DefenseAgent's `from_profile` pattern."""
        return cls(profile, **kwargs)


