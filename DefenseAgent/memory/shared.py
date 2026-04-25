from typing import Any

from ms_agent.memory.memory_manager import SharedMemoryManager as MsSharedMemoryManager

from DefenseAgent.config.profile import AgentProfile
from DefenseAgent.memory._bridge import profile_to_dictconfig


class SharedMemoryManager(MsSharedMemoryManager):
    """Inherits ms-agent's process-wide singleton; adds a `from_profile` helper that translates our AgentProfile to its DictConfig."""

    @classmethod
    async def get_for_profile(
        cls,
        profile: AgentProfile,
        *,
        mem_instance_type: str = "default_memory",
        **profile_overrides: Any,
    ) -> Any:
        """Resolve the singleton keyed by (mem_instance_type, user_id, path); creates one from `profile` on first call."""
        config = profile_to_dictconfig(profile, **profile_overrides)
        return await cls.get_shared_memory(config, mem_instance_type)
