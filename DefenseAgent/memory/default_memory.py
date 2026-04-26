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


class _Mem0AddProxy:
    """Wraps a mem0 client so `add()` routes `memory_type` into `metadata` and pins `infer=False`; every other attribute proxies through to the underlying client unchanged. Lets DefenseAgent enforce mem0-level conventions without monkey-patching the live `mem0.Memory.add` attribute (which would clobber MagicMock attributes in tests)."""

    def __init__(self, underlying: Any) -> None:
        """Capture the wrapped mem0 client without touching its attributes."""
        object.__setattr__(self, "_underlying", underlying)

    def add(self, messages: Any, **kwargs: Any) -> Any:
        """Route `memory_type=X` → `metadata={'memory_type': X}` (mem0's native field only accepts 'procedural_memory'), default `infer=False`, then delegate to the underlying client's `add`."""
        memory_type_arg = kwargs.pop("memory_type", None)
        if memory_type_arg and "metadata" not in kwargs:
            kwargs["metadata"] = {"memory_type": memory_type_arg}
        kwargs.setdefault("infer", False)
        return self._underlying.add(messages, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Forward any attribute we don't override to the wrapped client (search, get_all, delete, _create_memory, etc.)."""
        return getattr(self._underlying, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Forward attribute assignments (e.g. ms-agent setting `_telemetry_vector_store`) to the underlying client so its internal state stays consistent."""
        if name == "_underlying":
            object.__setattr__(self, name, value)
        else:
            setattr(self._underlying, name, value)


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
        self._wrap_mem0_add()
        self.profile = profile

    def _init_memory_obj(self):
        """Build mem0.Memory directly from the DictConfig's `mem0_config` block, bypassing ms-agent's hardcoded service-URL translation so custom embedder/LLM endpoints (OpenRouter, vLLM, etc.) are honoured."""
        import mem0
        from omegaconf import OmegaConf
        cfg = OmegaConf.to_container(self.config.mem0_config, resolve=True)
        return mem0.Memory.from_config(cfg)

    def _wrap_mem0_add(self) -> None:
        """Replace `self.memory` with a thin proxy whose `add()` enforces DefenseAgent's mem0-level conventions no matter who calls it (including ms-agent's inherited `add_single`): custom `memory_type` values are tucked into `metadata` (mem0's native `memory_type` kwarg only accepts 'procedural_memory'), and `infer=False` is the default so trajectories are stored verbatim. Wrapping here lets us delegate the full add()/add_single() pipeline to ms-agent (block hashing, history_mode, cache persistence) without duplicating its ~50 lines. Every other attribute (search, get_all, delete, ...) is proxied through unchanged via `__getattr__`."""
        if self.memory is None:
            return
        self.memory = _Mem0AddProxy(self.memory)

    async def run(self, messages: list[Message], **kwargs: Any) -> list[Message]:
        """Passthrough in DefenseAgent — ms-agent's run() injects retrieved memories as a `system` message, which collides with `BaseAgent._build_system_prompt()` already passing identity via the LLM's `system=` kwarg. The LLM accesses memory explicitly through the built-in `memory_recall` tool instead."""
        return messages

    async def add(
        self,
        messages: list[Message],
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        memory_type: str | None = None,
    ) -> None:
        """Convert at the Message boundary, then defer to ms-agent's `add()` so its block-hash dedup (`_analyze_messages`) and `history_mode` (add vs overwrite, sourced from `profile.memory.history_mode`) actually take effect. DefenseAgent's mem0-level tweaks — routing custom `memory_type` through `metadata` and pinning `infer=False` for verbatim storage — are applied transparently by `_wrap_mem0_add()` in `__init__`. ignore_roles filtering is handled by ms-agent's inherited `parse_messages()` per block."""
        ms_messages = messages_ours_to_theirs(messages)
        await super().add(
            ms_messages,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            memory_type=memory_type,
        )

    def search(self, query: str, meta_infos: list[dict[str, Any]] | None = None) -> list[str]:
        """Override ms-agent's search() with mem0's new `filters=` API and return the same list[str] shape so inherited callers (notably ms-agent's `run()`) keep working."""
        if not query:
            return []
        if meta_infos is None:
            meta_infos = [{}]
        out: list[str] = []
        for info in meta_infos:
            filters = {
                "user_id": info.get("user_id") or self.user_id,
                "agent_id": info.get("agent_id") or self.agent_id,
                "run_id": info.get("run_id") or self.run_id,
            }
            limit = info.get("limit", self.search_limit)
            response = self.memory.search(query, filters=filters, limit=limit)
            results = response.get("results", []) if isinstance(response, dict) else []
            out.extend(r.get("memory", "") for r in results)
        return out

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
        filters = {
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "run_id": self.run_id,
        }
        response = self.memory.search(
            query,
            filters=filters,
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
        """Pull every record under the (user_id, agent_id, run_id) tuple via mem0's `filters=` API, optionally narrowed by `memory_type`."""
        filters = {
            "user_id": user_id or self.user_id,
            "agent_id": agent_id or self.agent_id,
            "run_id": run_id or self.run_id,
        }
        response = self.memory.get_all(filters=filters)
        results = response.get("results", []) if isinstance(response, dict) else []
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


