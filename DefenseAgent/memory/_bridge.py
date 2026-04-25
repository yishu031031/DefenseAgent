import os
import re
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from ms_agent.llm.utils import Message as MsMessage

from DefenseAgent.config.profile import AgentProfile
from DefenseAgent.llm.types import Message as OurMessage
from DefenseAgent.llm.types import ToolCall


def profile_to_dictconfig(
    profile: AgentProfile,
    *,
    user_id: str = "default_user",
    agent_id: str | None = None,
    run_id: str = "default_run",
    storage_path: str | Path | None = None,
) -> Any:
    """Translate our pydantic AgentProfile into the OmegaConf DictConfig that ms-agent's Memory subclasses read; carries the ready-to-use `mem0_config` dict so DefenseAgent.DefaultMemory can bypass ms-agent's hardcoded service-URL translation."""
    resolved_path = _resolve_storage_path(profile, storage_path)
    resolved_agent_id = agent_id or profile.id
    storage_dir = str(resolved_path / "default_memory")
    llm_cfg = _llm_config_from_env()
    embedder_cfg = _embedder_config_from_env()
    return OmegaConf.create({
        "output_dir": str(resolved_path),
        "compress": True,
        "is_retrieve": profile.memory.is_retrieve,
        "memory": {
            "default_memory": {
                "user_id": user_id,
                "agent_id": resolved_agent_id,
                "run_id": run_id,
                "history_mode": profile.memory.history_mode,
                "ignore_roles": list(profile.memory.ignore_roles),
                "ignore_fields": list(profile.memory.ignore_fields),
                "search_limit": profile.memory.search_limit,
                "path": storage_dir,
            },
            "context_compressor": {
                "context_limit": profile.memory.context_limit,
                "prune_protect": profile.memory.prune_protect,
                "prune_minimum": profile.memory.prune_minimum,
                "reserved_buffer": profile.memory.reserved_buffer,
                "enable_summary": profile.memory.enable_summary,
            },
        },
        "llm": llm_cfg,
        "embedder": embedder_cfg,
        "mem0_config": _mem0_config(embedder_cfg, llm_cfg, storage_dir),
    })


def _mem0_config(
    embedder_cfg: dict[str, Any],
    llm_cfg: dict[str, Any],
    storage_dir: str,
) -> dict[str, Any]:
    """Assemble the dict mem0.Memory.from_config() expects (embedder + llm + qdrant on-disk vector store). The embedder's `embedding_dims` is propagated to the vector store so qdrant collections match the embedder's output dimensionality."""
    collection = re.sub(r"[^a-zA-Z0-9_]+", "_", storage_dir).strip("_") or "default"
    vs_config: dict[str, Any] = {
        "path": storage_dir,
        "on_disk": True,
        "collection_name": collection,
    }
    embedder_inner = embedder_cfg.get("config", {})
    if "embedding_dims" in embedder_inner:
        vs_config["embedding_model_dims"] = embedder_inner["embedding_dims"]
    return {
        "embedder": embedder_cfg,
        "llm": llm_cfg,
        "vector_store": {"provider": "qdrant", "config": vs_config},
    }


def msg_ours_to_theirs(msg: OurMessage) -> MsMessage:
    """Copy field-by-field from a DefenseAgent Message into an ms-agent Message; preserves tool_calls + role/content."""
    tool_calls = None
    if msg.tool_calls:
        tool_calls = [
            {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
            for tc in msg.tool_calls
        ]
    return MsMessage(
        role=msg.role,
        content=msg.content or "",
        tool_calls=tool_calls,
        tool_call_id=msg.tool_call_id,
        name=msg.name,
    )


def msg_theirs_to_ours(msg: MsMessage) -> OurMessage:
    """Copy field-by-field from an ms-agent Message back into our DefenseAgent Message."""
    tool_calls: list[ToolCall] = []
    raw_calls = getattr(msg, "tool_calls", None)
    if raw_calls:
        for tc in raw_calls:
            if isinstance(tc, dict):
                tool_calls.append(
                    ToolCall(
                        id=tc.get("id", ""),
                        name=tc.get("name", ""),
                        arguments=tc.get("arguments", {}) or {},
                    )
                )
            else:
                tool_calls.append(
                    ToolCall(
                        id=getattr(tc, "id", ""),
                        name=getattr(tc, "name", ""),
                        arguments=getattr(tc, "arguments", {}) or {},
                    )
                )
    return OurMessage(
        role=msg.role,
        content=msg.content or "",
        tool_calls=tool_calls,
        tool_call_id=getattr(msg, "tool_call_id", None),
        name=getattr(msg, "name", None),
    )


def messages_ours_to_theirs(messages: list[OurMessage]) -> list[MsMessage]:
    """Map our Message list to ms-agent's list (preserves order)."""
    return [msg_ours_to_theirs(m) for m in messages]


def messages_theirs_to_ours(messages: list[MsMessage]) -> list[OurMessage]:
    """Map an ms-agent Message list back to ours (preserves order)."""
    return [msg_theirs_to_ours(m) for m in messages]


def record_memory_type(record: dict[str, Any]) -> str | None:
    """Pull memory_type out of a mem0 record; supports top-level and metadata-nested forms."""
    if "memory_type" in record:
        return record["memory_type"]
    return (record.get("metadata") or {}).get("memory_type")


def _resolve_storage_path(
    profile: AgentProfile,
    storage_path: str | Path | None,
) -> Path:
    """Pick the explicit storage_path, then profile.memory.storage_path, then `<profile.source_dir>/memory/`."""
    if storage_path is not None:
        return Path(storage_path).resolve()
    if profile.memory.storage_path:
        return Path(profile.memory.storage_path).resolve()
    if profile.source_dir is None:
        raise ValueError(
            "profile has no source_dir; pass storage_path explicitly when "
            "loading an in-memory profile"
        )
    return (profile.source_dir / "memory").resolve()


def _llm_config_from_env() -> dict[str, Any]:
    """Build the mem0 `llm` config dict from AGENT_LAB_LLM_PROVIDER + per-provider .env block. mem0 only natively understands `anthropic` and `openai`; every other provider (deepseek, qwen, vllm, modelscope, openrouter) is routed through mem0's `openai` provider with the matching base_url."""
    provider = os.environ.get("AGENT_LAB_LLM_PROVIDER", "").strip().lower()
    if not provider:
        raise ValueError(
            "AGENT_LAB_LLM_PROVIDER is not set; mem0 needs an LLM for fact extraction"
        )
    block = provider.upper()
    api_key = os.environ.get(f"{block}_API_KEY", "")
    base_url = os.environ.get(f"{block}_BASE_URL", "")
    model = os.environ.get(f"{block}_MODEL", "")
    if not api_key or not model:
        raise ValueError(
            f"{block}_API_KEY and {block}_MODEL must be set in .env for mem0"
        )
    if provider == "anthropic":
        return {"provider": "anthropic", "config": {"api_key": api_key, "model": model}}
    cfg: dict[str, Any] = {"api_key": api_key, "model": model}
    if base_url:
        cfg["openai_base_url"] = base_url
    return {"provider": "openai", "config": cfg}


def _embedder_config_from_env() -> dict[str, Any]:
    """Build the mem0 `embedder` config dict from EMBEDDING_* env vars (always OpenAI-compatible)."""
    api_key = os.environ.get("EMBEDDING_API_KEY", "")
    base_url = os.environ.get("EMBEDDING_BASE_URL", "")
    model = os.environ.get("EMBEDDING_MODEL", "")
    if not api_key or not model:
        raise ValueError(
            "EMBEDDING_API_KEY and EMBEDDING_MODEL must be set in .env"
        )
    cfg: dict[str, Any] = {"api_key": api_key, "model": model}
    if base_url:
        cfg["openai_base_url"] = base_url
    raw_dims = os.environ.get("EMBEDDING_DIMS", "").strip()
    if raw_dims:
        try:
            cfg["embedding_dims"] = int(raw_dims)
        except ValueError:
            pass
    else:
        cfg["embedding_dims"] = 4096
    return {"provider": "openai", "config": cfg}
