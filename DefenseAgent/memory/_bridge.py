import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from ms_agent.llm.utils import Message as MsMessage

from DefenseAgent.config.profile import AgentProfile
from DefenseAgent.llm.types import Message as OurMessage
from DefenseAgent.llm.types import ToolCall


@dataclass
class MemoryBackendConfig:
    """Pure-code description of mem0's LLM + embedder providers — what env vars
    used to supply implicitly. Pass one of these to `Mem0Memory.create`
    (or `profile_to_dictconfig`'s `backend=` kwarg) to bypass the .env path
    entirely. SDK callers, tests, and multi-tenant servers should use this."""

    llm_provider: str          # "anthropic" / "openai" / "deepseek" / ...
    llm_api_key: str
    llm_model: str
    llm_base_url: str = ""
    embedding_provider: str = "openai"
    embedding_api_key: str = ""
    embedding_model: str = ""
    embedding_base_url: str = ""
    embedding_dims: int = 4096


def profile_to_dictconfig(
    profile: AgentProfile,
    *,
    backend: MemoryBackendConfig | None = None,
    user_id: str = "default_user",
    agent_id: str | None = None,
    run_id: str = "default_run",
    storage_path: str | Path | None = None,
) -> Any:
    """Translate our pydantic AgentProfile into the OmegaConf DictConfig that ms-agent's Memory subclasses read; carries the ready-to-use `mem0_config` dict so DefenseAgent.Mem0Memory can bypass ms-agent's hardcoded service-URL translation. Two LLM/embedder shapes coexist on the config: a flat one at `config.llm` for ms-agent's ContextCompressor (which reads `config.llm.model` / `config.llm.openai_api_key` etc.), and a nested mem0-shape inside `config.mem0_config` for `mem0.Memory.from_config()`. Pass `backend=` for pure-code construction; omit it to fall back to env-variable resolution."""
    if backend is None:
        backend = _backend_from_env()
    resolved_path = _resolve_storage_path(profile, storage_path)
    resolved_agent_id = agent_id or profile.id
    storage_dir = str(resolved_path / "default_memory")
    mem0_llm = _llm_config_from_backend(backend)
    mem0_embedder = _embedder_config_from_backend(backend)
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
        "llm": _flat_llm_config_from_backend(backend),
        "mem0_config": _mem0_config(mem0_embedder, mem0_llm, storage_dir),
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


def record_tier(record: dict[str, Any]) -> str | None:
    """Pull the tier label out of a mem0 record. Lives in metadata only — there
    is no top-level mem0 field for it. Returns None for legacy records written
    before the tier dimension existed (callers should treat that as EPISODIC)."""
    return (record.get("metadata") or {}).get("tier")


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


# ---------------------------------------------------------------------------
# Backend → config-block translators (pure functions; no env access).
# ---------------------------------------------------------------------------


def _flat_llm_config_from_backend(backend: MemoryBackendConfig) -> dict[str, Any]:
    """Build the flat-shape LLM config that ms-agent's ContextCompressor reads. Fields mirror ms-agent's `openai_llm.py`/`anthropic_llm.py` lookups: `service`, `model`, `<service>_api_key`, `<service>_base_url`. Everything OpenAI-compatible is routed through `service='openai'` so the matching base_url and api_key keys are picked up."""
    service = "anthropic" if backend.llm_provider == "anthropic" else "openai"
    cfg: dict[str, Any] = {
        "service": service,
        "model": backend.llm_model,
        f"{service}_api_key": backend.llm_api_key,
    }
    if backend.llm_base_url:
        cfg[f"{service}_base_url"] = backend.llm_base_url
    return cfg


def _llm_config_from_backend(backend: MemoryBackendConfig) -> dict[str, Any]:
    """Build the mem0 `llm` config dict from a MemoryBackendConfig. mem0 only natively understands `anthropic` and `openai`; every other provider (deepseek, qwen, vllm, modelscope, openrouter) is routed through mem0's `openai` provider with the matching base_url."""
    if backend.llm_provider == "anthropic":
        return {
            "provider": "anthropic",
            "config": {"api_key": backend.llm_api_key, "model": backend.llm_model},
        }
    cfg: dict[str, Any] = {
        "api_key": backend.llm_api_key,
        "model": backend.llm_model,
    }
    if backend.llm_base_url:
        cfg["openai_base_url"] = backend.llm_base_url
    return {"provider": "openai", "config": cfg}


def _embedder_config_from_backend(backend: MemoryBackendConfig) -> dict[str, Any]:
    """Build the mem0 `embedder` config dict from a MemoryBackendConfig (always OpenAI-compatible — Anthropic doesn't ship embeddings)."""
    if not backend.embedding_api_key or not backend.embedding_model:
        raise ValueError(
            "MemoryBackendConfig.embedding_api_key and embedding_model are required; "
            "mem0 cannot run without an embedder."
        )
    cfg: dict[str, Any] = {
        "api_key": backend.embedding_api_key,
        "model": backend.embedding_model,
        "embedding_dims": backend.embedding_dims,
    }
    if backend.embedding_base_url:
        cfg["openai_base_url"] = backend.embedding_base_url
    return {"provider": backend.embedding_provider, "config": cfg}


# ---------------------------------------------------------------------------
# Env → backend (legacy compatibility shim).
# ---------------------------------------------------------------------------


def _flat_llm_config_from_env() -> dict[str, Any]:
    """Build the flat-shape LLM config from env vars only — no embedder required.

    Thin convenience for `rag/_bridge.py` and any caller that needs ms-agent's
    flat LLM dict but isn't building memory (so EMBEDDING_* doesn't have to be
    set). Reads `AGENT_LAB_LLM_PROVIDER` + the matching `<PROVIDER>_API_KEY` /
    `_BASE_URL` / `_MODEL` block.
    """
    llm_provider = os.environ.get("AGENT_LAB_LLM_PROVIDER", "").strip().lower()
    if not llm_provider:
        raise ValueError(
            "AGENT_LAB_LLM_PROVIDER is not set; cannot build LLM config."
        )
    block = llm_provider.upper()
    api_key = os.environ.get(f"{block}_API_KEY", "")
    base_url = os.environ.get(f"{block}_BASE_URL", "")
    model = os.environ.get(f"{block}_MODEL", "")
    if not api_key or not model:
        raise ValueError(
            f"{block}_API_KEY and {block}_MODEL must be set in .env"
        )
    # Reuse the backend-shaped translator with a temporary backend that has no
    # embedder fields populated — the flat helper doesn't read them.
    backend = MemoryBackendConfig(
        llm_provider=llm_provider,
        llm_api_key=api_key,
        llm_model=model,
        llm_base_url=base_url,
    )
    return _flat_llm_config_from_backend(backend)


def _backend_from_env() -> MemoryBackendConfig:
    """Build a MemoryBackendConfig by reading the same env vars the codebase used to read directly. Called from `profile_to_dictconfig` when no explicit `backend=` is supplied — preserves the original .env-driven behavior for callers that haven't migrated yet."""
    llm_provider = os.environ.get("AGENT_LAB_LLM_PROVIDER", "").strip().lower()
    if not llm_provider:
        raise ValueError(
            "AGENT_LAB_LLM_PROVIDER is not set; mem0 needs an LLM for fact extraction. "
            "Either set it in .env, or pass `backend=MemoryBackendConfig(...)` explicitly."
        )
    block = llm_provider.upper()
    llm_api_key = os.environ.get(f"{block}_API_KEY", "")
    llm_base_url = os.environ.get(f"{block}_BASE_URL", "")
    llm_model = os.environ.get(f"{block}_MODEL", "")
    if not llm_api_key or not llm_model:
        raise ValueError(
            f"{block}_API_KEY and {block}_MODEL must be set in .env for mem0"
        )

    embedding_api_key = os.environ.get("EMBEDDING_API_KEY", "")
    embedding_base_url = os.environ.get("EMBEDDING_BASE_URL", "")
    embedding_model = os.environ.get("EMBEDDING_MODEL", "")
    if not embedding_api_key or not embedding_model:
        raise ValueError(
            "EMBEDDING_API_KEY and EMBEDDING_MODEL must be set in .env"
        )
    embedding_dims = 4096
    raw_dims = os.environ.get("EMBEDDING_DIMS", "").strip()
    if raw_dims:
        try:
            embedding_dims = int(raw_dims)
        except ValueError:
            pass

    return MemoryBackendConfig(
        llm_provider=llm_provider,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        embedding_provider="openai",
        embedding_api_key=embedding_api_key,
        embedding_model=embedding_model,
        embedding_base_url=embedding_base_url,
        embedding_dims=embedding_dims,
    )
