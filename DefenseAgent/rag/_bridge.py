import os
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from DefenseAgent.config.profile import AgentProfile
from DefenseAgent.memory._bridge import _flat_llm_config_from_env
from DefenseAgent.rag.base import RAGConfigError


_DEFAULT_HF_EMBEDDING = "Qwen/Qwen3-Embedding-0.6B"


def profile_to_rag_dictconfig(
    profile: AgentProfile,
    *,
    storage_path: str | Path | None = None,
    documents_path: str | Path | None = None,
) -> Any:
    """Translate our pydantic AgentProfile into the OmegaConf DictConfig that ms-agent's `LlamaIndexRAG` reads.

    Embedding fields (`embedding`, `embedding_api_key`, `embedding_base_url`, `embedding_dims`) are resolved per-field with profile-first / env-fallback (mirrors the LLMConfig pattern); the resolved values land on the DictConfig so both the OpenAI-compat path and ms-agent's HuggingFace path see consistent inputs. Carries a flat `llm` block (only consumed when `retrieve_only=False`), the `rag` knobs subset, and a top-level `use_huggingface` flag.
    """
    rag_cfg = profile.rag
    storage_dir = _resolve_storage_path(profile, storage_path)
    documents_dir = _resolve_documents_path(profile, documents_path)
    embedding = _resolve_embedding_fields(rag_cfg)
    config: dict[str, Any] = {
        "rag": {
            "embedding": embedding["model"],
            "embedding_provider": rag_cfg.embedding_provider,
            "embedding_api_key": embedding["api_key"],
            "embedding_base_url": embedding["base_url"],
            "embedding_dims": embedding["dims"],
            "chunk_size": rag_cfg.chunk_size,
            "chunk_overlap": rag_cfg.chunk_overlap,
            "retrieve_only": rag_cfg.retrieve_only,
            "storage_dir": str(storage_dir),
        },
        "use_huggingface": rag_cfg.use_huggingface,
    }
    if not rag_cfg.retrieve_only:
        config["llm"] = _flat_llm_config_from_env()
    if documents_dir is not None:
        config["documents_dir"] = str(documents_dir)
    return OmegaConf.create(config)


def _resolve_embedding_fields(rag_cfg: Any) -> dict[str, Any]:
    """Per-field profile-then-env resolution for the embedder. Profile values win when set + non-empty after `.strip()`; otherwise fall back to `EMBEDDING_MODEL` / `EMBEDDING_API_KEY` / `EMBEDDING_BASE_URL` / `EMBEDDING_DIMS` from .env. Returns a dict of resolved values; missing values come back as None or (for the HuggingFace model name fallback) the ms-agent default."""
    model = _profile_or_env_str(rag_cfg.embedding, "EMBEDDING_MODEL")
    api_key = _profile_or_env_str(rag_cfg.embedding_api_key, "EMBEDDING_API_KEY")
    base_url = _profile_or_env_str(rag_cfg.embedding_base_url, "EMBEDDING_BASE_URL")
    dims = _profile_or_env_int(rag_cfg.embedding_dims, "EMBEDDING_DIMS")
    if rag_cfg.embedding_provider == "huggingface" and not model:
        # The HF path needs *some* model; use the ms-agent default rather than crashing
        # at index time. The OpenAI path defers the missing-key error to the resolver
        # inside _install_openai_compat_embedding so the message stays accurate.
        model = _DEFAULT_HF_EMBEDDING
    return {"model": model, "api_key": api_key, "base_url": base_url, "dims": dims}


def _profile_or_env_str(profile_value: str | None, env_var: str) -> str | None:
    """Return the profile value when it's a non-empty string after `.strip()`; otherwise the env var; otherwise None."""
    if isinstance(profile_value, str) and profile_value.strip():
        return profile_value.strip()
    raw = os.environ.get(env_var, "")
    return raw.strip() if raw and raw.strip() else None


def _profile_or_env_int(profile_value: int | None, env_var: str) -> int | None:
    """Same as `_profile_or_env_str` but for the integer dim count; ignores unparseable env values."""
    if isinstance(profile_value, int):
        return profile_value
    raw = os.environ.get(env_var, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _resolve_storage_path(
    profile: AgentProfile,
    storage_path: str | Path | None,
) -> Path:
    """Pick the explicit storage_path, then `profile.rag.storage_dir`, then `<profile.source_dir>/rag/`."""
    if storage_path is not None:
        return Path(storage_path).resolve()
    if profile.rag.storage_dir:
        return Path(profile.rag.storage_dir).resolve()
    if profile.source_dir is None:
        raise RAGConfigError(
            "profile has no source_dir; pass storage_path explicitly when "
            "loading an in-memory profile"
        )
    return (profile.source_dir / "rag").resolve()


def _resolve_documents_path(
    profile: AgentProfile,
    documents_path: str | Path | None,
) -> Path | None:
    """Pick the explicit documents_path; else use `profile.rag.documents_dir` (absolute paths are taken as-is, relative paths anchor to `profile.source_dir`). Returns None when nothing is configured (no auto-load)."""
    if documents_path is not None:
        return Path(documents_path).resolve()
    if not profile.rag.documents_dir:
        return None
    candidate = Path(profile.rag.documents_dir)
    if candidate.is_absolute():
        return candidate.resolve()
    if profile.source_dir is None:
        raise RAGConfigError(
            "profile.rag.documents_dir is a relative path but profile has no "
            "source_dir to anchor it; either pass documents_path explicitly, "
            "or set profile.rag.documents_dir to an absolute path"
        )
    return (profile.source_dir / profile.rag.documents_dir).resolve()
