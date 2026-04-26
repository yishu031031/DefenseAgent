from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from DefenseAgent.config.profile import AgentProfile
from DefenseAgent.memory._bridge import _flat_llm_config_from_env
from DefenseAgent.rag.base import RAGConfigError


def profile_to_rag_dictconfig(
    profile: AgentProfile,
    *,
    storage_path: str | Path | None = None,
    documents_path: str | Path | None = None,
) -> Any:
    """Translate our pydantic AgentProfile into the OmegaConf DictConfig that ms-agent's `LlamaIndexRAG` reads.

    Carries a flat `llm` block (only consumed when `retrieve_only=False`, since ms-agent's RAG
    falls back to its own LLM call for synthesis), the `rag` knobs subset (`embedding`, `chunk_size`,
    `chunk_overlap`, `retrieve_only`, `storage_dir`), and a top-level `use_huggingface` flag that
    tells ms-agent's `_setup_embedding_model` whether to skip modelscope's `snapshot_download`.
    """
    rag_cfg = profile.rag
    storage_dir = _resolve_storage_path(profile, storage_path)
    documents_dir = _resolve_documents_path(profile, documents_path)
    config: dict[str, Any] = {
        "rag": {
            "embedding": rag_cfg.embedding,
            "embedding_provider": rag_cfg.embedding_provider,
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
    """Pick the explicit documents_path, then `<profile.source_dir>/<profile.rag.documents_dir>`. Returns None when nothing is configured (no auto-load)."""
    if documents_path is not None:
        return Path(documents_path).resolve()
    if not profile.rag.documents_dir:
        return None
    if profile.source_dir is None:
        raise RAGConfigError(
            "profile.rag.documents_dir is set but profile has no source_dir; "
            "pass documents_path explicitly when loading an in-memory profile"
        )
    return (profile.source_dir / profile.rag.documents_dir).resolve()
