from ms_agent.rag.utils import rag_mapping as _ms_rag_mapping

from DefenseAgent.rag.base import (
    RAG,
    RAGConfigError,
    RAGError,
    RAGProviderError,
)
from DefenseAgent.rag.llama_index_rag import LlamaIndexRAG


# Override ms-agent's LlamaIndexRAG entry with our profile-aware subclass so
# any name-based lookup (e.g. ms-agent's LLMAgent) resolves to ours.
rag_mapping = {**_ms_rag_mapping, "LlamaIndexRAG": LlamaIndexRAG}


__all__ = [
    "RAG",
    "LlamaIndexRAG",
    "rag_mapping",
    "RAGError",
    "RAGConfigError",
    "RAGProviderError",
]
