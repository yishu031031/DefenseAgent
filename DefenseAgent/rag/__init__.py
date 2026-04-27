from ms_agent.rag.utils import rag_mapping as _ms_rag_mapping

from DefenseAgent.rag.base import (
    RAG,
    RAGConfigError,
    RAGError,
    RAGProviderError,
)
from DefenseAgent.rag.extraction import (
    HtmlExtractor,
    PyPdfExtractor,
    StructuredChunk,
    StructuredDocExtractor,
    StructuredExtractor,
    StructuredResource,
)
from DefenseAgent.rag.llama_index_rag import LlamaIndexRAG
from DefenseAgent.rag.renderer import (
    ImageRenderer,
    ResourceRenderer,
    TableRenderer,
    default_renderers,
)


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
    "StructuredChunk",
    "StructuredResource",
    "StructuredExtractor",
    "StructuredDocExtractor",
    "PyPdfExtractor",
    "HtmlExtractor",
    "ResourceRenderer",
    "TableRenderer",
    "ImageRenderer",
    "default_renderers",
]
