"""Resource renderers — pluggable "kind → string" serializers for RAG resources.

When the LLM calls `rag_get_resource`, the agent layer hands the resource off
to a renderer keyed by its `kind`. Built-in renderers cover `image` and
`table`; SDK callers add their own (`csv`, `audio`, `chart`, ...) by
implementing the `ResourceRenderer` protocol and calling
`LlamaIndexRAG.register_renderer()`.

Renderers are async to allow I/O-heavy formatting (e.g. transcribing an audio
file) without blocking the agent loop.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from DefenseAgent.rag.extraction import StructuredResource


@runtime_checkable
class ResourceRenderer(Protocol):
    """How to materialize a resource for LLM consumption.

    Implementations must declare the `kind` they handle (matching
    `StructuredResource.kind`) and an async `render` that returns a string the
    LLM can read directly. The agent layer is responsible for tagging the
    output with the resource id and any other framing the LLM should see.
    """

    kind: str

    async def render(self, resource: StructuredResource) -> str: ...


# ---------- built-in renderers ----------


class TableRenderer:
    """Renders a `kind="table"` resource by reading the persisted markdown."""

    kind = "table"

    async def render(self, resource: StructuredResource) -> str:
        """Read the table file as utf-8 text and prepend an id/caption header."""
        try:
            content = resource.path.read_text(encoding="utf-8")
        except Exception as e:  # noqa: BLE001 - surfaces the failure to the LLM
            return (
                f"(failed to read table at {resource.path}: "
                f"{type(e).__name__}: {e})"
            )
        header = f"table [{resource.id}]"
        if resource.caption:
            header += f' "{resource.caption}"'
        return f"{header}\n\n{content}"


class ImageRenderer:
    """Renders a `kind="image"` resource as a path + metadata string.

    Does NOT read image bytes — agent text turns can't directly inline images.
    The host application is expected to feed `resource.path` to a
    vision-capable model when visual analysis is needed.
    """

    kind = "image"

    async def render(self, resource: StructuredResource) -> str:
        size = resource.path.stat().st_size if resource.path.is_file() else 0
        cap = f' "{resource.caption}"' if resource.caption else ""
        mime = f", mime={resource.mime_type}" if resource.mime_type else ""
        return (
            f"image [{resource.id}]{cap} at {resource.path} "
            f"({size} bytes{mime}).\n"
            "Note: agent does not auto-load image bytes into the next LLM "
            "turn. The host application can pass this path to a "
            "vision-capable model if visual analysis is needed."
        )


def default_renderers() -> dict[str, ResourceRenderer]:
    """Return a fresh dict of the built-in renderers, keyed by `kind`."""
    return {
        "image": ImageRenderer(),
        "table": TableRenderer(),
    }
