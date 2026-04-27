"""Example: a `kind="csv"` ResourceRenderer.

Drop-in copy for SDK users who store CSV resources in their RAG index. Reads
the file with pandas and emits a markdown table the LLM can read directly.

Usage:
    from DefenseAgent.rag import LlamaIndexRAG
    from scripts.extras.csv_renderer import CsvRenderer

    rag = await LlamaIndexRAG.from_profile(profile)
    rag.register_renderer(CsvRenderer())
    # When the LLM calls rag_get_resource(rid="...csv...") the CsvRenderer takes over.

Requires `pandas` (and `tabulate` for `df.to_markdown()`):
    pip install pandas tabulate
"""
from __future__ import annotations

from DefenseAgent.rag.extraction import StructuredResource


class CsvRenderer:
    """Renders a `kind="csv"` resource as a markdown table.

    Honors `resource.extra["max_rows"]` (default 50) to keep the LLM context
    manageable on huge spreadsheets. Falls back to a row-count summary when
    the CSV exceeds the cap.
    """

    kind = "csv"

    async def render(self, resource: StructuredResource) -> str:
        try:
            import pandas as pd
        except ImportError:
            return (
                f"(csv renderer requires `pip install pandas tabulate`; "
                f"resource [{resource.id}] available at {resource.path})"
            )

        try:
            df = pd.read_csv(resource.path)
        except Exception as e:  # noqa: BLE001 - surface to LLM
            return (
                f"(failed to read csv [{resource.id}] at {resource.path}: "
                f"{type(e).__name__}: {e})"
            )

        max_rows = int(resource.extra.get("max_rows", 50))
        truncated = len(df) > max_rows
        body_df = df.head(max_rows) if truncated else df

        try:
            md = body_df.to_markdown(index=False)
        except ImportError:
            md = body_df.to_string(index=False)

        header = f"csv [{resource.id}]"
        if resource.caption:
            header += f' "{resource.caption}"'
        suffix = (
            f"\n\n(showing first {max_rows} of {len(df)} rows)"
            if truncated else ""
        )
        return f"{header} ({len(df)} rows × {len(df.columns)} cols)\n\n{md}{suffix}"
