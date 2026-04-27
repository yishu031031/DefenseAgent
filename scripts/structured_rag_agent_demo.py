"""End-to-end multimodal RAG agent demo: extract → ingest → search → fetch.

Demonstrates every piece of the v0.1 SDK-friendly RAG pipeline working together:

  1. Custom renderer registration (`CsvRenderer` for `kind="csv"`)
  2. Built-in HTML extractor producing `<resource_info>` markers for images
  3. Heterogeneous `add_*` calls (structured chunks + plain text) all additive
  4. `rag_search` returning hits with a per-hit resource manifest
  5. `rag_get_resource` dispatching to the right renderer (Image / Table / Csv)

No real LLM calls are made: a `ScriptedLLM` plays back tool-call decisions a
real agent would make. This keeps the demo deterministic, offline, and free.
The real value is showing how `LlamaIndexRAG.register_renderer()` +
`StructuredDocExtractor.register()` let SDK users plug in custom kinds with
**zero source-tree edits**.

Usage (from project root, .venv3 active):
    python scripts/structured_rag_agent_demo.py

Required env (mem0 + RAG embedding):
    EMBEDDING_API_KEY=...
    EMBEDDING_BASE_URL=...
    EMBEDDING_MODEL=...
    EMBEDDING_DIMS=...   (must match the model)
"""
from __future__ import annotations

import asyncio
import re
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env", override=False)

from DefenseAgent.agent import (
    AgentConfig,
    MEMORY_RECALL_TOOL_NAME,
    RAG_GET_RESOURCE_TOOL_NAME,
    RAG_SEARCH_TOOL_NAME,
    ReActAgent,
)
from DefenseAgent.config import AgentProfile
from DefenseAgent.llm.types import LLMResponse, Message, TokenUsage, ToolCall
from DefenseAgent.rag import (
    LlamaIndexRAG,
    StructuredChunk,
    StructuredDocExtractor,
    StructuredResource,
)
from DefenseAgent.tools import ToolRegistry

# Pull in the reusable example renderer + extractor from scripts/extras/.
from scripts.extras.csv_renderer import CsvRenderer


# ============================================================
# 1. Setup: temp workspace + sample documents
# ============================================================


def _build_workspace() -> tuple[Path, Path]:
    """Create an isolated tempdir with sample HTML + CSV docs we'll ingest."""
    workspace = Path(tempfile.mkdtemp(prefix="structured_rag_agent_"))
    docs = workspace / "docs"
    docs.mkdir()

    # ---- HTML doc with image + version table ----
    img_path = docs / "diagram.png"
    # 1×1 transparent PNG so the demo doesn't need real image data
    img_path.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
        b"\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xf8\xff\xff?\x00\x05\xfe\x02\xfe"
        b"\xa3\x9d\x18\xfa\x00\x00\x00\x00IEND\xaeB`\x82"
    )

    (docs / "report.html").write_text(
        """\
<!DOCTYPE html>
<html><body>

<h1>CVE-2025-1234 Vulnerability Report</h1>
<p>Synthetic report demo for the structured RAG pipeline.</p>

<h2>Attack Chain</h2>
<p>Attackers chain three steps to achieve RCE; the diagram shows the flow.</p>
<img src="diagram.png" alt="End-to-end attack chain"/>

<h2>Affected Versions</h2>
<table>
<tr><th>Version</th><th>Status</th><th>Mitigation</th></tr>
<tr><td>1.0–1.4</td><td>Vulnerable</td><td>Upgrade to 1.5.2+</td></tr>
<tr><td>1.5.0</td><td>Partial fix</td><td>Apply patch CVE-2025-1234-A</td></tr>
<tr><td>1.5.2+</td><td>Patched</td><td>None required</td></tr>
</table>

</body></html>
""",
        encoding="utf-8",
    )

    # ---- CSV doc (custom kind = "csv", picked up via our injected extractor) ----
    csv_path = docs / "incidents.csv"
    csv_path.write_text(
        "date,severity,asset,resolved\n"
        "2025-01-04,high,api-gateway,true\n"
        "2025-01-09,critical,db-primary,true\n"
        "2025-01-15,medium,worker-pool,false\n"
        "2025-01-22,low,bastion,true\n",
        encoding="utf-8",
    )
    return workspace, csv_path


# ============================================================
# 2. A custom CsvExtractor — emits StructuredChunk objects
#    with kind="csv" resources so CsvRenderer takes over.
# ============================================================


class CsvExtractor:
    """Simple StructuredExtractor that ingests .csv files as a single chunk
    pointing at a `kind="csv"` resource. Persists the file inside the
    structured resources directory so the index stays portable.
    """

    def __init__(self, resources_dir: Path) -> None:
        self.resources_dir = Path(resources_dir)

    def supports(self, source: str | Path) -> bool:
        path = Path(source)
        return path.suffix.lower() == ".csv" and path.is_file()

    def extract(self, source: str | Path) -> list[StructuredChunk]:
        import hashlib
        path = Path(source).resolve()
        h = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        sub_dir = self.resources_dir / h
        sub_dir.mkdir(parents=True, exist_ok=True)
        dst = sub_dir / path.name
        if not dst.is_file():
            shutil.copyfile(path, dst)

        rid = f"{path.name}@{h}@csv0"
        # Use the first line as a synthetic caption so rag_search shows it.
        first_line = path.read_text(encoding="utf-8").splitlines()[0]
        resource = StructuredResource(
            id=rid,
            kind="csv",
            path=dst,
            caption=f"columns: {first_line}",
            mime_type="text/csv",
            extra={"max_rows": 20},   # CsvRenderer reads this knob
        )
        return [
            StructuredChunk(
                text=(
                    f"Security incident log ({path.name}): tracks date, severity, "
                    f"affected asset, and resolution status across recent incidents.\n\n"
                    f"<resource_info>{rid}</resource_info>"
                ),
                resources=[resource],
                metadata={"source": str(path), "section": 0},
            )
        ]


# ============================================================
# 3. Scripted LLM that simulates the conversation
# ============================================================


def _make_scripted_llm() -> MagicMock:
    """Hand-roll an LLM that emits the exact tool-call sequence a real ReAct
    run would produce against this dataset. Lets the demo run offline + fast."""

    def resp(content="", tool_calls=None) -> LLMResponse:
        return LLMResponse(
            content=content, tool_calls=list(tool_calls or []),
            usage=TokenUsage(20, 10, 30),
            stop_reason="tool_use" if tool_calls else "end_turn",
            raw={},
        )

    # 4-turn script:
    #   1. rag_search("attack chain")          → finds HTML chunk with image
    #   2. rag_get_resource(image rid)         → renders image path
    #   3. rag_search("incident severities")   → finds CSV chunk
    #   4. rag_get_resource(csv rid)           → renders csv as markdown
    #   5. final answer
    responses = [
        resp(content="Let me search for the attack chain.", tool_calls=[
            ToolCall(id="t1", name=RAG_SEARCH_TOOL_NAME,
                     arguments={"query": "attack chain RCE"}),
        ]),
        resp(content="Now let me fetch the diagram image.", tool_calls=[
            ToolCall(id="t2", name=RAG_GET_RESOURCE_TOOL_NAME,
                     arguments={"resource_id": "PLACEHOLDER_IMG"}),
        ]),
        resp(content="And the incident log.", tool_calls=[
            ToolCall(id="t3", name=RAG_SEARCH_TOOL_NAME,
                     arguments={"query": "incident severity recent"}),
        ]),
        resp(content="Let me pull the full csv.", tool_calls=[
            ToolCall(id="t4", name=RAG_GET_RESOURCE_TOOL_NAME,
                     arguments={"resource_id": "PLACEHOLDER_CSV"}),
        ]),
        resp(content=(
            "Findings:\n"
            "- The CVE-2025-1234 attack chain is documented (3-step RCE); see diagram.\n"
            "- Recent incident log shows 2 high/critical issues this month, both resolved.\n"
            "- Versions 1.0-1.4 require an upgrade; 1.5.0 needs a patch."
        )),
    ]

    llm = MagicMock(name="ScriptedLLM")
    llm.adapter = MagicMock()
    llm.adapter.model = "scripted"

    state = {"index": 0, "patched_image_rid": None, "patched_csv_rid": None}

    async def chat(messages, **kwargs):
        # 1. Sniff FIRST — scan every prior tool message for `• resource [RID] (kind)`
        #    lines (emitted by BaseAgent._format_rag_hit). Take the first RID we
        #    see for each kind we care about. Done before patching so the next
        #    response gets up-to-date state.
        for m in messages:
            if m.role != "tool" or not m.content:
                continue
            for line in m.content.splitlines():
                mt = re.match(r"\s*•\s*resource\s*\[([^\]]+)\]\s*\((\w+)\)", line)
                if not mt:
                    continue
                rid, kind = mt.group(1), mt.group(2)
                if kind == "image" and state["patched_image_rid"] is None:
                    state["patched_image_rid"] = rid
                elif kind == "csv" and state["patched_csv_rid"] is None:
                    state["patched_csv_rid"] = rid

        # 2. Pick the next response and patch placeholder RIDs with whatever
        #    the sniff phase has accumulated by now.
        i = state["index"]
        state["index"] += 1
        r = responses[i]
        for tc in r.tool_calls:
            if tc.arguments.get("resource_id") == "PLACEHOLDER_IMG":
                tc.arguments["resource_id"] = (
                    state["patched_image_rid"] or "(no image rid sniffed yet)"
                )
            elif tc.arguments.get("resource_id") == "PLACEHOLDER_CSV":
                tc.arguments["resource_id"] = (
                    state["patched_csv_rid"] or "(no csv rid sniffed yet)"
                )
        return r

    llm.chat = chat
    return llm


# ============================================================
# 4. Demo orchestration
# ============================================================


def _banner(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


async def main() -> int:
    _banner("Setup — building demo workspace")
    workspace, csv_path = _build_workspace()
    print(f"  workspace: {workspace}")
    print(f"  sample HTML: {workspace / 'docs' / 'report.html'}")
    print(f"  sample CSV:  {csv_path}")

    # Build an in-memory profile pointing at our temp docs.
    profile = AgentProfile(
        id="rag_demo_001",
        name="RAG Demo Agent",
        age=30,
        traits="systematic, thorough",
        backstory="Security analyst exercising the multimodal RAG pipeline.",
        initial_plan="Demonstrate hybrid extract → search → fetch flow.",
        rag={
            "enabled": True,
            "documents_dir": str(workspace / "docs"),
            "storage_dir": str(workspace / "rag"),
            "top_k": 3,
        },
    )

    _banner("Step 1 — Build extractor with the example DocxExtractor + custom CsvExtractor")
    extractor = StructuredDocExtractor(profile, resources_dir=workspace / "rag" / "resources")
    extractor.register(CsvExtractor(resources_dir=extractor.resources_dir))   # custom .csv backend
    print(f"  extractor backends (in order): {[type(b).__name__ for b in extractor._backends]}")

    _banner("Step 2 — Build RAG with extractor + CsvRenderer registered")
    try:
        rag = await LlamaIndexRAG.from_profile(
            profile, load_env=False, auto_load=False, extractor=extractor,
        )
    except Exception as e:
        print(f"  Could not build LlamaIndexRAG: {type(e).__name__}: {e}")
        print("  Make sure llama-index-core, llama-index-embeddings-openai-like,")
        print("  and EMBEDDING_* env vars are set. Demo aborted.")
        return 1
    rag.register_renderer(CsvRenderer())
    print(f"  registered renderer kinds: {sorted(rag._renderers.keys())}")

    _banner("Step 3 — Ingest documents (HTML + CSV)")
    files = list((workspace / "docs").iterdir())
    structured = [f for f in files if extractor.supports(f)]
    print(f"  files routed to structured extraction: {[f.name for f in structured]}")
    chunks = extractor.extract(structured)
    print(f"  produced {len(chunks)} chunk(s); resource ids:")
    for c in chunks:
        for r in c.resources:
            print(f"    - {r.id}  (kind={r.kind})")
    await rag.add_structured_chunks(chunks)
    await rag.save_index()
    print("  index persisted.")

    _banner("Step 4 — Build agent with our RAG injected")
    config = AgentConfig(
        profile=profile,
        load_env=False,
        use_memory=False,
        use_reflection=False,
        use_compressor=False,
        use_logger=False,
        use_rag=True,
        rag=rag,
        llm=_make_scripted_llm(),
        tool_registry=ToolRegistry(),
    )
    agent = ReActAgent(config)
    print(f"  agent tools: {sorted(agent._agent_tools.keys())}")

    _banner("Step 5 — Run the agent (scripted LLM walks through both resource kinds)")
    result = await agent.run(
        "Summarize the CVE-2025-1234 attack chain, show the diagram, "
        "and pull the recent incident log.",
        max_steps=10,
    )
    print(f"\nFinal answer:\n{result.final_answer}\n")

    print("Trace (kind / tool / brief):")
    for s in result.steps:
        if s.kind == "tool_call":
            for tc in s.tool_calls:
                arg_preview = next(iter(tc.arguments.values()), "")
                print(f"  [step {s.index}] tool_call → {tc.name}({arg_preview!r})")
        elif s.kind == "tool_result":
            for tr in s.tool_results:
                preview = (tr.content or "").splitlines()[0][:100]
                print(f"  [step {s.index}] tool_result ({tr.name}) → {preview}")
        elif s.kind == "answer":
            print(f"  [step {s.index}] answer ({(s.usage.total_tokens if s.usage else 0)} tok)")

    print(f"\nTotal tokens: {result.usage.total_tokens}")
    print(f"Workspace kept at: {workspace}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
