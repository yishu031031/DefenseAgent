"""Structured-extraction end-to-end demo: HTML → chunks → resource lookup.

Builds a synthetic CVE report (HTML with an embedded image + version table)
in a fresh per-run workspace, then walks the Module 5 multimodal RAG pipeline:

    StructuredDocExtractor.extract()
        → list[StructuredChunk]                  ← text + persisted resources
        → LlamaIndexRAG.add_structured_chunks()  (only with --with-rag)
        → LlamaIndexRAG.retrieve()
        → LlamaIndexRAG.get_resource_path(rid)
        → PIL.Image.open(path)

Two run modes:
  - default:    extract + manual resource lookup. No API key, no llama-index needed.
  - --with-rag: also ingest into a real VectorStoreIndex and retrieve.
                Requires llama-index plus EMBEDDING_API_KEY / EMBEDDING_MODEL in .env.

Usage (from project root, .venv3 active):
    python scripts/structured_extraction_demo.py
    python scripts/structured_extraction_demo.py --with-rag
"""
from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

# Load .env so EMBEDDING_API_KEY / EMBEDDING_MODEL etc. land in os.environ
# before the RAG step checks for them.
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

from DefenseAgent.config import AgentProfile
from DefenseAgent.rag import StructuredChunk, StructuredDocExtractor


# ---------- workspace + sample document ----------


WORKSPACE = Path(__file__).resolve().parent.parent / "demo_workspace" / "structured_extraction"
SAMPLE_DIR = WORKSPACE / "sample_docs"


SAMPLE_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"/><title>CVE-2024-0001 Report</title></head>
<body>

<h1>CVE-2024-0001: Sample Vulnerability Report</h1>
<p>This synthetic CVE record is used by the structured-extraction demo.</p>

<h2>Attack Chain</h2>
<p>Attackers chain three steps to achieve RCE; the diagram below shows the
full flow from initial probe to privileged shell.</p>
<img src="diagram.png" alt="End-to-end attack chain"/>

<h2>Affected Versions</h2>
<table>
  <tr><th>Version</th><th>Status</th><th>Mitigation</th></tr>
  <tr><td>1.0 - 1.4</td><td>Vulnerable</td><td>Upgrade to 1.5.2+</td></tr>
  <tr><td>1.5.0</td><td>Partial fix</td><td>Apply patch CVE-2024-0001-A</td></tr>
  <tr><td>1.5.2+</td><td>Patched</td><td>None required</td></tr>
</table>

<h2>Indicators of Compromise</h2>
<p>Look for unexpected outbound LDAP requests on port 389 from application
hosts that have no business making them.</p>

</body>
</html>
"""


def _make_sample_image(path: Path) -> bool:
    """Render a synthetic diagram PNG via Pillow; returns False (and writes a
    placeholder) when Pillow isn't installed so the demo can still proceed."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        path.write_bytes(b"\x89PNG\r\n\x1a\n[placeholder; install Pillow for a real image]")
        return False
    img = Image.new("RGB", (320, 160), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, 310, 150], outline="black", width=2)
    draw.text((24, 32), "Attack Flow Diagram", fill="black")
    draw.text((24, 60), "1. Initial probe", fill="black")
    draw.text((24, 84), "2. LDAP injection -> JNDI lookup", fill="black")
    draw.text((24, 108), "3. Remote class load -> RCE", fill="black")
    img.save(path, format="PNG")
    return True


def _setup_workspace() -> Path:
    """(Re)create the demo workspace and drop the sample HTML + image into it."""
    if WORKSPACE.exists():
        shutil.rmtree(WORKSPACE)
    SAMPLE_DIR.mkdir(parents=True)
    html_path = SAMPLE_DIR / "report.html"
    html_path.write_text(SAMPLE_HTML, encoding="utf-8")
    image_real = _make_sample_image(SAMPLE_DIR / "diagram.png")
    return html_path, image_real


def _make_demo_profile() -> AgentProfile:
    """In-memory AgentProfile anchored at WORKSPACE so resources land under it."""
    profile = AgentProfile(
        id="demo",
        name="Extraction Demo",
        age=1,
        traits="precise, deterministic, transparent",
        backstory="A throwaway profile used only by structured_extraction_demo.py.",
        initial_plan="Extract a CVE HTML report and prove the resource lookup works.",
        rag={
            "enabled": True,
            "storage_dir": str(WORKSPACE / "rag"),
        },
    )
    profile._source_path = WORKSPACE / "_demo.yaml"
    return profile


# ---------- pretty printing ----------


def _banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def _print_chunks(chunks: list[StructuredChunk]) -> None:
    print(f"\nExtracted {len(chunks)} chunk(s) total.\n")
    for i, c in enumerate(chunks, 1):
        print(f"--- chunk #{i} -----------------------------------------------------------")
        print(f"  metadata:  {c.metadata}")
        print(f"  resources: {len(c.resources)} item(s)")
        for r in c.resources:
            print(f"    - id={r.id}")
            print(f"      kind={r.kind}  mime={r.mime_type}")
            print(f"      path={r.path}")
            if r.caption:
                print(f"      caption={r.caption!r}")
        print(f"  text ({len(c.text)} chars):")
        for line in c.text.splitlines():
            print(f"      | {line}")
        print()


# ---------- mode 1: extraction-only ----------


def _demo_extraction(profile: AgentProfile, html_path: Path) -> list[StructuredChunk]:
    _banner("Step 1 - Extract structured chunks from HTML")
    extractor = StructuredDocExtractor(profile)
    chunks = extractor.extract([html_path])
    _print_chunks(chunks)

    _banner("Step 2 - Resource lookup (manual id -> path map, no RAG yet)")
    lookup = {r.id: r for c in chunks for r in c.resources}
    if not lookup:
        print("  (no resources extracted; check that diagram.png exists)")
        return chunks
    for rid, res in lookup.items():
        size = res.path.stat().st_size if res.path.is_file() else "missing"
        print(f"  - {rid}")
        print(f"      kind={res.kind}  caption={res.caption!r}")
        print(f"      path={res.path}  ({size} bytes)")
        _try_describe_image(res.path)
    return chunks


def _try_describe_image(path: Path) -> None:
    """Open the image with Pillow (when installed) and print basic stats."""
    try:
        from PIL import Image
    except ImportError:
        print("      (Pillow not installed; skip image inspection)")
        return
    try:
        with Image.open(path) as img:
            print(f"      Image.open OK: format={img.format} size={img.size} mode={img.mode}")
    except Exception as e:  # noqa: BLE001 - placeholder bytes will fail; that's OK
        print(f"      Image.open failed: {type(e).__name__}: {e}")


# ---------- mode 2: end-to-end with real RAG ----------


async def _demo_with_real_rag(
    profile: AgentProfile, chunks: list[StructuredChunk]
) -> None:
    _banner("Step 3 - Ingest chunks into LlamaIndexRAG (real index, real embeddings)")

    if not (os.environ.get("EMBEDDING_API_KEY") and os.environ.get("EMBEDDING_MODEL")):
        print("  EMBEDDING_API_KEY / EMBEDDING_MODEL are not both set in the env.")
        print("  Skipping the RAG section. Fill them in .env to enable this step.")
        return

    try:
        from DefenseAgent.rag import LlamaIndexRAG
    except Exception as e:
        print(f"  Could not import LlamaIndexRAG ({type(e).__name__}: {e})")
        return

    try:
        rag = await LlamaIndexRAG.from_profile(profile, auto_load=False)
    except Exception as e:
        print(f"  Could not build LlamaIndexRAG ({type(e).__name__}: {e})")
        print("  Make sure llama-index-core and llama-index-embeddings-openai-like are installed.")
        return

    await rag.add_structured_chunks(chunks)
    print(f"  Ingested {len(chunks)} chunk(s) into the vector index.")

    _banner("Step 4 - Retrieve with a sample query")
    query = "What is the attack chain and which versions are vulnerable?"
    print(f"  query: {query!r}\n")

    try:
        hits = await rag.retrieve(query, limit=3)
    except Exception as e:
        print(f"  retrieve() failed: {type(e).__name__}: {e}")
        return

    print(f"  Got {len(hits)} hit(s):")
    for i, h in enumerate(hits, 1):
        snippet = h["text"][:160].replace("\n", " ")
        if len(h["text"]) > 160:
            snippet += "..."
        print(f"\n  hit#{i} score={h['score']:.4f}")
        print(f"    text: {snippet}")
        rids = h.get("metadata", {}).get("resource_ids", []) or []
        print(f"    resource_ids: {rids}")

    _banner("Step 5 - Look up each hit's images via get_resource_path()")
    seen: set[str] = set()
    for h in hits:
        for rid in h.get("metadata", {}).get("resource_ids", []) or []:
            if rid in seen:
                continue
            seen.add(rid)
            path = rag.get_resource_path(rid)
            print(f"  - {rid}")
            print(f"    -> {path}")
            if path and path.is_file():
                _try_describe_image(path)


# ---------- entry point ----------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="HTML -> StructuredChunk -> RAG -> resource lookup demo.",
    )
    parser.add_argument(
        "--with-rag",
        action="store_true",
        help="also ingest into a real LlamaIndexRAG and run retrieve() (needs API key).",
    )
    args = parser.parse_args()

    _banner("Setup - building demo workspace")
    html_path, image_real = _setup_workspace()
    profile = _make_demo_profile()
    print(f"  workspace: {WORKSPACE}")
    print(f"  sample HTML: {html_path}")
    print(f"  sample image: {'real PNG via Pillow' if image_real else 'placeholder bytes (install Pillow for a real one)'}")

    chunks = _demo_extraction(profile, html_path)

    if args.with_rag:
        asyncio.run(_demo_with_real_rag(profile, chunks))
    else:
        _banner("Done")
        print("  Tip: rerun with --with-rag to also exercise vector ingestion + retrieval.")
        print(f"  All artifacts live under: {WORKSPACE}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
