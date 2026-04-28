"""Pretty-print an agent's memory stream in chronological order (no embeddings).

Usage:
    python scripts/dump_memory.py                                                              # defaults to the example agent's stream
    python scripts/dump_memory.py DefenseAgent/examples/example_agent/memory/stream.db --kind observation
    python scripts/dump_memory.py DefenseAgent/examples/example_agent/memory/stream.db --limit 20

Reads the SQLite file directly — no LLM calls, no embedding decode. Each record
prints as one block:

    [kind         imp= 7.0] 2026-04-24T18:13:55+00:00  <record_id>
        content text (wrapped)
        metadata = {...}          # only when non-empty

If the file does not exist yet, the script exits 2 with a clear message so you
don't accidentally probe a stale path.
"""
import argparse
import json
import sqlite3
import sys
import textwrap
from pathlib import Path


from DefenseAgent.examples import EXAMPLE_AGENT_DIR

DEFAULT_DB = EXAMPLE_AGENT_DIR / "memory" / "stream.db"


_SELECT_SQL_TEMPLATE = """\
SELECT id, content, kind, importance, timestamp, metadata_json
FROM   memory_records
{where}
ORDER  BY timestamp ASC, id ASC
{limit}
"""


def main(argv: list[str]) -> int:
    """Parse CLI args, query the SQLite file, print records in time order; returns an exit code."""
    parser = argparse.ArgumentParser(
        description="Dump an agent's memory stream to stdout (no embeddings)."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=str(DEFAULT_DB),
        help="Path to stream.db (defaults to the example agent's bundle).",
    )
    parser.add_argument(
        "--kind",
        choices=["observation", "fact", "preference", "plan", "reflection"],
        help="Filter by memory kind.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Show at most N records (0 = all).",
    )
    args = parser.parse_args(argv)

    db_file = Path(args.path)
    if not db_file.is_file():
        print(f"[dump_memory] no stream at {db_file}", file=sys.stderr)
        print(
            "[dump_memory] run a demo or the agent first to populate it.",
            file=sys.stderr,
        )
        return 2

    rows = _query(db_file, kind=args.kind, limit=args.limit)
    if not rows:
        print(f"[dump_memory] {db_file}: (empty)")
        return 0

    print(f"[dump_memory] {db_file}")
    print(f"[dump_memory] {len(rows)} record(s)\n")
    for row in rows:
        _print_row(row)
    return 0


def _query(path: Path, *, kind: str | None, limit: int) -> list[tuple]:
    """Return filtered rows in chronological order; opens a read-only connection."""
    where_sql = "WHERE kind = ?" if kind else ""
    limit_sql = f"LIMIT {int(limit)}" if limit and limit > 0 else ""
    sql = _SELECT_SQL_TEMPLATE.format(where=where_sql, limit=limit_sql)
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        if kind:
            cursor = conn.execute(sql, (kind,))
        else:
            cursor = conn.execute(sql)
        return list(cursor)
    finally:
        conn.close()


def _print_row(row: tuple) -> None:
    """Render one row as a header line + wrapped content block + metadata if non-empty."""
    rid, content, kind, importance, timestamp, metadata_json = row
    print(f"[{kind:<12} imp={importance:>4.1f}]  {timestamp}  {rid}")
    wrapped = textwrap.fill(
        content,
        width=88,
        initial_indent="    ",
        subsequent_indent="    ",
    )
    print(wrapped)
    if metadata_json and metadata_json != "{}":
        try:
            parsed = json.loads(metadata_json)
        except json.JSONDecodeError:
            parsed = metadata_json
        print(f"    metadata = {parsed}")
    print()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
