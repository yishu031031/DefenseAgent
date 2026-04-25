import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterator

import numpy as np

from DefenseAgent.memory.stream import MemoryKind, MemoryRecord


_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS memory_records (
    id            TEXT PRIMARY KEY,
    content       TEXT NOT NULL,
    kind          TEXT NOT NULL,
    importance    REAL NOT NULL,
    timestamp     TEXT NOT NULL,
    embedding     BLOB NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_memory_kind      ON memory_records(kind);
CREATE INDEX IF NOT EXISTS idx_memory_timestamp ON memory_records(timestamp);
"""


_INSERT_SQL = """\
INSERT INTO memory_records
    (id, content, kind, importance, timestamp, embedding, metadata_json)
VALUES
    (?, ?, ?, ?, ?, ?, ?)
"""


_SELECT_ALL_SQL = """\
SELECT id, content, kind, importance, timestamp, embedding, metadata_json
FROM   memory_records
ORDER  BY timestamp ASC, id ASC
"""


def open_db(path: str | Path) -> sqlite3.Connection:
    """Open (or create) a SQLite file at `path`, apply schema + pragmas, return the connection."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(file_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(_SCHEMA_SQL)
    conn.commit()
    return conn


def write_record(conn: sqlite3.Connection, record: MemoryRecord) -> None:
    """Persist one MemoryRecord; commits immediately so a crash leaves at most the last in-flight add behind."""
    conn.execute(
        _INSERT_SQL,
        (
            record.id,
            record.content,
            record.kind,
            float(record.importance),
            record.timestamp.isoformat(),
            _embedding_to_blob(record.embedding),
            json.dumps(record.metadata, ensure_ascii=False, sort_keys=True),
        ),
    )
    conn.commit()


def load_records(conn: sqlite3.Connection) -> Iterator[MemoryRecord]:
    """Yield every persisted record in chronological (timestamp, id) order for rehydration at startup."""
    for row in conn.execute(_SELECT_ALL_SQL):
        yield _row_to_record(row)


def _row_to_record(row: tuple) -> MemoryRecord:
    """Reassemble a MemoryRecord from one SELECT row."""
    rid, content, kind, importance, timestamp, embedding_blob, metadata_json = row
    return MemoryRecord(
        id=rid,
        content=content,
        kind=_parse_kind(kind),
        importance=float(importance),
        timestamp=datetime.fromisoformat(timestamp),
        embedding=_embedding_from_blob(embedding_blob),
        metadata=json.loads(metadata_json) if metadata_json else {},
    )


def _parse_kind(value: str) -> MemoryKind:
    """Narrow a plain TEXT value into the MemoryKind Literal; unknown kinds raise."""
    allowed = {"observation", "fact", "preference", "plan", "reflection"}
    if value not in allowed:
        raise ValueError(f"unknown memory kind in DB: {value!r}")
    return value  # type: ignore[return-value]


def _embedding_to_blob(vec: list[float]) -> bytes:
    """Pack a Python float list into a numpy float32 byte string (4 bytes per dim)."""
    return np.asarray(vec, dtype=np.float32).tobytes()


def _embedding_from_blob(blob: bytes) -> list[float]:
    """Unpack a float32 byte string back into a Python float list."""
    return np.frombuffer(blob, dtype=np.float32).tolist()
