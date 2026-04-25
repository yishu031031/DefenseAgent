import math
import re
import sqlite3
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

from DefenseAgent.memory.embedding import (
    EmbeddingAdapter,
    MemoryError,
    MemoryNotFoundError,
)


MemoryKind = Literal["observation", "fact", "preference", "plan", "reflection"]


@dataclass
class MemoryRecord:
    """One immutable memory entry: id, content, timestamp, kind, importance, embedding, metadata."""
    id: str
    content: str
    timestamp: datetime
    kind: MemoryKind
    importance: float
    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


def cosine(a: list[float], b: list[float]) -> float:
    """Return cosine similarity in [-1, 1]; raises MemoryError on dim mismatch, 0.0 on zero vectors."""
    if len(a) != len(b):
        raise MemoryError(f"embedding dimension mismatch: {len(a)} vs {len(b)}")
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase `text`, split on non-alphanumerics, drop 1-char tokens; no stemming or stopwords."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) >= 2]


class BM25Index:
    """Okapi BM25 index with incremental add() and per-query score() over the whole corpus."""

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        """Store the standard BM25 hyperparameters; initialize empty per-doc and corpus state."""
        self.k1 = k1
        self.b = b
        self._doc_terms: dict[str, list[str]] = {}
        self._doc_freqs: dict[str, Counter[str]] = {}
        self._df: Counter[str] = Counter()
        self._total_doc_len = 0

    def __len__(self) -> int:
        """Return the number of documents currently in the index."""
        return len(self._doc_terms)

    def add(self, doc_id: str, text: str) -> None:
        """Tokenize `text`, register the doc under `doc_id`, update corpus stats."""
        tokens = tokenize(text)
        self._doc_terms[doc_id] = tokens
        self._doc_freqs[doc_id] = Counter(tokens)
        self._total_doc_len += len(tokens)
        for term in set(tokens):
            self._df[term] += 1

    def score(self, query: str) -> dict[str, float]:
        """Return {doc_id: bm25_score} for every doc in the index (0.0 for non-matchers)."""
        if not self._doc_terms:
            return {}

        query_terms = tokenize(query)
        scores = {doc_id: 0.0 for doc_id in self._doc_terms}
        if not query_terms:
            return scores

        n_docs = len(self._doc_terms)
        avgdl = self._total_doc_len / n_docs

        for term in query_terms:
            df_t = self._df.get(term, 0)
            if df_t == 0:
                continue
            idf = math.log((n_docs - df_t + 0.5) / (df_t + 0.5) + 1.0)

            for doc_id, freqs in self._doc_freqs.items():
                tf = freqs.get(term, 0)
                if tf == 0:
                    continue
                doc_len = len(self._doc_terms[doc_id])
                if avgdl:
                    length_norm = 1 - self.b + self.b * doc_len / avgdl
                else:
                    length_norm = 1 - self.b + self.b * doc_len
                denom = tf + self.k1 * length_norm
                scores[doc_id] += idf * (tf * (self.k1 + 1)) / denom

        return scores


def _default_clock() -> datetime:
    """Return the current UTC time."""
    return datetime.now(timezone.utc)


class MemoryStream:
    """Append-only record store with a BM25 index, same-kind dedup, and optional SQLite persistence."""

    def __init__(
        self,
        embedding_adapter: EmbeddingAdapter,
        *,
        clock: Callable[[], datetime] | None = None,
        dedup_threshold: float | None = 0.95,
        db_path: str | Path | None = None,
    ) -> None:
        """Bind the adapter/clock/dedup; when `db_path` is set, open SQLite and rehydrate prior records."""
        self.embedding_adapter = embedding_adapter
        self.dedup_threshold = dedup_threshold
        if clock is None:
            self._clock = _default_clock
        else:
            self._clock = clock
        self._records: list[MemoryRecord] = []
        self._records_by_id: dict[str, MemoryRecord] = {}
        self._records_by_kind: dict[MemoryKind, list[MemoryRecord]] = {}
        self._bm25 = BM25Index()
        self._db: sqlite3.Connection | None = None
        self.db_path: Path | None = None
        if db_path is not None:
            self._open_db(db_path)

    def _open_db(self, db_path: str | Path) -> None:
        """Open or create the SQLite file, rehydrate every persisted record into the in-memory indexes."""
        from DefenseAgent.memory import sqlite_store

        self.db_path = Path(db_path)
        self._db = sqlite_store.open_db(self.db_path)
        for record in sqlite_store.load_records(self._db):
            self._index_record(record)

    def close(self) -> None:
        """Close the SQLite connection (no-op if this stream was built without a db_path)."""
        if self._db is not None:
            self._db.close()
            self._db = None

    def __len__(self) -> int:
        """Return the number of stored records."""
        return len(self._records)

    def get_all(self) -> list[MemoryRecord]:
        """Return every record in insertion (chronological) order."""
        return list(self._records)

    def get_recent(self, n: int) -> list[MemoryRecord]:
        """Return the last `n` records in most-recent-first order; empty when n ≤ 0."""
        if n <= 0:
            return []
        return list(reversed(self._records[-n:]))

    def get_by_id(self, record_id: str) -> MemoryRecord:
        """Return the record with `record_id`; raises MemoryNotFoundError when missing."""
        if record_id not in self._records_by_id:
            raise MemoryNotFoundError(f"no record with id={record_id!r}")
        return self._records_by_id[record_id]

    @property
    def bm25(self) -> BM25Index:
        """Expose the BM25 index so the retriever can score queries against it."""
        return self._bm25

    async def add(
        self,
        content: str,
        *,
        kind: MemoryKind = "observation",
        importance: float = 5.0,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        """Embed, dedup-check (same kind), append, return the stored (or pre-existing) record."""
        embedding = await self.embedding_adapter.embed(content)

        if self.dedup_threshold is not None:
            duplicate = self._find_same_kind_duplicate(embedding, kind)
            if duplicate is not None:
                return duplicate

        if metadata is None:
            metadata_copy: dict[str, Any] = {}
        else:
            metadata_copy = dict(metadata)

        record = MemoryRecord(
            id=uuid.uuid4().hex,
            content=content,
            timestamp=self._clock(),
            kind=kind,
            importance=importance,
            embedding=embedding,
            metadata=metadata_copy,
        )
        self._append_record(record)
        return record

    def add_record(self, record: MemoryRecord) -> None:
        """Append a fully-formed record directly (bypasses embedding + dedup)."""
        self._append_record(record)

    def _index_record(self, record: MemoryRecord) -> None:
        """Insert `record` into the in-memory indexes (records list, id/kind dicts, BM25)."""
        self._records.append(record)
        self._records_by_id[record.id] = record
        self._records_by_kind.setdefault(record.kind, []).append(record)
        self._bm25.add(record.id, record.content)

    def _append_record(self, record: MemoryRecord) -> None:
        """Index `record` in memory and persist to SQLite when this stream has a db_path."""
        self._index_record(record)
        if self._db is not None:
            from DefenseAgent.memory import sqlite_store

            sqlite_store.write_record(self._db, record)

    def _find_same_kind_duplicate(
        self,
        embedding: list[float],
        kind: MemoryKind,
    ) -> MemoryRecord | None:
        """Return the first same-kind record whose cosine similarity meets dedup_threshold, else None."""
        threshold = self.dedup_threshold
        for existing in self._records_by_kind.get(kind, ()):
            if cosine(existing.embedding, embedding) >= threshold:
                return existing
        return None
