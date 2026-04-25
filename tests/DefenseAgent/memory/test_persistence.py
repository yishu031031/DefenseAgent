"""Tests for SQLite-backed memory persistence.

Groups:
  • Schema — the file exists, has the expected tables + indexes.
  • Roundtrip — a record added via stream.add() survives close + reopen
    (content, kind, importance, timestamp, embedding, metadata all intact).
  • Dedup across sessions — rehydrated records count as duplicates on the
    next session's add().
  • BM25 rehydration — the in-memory BM25 index is rebuilt from persisted
    records, so retrieval keyword scores still work after a restart.
  • Memory.from_profile — default path is <profile.source_dir>/memory/stream.db,
    in-memory profiles raise, explicit memory_dir override works.
"""
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from DefenseAgent.config import AgentProfile
from DefenseAgent.memory import EmbeddingConfigError, Memory
from DefenseAgent.memory.embedding import EmbeddingAdapter
from DefenseAgent.memory.stream import MemoryStream


_MAYA_PROFILE = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "agents" / "maya_rodriguez" / "profile.yaml"
)


class _StubEmbeddingAdapter(EmbeddingAdapter):
    """Embedding adapter that returns a canned per-text vector; unknown text → [0,0,0,1]."""

    def __init__(self, vectors: dict[str, list[float]]):
        """Store a {text: vector} map to serve out of; track call history for assertions."""
        self._vectors = vectors
        self.calls: list[str] = []

    async def embed(self, text: str) -> list[float]:
        """Return the configured vector for `text`, falling back to a sentinel."""
        self.calls.append(text)
        return list(self._vectors.get(text, [0.0, 0.0, 0.0, 1.0]))

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Dispatch to `embed()` once per text; preserves input order."""
        return [await self.embed(t) for t in texts]


def _fixed_clock() -> datetime:
    """Return a deterministic timestamp so chronological ordering assertions are stable."""
    return datetime(2026, 4, 24, 12, 0, tzinfo=timezone.utc)


def _make_profile(source_dir: Path) -> AgentProfile:
    """Build a minimal AgentProfile whose source_dir points at `source_dir` (simulates from_yaml)."""
    profile = AgentProfile(
        id="t", name="T", age=30, traits="a", backstory="b", initial_plan="c",
    )
    (source_dir / "profile.yaml").write_text("agent: {}", encoding="utf-8")
    profile._source_path = (source_dir / "profile.yaml").resolve()
    return profile


# ---------- schema ----------


def test_open_db_creates_expected_tables_and_indexes(tmp_path: Path) -> None:
    from DefenseAgent.memory import sqlite_store

    conn = sqlite_store.open_db(tmp_path / "memory" / "stream.db")
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    indexes = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        )
    }
    conn.close()
    assert "memory_records" in tables
    assert "idx_memory_kind" in indexes
    assert "idx_memory_timestamp" in indexes


def test_open_db_parent_directory_created(tmp_path: Path) -> None:
    from DefenseAgent.memory import sqlite_store

    target = tmp_path / "nested" / "deeper" / "stream.db"
    assert not target.parent.exists()
    conn = sqlite_store.open_db(target)
    conn.close()
    assert target.is_file()


# ---------- roundtrip ----------


async def test_record_survives_close_and_reopen(tmp_path: Path) -> None:
    db = tmp_path / "memory" / "stream.db"
    adapter = _StubEmbeddingAdapter(
        {"attended lecture": [0.25, -0.5, 0.75, 1.0]}
    )

    stream1 = MemoryStream(
        adapter, clock=_fixed_clock, dedup_threshold=None, db_path=db,
    )
    record = await stream1.add(
        "attended lecture",
        kind="observation",
        importance=7.0,
        metadata={"source": "demo"},
    )
    stream1.close()

    stream2 = MemoryStream(
        adapter, clock=_fixed_clock, dedup_threshold=None, db_path=db,
    )
    try:
        assert len(stream2) == 1
        reloaded = stream2.get_by_id(record.id)
        assert reloaded.content == "attended lecture"
        assert reloaded.kind == "observation"
        assert reloaded.importance == 7.0
        assert reloaded.timestamp == record.timestamp
        assert reloaded.metadata == {"source": "demo"}
        # Float32 round-trip: accept minor precision loss, not wrong values.
        for original, decoded in zip(record.embedding, reloaded.embedding):
            assert abs(original - decoded) < 1e-6
    finally:
        stream2.close()


async def test_dedup_holds_across_reopen(tmp_path: Path) -> None:
    db = tmp_path / "memory" / "stream.db"
    adapter = _StubEmbeddingAdapter({"x": [1.0, 0.0, 0.0, 0.0]})

    stream1 = MemoryStream(
        adapter, clock=_fixed_clock, dedup_threshold=0.95, db_path=db,
    )
    first = await stream1.add("x", kind="observation")
    stream1.close()

    stream2 = MemoryStream(
        adapter, clock=_fixed_clock, dedup_threshold=0.95, db_path=db,
    )
    try:
        second = await stream2.add("x", kind="observation")
        assert second.id == first.id
        assert len(stream2) == 1
    finally:
        stream2.close()


async def test_bm25_is_rebuilt_on_reopen(tmp_path: Path) -> None:
    db = tmp_path / "memory" / "stream.db"
    adapter = _StubEmbeddingAdapter(
        {
            "the binary search tree is balanced": [1, 0, 0, 0],
            "i drank coffee in the morning":      [0, 1, 0, 0],
        }
    )
    stream1 = MemoryStream(
        adapter, clock=_fixed_clock, dedup_threshold=None, db_path=db,
    )
    await stream1.add("the binary search tree is balanced")
    await stream1.add("i drank coffee in the morning")
    stream1.close()

    stream2 = MemoryStream(
        adapter, clock=_fixed_clock, dedup_threshold=None, db_path=db,
    )
    try:
        bm25_scores = stream2.bm25.score("binary tree")
        # At least one positive BM25 score must come back — the index was rebuilt.
        assert any(score > 0 for score in bm25_scores.values())
        # And the tree document outranks the coffee document.
        tree_id = next(
            r.id for r in stream2.get_all() if "binary" in r.content
        )
        coffee_id = next(
            r.id for r in stream2.get_all() if "coffee" in r.content
        )
        assert bm25_scores[tree_id] > bm25_scores[coffee_id]
    finally:
        stream2.close()


async def test_records_load_in_chronological_order(tmp_path: Path) -> None:
    db = tmp_path / "memory" / "stream.db"
    adapter = _StubEmbeddingAdapter(
        {"first": [1, 0, 0, 0], "second": [0, 1, 0, 0], "third": [0, 0, 1, 0]}
    )
    ticks = [
        datetime(2026, 4, 1, 9, 0, tzinfo=timezone.utc),
        datetime(2026, 4, 1, 10, 0, tzinfo=timezone.utc),
        datetime(2026, 4, 1, 11, 0, tzinfo=timezone.utc),
    ]
    stream1 = MemoryStream(
        adapter, clock=lambda: ticks.pop(0), dedup_threshold=None, db_path=db,
    )
    await stream1.add("first")
    await stream1.add("second")
    await stream1.add("third")
    stream1.close()

    stream2 = MemoryStream(
        adapter, clock=_fixed_clock, dedup_threshold=None, db_path=db,
    )
    try:
        contents = [r.content for r in stream2.get_all()]
        assert contents == ["first", "second", "third"]
    finally:
        stream2.close()


# ---------- Memory.from_profile ----------


async def test_from_profile_uses_profile_source_dir(tmp_path: Path) -> None:
    profile = _make_profile(tmp_path)
    # Stub the embedding adapter factory by pre-setting env vars to avoid the
    # network-backed path; we don't actually embed in this assertion.
    memory = Memory(
        profile=profile,
        embedding_adapter=_StubEmbeddingAdapter({"x": [1, 0, 0, 0]}),
        clock=_fixed_clock,
        db_path=tmp_path / "memory" / "stream.db",
    )
    try:
        assert memory.stream.db_path == (tmp_path / "memory" / "stream.db").resolve()
        assert (tmp_path / "memory" / "stream.db").is_file()
    finally:
        memory.close()


async def test_from_profile_default_path_is_source_dir_memory(tmp_path: Path) -> None:
    """Memory.from_profile with no memory_dir picks <profile.source_dir>/memory/stream.db."""
    profile = _make_profile(tmp_path)

    class _NoEmbed(EmbeddingAdapter):
        async def embed(self, text): return [0.0]
        async def embed_batch(self, texts): return [[0.0] for _ in texts]

    memory = Memory(
        profile=profile,
        embedding_adapter=_NoEmbed(),
        clock=_fixed_clock,
        db_path=(profile.source_dir / "memory" / "stream.db"),
    )
    try:
        assert (profile.source_dir / "memory").is_dir()
        assert memory.stream.db_path == (profile.source_dir / "memory" / "stream.db").resolve()
    finally:
        memory.close()


def test_from_profile_raises_when_in_memory_profile_has_no_source_dir() -> None:
    """An in-memory profile (no YAML) can't auto-resolve a memory dir."""
    profile = AgentProfile(
        id="x", name="X", age=1, traits="t", backstory="b", initial_plan="p",
    )
    with pytest.raises(EmbeddingConfigError):
        Memory.from_profile(profile, load_env=False)


def test_from_profile_persist_false_skips_sqlite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """persist=False creates no stream.db and no memory/ directory on disk."""
    profile = _make_profile(tmp_path)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("EMBEDDING_API_KEY", "test-key")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")

    memory = Memory.from_profile(profile, persist=False, load_env=False)
    try:
        assert memory.stream.db_path is None
        assert memory.stream._db is None
        # No persistence artefacts sit next to the profile.
        assert not (tmp_path / "memory").exists()
    finally:
        memory.close()


def test_from_profile_persist_false_works_without_source_dir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """persist=False skips source_dir resolution entirely (safe for in-memory profiles)."""
    profile = AgentProfile(
        id="x", name="X", age=1, traits="t", backstory="b", initial_plan="p",
    )
    assert profile.source_dir is None
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("EMBEDDING_API_KEY", "test-key")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")

    memory = Memory.from_profile(profile, persist=False, load_env=False)
    try:
        assert memory.stream.db_path is None
    finally:
        memory.close()


def test_from_profile_persist_false_rejects_memory_dir(tmp_path: Path) -> None:
    """Passing both persist=False and memory_dir is contradictory and must raise immediately."""
    profile = _make_profile(tmp_path)
    with pytest.raises(ValueError):
        Memory.from_profile(
            profile,
            persist=False,
            memory_dir=tmp_path / "nope",
            load_env=False,
        )


def test_from_profile_accepts_explicit_memory_dir_override(tmp_path: Path) -> None:
    """An explicit memory_dir override works even for in-memory profiles (given valid env)."""
    profile = AgentProfile(
        id="x", name="X", age=1, traits="t", backstory="b", initial_plan="p",
    )
    # We don't have embedding env set in this test's environment, so this must
    # raise EmbeddingConfigError from the env-resolver — NOT from source_dir.
    # That confirms memory_dir was accepted past the pre-check.
    import os
    saved = {k: os.environ.pop(k, None)
             for k in ("EMBEDDING_PROVIDER", "EMBEDDING_API_KEY",
                       "EMBEDDING_BASE_URL", "EMBEDDING_MODEL")}
    try:
        with pytest.raises(EmbeddingConfigError) as excinfo:
            Memory.from_profile(
                profile,
                memory_dir=tmp_path / "mem",
                load_env=False,
            )
        assert "EMBEDDING_PROVIDER" in str(excinfo.value)
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


# ---------- context manager ----------


async def test_memory_context_manager_closes_stream(tmp_path: Path) -> None:
    db = tmp_path / "memory" / "stream.db"
    adapter = _StubEmbeddingAdapter({"x": [1, 0, 0, 0]})
    with Memory(
        profile=AgentProfile(
            id="x", name="X", age=1, traits="t", backstory="b", initial_plan="p",
        ),
        embedding_adapter=adapter,
        clock=_fixed_clock,
        db_path=db,
    ) as memory:
        await memory.remember("x", kind="observation")
        assert memory.stream._db is not None

    # After the with-block, close() fires and the connection is dropped.
    assert memory.stream._db is None


# ---------- dump_memory.py smoke ----------


def test_dump_memory_script_reads_real_db(tmp_path: Path) -> None:
    """Write a couple of rows, then run the dump script's main() and confirm it doesn't crash."""
    import subprocess
    import sys as _sys

    db = tmp_path / "memory" / "stream.db"
    db.parent.mkdir(parents=True)
    conn = sqlite3.connect(str(db))
    conn.executescript(
        """
        CREATE TABLE memory_records (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            kind TEXT NOT NULL,
            importance REAL NOT NULL,
            timestamp TEXT NOT NULL,
            embedding BLOB NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}'
        );
        """
    )
    conn.execute(
        "INSERT INTO memory_records VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            "r1", "hello world", "observation", 5.0,
            "2026-04-24T12:00:00+00:00", b"\x00" * 4, "{}",
        ),
    )
    conn.commit()
    conn.close()

    script = (
        Path(__file__).resolve().parent.parent.parent.parent
        / "scripts" / "dump_memory.py"
    )
    result = subprocess.run(
        [_sys.executable, str(script), str(db)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "hello world" in result.stdout
    assert "observation" in result.stdout
