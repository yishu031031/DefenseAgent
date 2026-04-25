# Module 4 — Memory (Writing, Storage, Retrieval) Design

**Date:** 2026-04-22
**Status:** Superseded for the implementation; the design rationale below is preserved for history.
**Module position:** 4 of N. Foundation for the cognitive loop, which reads and writes memories every step.

> **Amendment (2026-04-25): Replaced by ms-agent inheritance + mem0 backend.**
> The hand-rolled hybrid retrieval / RRF / kind-aware decay / SQLite persistence described in the rest of this spec **has been removed**. `DefenseAgent.memory` now consists of thin adapters that inherit from `ms_agent.memory` classes:
>
> - `DefenseAgent.memory.DefaultMemory(ms_agent.memory.DefaultMemory)` — wraps mem0 (vector storage, LLM fact extraction, message-block diff/rollback). Storage backend defaults to Qdrant on-disk at `<profile.source_dir>/memory/`. Retrieval is mem0's vector `search()`. The Park-style composite scoring (recency × importance × relevance), the BM25+dense RRF fusion, the typed `MemoryKind` Literal, and the kind-aware decay rules are all **gone** — mem0 doesn't model them. Memory typing now lives in mem0's free-form `memory_type` string field (e.g., `"trajectory"`, `"outcome"`, `"failure"`, `"reflection"`).
> - `DefenseAgent.memory.ContextCompressor(ms_agent.memory.condenser.ContextCompressor)` — token-overflow compaction with prune + summarize. New capability for DefenseAgent; the original spec had no token-budget compactor.
> - `DefenseAgent.memory.SharedMemoryManager(ms_agent.memory.memory_manager.SharedMemoryManager)` — process-wide singleton keyed by `(type, user_id, path)`.
>
> The boundary work (config translation `AgentProfile → omegaconf.DictConfig`, `Message ↔ ms_agent.Message` field copying) lives in `DefenseAgent.memory._bridge`. The `memory_mapping` dict re-exports ms-agent's registry of pluggable memory subclasses. Total adapter LOC: ~250 vs the ~600 we had before.
>
> **What broke:** `MemoryStream`, `MemoryRetriever`, `BM25Index`, `cosine`, `sqlite_store`, `MemoryRecord`, `MemoryKind`, `ScoredMemory` are deleted. `Memory.remember(content, kind=..., importance=...)` / `Memory.recall(query, top_k=...)` are gone — the new contract is `await memory.run(messages) -> messages` (ms-agent's pattern: ingest + search + inject in one call) plus `await memory.add(messages, memory_type=...)` for explicit ingestion and `memory.search_records(query, limit, memory_type)` for explicit retrieval. `MemoryConfig` lost `max_working_memory_tokens`, `retrieval_top_k`, `recency_weight`, `importance_weight`, `relevance_weight` and gained `search_limit`, `history_mode`, `context_limit`, `prune_protect`, `prune_minimum`, `reserved_buffer`, `enable_summary`, `is_retrieve`, `ignore_roles`, `ignore_fields`.
>
> **Rationale**: the user wanted to standardize on ms-agent's memory scheme to avoid carrying our own retrieval stack. Inheriting from their classes (rather than reimplementing) gets the full mem0 implementation including features we never had — message-block diff/rollback, LLM fact extraction, pluggable vector stores — while still letting us layer on DefenseAgent-specific helpers (`search_records`, `memory_type` filtering on `get_all`).
>
> The previous "Amendment (2026-04-24): SQLite persistence" block is also superseded — SQLite is gone, replaced by mem0's pluggable vector store.

> **Amendment (2026-04-24): SQLite persistence.** *(now historical)*
> The original spec deferred persistence until pause/resume needed it; that time arrived when we adopted per-agent bundles. `MemoryStream` now accepts an optional `db_path` — when supplied, every `add()` writes a row to `<db_path>` (SQLite, WAL, per-record commit) and every `__init__` rehydrates existing rows. `Memory.from_profile(profile)` is the new front door: it resolves `<profile.source_dir>/memory/stream.db` automatically so each agent gets its own database beside its profile. Embeddings are serialized via `numpy.float32.tobytes()` / `numpy.frombuffer()`; BM25 is still RAM-only and rebuilt from a table scan on open. The helper `scripts/dump_memory.py` prints a stream in chronological order without embeddings for debugging. The "No persistence" line in the original spec is superseded.

## Purpose

Give the harness a **queryable record of the agent's experience** with a modern retrieval pipeline. Three jobs, nothing more:

1. **Writing** — embed a new observation/fact/plan/preference/reflection, duplicate-check, timestamp, return a structured `MemoryRecord`.
2. **Storage** — an append-only stream of records plus an incrementally-updated inverted index (for BM25).
3. **Retrieval** — given a query, return the top-k memories using **hybrid retrieval**: dense embedding + sparse BM25 fused by Reciprocal Rank Fusion (RRF), then weighted by recency × importance × per-kind rules from the agent's profile.

This module does *not* reimplement the 2023 Generative Agents formula literally. It keeps the **three scoring axes** (recency, importance, relevance) because those are genuinely useful cognitive signals, but swaps vanilla cosine for a hybrid retriever that's the current industry standard.

**Explicitly out of scope for this module:**
- **Reflection** (LLM-generated synthesis of recent memories into higher-level insights) — Module 5.
- **LLM-based importance scoring** — Module 5 will add this. For now, callers supply importance manually; default 5.0.
- **Reranker stage** (cross-encoder on top-K) — useful when retrieval quality becomes the bottleneck; Module 5 or later.
- **File persistence** (JSON/SQLite dump and load) — deferred until pause/resume needs it.
- **MemGPT-style self-editing tiers** — requires tool-calling; Module 6+.
- **Fact extraction** from raw observations via LLM (Mem0-style) — crosses into Module 1/5 territory.

## The four big ideas that shape the design

1. **Memory is append-only.** Records are immutable once created. Reflections and updates are *new* records, never edits. Keeps storage/retrieval simple and time-travel trivial.

2. **Retrieval is the quality lever.** With thousands of memories, picking ~10 for the LLM's context is where quality comes from. The retriever is where most of the module's code lives.

3. **Hybrid retrieval beats pure cosine.** Dense embeddings are great for paraphrase but miss keyword-ish queries ("what did I say about Greg?"). BM25 catches exactly those. RRF fuses the two rankings without needing to calibrate score scales across rankers. This is standard production practice in 2025.

4. **Memory kind changes behavior.** Not every record is an event. Facts ("I'm allergic to peanuts") and preferences ("I hate morning classes") are *stable*; applying recency decay to them would be wrong. Plans have status. The retriever branches on `kind` to apply the right rule.

## Design

### Data model

```python
# DefenseAgent/memory/types.py

MemoryKind = Literal[
    "observation",   # raw event ("attended the 9 AM lecture")            — decays with time
    "fact",          # stable proposition ("I'm in my second year")        — no recency decay
    "preference",    # persistent disposition ("hate morning classes")     — no recency decay
    "plan",          # intent ("study graphs tonight")                     — filtered by metadata["status"]
    "reflection",    # synthesized insight (Module 5 produces these)       — decays like observations
]

@dataclass
class MemoryRecord:
    id: str                    # UUID4 hex, assigned at creation
    content: str               # human-readable text
    timestamp: datetime        # UTC, millisecond precision
    kind: MemoryKind           # determines retrieval behavior
    importance: float          # 1.0–10.0, caller-supplied (Module 5 adds LLM scoring)
    embedding: list[float]     # non-empty
    metadata: dict             # freeform; plans use metadata["status"] = "active" | "done"

@dataclass
class ScoredMemory:
    """A memory record with its retrieval score components — returned by MemoryRetriever."""
    record: MemoryRecord
    score: float                # composite weighted score
    recency_score: float        # [0, 1]; 1.0 for facts/preferences (no decay)
    importance_score: float     # [0.1, 1.0]
    relevance_score: float      # [0, 1] — normalized hybrid RRF score
    dense_rank: int | None      # 1-indexed rank in dense ranking; None if not in top-candidates
    sparse_rank: int | None     # 1-indexed rank in BM25 ranking; None if missing
```

`ScoredMemory` exposes every retrieval component, including the per-ranker ranks, so callers and tests can understand exactly **why** a memory ranked where it did. Observability first.

### `.env` block (already added on 2026-04-22)

```dotenv
EMBEDDING_PROVIDER=qwen                 # one of: openai, qwen, vllm
EMBEDDING_API_KEY=                      # blank until populated
EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_MODEL=text-embedding-v3
```

Validation (factory-level):
- `EMBEDDING_PROVIDER` empty or not in the supported set → `EmbeddingConfigError`.
- `EMBEDDING_MODEL` empty → `EmbeddingConfigError`.
- `EMBEDDING_BASE_URL` empty for `qwen` or `vllm` → error. Empty for `openai` → OK (SDK default).
- `EMBEDDING_API_KEY` empty for `openai`/`qwen` → error. Empty for `vllm` → defaulted to `"token-not-needed"`.

### Embedding adapter

```python
# DefenseAgent/memory/embedding_adapter.py

class EmbeddingAdapter(ABC):
    @abstractmethod
    async def embed(self, text: str) -> list[float]: ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


class OpenAICompatibleEmbeddingAdapter(EmbeddingAdapter):
    """Covers OpenAI, Qwen (DashScope), vLLM — they all speak /embeddings."""
    def __init__(self, *, api_key: str, base_url: str, model: str, client=None): ...


def make_embedding_adapter_from_env(dotenv_path=None, *, load_env=True) -> EmbeddingAdapter:
    """Read EMBEDDING_* env block and build the configured adapter."""
```

**Design decisions:**
- **Client-injection test seam**, identical to Module 1's chat adapters. Tests pass a `MagicMock`; production creates `AsyncOpenAI` lazily.
- **No override tier on `EMBEDDING_*`** for this pass. One block. Can add later if needed.
- **Dimension unchecked.** Different models return different dimensions; the stream records `len(embedding)` implicitly. Mixing dimensions in one stream breaks cosine — single-embedding-model-per-stream is the caller's responsibility.

### BM25 index (pure Python)

```python
# DefenseAgent/memory/bm25.py

class BM25Index:
    """Okapi BM25 over the content of memory records. Pure Python, ~40 lines core.

    k1 = 1.5, b = 0.75 (standard defaults).
    """
    def __init__(self, *, k1: float = 1.5, b: float = 0.75): ...

    def add(self, doc_id: str, text: str) -> None:
        """Tokenize `text`, update per-document term counts, update corpus stats."""

    def score(self, query: str) -> dict[str, float]:
        """Return {doc_id: bm25_score} for every doc (includes zeros for non-matchers)."""

    def __len__(self) -> int: ...

def tokenize(text: str) -> list[str]:
    """Lowercase + split on non-alnum + strip tiny tokens. No stemming for this pass."""
```

**Why pure Python, not `rank-bm25`:**
1. Pedagogical value — BM25 is a short, classical formula that's worth reading.
2. Zero new dependencies.
3. The incremental update semantics (adds happen one at a time) are straightforward.

Tokenization is intentionally simple: lowercase, split on non-alphanumerics, drop 1-char tokens. No stemming, no stopwords, no language detection. Modern BM25 variants add those; we don't need them yet.

Corpus stats kept alongside the index:
- `doc_count` — total documents.
- `avg_doc_len` — running average of tokens per doc.
- `df[term]` — document frequency per term.
- `idf[term]` — recomputed lazily on score() from `df[term]` and `doc_count`.

### `MemoryStream`

```python
# DefenseAgent/memory/stream.py

class MemoryStream:
    def __init__(
        self,
        embedding_adapter: EmbeddingAdapter,
        *,
        clock: Callable[[], datetime] | None = None,
        dedup_threshold: float | None = 0.95,   # cosine threshold; None disables
    ): ...

    async def add(
        self,
        content: str,
        *,
        kind: MemoryKind = "observation",
        importance: float = 5.0,
        metadata: dict | None = None,
    ) -> MemoryRecord:
        """Embed, dedup-check, timestamp, append, return the record.

        If dedup_threshold is set and a memory of the same kind has cosine
        similarity > threshold with the new embedding, the existing record is
        returned unchanged and nothing is added. The caller can detect dedup
        by comparing `returned.id` against the record's previous absence.
        """

    def add_record(self, record: MemoryRecord) -> None:
        """Append a pre-built record (e.g. from a future persistence layer).
        Also updates the BM25 index. Bypasses dedup check by design.
        """

    def get_recent(self, n: int) -> list[MemoryRecord]: ...
    def get_all(self) -> list[MemoryRecord]: ...
    def get_by_id(self, record_id: str) -> MemoryRecord: ...        # raises MemoryNotFoundError
    def __len__(self) -> int: ...

    # Internal: the BM25Index lives here so adds can update it.
    # Exposed via a read-only property for the retriever:
    @property
    def bm25(self) -> BM25Index: ...
```

**Design decisions:**
- **`add()` is async** (embedding call); **`add_record()` is sync** (no embedding).
- **Dedup within-kind only.** An observation and a fact with similar content are semantically different; dedup across kinds would conflate them.
- **Dedup returns the existing record.** No error, no importance bump, no message. Callers who care can inspect `record.timestamp != now` to detect dedup.
- **BM25 index is owned by the stream**, updated incrementally on `add()` / `add_record()`. The retriever queries it.

### `MemoryRetriever` — hybrid retrieval

```python
# DefenseAgent/memory/retriever.py

RRF_K = 60   # Cormack et al. (2009), the accepted default for RRF

class MemoryRetriever:
    def __init__(
        self,
        stream: MemoryStream,
        embedding_adapter: EmbeddingAdapter,
        memory_config: MemoryConfig,                            # from Module 2
        *,
        clock: Callable[[], datetime] | None = None,
        recency_half_life_hours: float = 24.0,
    ): ...

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,                                # defaults to memory_config.retrieval_top_k
        kinds: tuple[MemoryKind, ...] | None = None,             # None = all kinds
    ) -> list[ScoredMemory]:
        """Hybrid-retrieve top-k memories for `query`."""
```

#### Retrieval pipeline

```
retrieve(query, top_k, kinds)
│
├─ Step 1: candidate set
│     records = stream.get_all()
│     • filter out plans with metadata["status"] == "done"
│     • if `kinds` is provided, filter records by kind
│
├─ Step 2: dense ranking
│     query_emb = embedding_adapter.embed(query)
│     for each record: cosine = cos_sim(query_emb, record.embedding); clip to [0, 1]
│     rank records by cosine descending → `dense_rank[record.id]`
│
├─ Step 3: sparse ranking
│     scores = stream.bm25.score(query)              # one scan
│     rank records by scores[record.id] desc → `sparse_rank[record.id]`
│
├─ Step 4: RRF fusion
│     raw_rrf[r.id] = 1/(RRF_K + dense_rank[r.id]) + 1/(RRF_K + sparse_rank[r.id])
│     rrf_max = max(raw_rrf.values())
│     hybrid_relevance[r.id] = raw_rrf[r.id] / rrf_max   # normalized to [0, 1]
│
├─ Step 5: per-kind adjustment
│     recency_score = match kind:
│         "fact", "preference" → 1.0                                            # no decay
│         else                 → 2 ** (-age_hours / half_life)                  # exponential
│     importance_score = record.importance / 10.0
│
├─ Step 6: composite
│     final[r.id] = (
│         w_recency    * recency_score +
│         w_importance * importance_score +
│         w_relevance  * hybrid_relevance
│     )
│     (weights from profile.memory.{recency,importance,relevance}_weight)
│
├─ Step 7: sort by final desc, truncate to top_k
│
└─ return list[ScoredMemory]
```

#### Why RRF specifically

Reciprocal Rank Fusion (Cormack, Clarke, Büttcher, SIGIR 2009) solves a real problem: dense cosine scores and BM25 scores live on incomparable scales. Normalizing each to [0, 1] independently is sensitive to outliers. RRF ignores scores and works only on ranks:

$$\text{rrf}(d) = \sum_{r \in \text{rankers}} \frac{1}{k + \text{rank}_r(d)}$$

With `k = 60`, a doc ranked 1st in both rankers scores ~`1/61 + 1/61 ≈ 0.033`; a doc ranked 100th in both scores ~`2/160 ≈ 0.013`. Robust, monotonic, calibration-free. Then we min-max normalize the RRF values to [0, 1] so it composes cleanly with the recency and importance scores.

#### Per-kind retrieval rules (summary)

| Kind          | Recency decay?         | Status filter?           | Default importance in demo |
|---------------|------------------------|--------------------------|----------------------------|
| observation   | yes (exp, 24h half-life) | —                        | 5                          |
| fact          | **no** (recency = 1.0)  | —                        | 6–9                        |
| preference    | **no** (recency = 1.0)  | —                        | 7–9                        |
| plan          | yes                     | **filtered out if "done"** | 6–8                        |
| reflection    | yes                     | —                        | 7–10                       |

#### Cosine similarity

Pure-Python, stdlib math only:

```python
def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b): raise MemoryError("embedding dimension mismatch")
    dot = sum(x*y for x, y in zip(a, b))
    na  = math.sqrt(sum(x*x for x in a)) or 1e-12
    nb  = math.sqrt(sum(x*x for x in b)) or 1e-12
    return dot / (na * nb)
```

At 1024-dim × a few hundred memories per query, this runs in well under 50ms. If we ever need more, the code is ready to swap for a numpy implementation.

### Errors

```python
# DefenseAgent/memory/errors.py

class MemoryError(Exception): ...                  # base
class EmbeddingConfigError(MemoryError): ...       # .env misconfigured
class EmbeddingProviderError(MemoryError): ...     # provider API error; __cause__ preserved
class MemoryNotFoundError(MemoryError): ...        # get_by_id() miss
```

## File layout

> **Note (2026-04-24):** The original 7-file layout below was consolidated to 5 files — see the [walkthrough](../../walkthroughs/module-04-memory.md#2-directory-map) for the current structure. Public API (exports from `DefenseAgent.memory`) is unchanged.

```
DefenseAgent/memory/
├── __init__.py                  # re-exports public API
├── types.py                     # MemoryKind, MemoryRecord, ScoredMemory
├── errors.py                    # MemoryError + 3 subclasses
├── embedding_adapter.py         # EmbeddingAdapter + OpenAICompatibleEmbeddingAdapter + factory
├── bm25.py                      # BM25Index + tokenize()
├── stream.py                    # MemoryStream (owns embedding adapter + BM25 index, does dedup)
└── retriever.py                 # MemoryRetriever + RRF + cosine + per-kind scoring

tests/DefenseAgent/memory/
├── __init__.py
├── test_types.py
├── test_errors.py
├── test_embedding_adapter.py    # mocked openai client
├── test_bm25.py                 # tokenizer, IDF, score ordering, edge cases
├── test_stream.py               # stub embedding adapter, deterministic clock, dedup cases
└── test_retriever.py            # full hybrid pipeline on hand-built records

tests/DefenseAgent/integration/
└── test_memory_integration.py   # load profile → build stream → add records → retrieve with real MemoryConfig

scripts/
└── memory_demo.py               # day-in-Maya's-life observations + facts + prefs + plans, several queries
```

## Dependencies

**No new runtime dependencies.**

- `openai` SDK — already present (embedding adapter uses it).
- `stdlib math`, `uuid`, `datetime` — already in the stdlib.
- BM25 — pure Python, no package.

`requirements.txt` unchanged.

## Integration with earlier modules

| Module | How it's used | Do we modify that module? |
|---|---|---|
| Module 1 (`llm`) | **Not used at all in Module 4.** Module 5 will wrap `LLMAdapter.chat()` for importance scoring and reflection. | No |
| Module 2 (`config`) | `MemoryRetriever` reads `MemoryConfig.{retrieval_top_k, recency_weight, importance_weight, relevance_weight}`. | No |
| Module 3 (`ops`) | Optional: callers can wrap `stream.add()` / `retriever.retrieve()` in `logger.info("memory.added", ...)` calls. | No |

Zero retrofits.

## Testing strategy

All tests offline. Key seams:

- **Embedding adapter** tests use the openai client-injection seam exactly like Module 1.
- **Stream / retriever / integration** tests inject a `StubEmbeddingAdapter` that returns pre-assigned vectors so cosine results are hand-verifiable.
- **BM25** tests are purely deterministic (no randomness, no external calls).
- **Clock** is injected everywhere a timestamp or age appears.

### Coverage outline

**`test_types.py`** — field defaults, each kind literal accepted, dataclass roundtrip.

**`test_errors.py`** — hierarchy, `__cause__` preservation.

**`test_embedding_adapter.py`** — happy-path `embed()` / `embed_batch()`, every `.env` error branch, provider exception wrapping, both `embed` and `embed_batch` translate request/response correctly.

**`test_bm25.py`** —
- `tokenize` drops tiny tokens and splits on non-alnum.
- Single-doc corpus: BM25 score monotonic in term frequency.
- Multi-doc corpus: IDF higher for rare terms.
- Query with no matches: all zeros.
- `len()` reflects doc count.

**`test_stream.py`** —
- `add()` generates embedding and appends.
- Timestamp comes from injected clock.
- `get_recent(n)` returns most-recent-first.
- `get_by_id` hit + miss.
- `add_record` bypasses dedup.
- Dedup: near-duplicate of same kind is not added; far-apart vector is added; duplicate of *different* kind IS added.
- `dedup_threshold=None` disables dedup.

**`test_retriever.py`** —
- Dense-only win (BM25 rank ties): dense ranking wins.
- Sparse-only win (identical embeddings, differing tokens): BM25 wins.
- RRF fusion correctness on hand-built ranks.
- Recency-weight dominance ranks newer over older.
- Importance-weight dominance ranks higher-importance over lower.
- Facts bypass recency decay.
- Preferences bypass recency decay.
- Plans with `metadata["status"] == "done"` are excluded.
- `kinds` filter returns only requested kinds.
- `top_k` default comes from `MemoryConfig.retrieval_top_k`.
- Empty stream returns `[]` without error.
- `ScoredMemory` exposes the expected component fields.

**`test_memory_integration.py`** (lives under `tests/DefenseAgent/integration/`) — load Maya's profile, build stream + retriever from her `MemoryConfig`, add a handful of hand-written memories spanning all five kinds, verify which memories retrieval picks for a specific query and that the ranking respects her weights.

No real API calls in any test.

## Execution flows

### Writing a memory (one `await stream.add(...)` call)

```
stream.add(
    "I attended the 9 AM data structures lecture.",
    kind="observation",
    importance=6,
)
│
├─ 1. query_emb = embedding_adapter.embed(content)
│     → HTTPS POST to provider's /embeddings (e.g. 1024-float vector for text-embedding-v3)
│
├─ 2. dedup check
│     for each existing record with same kind:
│         if cosine(query_emb, record.embedding) > dedup_threshold:
│             return record           # ← no new record; caller sees existing one
│
├─ 3. build MemoryRecord
│     record = MemoryRecord(
│         id=uuid4().hex,
│         content=content,
│         timestamp=clock(),
│         kind=kind,
│         importance=importance,
│         embedding=query_emb,
│         metadata=metadata or {},
│     )
│
├─ 4. append
│     self._records.append(record)
│     self._bm25.add(record.id, content)     # update BM25 index
│
└─ return record
```

### Retrieving (one `await retriever.retrieve(query)` call)

```
retriever.retrieve("how am I doing on the homework?", top_k=5)
│
├─ 1. candidates = [r for r in stream if not (r.kind == "plan" and r.metadata.get("status") == "done")]
│     (optionally filter by `kinds` parameter)
│
├─ 2. query_emb = embedding_adapter.embed(query)          # ONE embedding call per retrieval
│
├─ 3. dense_ranks  = rank candidates by cosine(query_emb, r.embedding) desc
│     sparse_ranks = rank candidates by stream.bm25.score(query)        desc
│
├─ 4. for each candidate r:
│         raw_rrf[r] = 1/(60 + dense_rank[r]) + 1/(60 + sparse_rank[r])
│     hybrid_relevance[r] = raw_rrf[r] / max(raw_rrf)
│
├─ 5. for each candidate r:
│         recency    = 1.0 if r.kind in ("fact", "preference") else 2**(-age_hours/24)
│         importance = r.importance / 10
│         final      = w_r*recency + w_i*importance + w_v*hybrid_relevance[r]
│
├─ 6. sort candidates by final desc, take top_k
│
└─ return [ScoredMemory(record=r, score=final, recency_score=..., ...) for r in top_k]
```

### Failure modes

| Situation | Behavior |
|---|---|
| `.env` missing `EMBEDDING_PROVIDER` | factory raises `EmbeddingConfigError` |
| Provider returns 429/500 on embed | adapter raises `EmbeddingProviderError`; propagates to `add()` or `retrieve()` |
| `get_by_id("bogus")` | raises `MemoryNotFoundError` |
| Embedding dim mismatch (user changed model mid-stream) | `cosine()` raises `MemoryError` with a clear diagnostic |
| Empty stream | `retrieve()` returns `[]` without error |
| Duplicate `add()` (above threshold) | returns the existing record; no new record, no error |
| Query matches no BM25 term | sparse ranks are all tied; RRF still works via the dense ranking |

## Future extensions (Module 5+)

- **Reflection engine.** `ReflectionEngine(stream, llm_adapter, retriever, profile)` — schedules, prompts, synthesizes reflection-kind records with importance drawn from the LLM's confidence.
- **LLM-based importance scoring.** `score_importance(text, llm_adapter)` — thin `chat()` wrapper. Callers feed the result into `stream.add(..., importance=score)`.
- **Reranker.** Optional cross-encoder or API reranker over top-30 candidates from the hybrid stage, trimming to top-k. Big quality win when quality is bottlenecked on retrieval.
- **Persistence.** `save_to(path)` / `load_from(path)` — JSON lines. Records serialize cleanly (stdlib-friendly types).
- **Entity index.** Extract entities at write time, index them separately, enable "all memories mentioning X" fast-path.
- **Fact extraction (Mem0-style).** Use LLM to extract atomic facts from raw observations; store facts instead of or in addition to the raw event.
- **Sleep-time consolidation.** Background task that re-summarizes daily observations into higher-level facts and reflections.

## Token-budget retrieval (revised 2026-04-22 PM)

`MemoryConfig.max_working_memory_tokens` (already declared in Module 2,
default 4000) is now a **second binding constraint** on `retrieve()`
alongside `retrieval_top_k`. Before, it was dead config.

### How it works

After the retriever has sorted the composite-scored list:

1. Walk the list in rank order.
2. Estimate each record's token cost with `_estimate_tokens(content)` —
   a cheap heuristic (`max(1, len(content) // 4)` ≈ OpenAI tokens for
   English-ish text; no `tiktoken` dep).
3. Accumulate records while `running_tokens + cost <= budget` AND the
   count is still below `top_k`.
4. **Exception:** the top-1 is always included, even if it alone exceeds
   the budget. A tight budget should never yield an empty result from a
   non-empty candidate set.

Whichever cap kicks in first — `top_k` or `max_working_memory_tokens` —
binds the return length.

### Why a simple heuristic, not tiktoken

The user's directive for this pass: "keeping this initial version
lightweight." A proper tokenizer:
- Adds a new runtime dep (`tiktoken` is 2 MB + model downloads)
- Binds us to an OpenAI-style tokenizer, which Anthropic/Google/Qwen
  don't share
- Gives precise counts for ranking decisions where a ~20% error band
  is already fine

The `len // 4` rule:
- No deps
- ~85% accurate for English at low variance
- Good enough for "don't overflow the LLM's context window"

Callers who need exact counts can wrap the retriever with their own
pre-filter using whatever tokenizer their LLM uses.

### What this unlocks

Callers feeding retrieved memories into an LLM context can now specify
a per-query context budget via `profile.memory.max_working_memory_tokens`
and trust the retriever to return a set that fits. Previously the only
cap was `retrieval_top_k`, which knew nothing about content length.

## Future extensions (Module 6+)

- **Reflection engine.** Now landed in Module 5 (`Reflector`).
- **LLM-based importance scoring.** Now landed in Module 5 (`Reflector.score_importance`).
- **Reranker.** Optional cross-encoder or API reranker over top-30 candidates from the hybrid stage, trimming to top-k. Big quality win when retrieval is the bottleneck.
- **Persistence.** `save_to(path)` / `load_from(path)` — JSON lines. Records serialize cleanly.
- **Entity index.** Extract entities at write time, index separately, enable "all memories mentioning X" fast-path.
- **Tiered memory (MemGPT-style).** Two-tier core/archival split with self-editing. A distinct future module — not a small extension to this one.
- **Precise tokenizer.** `tiktoken`-based counter as an opt-in replacement for `_estimate_tokens`.

## Open questions

None after the user's decisions on 2026-04-22:
- All three upgrades (hybrid retrieval / typed schema / dedup) adopted.
- BM25 implemented in pure Python, no `rank-bm25` dep.
- Hybrid retrieval only — no "classic" retriever shipped for comparison.
- `.env` block added with Qwen/DashScope placeholders; provider can change later without spec updates.
- Reflection and LLM-based importance scoring explicitly Module 5 (now built).
- File persistence deferred.
- `max_working_memory_tokens` wired as a second binding constraint on retrieve() (added 2026-04-22 PM); tokenizer is `len // 4` heuristic, not `tiktoken`.
