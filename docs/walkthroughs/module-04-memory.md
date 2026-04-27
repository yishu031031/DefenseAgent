# Module 4 Walkthrough — Memory

> Companion to the [design spec](../superpowers/specs/2026-04-22-module-04-memory-design.md).

> **Replaced (2026-04-25):** The hand-rolled hybrid retrieval / BM25 / RRF / SQLite stream described below has been **removed**. Memory is now thin adapters inheriting from `ms_agent.memory` classes over a `mem0` backend.
>
> ### Current state in 5 lines
>
> - `DefenseAgent.memory.Mem0Memory(ms_agent.DefaultMemory)` — front door; wraps mem0. `await memory.add(messages, memory_type=...)` to ingest, `memory.search_records(query, limit, memory_type)` to retrieve, `await memory.run(messages)` for ms-agent's combined ingest+search+inject contract.
> - `DefenseAgent.memory.ContextCompressor(ms_agent.ContextCompressor)` — token-overflow compaction (prune tool outputs + LLM summary).
> - `DefenseAgent.memory.SharedMemoryManager(ms_agent.SharedMemoryManager)` — process-wide singleton.
> - `DefenseAgent.memory._bridge` — `profile_to_dictconfig()` (AgentProfile → omegaconf.DictConfig) and `messages_ours_to_theirs()` / `messages_theirs_to_ours()` (DefenseAgent.Message ↔ ms_agent.Message field copy at the boundary).
> - `DefenseAgent.memory.memory_mapping` — re-export of ms-agent's registry of pluggable memory subclasses.
>
> ### What's gone
>
> `MemoryStream`, `MemoryRetriever`, `BM25Index`, `cosine`, `sqlite_store`, `MemoryRecord`, `MemoryKind`, `ScoredMemory`, `Memory.remember()`, `Memory.recall()`, `Memory.from_env()`, kind-aware recency decay, the RRF fusion, the Park-style composite scoring axes, all of it. mem0 stores plain memory strings keyed by `(user_id, agent_id, run_id, memory_type)` and ranks by cosine similarity only. Where we used to filter by `kind="reflection"`, we now filter by `memory_type="reflection"` (free-form string).
>
> ### What replaces it
>
> Tags via `memory_type`. Outcomes are tagged `"outcome"` (success) or `"failure"` (max_steps exhaustion / bad plan). Trajectory steps from ReActAgent are tagged `"trajectory"`. Reflections from the Reflector are tagged `"reflection"`. The Reflector's `_get_unreflected_records` filters mem0 records by `memory_type != "reflection"` instead of the old typed-Literal `kind` field.
>
> Storage is `<profile.source_dir>/memory/` with mem0 owning the contents (Qdrant on-disk by default, plus a SQLite history db). Embeddings come from the `EMBEDDING_*` env block, the LLM (used for fact extraction) from `AGENT_LAB_LLM_PROVIDER` + per-provider env block. The bridge translates both into mem0's config dict.
>
> ### What sections below still apply
>
> Sections 1–3 (problem framing, the four design ideas) are **historical reasoning** about why we built our own retrieval — preserved because the user explicitly chose to replace it with ms-agent's simpler vector-only model. Sections 4–11 (file walks of `embedding.py`, `stream.py`, `retriever.py`, `memory.py`, `sqlite_store.py`) describe code that **no longer exists**; they're left as a record of the prior implementation but should not be used as a reference for the current code.
>
> For the current module's behavior, read [DefenseAgent/memory/_bridge.py](../../DefenseAgent/memory/_bridge.py) (~150 LOC) and [DefenseAgent/memory/default_memory.py](../../DefenseAgent/memory/default_memory.py) (~100 LOC). The actual memory engine lives in `ms_agent/memory/default_memory.py` (~700 LOC) and `mem0`.

---

## CORE CLASS: `Memory`

Start here. The module's single public entry point:

```python
from DefenseAgent.config import AgentProfile
from DefenseAgent.memory import Memory

profile = AgentProfile.from_yaml("agents/maya_rodriguez/profile.yaml")
memory  = Memory.from_env(profile)

await memory.remember("I attended the 9 AM lecture.", importance=6)
results = await memory.recall("how's class going?")
```

`Memory` owns three collaborators. They are public attributes so tests and advanced callers can reach past the facade:

- `memory.embedding_adapter` — turns text into vectors
- `memory.stream` — append-only storage + BM25 index + dedup
- `memory.retriever` — hybrid retrieval pipeline (dense + sparse + RRF)

---

## 1. What problem this module solves

Agents accumulate thousands of observations, facts, plans, and reflections over a run. Two questions have to be answered well:

1. **Writing** — how do we record a new experience so later retrieval can find it?
2. **Retrieval** — given a query, which ~10 records belong in the LLM's context window *right now*?

Vanilla cosine similarity alone isn't enough for question 2:

- A semantically close but stale memory may matter less than a slightly-less-relevant memory from 10 minutes ago.
- A keyword-heavy query ("what did I say about Greg?") scores poorly on dense embeddings.
- Facts ("I'm allergic to peanuts") should never decay the way events do.

**Module 4 gives the harness:**

- Typed records with a **5-value kind** (`observation / fact / preference / plan / reflection`).
- Append-only storage with **near-duplicate suppression** on write.
- **Hybrid retrieval** — dense embeddings + BM25 + Reciprocal Rank Fusion.
- **Per-kind retrieval rules** — facts bypass recency decay, done plans are filtered out, etc.
- A single facade (`Memory`) that owns all of the above.

---

## 2. Directory map

```
DefenseAgent/memory/                   # 6 files, one concern per file
├── __init__.py                         # re-exports the public API
├── memory.py                           # Memory facade + .from_env + .from_profile   ← START HERE
├── stream.py                           # storage layer:
│                                       #   • MemoryKind, MemoryRecord
│                                       #   • BM25Index + tokenize()
│                                       #   • cosine()
│                                       #   • MemoryStream (add / get / dedup / optional SQLite persistence)
├── retriever.py                        # retrieval layer:
│                                       #   • ScoredMemory
│                                       #   • estimate_tokens, ranks_descending
│                                       #   • RRF_K, _NO_DECAY_KINDS
│                                       #   • MemoryRetriever
├── sqlite_store.py                     # persistence (added 2026-04-24):
│                                       #   • schema + pragmas
│                                       #   • write_record / load_records
│                                       #   • numpy float32 blob serialization
└── embedding.py                        # external I/O:
                                        #   • MemoryError + 3 subclasses
                                        #   • EmbeddingAdapter (ABC)
                                        #   • OpenAICompatibleEmbeddingAdapter

tests/DefenseAgent/memory/
├── __init__.py
├── test_memory.py                      # facade + stream + retriever + cosine
├── test_bm25.py                        # BM25 index + tokenizer
├── test_embedding.py                   # embedding adapter
└── test_persistence.py                 # SQLite round-trip + from_profile (12 tests)

scripts/
├── memory_demo.py                      # seeds Maya with 14 memories, runs 3 queries
└── dump_memory.py                      # pretty-prints a stream.db chronologically (no embeddings)
```

Dependency direction is one-way, no cycles:

```
embedding.py  ←  stream.py  ←  retriever.py  ←  memory.py  ←  __init__.py
```

`embedding.py` is at the bottom, which is why the `MemoryError` base class lives there — `stream.py` needs to raise `MemoryNotFoundError` (a subclass), and `embedding.py` is the one file `stream` already imports from.

---

## 3. The four ideas that shape the design

### 3.1 Memory is append-only

Records are immutable once created. Reflections (Module 5) are stored as *new* records with `kind="reflection"`, never as edits to older observations. No consistency rules, no migrations, time-travel is trivial (just filter by `timestamp < T`).

### 3.2 Retrieval is the quality lever

Picking ~10 records for the LLM's context is where agent intelligence comes from. The retriever is where most of the module's code lives, and most of the spec's design decisions are about scoring.

### 3.3 Hybrid retrieval beats pure cosine

Dense embeddings are great for paraphrase but miss keyword-heavy queries. BM25 is the inverse. Our retriever runs both and fuses them with **Reciprocal Rank Fusion** (Cormack et al. 2009):

$$\text{raw\_rrf}(m) = \frac{1}{60 + \text{dense\_rank}(m)} + \frac{1}{60 + \text{sparse\_rank}(m)}$$

Then we normalize the fused score to `[0, 1]` and multiply it by the profile's `relevance_weight`. No score-scale calibration needed — ranks are unitless.

### 3.4 Memory kind changes retrieval behavior

| Kind | Recency decay? | Filtered out when | Example |
|---|---|---|---|
| `observation` | yes (exponential, 24 h half-life) | never | "attended the 9 AM lecture" |
| `reflection` | yes | never | "I learn faster when stuck" |
| `plan` | yes | `metadata["status"] == "done"` | "finish homework by Friday" |
| `fact` | **no** (always 1.0) | never | "I'm a second-year CS major" |
| `preference` | **no** (always 1.0) | never | "I hate 8 AM classes" |

Facts and preferences are *stable*; treating them the same as events would make them fade out, which is wrong.

---

## 4. File: `embedding.py` — errors + embedding I/O

### 4.1 The error hierarchy

```python
class MemoryError(Exception): ...                 # base
class MemoryNotFoundError(MemoryError): ...       # get_by_id() miss
class EmbeddingConfigError(MemoryError): ...      # EMBEDDING_* env misconfigured
class EmbeddingProviderError(MemoryError): ...    # provider API error; __cause__ preserved
```

All four live here, together, at the top of `embedding.py`. Callers can catch `MemoryError` to handle everything the memory module throws.

### 4.2 `EmbeddingAdapter` — abstract base

```python
class EmbeddingAdapter(ABC):
    @abstractmethod
    async def embed(self, text: str) -> list[float]: ...
    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...
```

Two-method interface. Concrete `OpenAICompatibleEmbeddingAdapter` in the same file covers OpenAI, Qwen (DashScope), and vLLM — they all speak the OpenAI `/embeddings` wire protocol.

### 4.3 `OpenAICompatibleEmbeddingAdapter`

The wire layer. Construction takes `api_key`, `base_url`, `model` and optionally a pre-built client (test seam). `embed()` and `embed_batch()` both wrap provider failures in `EmbeddingProviderError` with the original exception chained as `__cause__`. `embed_batch()` also re-sorts the provider's response by the `index` field so inputs and outputs always align.

---

## 5. File: `stream.py` — types + storage

### 5.1 `MemoryKind` and `MemoryRecord`

```python
MemoryKind = Literal["observation", "fact", "preference", "plan", "reflection"]

@dataclass
class MemoryRecord:
    id: str                    # UUID4 hex, assigned at creation
    content: str
    timestamp: datetime        # UTC
    kind: MemoryKind           # drives retrieval behavior
    importance: float          # 1.0–10.0
    embedding: list[float]
    metadata: dict             # plans use {"status": "active" | "done"}
```

### 5.2 `cosine()` — similarity helper

```python
def cosine(a: list[float], b: list[float]) -> float:
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
```

Ceremony is around two edge cases:

- **Dimension mismatch** → hard raise. Means the caller mixed embedding models on one stream; silently scoring garbage is worse than crashing.
- **Zero-magnitude vector** → return `0.0`, not NaN.

Used by both `MemoryStream.add()` (dedup check) and `MemoryRetriever.retrieve()` (dense ranking).

### 5.3 `BM25Index` + `tokenize()` — sparse index

Tokenization is deliberately simple:

```python
_TOKEN_RE = re.compile(r"[a-z0-9]+")

def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) >= 2]
```

Lowercase + alphanumeric-only + drop 1-char tokens. No stemming, no stopword list — good enough for English-ish agent observations.

Formula (standard Okapi BM25, `k1=1.5`, `b=0.75`):

```
idf(t)       = log( (N - df(t) + 0.5) / (df(t) + 0.5) + 1 )
score(t, d)  = idf(t) * tf(t, d) * (k1 + 1)
             / ( tf(t, d) + k1 * (1 - b + b * |d| / avgdl) )
score(q, d)  = sum over t in q of score(t, d)
```

`BM25Index` keeps four incremental stats so `add()` and `score()` stay O(tokens), not O(corpus):

```python
self._doc_terms: dict[str, list[str]]       # tokens per doc (for |d|)
self._doc_freqs: dict[str, Counter[str]]    # tf(t, d) lookup
self._df:        Counter[str]               # df(t) across corpus
self._total_doc_len: int                    # for avgdl
```

### 5.4 `MemoryStream` — the write side

Three collections kept in sync:

```python
self._records: list[MemoryRecord]          # insertion order
self._records_by_id: dict[str, MemoryRecord]
self._bm25: BM25Index
```

#### `add()` — the whole flow

```python
async def add(self, content, *, kind="observation", importance=5.0, metadata=None):
    embedding = await self.embedding_adapter.embed(content)          # (1)

    if self.dedup_threshold is not None:                             # (2)
        duplicate = self._find_same_kind_duplicate(embedding, kind)
        if duplicate is not None:
            return duplicate                                         #     ← existing record

    record = MemoryRecord(                                           # (3)
        id=uuid.uuid4().hex,
        content=content,
        timestamp=self._clock(),
        kind=kind,
        importance=importance,
        embedding=embedding,
        metadata=dict(metadata) if metadata else {},
    )
    self._append_record(record)                                      # (4)
    return record
```

**Dedup is within-kind.** An `observation` and a `fact` with similar embeddings are NOT collapsed — different kinds carry different semantic weight.

**`_append_record` updates all three indexes:**

```python
def _append_record(self, record):
    self._records.append(record)
    self._records_by_id[record.id] = record
    self._bm25.add(record.id, record.content)
```

#### `add_record()` — the escape hatch

```python
def add_record(self, record: MemoryRecord) -> None:
    self._append_record(record)
```

Bypasses embedding + dedup. Useful for tests (inject records with forged timestamps) and for a future restore-from-snapshot path.

---

## 6. File: `retriever.py` — the read side

### 6.1 `ScoredMemory`

```python
@dataclass
class ScoredMemory:
    record: MemoryRecord
    score: float                # composite weighted score
    recency_score: float        # [0, 1]; 1.0 for facts/preferences
    importance_score: float     # [0.1, 1.0]
    relevance_score: float      # [0, 1], normalized hybrid RRF
    dense_rank: int | None      # 1-indexed rank in dense ranking
    sparse_rank: int | None     # 1-indexed rank in BM25 ranking
```

**Every scoring component is surfaced**, including the per-ranker ranks. Results are inspectable: in `memory_demo.py`'s output you can see exactly why memory N ranked above memory M.

### 6.2 `MemoryRetriever.retrieve()` — seven steps

```python
async def retrieve(self, query, top_k=None, kinds=None):
    # 1. candidate set — drops done plans, applies `kinds` filter
    candidates = self._select_candidates(kinds)
    if not candidates:
        return []

    k = top_k if top_k is not None else self.memory_config.retrieval_top_k
    if k <= 0:
        return []

    # 2. dense ranking
    query_embedding = await self.embedding_adapter.embed(query)
    dense_scores = {r.id: max(0.0, cosine(query_embedding, r.embedding))
                    for r in candidates}
    dense_rank = ranks_descending(dense_scores)

    # 3. sparse ranking (BM25)
    sparse_raw = self.stream.bm25.score(query)
    sparse_scores = {r.id: sparse_raw.get(r.id, 0.0) for r in candidates}
    sparse_rank = ranks_descending(sparse_scores)

    # 4 + 5. RRF fusion, min-max normalized to [0, 1]
    raw_rrf = {r.id: 1.0 / (RRF_K + dense_rank[r.id])
                   + 1.0 / (RRF_K + sparse_rank[r.id])
               for r in candidates}
    rrf_max = max(raw_rrf.values()) or 1.0
    hybrid_relevance = {rid: raw / rrf_max for rid, raw in raw_rrf.items()}

    # 6. per-kind recency + importance + weighted composite
    now = self._clock()
    scored = []
    for r in candidates:
        recency_score    = self._compute_recency_score(r, now)
        importance_score = r.importance / 10.0
        relevance_score  = hybrid_relevance[r.id]
        composite = (
            self.memory_config.recency_weight    * recency_score
            + self.memory_config.importance_weight * importance_score
            + self.memory_config.relevance_weight  * relevance_score
        )
        scored.append(ScoredMemory(
            record=r, score=composite,
            recency_score=recency_score,
            importance_score=importance_score,
            relevance_score=relevance_score,
            dense_rank=dense_rank[r.id],
            sparse_rank=sparse_rank[r.id],
        ))

    # 7. sort + walk under TWO binding caps: top_k AND token budget
    scored.sort(key=lambda s: s.score, reverse=True)
    return self._apply_token_budget(scored, k)
```

**The three weights come from `profile.memory`:** `recency_weight`, `importance_weight`, `relevance_weight`. Users tune retrieval behavior by editing the YAML.

**Two binding caps** (whichever binds first wins):

- **`retrieval_top_k`** — hard count cap from `profile.memory`.
- **`max_working_memory_tokens`** — token-budget cap from `profile.memory` (default 4000). The retriever uses `estimate_tokens()` (a cheap `len(content) // 4` heuristic) and stops before the next record would overflow.

**Exception:** the top-1 is always included, even if it alone exceeds the budget. A tight budget should never yield an empty result on a non-empty candidate set.

### 6.3 Per-kind recency decay

```python
_NO_DECAY_KINDS = frozenset({"fact", "preference"})

def _compute_recency_score(self, record, now):
    if record.kind in _NO_DECAY_KINDS:
        return 1.0
    age_hours = max(0.0, (now - record.timestamp).total_seconds() / 3600)
    return 2.0 ** (-age_hours / self.recency_half_life_hours)
```

Exponential decay with 24-hour half-life by default: 1.0 at t=0, 0.5 at 24 h, 0.25 at 48 h, effectively 0 past a week.

### 6.4 `_select_candidates` — the plan-status filter

```python
def _select_candidates(self, kinds):
    candidates = []
    for r in self.stream.get_all():
        if kinds is not None and r.kind not in kinds:
            continue
        if r.kind == "plan" and r.metadata.get("status") == "done":
            continue
        candidates.append(r)
    return candidates
```

---

## 7. File: `memory.py` — the facade

### 7.1 Construction

```python
class Memory:
    def __init__(self, profile, embedding_adapter, *,
                 stream=None, retriever=None,
                 clock=None, dedup_threshold=0.95,
                 recency_half_life_hours=24.0):
        self.profile = profile
        self.embedding_adapter = embedding_adapter
        self.stream = stream or MemoryStream(embedding_adapter,
                                             clock=clock,
                                             dedup_threshold=dedup_threshold)
        self.retriever = retriever or MemoryRetriever(self.stream,
                                                      embedding_adapter,
                                                      profile.memory,
                                                      clock=clock,
                                                      recency_half_life_hours=recency_half_life_hours)

    async def remember(self, content, *, kind="observation",
                       importance=5.0, metadata=None):
        return await self.stream.add(content, kind=kind,
                                     importance=importance, metadata=metadata)

    async def recall(self, query, *, top_k=None, kinds=None):
        return await self.retriever.retrieve(query, top_k=top_k, kinds=kinds)
```

The stream and retriever can be injected (tests, custom configurations). By default the facade builds them itself.

### 7.2 `Memory.from_env()` — env-driven construction

Reads the `EMBEDDING_*` block from `.env`:

```
EMBEDDING_PROVIDER=qwen              # openai | qwen | vllm
EMBEDDING_API_KEY=
EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_MODEL=text-embedding-v3
```

Validation rules:

- `EMBEDDING_PROVIDER` missing or unsupported → `EmbeddingConfigError`.
- `EMBEDDING_MODEL` missing → `EmbeddingConfigError`.
- `EMBEDDING_BASE_URL` missing for `qwen` or `vllm` → `EmbeddingConfigError` (OpenAI can default).
- `EMBEDDING_API_KEY` missing for `openai` or `qwen` → `EmbeddingConfigError`. For `vllm` it defaults to `"token-not-needed"`.

---

## 8. Execution flow: `scripts/memory_demo.py`

```
$ python scripts/memory_demo.py
│
├─ AgentProfile.from_yaml("agents/maya_rodriguez/profile.yaml")
├─ AgentLogger.from_profile(profile, log_file="logs/...memory.log")
├─ Memory.from_env(profile)
│        • reads EMBEDDING_* block
│        • constructs OpenAICompatibleEmbeddingAdapter
│        • constructs MemoryStream + MemoryRetriever
│
├─ seed 14 memories spanning all 5 kinds (clock back-dated per entry)
│        for each:  stream.add(content, kind, importance, metadata)
│                     └─ embed() + dedup + _append_record + bm25.add
│                   logger.info("memory.added", ...)
│
└─ 3 queries
         memory.recall(query, top_k=5)
           └─ retriever.retrieve(...)
                ├─ _select_candidates()        # drops done plans
                ├─ embed(query)                # 1 API call
                ├─ dense_scores  via cosine()
                ├─ sparse_scores via stream.bm25.score()
                ├─ RRF fusion + min-max normalize
                ├─ per-kind recency + importance + weighted sum
                └─ sorted, top-k under token budget
```

### Sample output

```
QUERY: How am I doing on the data structures homework?
========================================================================
  [observation   imp= 8.0] score=2.478  rec=0.94 imp_n=0.80 rel=1.00  (dense#1 bm25#1)
    Got stuck on problem 3 for an hour, finally worked it out with the TA.
  [observation   imp= 7.0] score=2.334  rec=0.89 imp_n=0.70 rel=0.96  (dense#2 bm25#2)
    Spent two hours in the library working on the BST homework set.
  [plan          imp= 8.0] score=2.172  rec=0.71 imp_n=0.80 rel=0.87  (dense#4 bm25#3)
    Finish the BST homework set by Friday.
  ...
```

The top memory ranked #1 in both dense and sparse — semantically aligned AND keyword-matched. The plan has lower recency (12 h old) but still scores well because `importance=8.0` and the keywords match.

### Where things can fail

| Situation | Surface |
|---|---|
| `EMBEDDING_PROVIDER` unset or unknown | `EmbeddingConfigError` at `Memory.from_env` |
| Missing `EMBEDDING_API_KEY` / `_MODEL` for non-vLLM | `EmbeddingConfigError` at `Memory.from_env` |
| Network / 429 / auth failure during `embed()` | `EmbeddingProviderError`, original exception as `__cause__` |
| `stream.get_by_id("bogus")` | `MemoryNotFoundError` |
| Different embedding models mixed in one stream | `MemoryError` from `cosine()` on dim mismatch |

---

## 9. Test coverage map

| File | Tests | Highlights |
|---|---|---|
| `test_memory.py` | 67 | Facade + stream + retriever + cosine; per-kind rules, hybrid tie-breakers, token-budget cap, `from_env` every branch |
| `test_bm25.py` | 18 | Tokenization, incremental corpus stats, IDF, known-value BM25 scores, edge cases |
| `test_embedding.py` | 4 | `embed()`/`embed_batch()` happy paths, batch reorder correctness, provider exception wrapping |

All tests are fully offline. Real embeddings only happen in `memory_demo.py`.

---

## 10. Things worth noticing

- **Every scoring component is exposed on `ScoredMemory`**, including per-ranker ranks. Observability-by-design: readers of the demo can see WHY a memory ranked where it did.

- **BM25 and cosine are both pure Python** — no `numpy`, no `rank-bm25`. At harness scale (thousands of memories, embedding dim ~1000) both are well under 50 ms per query.

- **Dedup is within-kind, on the write path.** It runs before any ranking, so a duplicate observation doesn't pollute future retrieval by adding near-identical vectors to the pool.

- **Persistence is per-agent and opt-in.** `Memory.from_profile(profile)` defaults to `<profile.source_dir>/memory/stream.db` — each agent bundle owns its own SQLite file. RAM-only stays the default when callers use `from_env()` or construct `Memory` / `MemoryStream` without a `db_path`. See §11 below for the wire format and BM25 rehydration story.

- **Module 1 (LLM) is not imported.** The memory module has zero coupling to `DefenseAgent.llm`. That is what makes Module 5 (Reflection) a meaningful addition: it's the module that bridges LLM and Memory.

- **Two binding caps on retrieval.** For a while `max_working_memory_tokens` was dead config — only `retrieval_top_k` capped results. Now both apply, with the top-1-always-included rule so a tight budget can't blank out the result.

---

## 11. Persistence: `sqlite_store.py` + `MemoryStream(db_path=...)` *(added 2026-04-24)*

### 11.1 When does persistence kick in

Three entry points, in increasing level of wiring:

| Entry point | Default behavior | When to use it |
|---|---|---|
| `MemoryStream(adapter, db_path=None)` | RAM-only (original behavior) | Tests, short-lived demos |
| `Memory.from_env(profile, db_path=<path>)` | Optional SQLite at the path you pass | Mixed setups where you want to control the file location |
| `Memory.from_profile(profile)` | **SQLite at `<profile.source_dir>/memory/stream.db`** | The typical production path — every agent gets its own DB beside its profile |
| `Memory.from_profile(profile, persist=False)` | RAM-only, no file on disk | Throwaway experiments against an existing agent bundle — the agent's persistent stream.db is left untouched |

`from_profile` with the default `persist=True` requires the profile to have been loaded via `AgentProfile.from_yaml(...)` (so `profile.source_dir` is populated). For in-memory profiles, either pass `memory_dir=...` explicitly or pass `persist=False`. Passing both `persist=False` and `memory_dir=` is a contradiction and raises `ValueError`.

### 11.2 Schema

```sql
CREATE TABLE memory_records (
    id            TEXT PRIMARY KEY,
    content       TEXT NOT NULL,
    kind          TEXT NOT NULL,
    importance    REAL NOT NULL,
    timestamp     TEXT NOT NULL,     -- ISO-8601 UTC, lex-sortable
    embedding     BLOB NOT NULL,     -- numpy float32 bytes
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX idx_memory_kind      ON memory_records(kind);
CREATE INDEX idx_memory_timestamp ON memory_records(timestamp);

PRAGMA journal_mode = WAL;
PRAGMA synchronous  = NORMAL;
```

`metadata_json` is required even though it wasn't in the original schema sketch — plans use `metadata["status"] = "done"` for retrieval filtering, and dropping it would silently break that rule after a reload.

### 11.3 Embedding serialization

```python
def _embedding_to_blob(vec: list[float]) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()

def _embedding_from_blob(blob: bytes) -> list[float]:
    return np.frombuffer(blob, dtype=np.float32).tolist()
```

Float32 (not float64). Qwen/OpenAI embedding models emit float32 precision; halving storage is free. A 4096-dim Qwen embedding is ~16KB per record. Round-trip is exact to within `1e-6`.

### 11.4 Rehydration on open

```python
def _open_db(self, db_path):
    self._db = sqlite_store.open_db(db_path)
    for record in sqlite_store.load_records(self._db):
        self._index_record(record)         # in-memory indexing only, no re-write
```

Records are fetched in `(timestamp, id)` order so `get_all()` still returns chronological insertion order. `_index_record` (as opposed to `_append_record`) updates the in-memory structures *without* re-writing to the DB — important because we're reading from the same file we'd be writing to.

The BM25 index is rebuilt from the scan: every call to `_index_record` includes `self._bm25.add(record.id, record.content)`. At harness scale (low-thousands of records per agent), this takes milliseconds on startup. If it ever becomes slow, the spec notes FTS5 as the migration target — but we're not there yet.

### 11.5 Dedup across sessions

The `_records_by_kind` index is populated during rehydration, so `_find_same_kind_duplicate` correctly catches near-dup embeddings against records from a previous run. The dedup threshold (default 0.95) applies both to fresh adds and to adds that match a rehydrated embedding — exactly the same code path.

### 11.6 Inspection: `scripts/dump_memory.py`

The DB is inspectable but the embedding BLOBs make raw `sqlite3` dumps unreadable. `dump_memory.py` issues a read-only connection, filters out the embedding column, and renders each record as:

```
[observation  imp= 8.0]  2026-04-24T18:45:06+00:00  <uuid>
    Maya finished the BST homework problem 3 with the TA.
```

Supports `--kind <kind>` and `--limit N` filters. Zero dependencies beyond the standard library — can be run against a live DB while an agent is still writing to it (WAL makes that safe).

### 11.7 What this buys (and what it doesn't)

- ✅ Per-agent isolation on disk: `agents/maya/memory/` and `agents/alice/memory/` never mix.
- ✅ Crash-safety: every `remember()` commits; losing the last in-flight call is the worst case.
- ✅ Schema evolution: `ALTER TABLE` beats rewriting JSONL when we add fields.
- ❌ **No eviction / retention.** An agent that runs for months will grow unbounded. When we hit that, the natural knob is a cutoff age + archival table; not needed yet.
- ❌ **No concurrent writers across processes.** SQLite WAL allows concurrent *readers*, but two live `Memory` instances writing to the same `stream.db` would race on dedup (both embed "x", both see no duplicate, both insert). If we ever co-run two processes against one agent, we'll add a write lock or move to FTS5 + a single writer.
