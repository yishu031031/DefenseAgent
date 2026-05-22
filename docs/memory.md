# Memory (since 0.2.0)

A tier-aware memory architecture inspired by Hello-Agents, layered on top of mem0 + Qdrant. The agent reaches every tier through a single `MemoryOrchestrator` facade; reads default to hybrid scoring across tiers; a feature-flagged `MemoryConsolidator` promotes high-importance items between lifecycle tiers.

## Architecture

```
                  MemoryOrchestrator
                          │
        ┌─────────┬───────┴──────┬──────────────┐
        ▼         ▼              ▼              ▼
    WORKING   EPISODIC       SEMANTIC      PROCEDURAL
    (in-mem   (trajectories  (reflections  (SOPs /
     deque)    & events)      & lessons)    attack patterns)
        │         │              │              │
        │         └────────┬─────┴──────────────┘
        │                  ▼
        │            mem0 + Qdrant
        │           (one collection,
        │            tier as metadata)
        │
        └─►  MemoryConsolidator (opt-in background promotion + boost)
```

| Tier | Storage | Capacity / TTL | Typical content |
|------|---------|----------------|----------------|
| **WORKING** | in-memory deque | `working_capacity` items, `working_ttl_seconds` TTL | Current-session scratchpad, ephemeral observations |
| **EPISODIC** | Qdrant (tier=`episodic`) | `episodic_capacity` | Raw events, agent trajectories, tool traces |
| **SEMANTIC** | Qdrant (tier=`semantic`) | `semantic_capacity` | Distilled facts, reflections, lessons learned |
| **PROCEDURAL** | Qdrant (tier=`procedural`) | `procedural_capacity` | SOPs, attack patterns, workflows, playbooks |

## On-disk layout

First `run()` against a fresh storage path produces:

```
my_profile/
└── memory/                              # = storage_path (default <profile_dir>/memory/)
    ├── stream.db                        # SQLite — full block stream (verbatim Messages)
    ├── cache.json                       # block hashes for ms-agent's dedup
    └── default_memory/
        └── collection/<agent_id>/       # local Qdrant vector index
            ├── storage.sqlite
            ├── *.lock
            └── ...
```

SQLite keeps the **full conversation history** in insertion order; Qdrant keeps the **vector embeddings**. Records are partitioned by the triple **`(user_id, agent_id, run_id)`** — multiple sessions of the same agent stay cleanly separated. The tier label is metadata only; all four persistent tiers share one collection.

## Configuration

```yaml
memory:
  # ms-agent / mem0 baseline (unchanged)
  is_retrieve: true
  history_mode: add                  # 'add' | 'overwrite'
  search_limit: 10
  ignore_roles: [tool, system]
  ignore_fields: [reasoning_content]
  context_limit: 128000
  prune_protect: 40000
  prune_minimum: 20000
  reserved_buffer: 20000
  enable_summary: true
  storage_path:

  # --- Tier-aware extensions (0.2.0) ---
  default_importance: 0.5            # Used when caller doesn't set one.

  scoring:                           # Hybrid retrieval weights.
    similarity: 0.55                 # mem0 cosine
    recency: 0.20                    # exp decay
    importance: 0.15                 # stored on the record
    frequency: 0.10                  # log-saturating
    recency_half_life_days: 7.0

  tier_limits:
    working_capacity: 50
    working_ttl_seconds: 3600
    episodic_capacity: 1000
    semantic_capacity: 5000
    procedural_capacity: 500

  consolidation:                     # Disabled by default — opt-in.
    enabled: false
    interval_seconds: 300
    promote_to_episodic_threshold: 0.5
    promote_to_semantic_threshold: 0.7
    promote_to_procedural_threshold: 0.85
    importance_boost_on_promotion: 1.1
    forget_below_importance: 0.3
    forget_idle_days: 30
```

## `MemoryOrchestrator` API

The agent's `self.memory` is always a `MemoryOrchestrator` (since 0.2.0). The agent builder auto-wraps any injected `Mem0Memory` so downstream code uniformly sees the tier-aware facade.

### Writes

```python
from DefenseAgent.memory import MemoryTier
from DefenseAgent.llm.types import Message

# Convenience methods — one per tier:
await agent.memory.add_episodic(messages, memory_type="trajectory", importance=0.4)
await agent.memory.add_semantic(messages, memory_type="reflection", importance=0.85)
await agent.memory.add_procedural(messages, memory_type="sop", importance=0.9)

# Or the general form:
await agent.memory.add(
    messages,
    tier=MemoryTier.WORKING,          # default: EPISODIC
    memory_type="alert",
    importance=0.7,                   # default: profile.memory.default_importance
    source_run_id="run-7",            # optional — for cross-tier provenance
    extra={"attack_phase": "recon"},  # optional — arbitrary metadata
)
```

WORKING writes never touch mem0 — they go straight to the in-memory deque. Everything else hits Qdrant with the tier baked into metadata.

### Reads

```python
# Cross-tier recall, hybrid scoring (new default):
items = agent.memory.recall("query", limit=5)

# Single-tier recall:
items = agent.memory.recall("query", tier=MemoryTier.SEMANTIC, limit=10)

# Opt back into vector-only scoring (legacy behavior):
items = agent.memory.recall("query", scoring="vector")
```

`items` is `list[MemoryItem]` — typed records with `content`, `tier`, `memory_type`, `importance`, `created_at`, `last_accessed_at`, etc. For raw mem0 dicts (back-compat with pre-0.2.0 callers), use `agent.memory.search_records(...)`.

### Built-in tool

When `memory.is_retrieve: true` the LLM gets:

```json
{
  "name": "memory_recall",
  "input_schema": {
    "query": "string",
    "top_k": "int (1..20, default 5)"
  }
}
```

The handler renders each hit as `- [tier/memory_type] content` so the model sees both the lifecycle tier and the finer label.

## Hybrid scoring formula

```
score = similarity × w_sim          (cosine from mem0, clamped [0, 1])
      + recency_decay × w_rec       (exp(-ln(2) × age_days / half_life))
      + importance × w_imp          (the record's stored importance)
      + frequency × w_freq          (1 - 1/(1 + access_count), saturating)
```

Weights live in `profile.memory.scoring`. They need not sum to 1 — overweight a dimension if your offline eval says so.

### Candidate widening

When `scoring="hybrid"`, the orchestrator asks mem0 for **`limit × candidate_multiplier`** candidates (default `3×`) before re-ranking. This gives high-importance / recent records a chance to surface even when their raw cosine isn't top-K.

## Memory types per tier (typical)

Tier and `memory_type` are **orthogonal** dimensions. Tier is the lifecycle bucket; `memory_type` is a finer semantic label. Conventions in the codebase:

| Tier | Common `memory_type` values | Source |
|------|----------------------------|--------|
| WORKING | `scratch`, `observation`, `intermediate` | Direct writes during a single run |
| EPISODIC | `trajectory`, `observation`, `outcome`, `failure`, `alert` | `BaseAgent` trajectory/outcome saving; raw event ingestion |
| SEMANTIC | `reflection`, `lesson`, `fact`, `summary` | `Reflector.reflect_now()` (Park-et-al-style periodic reflection) |
| PROCEDURAL | `sop`, `playbook`, `pattern`, `attack_chain` | Manual or domain-specific ingestion |

`memory_recall` returns hits prefixed: `- [semantic/reflection] you tend to over-explain on tool failures`.

## Consolidation lifecycle

The optional `MemoryConsolidator` promotes high-importance items along the lifecycle pipeline:

```
WORKING ─(importance ≥ promote_to_episodic_threshold)─→ EPISODIC
EPISODIC ─(≥ promote_to_semantic_threshold)──────────→ SEMANTIC
SEMANTIC ─(≥ promote_to_procedural_threshold)────────→ PROCEDURAL
```

On promotion: importance is multiplied by `importance_boost_on_promotion` (capped at 1.0), `consolidated_from` is stamped on the new record for audit, and `consolidated_from_tier` metadata identifies the source tier.

### One-shot or background

```python
from DefenseAgent.memory import MemoryConsolidator

consolidator = MemoryConsolidator(agent.memory)

# One pass, deterministic — returns ConsolidationStats:
stats = await consolidator.run_once()
# stats.promoted_to_episodic, .promoted_to_semantic, .promoted_to_procedural
# stats.skipped_already_promoted, .skipped_below_threshold, .errors

# Background loop — fires every `interval_seconds`:
await consolidator.start()
# ... agent runs ...
await consolidator.stop()      # graceful drain
```

### Idempotency

The consolidator tracks promoted record IDs **in memory** to avoid double-promoting in the same process. Restart resets this set — first pass after restart may re-promote items already moved up in the prior process. For production durability, wrap the consolidator and persist the `_promoted_ids` set externally (or accept the cost of an extra promotion per restart).

### Error tolerance

A failure inside one promotion (e.g. Qdrant transient error) increments `stats.errors` and the loop moves on — one bad record doesn't kill the pass. The background loop additionally swallows any exception from `run_once()` itself so the loop survives to the next interval.

## Working layer semantics

The WORKING tier is intentionally simple — no embeddings, no vector store. `WorkingMemory` is a deque with:

- **Capacity**: `working_capacity` — when at cap, FIFO eviction (oldest item drops). Importance is **not** consulted here — short-term memory is meant to roll over; the consolidator is what promotes important items before they expire.
- **TTL**: `working_ttl_seconds` — items older than this expire on the next `add()` or `search()` call.
- **Search**: substring match against content (case-insensitive). For semantic recall, route to persistent tiers via `recall(tier=MemoryTier.EPISODIC|...)`.

WORKING items get a `[role] content` prefix when ingested via `orchestrator.add(messages=...)` so substring search can preserve who-said-what.

### When you'd use it

- LLM-mid-loop scratch notes that should NOT pollute Qdrant
- A short-lived "current focus" buffer fed by other parts of the agent
- Anything you want to age out automatically without writing eviction logic

## `ContextCompressor` — separate concern

Independent from `MemoryOrchestrator`: `ContextCompressor` protects each LLM call from overflowing the context window. It runs **before** every LLM call on the working messages (what would go into `chat()` this turn), not against stored memory.

The four numbers interlock:

```
total tokens in working messages
        │
        │  if  total + reserved_buffer  >  context_limit
        │      then prune
        ▼
prune phase:
   ── keep most recent prune_protect tokens untouched (recent turns matter most)
   ── compress older turns down so total ≥ prune_minimum
   ── if enable_summary=true, the older block becomes a single LLM-generated summary turn
   ── if false, older turns are dropped without replacement
```

So `context_limit: 128000` + `reserved_buffer: 20000` means "start pruning when working messages cross 108K tokens". `prune_protect: 40000` says "never touch the most recent 40K tokens". Tune the four together; raising `context_limit` past your model's actual window causes API rejections with no upside.

## Back-compat with pre-0.2.0 callers

The orchestrator exposes back-compat shims so existing code keeps working:

```python
# Old (pre-0.2.0) — still works:
records = agent.memory.search_records("query", limit=5)
records = agent.memory.get_all(memory_type="reflection")

# New (0.2.0+) — typed, tier-aware:
items = agent.memory.recall("query", limit=5)
items = agent.memory.get_items(tier=MemoryTier.SEMANTIC)
```

Legacy mem0 records written before 0.2.0 (no `tier` field in metadata) decode as `tier=EPISODIC`, `importance=0.5` — nothing breaks.

## Smoke tests

The repo ships two memory smoke scripts:

```bash
# No external services — uses MagicMock for mem0:
python scripts/smoke_new_memory.py

# Real Ollama bge-m3 + real local Qdrant — needs OLLAMA running:
python scripts/smoke_real_backend.py [--keep]
```

The second one is the gold standard for verifying the full memory stack end-to-end before a release.
