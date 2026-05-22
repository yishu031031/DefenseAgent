# Configuration

The complete `profile.yaml` schema and how DefenseAgent resolves every field at load time.

## Profile bundle layout

A profile is a directory:

```
my_profile/
├── profile.yaml          # required — the schema below
├── prompts/              # optional — system-prompt templates
│   └── system.md
├── python_tools/         # optional — local Python tool entry points
│   └── calc.py
├── skills/               # optional — SKILL.md-style tool packs
│   └── tabular-report/
├── memory/               # auto-created at runtime if memory.is_retrieve=true
└── rag_corpus/           # documents indexed when rag.enabled=true
```

`AgentConfig(profile=Path("…/my_profile/profile.yaml"))` resolves every relative path inside the profile against the profile's directory, so the bundle is self-contained and movable.

Each block under `agent:` is independent and optional except identity. All fields are validated by Pydantic with `extra="forbid"` — typos in field names fail loudly at `AgentProfile.from_yaml()`.

## Resolution order

For every field that can live in both YAML and `.env`:

1. `<field>:` in profile YAML (whitespace-only counts as unset)
2. Cross-provider env tier — `LLM_API_KEY` / `LLM_MODEL_ID` / `LLM_BASE_URL`
3. Per-provider env tier — `<PROVIDER>_API_KEY` / `<PROVIDER>_MODEL` / `<PROVIDER>_BASE_URL`
4. Schema default

First non-empty wins.

## Identity

Only `id` and `name` are required. The other four fields flavour the agent's persona and have safe defaults.

```yaml
agent:
  id: "agent_001"     # str, min_length=1. REQUIRED.
  name: "Nova Patel"  # str, min_length=1. REQUIRED.
  age: 27             # int ≥ 0 | null. Optional, default null.
  traits: "..."       # str. Optional, default "".
  backstory: "..."    # str. Optional, default "".
  initial_plan: "..." # str. Optional, default "".
```

All six are exposed as `{id} {name} {age} {traits} {backstory} {initial_plan}` placeholders in the prompt template — see [`prompt:`](#prompt) below.

### Field semantics

| Field | Required? | Used for |
|---|---|---|
| `id` | **yes** | (1) `agent_id` partition key in mem0 — records scoped to this id. (2) Log file name: `<log_dir>/<id>.log`. (3) `{id}` placeholder. **Pick a stable identifier — changing `id` orphans existing memory.** |
| `name` | **yes** | `{name}` placeholder. Auto-built identity prompt opens with `You are <name>, ...`. |
| `age` | optional (null) | `{age}` placeholder. Useful for role-play personas. |
| `traits` | optional ("") | `{traits}` placeholder. Personality / tone / approach. |
| `backstory` | optional ("") | `{backstory}` placeholder. Long-form narrative. The most useful field for grounding the LLM. |
| `initial_plan` | optional ("") | `{initial_plan}` placeholder. What the agent is currently working on. |

### Auto-built identity block

With minimal `id: "bot"` + `name: "Helper"` you get just:

```
You are Helper.
```

Add `traits: "concise, technical"` and you get:

```
You are Helper.
Traits: concise, technical
```

Unset fields skip their lines entirely — no awkward "You are Helper, a -year-old. Traits: " sentences.

### Validation failure modes

The schema is strict — bad input fails at load with `ConfigValidationError`, not at `agent.run()`:

| Input | Result |
|---|---|
| `id: ""` or `id: "   "` | `string_too_short` |
| missing `id` or missing `name` | `missing` validation error |
| missing `age` / `traits` / `backstory` / `initial_plan` | accepted — defaults fire |
| `age: -1` | `greater_than_equal` violation |
| `age: 27.5` | `int_type` violation |
| any extra/unknown field | `extra_forbidden` |

## `llm:`

```yaml
llm:
  provider:           # str | null. One of: openai | anthropic | deepseek | qwen | google | vllm.
                      # Falls back to AGENT_LAB_LLM_PROVIDER.
  model:              # str | null. Provider-specific model id.
                      # Falls back to <PROVIDER>_MODEL or LLM_MODEL_ID.
  base_url:           # str | null. Provider endpoint.
                      # Falls back to <PROVIDER>_BASE_URL or LLM_BASE_URL.
  api_key:            # str | null. Falls back to <PROVIDER>_API_KEY.
                      # Recommend leaving blank in shared profiles.
```

All four are `str | None`. Whitespace-only values count as unset, so a half-edited YAML can't shadow correct env state.

**Recommended shape**: profile sets `provider` + `model` (those are part of the agent's identity); `.env` supplies `api_key` + `base_url` (operator concerns):

```yaml
llm:
  provider: deepseek
  model: deepseek-chat
```

```bash
# .env
DEEPSEEK_API_KEY=sk-…
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
```

For the full provider table, embedding pairings and per-provider notes, see [`providers.md`](providers.md).

## `cognitive:`

```yaml
cognitive:
  max_steps_per_cycle: 10     # int ≥ 1, default 10. ReAct tool-call loop cap.
  reflection_threshold: 5     # int ≥ 1, default 5. Unreflected-memory count to trigger reflect.
  importance_threshold: 7     # float [1, 10], default 7. Floor for "important" records.
  planning_horizon: "1 day"   # str min_length=1, default "1 day". Surfaced in prompts.
```

### `max_steps_per_cycle`

A "step" in `ReActAgent` is one (tool-call → tool-result) round-trip. When the cap is hit:

```python
result = await agent.run("multi-step task")
# result.stopped_reason == "max_steps"
# result.final_answer   ← the LLM's last partial output
# result.steps          ← full trace
```

Override per call: `await agent.run(task, max_steps=20)`. `SimpleAgent` ignores it; `PlanAndSolveAgent` reads it as the **plan length cap** (not substep cap — that's `AgentConfig.max_substeps_per_step`, default 3).

Tuning:
- Simple Q&A with one tool call: `3` is plenty
- ReAct over multi-tool research: `10–20`
- Long-horizon iteration: raise cautiously — every step is a paid LLM call

### `reflection_threshold` and the reflection cycle

After every `run()`, if `reflect_after_run: true` (default), `Reflector.maybe_reflect()` checks: at least `reflection_threshold` non-reflection records since last reflection? If yes:

1. Pull every mem0 record where `memory_type != 'reflection'`
2. `InsightSynthesizer` distills them into N (default 3) bullet-shaped insights
3. Each insight is written back **into the SEMANTIC tier** tagged `memory_type='reflection'`, with importance normalised to `[0, 1]` from the legacy 1-10 scale

Reflections are visible to subsequent `memory_recall` calls.

### When reflection actually pays off

Reflection costs at least 2 extra LLM calls per cycle. Be honest about your scenario:

| Scenario | Reflection helps? | Recommendation |
|---|---|---|
| **One-off script** — runs once and exits | **No.** Reflection writes 3 records, process ends. Pure waste. | `AgentConfig(reflect_after_run=False)` |
| **Demo / quickstart** | No. | Same. |
| **Same `agent_id` across many sessions** — long-running assistant, recurring batches | **Yes.** Reflections from session N surface in N+1. | Keep default. |
| **Generative-Agents-style simulations** | **Yes — by design.** This is what `Reflector` was built for ([Park et al. 2023](https://arxiv.org/abs/2304.03442)). | Keep default. Maybe lower threshold. |
| **High-volume short tasks** | **Maybe.** Helpful only if reflections survive across tickets. | Run with reflection on, inspect via `scripts/dump_memory.py`, decide. |

**Precondition for any "yes" case**: the LLM must actually call `memory_recall`. Reflections aren't auto-injected — a system prompt that says "before answering, call `memory_recall`" makes reflection useful; one that doesn't may waste the entire mechanism.

To disable entirely:

```python
AgentConfig(profile=..., reflect_after_run=False)        # skip reflection cycle only
AgentConfig(profile=..., use_reflection=False)           # skip Reflector construction
```

### `importance_threshold`

LLM-based 1–10 rating per record. During reflection, records below this threshold are filtered out before being fed to the synthesizer. Default 7 is conservative; lower to 5 if your records skew lower-impact.

### `planning_horizon`

Free-form string surfaced in the auto-built identity prompt. Defaults to `"1 day"`. Examples:
- `"this hour"` for short-window operational agents
- `"this sprint"` for engineering agents

Visible only if your prompt template includes the auto-built block.

## `memory:`

```yaml
memory:
  is_retrieve: true                  # bool, default true. Wires up memory_recall.
  history_mode: add                  # 'add' | 'overwrite'. Latter enables diff/rollback.
  search_limit: 10                   # int ≥ 1, default 10. Max records per memory_recall.
  ignore_roles: [tool, system]       # list[str], default ['tool', 'system'].
  ignore_fields: [reasoning_content] # list[str], default ['reasoning_content'].
  context_limit: 128000              # int ≥ 1024, default 128000. Tokens before prune.
  prune_protect: 40000               # int ≥ 0, default 40000. Never touched during prune.
  prune_minimum: 20000               # int ≥ 0, default 20000. Floor after prune.
  reserved_buffer: 20000             # int ≥ 0, default 20000. Safety margin.
  enable_summary: true               # bool, default true. ContextCompressor LLM-summarises.
  storage_path:                      # str | null. Default: <profile_dir>/memory/.

  # --- Tier-aware extensions (since 0.2.0) ---
  default_importance: 0.5            # float [0,1]. Importance attached when caller doesn't set one.
  scoring:                           # Hybrid scoring weights for retrieval.
    similarity: 0.55
    recency: 0.20
    importance: 0.15
    frequency: 0.10
    recency_half_life_days: 7.0
  tier_limits:
    working_capacity: 50             # In-memory deque size.
    working_ttl_seconds: 3600        # Working items expire after this.
    episodic_capacity: 1000
    semantic_capacity: 5000
    procedural_capacity: 500
  consolidation:                     # Background lifecycle job. Disabled by default.
    enabled: false
    interval_seconds: 300
    promote_to_episodic_threshold: 0.5
    promote_to_semantic_threshold: 0.7
    promote_to_procedural_threshold: 0.85
    importance_boost_on_promotion: 1.1
    forget_below_importance: 0.3
    forget_idle_days: 30
```

Requires `defense-agent[memory]`. For the full tier-aware architecture (orchestrator API, scoring formula, lifecycle semantics), see [`memory.md`](memory.md).

## `rag:`

```yaml
rag:
  enabled: false                   # bool, default false. Flip to true → wires rag_search.
  documents_dir: rag_corpus        # str | null. Relative to profile dir. Auto-indexed on first run().
  storage_dir: rag_index           # str | null. Where the FAISS index persists.
  embedding_provider: openai       # 'openai' | 'huggingface', default 'openai'.
  embedding:                       # str | null. → EMBEDDING_MODEL.
  embedding_api_key:               # str | null. → EMBEDDING_API_KEY.
  embedding_base_url:              # str | null. → EMBEDDING_BASE_URL.
  embedding_dims:                  # int ≥ 1, null. → EMBEDDING_DIMS.
  chunk_size: 512                  # int ≥ 1, default 512. Tokens per chunk during ingestion.
  chunk_overlap: 50                # int ≥ 0, default 50. Token overlap.
  top_k: 5                         # int ≥ 1, default 5. Default rag_search top_k.
  score_threshold: 0.0             # float [0.0, 1.0], default 0.0. Min score returned.
  retrieve_only: true              # bool, default true. False → RAG also synthesises an answer.
  use_huggingface: false           # bool, default false. ms-agent's HF download path.
```

Requires `defense-agent[rag]`.

### Bootstrap flow

The first time `agent.run()` fires under `rag.enabled: true`:

1. **Discover documents** — every file under `documents_dir` (default `rag_corpus/`)
2. **Extract structured chunks** — `StructuredDocExtractor` walks each file with registered extractor backends (`PyPdfExtractor`, `HtmlExtractor`, …). Markdown/text go through LlamaIndex's default loader.
3. **Tokenise + chunk** — each chunk sub-split using `chunk_size` with `chunk_overlap`
4. **Embed + index** — vectors land in a persistent FAISS index under `storage_dir`
5. **Persist** — so subsequent runs skip steps 1–4

End-state:

```
my_profile/
├── rag_corpus/
│   ├── runbook.pdf
│   ├── architecture.html
│   └── notes.md
└── rag_index/
    ├── default__vector_store.json
    ├── docstore.json
    └── _resources/
```

To re-index after document changes: delete `storage_dir` and run again. No incremental indexing — whole-or-nothing.

### Document format support

| Source | Backend | Extracts |
|---|---|---|
| `.pdf` | `PyPdfExtractor` (pdfplumber) | Text, tables as Markdown, embedded images |
| `.html` | `HtmlExtractor` (beautifulsoup4) | Body text by section, tables, `<img>` references |
| `.md` / `.txt` / `.rst` | LlamaIndex default | Plain-text chunks |
| `.docx` / `.epub` / others | LlamaIndex default (best-effort) | Plain-text chunks |

Extractors are pluggable — subclass `StructuredExtractor` (must implement `supports(source)` and `extract(source) -> list[StructuredChunk]`), register on the extractor:

```python
from DefenseAgent.rag.extraction import StructuredDocExtractor

class MyCsvExtractor:
    def supports(self, source): return str(source).endswith(".csv")
    def extract(self, source): return [...]   # list[StructuredChunk]

extractor = StructuredDocExtractor(...)
extractor.register(MyCsvExtractor(), prepend=True)
```

### `rag_search` tool — what the LLM sees

```json
{
  "name": "rag_search",
  "input_schema": {
    "query": "string",
    "top_k": "int (default <profile.rag.top_k>)"
  }
}
```

- **`retrieve_only: true`** (default) returns ranked chunks: `[score=0.84] <chunk text>`. Cheap, agent decides what to do with them.
- **`retrieve_only: false`** runs LlamaIndex's QA synthesizer — a second LLM call composes one answer. More expensive, less flexible.

`score_threshold:` filters before returning (default 0.0 → keep everything).

## `prompt:`

```yaml
prompt:
  path: prompts/system.md          # str | null. File relative to profile dir.
  system:                          # str | null. Inline alternative to `path:`.
  extra_instructions:              # str | null. Appended after the resolved identity.
```

### Three resolution paths

The agent resolves the system prompt in this order, **first non-empty wins**:

1. **Inline `system:`** — a literal string in YAML. For ad-hoc agents.
2. **`path:`** — file resolved relative to the profile's directory.
3. **Auto-built identity block** — generated from `id`/`name`/`age`/`traits`/`backstory`/`initial_plan`.

In all three, `extra_instructions:` is appended at the end with a blank-line separator.

### Concrete `prompts/system.md`

```markdown
You are {name}, a {age}-year-old {traits} field engineer turned AI researcher.

# Background
{backstory}

# Today
{initial_plan}

# How to behave
- Speak in first person, concise. Sentences, not paragraphs.
- When the answer needs information from earlier conversations or stored facts,
  call `memory_recall` instead of guessing.
- When the answer needs work done (file lookups, web searches, computations),
  call the appropriate tool.
- Stay in character.
```

The six placeholders are rendered via Python's `str.format`. Anything else (`{plan}`, `{date}`, `{user}`) would `KeyError` — the agent catches this, logs a warning, and falls back to the auto-built block.

### Failure modes

| Problem | Behaviour |
|---|---|
| `path:` points at a non-existent file | `ConfigValidationError` at profile load |
| Template references an unknown placeholder | Logged warning + fallback to auto-built block; run continues |
| Both `system:` and `path:` set | `ConfigValidationError` — pick one |
| Both empty + identity fields incomplete | Auto-build would already have failed at identity validation |

`AgentConfig.extra_instructions` (Python-side override) takes precedence over `profile.prompt.extra_instructions`.

## Switching providers without code changes

Same profile, same code, three different providers — only `.env` changes:

```bash
# variant A — DeepSeek
AGENT_LAB_LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-…
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
```

```bash
# variant B — DashScope/Qwen
AGENT_LAB_LLM_PROVIDER=qwen
QWEN_API_KEY=sk-…
QWEN_MODEL=qwen-plus
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

```bash
# variant C — local vLLM
AGENT_LAB_LLM_PROVIDER=vllm
VLLM_API_KEY=EMPTY
VLLM_MODEL=Qwen/Qwen2.5-72B-Instruct
VLLM_BASE_URL=http://localhost:8000/v1
```

Provided your profile leaves `llm.provider` / `llm.model` blank, the agent picks up whichever set is active in `.env`. No reload, no code change.
