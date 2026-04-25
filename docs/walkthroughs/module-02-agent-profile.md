# Module 2 Walkthrough — Agent Profile Loader

> Companion to the [design spec](../superpowers/specs/2026-04-22-module-02-agent-profile-design.md). Explains the code line-by-line and traces the execution of `show_profile.py` and `profile_chat_demo.py`.

---

## CORE CLASS: `AgentProfile`

Start here. The canonical entry point is `AgentProfile.from_yaml(path)`:

```python
from DefenseAgent.config import AgentProfile

profile = AgentProfile.from_yaml("agents/maya_rodriguez/profile.yaml")
print(profile.name, profile.age)
print(profile.tools.skills)    # declared skill paths (resolved relative to the profile)
```

The free function `load_profile(path)` still exists and does exactly the same thing — it's kept for backwards compatibility, but `AgentProfile.from_yaml(...)` is the way new code should read.

The rest of this walkthrough covers the Pydantic schema, the loader's five-stage flow, and the error hierarchy — i.e. what `AgentProfile.from_yaml()` is actually doing under the hood.

---

## 1. What problem this module solves

Every agent has identity (id, name, traits, backstory) and tuning knobs (cognitive thresholds, memory weights). Those values live in a YAML file so they can be edited without touching code. But YAML by itself is untyped and forgiving — a typo like `traists:` instead of `traits:` would silently succeed and the harness would misbehave three modules later.

**Module 2 does three things:**
1. Defines a **typed Pydantic model** (`AgentProfile`) that names and validates every field.
2. Provides a **strict loader** (`AgentProfile.from_yaml`) that parses YAML into that model and fails loudly on any mismatch.
3. Ships **reference bundles** under `agents/<id>/` so demos and tests have something concrete to load — and so each agent's tool set (skills + MCP servers) is self-contained in its own directory.

---

## 2. Directory map *(revised 2026-04-24 — agent bundles)*

```
DefenseAgent/config/                # loader CODE (no data)
├── __init__.py                      # public API re-exports
└── profile.py                       # models + ConfigError hierarchy + from_yaml

agents/                              # user-editable DATA — one bundle per agent
├── alice_chen/
│   └── profile.yaml                 # reference profile — a data scientist
└── maya_rodriguez/
    ├── profile.yaml                 # identity + cognitive + memory + tools
    └── skills/                      # private skills (loaded by ToolRegistry.from_profile)
        └── tabular-report/
            ├── SKILL.md
            ├── scripts/
            └── templates/

scripts/
├── show_profile.py                  # load a profile + pretty-print it
├── profile_chat_demo.py             # load a profile + use it with the LLM
└── tools_demo.py                    # load a profile + build its ToolRegistry
```

**Key layout decisions:**

- **Loader code and agent data live in different directories.** `DefenseAgent/config/` has no YAML; `agents/` has no Python.
- **Each agent is a bundle.** Identity (`profile.yaml`), private skills, and MCP declarations all live inside `agents/<agent_id>/`. Creating a new agent is `cp -r agents/maya_rodriguez agents/new_agent` + edit the profile — nothing else in the repo cares.
- **Tool paths resolve from the profile's directory.** Declaring `skills: [skills/foo]` in `agents/maya/profile.yaml` means `agents/maya/skills/foo`, *not* a shared top-level `skills/`. Agents cannot accidentally see each other's tools.

---

## 3. Pydantic models (`profile.py`)

Three nested BaseModel subclasses. Each sets `model_config = ConfigDict(extra="forbid")` — unknown keys become validation errors.

### `CognitiveConfig`
```python
class CognitiveConfig(BaseModel):
    model_config = _STRICT

    max_steps_per_cycle: int   = Field(ge=1, default=10)
    reflection_threshold: int  = Field(ge=1, default=5)
    importance_threshold: float = Field(ge=1, le=10, default=7)
    planning_horizon: str      = Field(min_length=1, default="1 day")
```

| Field | Meaning | Why the constraint |
|---|---|---|
| `max_steps_per_cycle` | Max actions in one wake cycle | ≥1 — a zero-step cycle makes no sense |
| `reflection_threshold` | Trigger reflection after N new memories | ≥1 — reflecting on zero memories is empty |
| `importance_threshold` | Memories above this (1–10) count as "important" | 1–10 matches the scoring scale |
| `planning_horizon` | Free-form string like "1 day" | Free-form because parsing is a future concern |

All fields have defaults, so you can omit the whole `cognitive:` block in YAML.

### `MemoryConfig`
```python
class MemoryConfig(BaseModel):
    model_config = _STRICT

    max_working_memory_tokens: int = Field(ge=1, default=4000)
    retrieval_top_k: int           = Field(ge=1, default=10)
    recency_weight: float          = Field(ge=0, default=1.0)
    importance_weight: float       = Field(ge=0, default=1.0)
    relevance_weight: float        = Field(ge=0, default=1.0)
```

Weights use `ge=0` (not `gt=0`) so users can disable a term by setting its weight to 0. For example, `recency_weight: 0.0` means "ignore recency entirely when scoring memories for retrieval."

### `AgentProfile`
```python
class AgentProfile(BaseModel):
    model_config = _STRICT

    id: str            = Field(min_length=1)
    name: str          = Field(min_length=1)
    age: int           = Field(ge=0)
    traits: str        = Field(min_length=1)
    backstory: str     = Field(min_length=1)
    initial_plan: str  = Field(min_length=1)
    cognitive: CognitiveConfig = Field(default_factory=CognitiveConfig)
    memory: MemoryConfig       = Field(default_factory=MemoryConfig)
```

**Why `default_factory`** on the nested models: using `= CognitiveConfig()` as a default creates ONE shared instance across all AgentProfile instances — mutating it would leak. `default_factory=CognitiveConfig` builds a fresh one per profile.

**All identity fields required**, nothing else. A minimal YAML needs only these six fields plus the `agent:` wrapper.

---

## 4. The loader (`profile.py`)

```python
def load_profile(path: str | Path) -> AgentProfile:
    file_path = Path(path)
    if not file_path.is_file():
        raise ConfigFileNotFoundError(f"profile file not found: {file_path}")

    raw_text = file_path.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(raw_text)
    except yaml.YAMLError as e:
        raise ConfigParseError(f"invalid YAML in {file_path}: {e}") from e

    if not isinstance(data, dict):
        raise ConfigParseError(f"expected top-level mapping ..., got {type(data).__name__}")
    if "agent" not in data:
        raise ConfigParseError(f"missing top-level 'agent:' key in {file_path}")
    agent_data = data["agent"]
    if not isinstance(agent_data, dict):
        raise ConfigParseError(f"'agent:' value must be a mapping ...")

    try:
        return AgentProfile.model_validate(agent_data)
    except ValidationError as e:
        raise ConfigValidationError(f"profile at {file_path} failed schema validation:\n{e}") from e
```

Read from top:

1. **`is_file()` before reading.** Both missing paths and directories fail fast as `ConfigFileNotFoundError`. (Note: `open()` on a directory would raise `IsADirectoryError`; we catch this earlier so callers only see `ConfigError` subclasses.)

2. **`yaml.safe_load`, not `yaml.load`.** This matters. `yaml.load` honors `!!python/object:os.system` and other tags that can execute arbitrary code at parse time. `safe_load` refuses them — it loads only simple types (strings, numbers, lists, dicts, bools). `test_safe_load_rejects_python_object_tag` guards this.

3. **Shape checks before schema.** An empty YAML file parses to `None`; a scalar like `"hi"` parses to a string; a top-level list parses to a list. None of these are dicts, so we'd get a weird pydantic error trying to validate them. Instead we fail with a clear `ConfigParseError` message naming what the top level actually was.

4. **`"agent" not in data` check.** Without this the user would get a confusing pydantic error about missing `id`, `name`, etc. We tell them directly: the top-level `agent:` key is missing.

5. **`model_validate` + wrapped ValidationError.** Pydantic's own `ValidationError` is noisy. We wrap it in `ConfigValidationError` with our own short message, but we preserve the pydantic error via `from e` so callers can still inspect field-level details via `e.__cause__`.

---

## 5. Errors (`errors.py`)

```
ConfigError (base)
├── ConfigFileNotFoundError   — path does not exist (or is a directory)
├── ConfigParseError          — file exists but isn't a valid YAML mapping with an 'agent:' key
└── ConfigValidationError     — YAML parses but schema fails; __cause__ is pydantic.ValidationError
```

Four distinct failure modes, each with its own class. Callers can be precise:
```python
try:
    profile = load_profile(path)
except ConfigFileNotFoundError:
    ...  # bad path
except ConfigParseError:
    ...  # fix your YAML
except ConfigValidationError as e:
    print(e.__cause__.errors())   # per-field pydantic detail
```

---

## 6. Example YAML — `agents/maya_rodriguez/profile.yaml`

```yaml
agent:
  id: "student_maya_001"
  name: "Maya Rodriguez"
  age: 20
  traits: "curious, persistent, collaborative"
  backstory: >
    A second-year Computer Science student at a state university.
    Grew up bilingual (Spanish/English)...
  initial_plan: >
    Wake up at 7:30, review yesterday's lecture notes over coffee,
    attend the 9 AM data structures lecture...
  cognitive:
    max_steps_per_cycle: 8
    reflection_threshold: 4
    importance_threshold: 6
    planning_horizon: "1 day"
  memory:
    max_working_memory_tokens: 3000
    retrieval_top_k: 8
    recency_weight: 1.0
    importance_weight: 1.2
    relevance_weight: 1.5
```

**YAML `>` folded scalar.** Lines after `>` are joined with spaces and a single trailing newline. This lets us write multi-line backstories without literal `\n` characters but still have them serialize cleanly. The loader `.strip()`s the result when building the system prompt so the trailing `\n` doesn't leak into the LLM input.

---

## 7. Execution flow: `scripts/show_profile.py`

Quick diagnostic tool. Takes an optional path; defaults to `agents/alice_chen/profile.yaml`.

```
$ python scripts/show_profile.py

┌─ main()
│
├─ 1. path = DEFAULT_PATH (= agents/alice_chen/profile.yaml)
│        or argv[1] if provided
│
├─ 2. profile = load_profile(path)
│     │
│     ├─ ConfigFileNotFoundError? print + return 1
│     ├─ ConfigParseError? print + return 1
│     └─ ConfigValidationError? print + return 1
│     (all three subclasses are caught by `except ConfigError`)
│
├─ 3. print("[show_profile] loaded <path>")
│
└─ 4. print(json.dumps(profile.model_dump(), indent=2))
       • model_dump() returns a pure-dict representation
       • json.dumps pretty-prints it (including nested cognitive/memory blocks)
```

Sample output:
```
[show_profile] loaded /Users/.../agent_lab/agents/alice_chen/profile.yaml
[show_profile] model: AgentProfile
---
{
  "id": "agent_001",
  "name": "Alice Chen",
  ...
  "cognitive": { ... },
  "memory": { ... }
}
```

This is the fastest way to answer "did my YAML edit break anything?" — if `load_profile` fails, you see exactly which error class + message.

---

## 8. Execution flow: `scripts/profile_chat_demo.py` — Module 1 ⊗ Module 2

The composition demo. Loads a profile (Module 2) + loads an adapter from .env (Module 1) + sends one turn of conversation.

```
$ python scripts/profile_chat_demo.py

┌─ main() (async)
│
├─ Step 1: load_profile(agents/maya_rodriguez/profile.yaml)
│          → AgentProfile(name="Maya Rodriguez", age=20, ...)
│          → fails fast with ConfigError if YAML is broken
│
├─ Step 2: make_adapter_from_env()
│          → reads AGENT_LAB_LLM_PROVIDER from .env
│          → reads {PROVIDER}_API_KEY / BASE_URL / MODEL (with LLM_* overrides)
│          → returns OpenAICompatibleAdapter or AnthropicAdapter
│          → fails fast with LLMConfigError if .env is incomplete
│
├─ Step 3: build_system_prompt(profile)
│          → returns a multi-line str built from profile.name, age, traits,
│            backstory, initial_plan + the "stay in character" instruction
│
├─ Step 4: adapter.chat(
│              messages=[Message(role="user", content=USER_QUESTION)],
│              system=system_prompt,
│              temperature=0.7,
│              max_tokens=256,
│          )
│          │
│          ├─ OpenAICompatibleAdapter._chat (for DeepSeek)
│          │   • merges system_prompt as first {"role":"system","content":...} on the wire
│          │   • serializes messages to OpenAI shape
│          │   • HTTPS POST to api.deepseek.com/v1/chat/completions
│          │   • parses choices[0].message.content → LLMResponse
│          │
│          └─ returns LLMResponse
│
└─ Step 5: print profile.name + resp.content + usage
          e.g.: "[demo] Maya Rodriguez: Morning was solid! ..."
```

**Why this is important as an example:** it shows the **exact composition pattern** that every future harness module will use. Cognitive loop, context manager, and memory will all be passed an `AgentProfile` and an `LLMAdapter` and will do roughly the same dance: *use profile to shape the prompt → call the adapter → interpret the response*. The demo is a hand-written version of a flow that will soon be automated by `core/`.

**Where could execution fail?**

| Failure | Stage | Exception class | Exit code |
|---|---|---|---|
| YAML file missing at the given path | Step 1 | `ConfigFileNotFoundError` | 2 |
| YAML file unparseable | Step 1 | `ConfigParseError` | 2 |
| YAML file valid but schema mismatches | Step 1 | `ConfigValidationError` | 2 |
| `.env` missing provider / key / model | Step 2 | `LLMConfigError` | 2 |
| Network failure, bad auth, rate limit | Step 4 | `LLMProviderError` | 1 |

All five error paths are caught by the demo; the exit codes distinguish "fix your config" (2) from "runtime failure, maybe retry" (1).

---

## 9. Test coverage map

| File | Tests | What's covered |
|---|---|---|
| `tests/DefenseAgent/config/test_errors.py` | 3 | Exception hierarchy + `__cause__` preservation |
| `tests/DefenseAgent/config/test_profile_models.py` | 34 | Every field's default, every validator (parametrized), extra-key rejection on each nested model, empty-string rejection, `age=0` edge case |
| `tests/DefenseAgent/config/test_loader.py` | 17 | Happy path (minimal + full YAML), every `ConfigError` subclass branch, YAML-safety guard, chained `__cause__` on validation error |
| `tests/DefenseAgent/integration/test_profile_llm_integration.py` | 4 | Shipped student profile round-trips; identity fields reach the adapter boundary; minimal profile composes with defaults; inline profile composes without depending on shipped YAMLs |

All tests are fully offline — YAML files go to `tmp_path`, the LLM uses a `StubAdapter`.

---

## 10. Things worth noticing

- **Data and code are separated on disk.** `DefenseAgent/config/` has no YAMLs. `agents/` has no Python. You can delete all of `agents/` and the loader code still runs (with a `ConfigFileNotFoundError` when you try to use it).

- **Each agent is self-contained.** `agents/maya_rodriguez/` holds Maya's profile, her skills, and her MCP declarations — nothing belonging to Maya lives anywhere else. Copy-paste the folder to create a new agent; the paths declared in the profile are all relative so they travel with the bundle.

- **`profile.source_dir` is the anchor for relative paths.** `AgentProfile.from_yaml` records the resolved YAML location on the instance. `ToolRegistry.from_profile` reads `profile.tools.skills` and joins each entry against `profile.source_dir` to get an absolute path before calling `add_skill`. Two agents can never accidentally see each other's skills because each one's paths only resolve below its own bundle.
- **Strict Pydantic mode makes typos loud.** `extra="forbid"` on every model means a `traists:` in YAML is a ValidationError before the profile ever reaches a consumer. This is pedagogically important — silent ignoring would make learning the schema harder.
- **`default_factory` on nested models.** Using `Field(default_factory=CognitiveConfig)` (not `= CognitiveConfig()`) is the right pydantic pattern and matters once tests start mutating profile objects across cases.
- **Loader doesn't know about the LLM.** `profile.py` imports `yaml` and `pydantic`. It does NOT import anything from `DefenseAgent.llm`. The composition happens only in `scripts/profile_chat_demo.py` and in the integration test — not inside either module. Future cross-module logic (building a system prompt from a profile, feeding a profile's memory weights to a retriever) will live in a future `DefenseAgent/core/` module, not here.
- **Integration test ships a `StubAdapter`.** It subclasses `LLMAdapter` directly. This is the first place outside `DefenseAgent/llm/` that has demonstrated the abstract-base-class contract works — any subclass of `LLMAdapter` that implements `chat()` drops in. This is the reason the adapter layer bothered to be abstract.
