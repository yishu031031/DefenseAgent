# Module 2 — Agent Profile Loader Design

**Date:** 2026-04-22
**Status:** Approved, ready for implementation
**Module position:** 2 of N. Consumed by later modules (cognitive loop, context manager, memory retriever) that need agent identity and tuning parameters. No runtime coupling to Module 1 (LLM).

> **Amendment (2026-04-24):**
> - `AgentProfile` now carries a `tools: ToolsConfig` field (`skills: list[str]`, `mcp: list[MCPServerConfig]`) so each agent declares its own tool set. Defaults to empty — profiles that predate this field still load unchanged.
> - `AgentProfile` instances remember where they were loaded from: `profile.source_path` / `profile.source_dir` are populated by `from_yaml`, so downstream code (notably `ToolRegistry.from_profile`) can resolve relative tool paths from the profile's directory.
> - Profile YAMLs moved from `profiles/<name>.yaml` into **agent bundles** at `agents/<id>/profile.yaml`, with each agent's `skills/` directory sitting next to its profile. The "File layout" block below reflects the current structure; the model table and loader sections are amended inline.

## Purpose

Turn a YAML file into a typed, validated `AgentProfile` object. The rest of the harness should never read YAML directly — all agent-configuration knobs flow through this module's models. A typo in the YAML must fail loud at load time, not three modules later during a cognitive step.

## Scope

### In scope
- Three Pydantic v2 models: `CognitiveConfig`, `MemoryConfig`, `AgentProfile`.
- `load_profile(path) -> AgentProfile` loader (reads file, parses YAML, validates, returns model).
- Strict validation (`extra = "forbid"`): unknown keys are rejected.
- Custom error hierarchy (`ConfigError` plus three subclasses) so callers can branch on failure mode.
- Default/reference YAML at `profiles/alice_chen.yaml`.
- Unit tests covering every model validator and every loader error branch.
- Sample script `scripts/show_profile.py` that loads the default file and pretty-prints the parsed model.

### Out of scope (deferred)
- `settings.yaml` (runtime settings like log level) — separate concern, add when a consumer shows up.
- Parsing `planning_horizon` into a structured duration (e.g., `timedelta`) — keep it as a free-form string until the cognitive loop actually uses it.
- Hot reload / file watching.
- Multi-profile loading / profile inheritance.
- Integration with the LLM module (there is no coupling yet).

## Design

### Models

All models set `model_config = ConfigDict(extra="forbid")` — unknown YAML keys raise a validation error.

#### `CognitiveConfig`
Tuning knobs for the cognitive loop. All fields have sensible defaults so a minimal profile can omit the whole block.

| Field                  | Type  | Constraint | Default |
|------------------------|-------|------------|---------|
| `max_steps_per_cycle`  | int   | ≥ 1        | 10      |
| `reflection_threshold` | int   | ≥ 1        | 5       |
| `importance_threshold` | float | 1 ≤ x ≤ 10 | 7       |
| `planning_horizon`     | str   | non-empty  | "1 day" |

#### `MemoryConfig`
Memory-system weights and budgets. All fields have sensible defaults.

| Field                        | Type  | Constraint | Default |
|------------------------------|-------|------------|---------|
| `max_working_memory_tokens`  | int   | ≥ 1        | 4000    |
| `retrieval_top_k`            | int   | ≥ 1        | 10      |
| `recency_weight`             | float | ≥ 0        | 1.0     |
| `importance_weight`          | float | ≥ 0        | 1.0     |
| `relevance_weight`           | float | ≥ 0        | 1.0     |

#### `MCPServerConfig` *(added 2026-04-24)*
Stdio launch parameters for one MCP server.

| Field     | Type              | Constraint | Default |
|-----------|-------------------|------------|---------|
| `command` | str               | non-empty  | (required) |
| `args`    | list[str]         | —          | `[]`    |
| `env`     | dict[str,str] \| None | —      | `None`  |
| `cwd`     | str \| None       | —          | `None`  |

#### `ToolsConfig` *(added 2026-04-24)*
Per-agent tool declarations. All fields default to empty lists so existing profiles continue to validate.

| Field    | Type                        | Default |
|----------|-----------------------------|---------|
| `skills` | list[str] (paths)           | `[]`    |
| `mcp`    | list[`MCPServerConfig`]     | `[]`    |

#### `AgentProfile`
Identity + nested configs. Identity fields are required; every nested block has a default factory so minimal profiles can omit them.

| Field          | Type               | Constraint        | Default                |
|----------------|--------------------|-------------------|------------------------|
| `id`           | str                | non-empty         | (required)             |
| `name`         | str                | non-empty         | (required)             |
| `age`          | int                | ≥ 0               | (required)             |
| `traits`       | str                | non-empty         | (required)             |
| `backstory`    | str                | non-empty         | (required)             |
| `initial_plan` | str                | non-empty         | (required)             |
| `cognitive`    | `CognitiveConfig`  | —                 | `CognitiveConfig()`    |
| `memory`       | `MemoryConfig`     | —                 | `MemoryConfig()`       |
| `tools`        | `ToolsConfig`      | —                 | `ToolsConfig()`        |

**Source tracking.** `from_yaml` populates two private-but-readable attributes on the returned instance:

- `profile.source_path: Path | None` — the resolved path the YAML was loaded from.
- `profile.source_dir: Path | None`  — its parent directory, the anchor used by `ToolRegistry.from_profile` to resolve relative skill paths.

In-memory profiles (built via `AgentProfile(...)` or `model_validate`) have both fields set to `None`. Callers that want to load tools for such profiles must pass an explicit `base_dir` to `ToolRegistry.from_profile`.

### YAML shape

The loader expects a single top-level key `agent:` whose value matches `AgentProfile`. Rationale: YAML files often evolve to carry multiple top-level sections (e.g., `agent:`, `environment:`, `world:`); namespacing under `agent:` from the start keeps the door open.

```yaml
agent:
  id: "agent_001"
  name: "Alice Chen"
  age: 28
  traits: "curious, methodical, empathetic"
  backstory: "A data scientist who recently moved to Lakeside..."
  initial_plan: "Wake up, review emails, work on the data analysis project"
  cognitive:
    max_steps_per_cycle: 10
    reflection_threshold: 5
    importance_threshold: 7
    planning_horizon: "1 day"
  memory:
    max_working_memory_tokens: 4000
    retrieval_top_k: 10
    recency_weight: 1.0
    importance_weight: 1.0
    relevance_weight: 1.0
```

### Loader

```python
def load_profile(path: str | Path) -> AgentProfile:
    """Read YAML at `path`, validate, return an AgentProfile.

    Raises:
        ConfigFileNotFoundError: path does not exist.
        ConfigParseError:        file exists but isn't valid YAML, or top-level isn't a mapping.
        ConfigValidationError:   YAML parses but doesn't match the AgentProfile schema
                                 (missing required field, wrong type, out-of-range value,
                                 unknown key, etc.).
    """
```

**Resolution order inside the loader:**
1. Open file → `ConfigFileNotFoundError` if missing.
2. `yaml.safe_load` → `ConfigParseError` on `yaml.YAMLError`, also if the result isn't a dict.
3. Require top-level `agent:` key → `ConfigParseError` if absent.
4. `AgentProfile.model_validate(data["agent"])` → wrap `pydantic.ValidationError` in `ConfigValidationError` with the original pydantic error as `__cause__`.

`yaml.safe_load` is used (not `yaml.load`) to reject `!!python/object` and similar code-execution paths.

### Errors (`DefenseAgent/config/errors.py`)

```python
class ConfigError(Exception): ...
class ConfigFileNotFoundError(ConfigError): ...
class ConfigParseError(ConfigError): ...
class ConfigValidationError(ConfigError):
    """YAML parsed but failed schema validation.

    The original pydantic ValidationError is attached via `raise ... from e`.
    """
```

### File layout *(revised 2026-04-24)*

```
DefenseAgent/config/                # loader CODE only
├── __init__.py                      # re-exports AgentProfile + Tools/MCP models + errors
└── profile.py                       # all models + ConfigError hierarchy + from_yaml

agents/                              # user-editable DATA — one bundle per agent
├── alice_chen/
│   └── profile.yaml
└── maya_rodriguez/
    ├── profile.yaml                 # identity + cognitive + memory + tools
    └── skills/                      # private skills (resolved by ToolRegistry.from_profile)
        └── tabular-report/
            ├── SKILL.md
            ├── scripts/
            │   └── generate.py
            └── templates/
                └── header.md

tests/DefenseAgent/config/
├── __init__.py
├── test_profile.py                  # model + loader validation
└── test_tools_config.py             # ToolsConfig / MCPServerConfig / source_path

scripts/
├── show_profile.py                  # load + pretty-print the default profile
├── profile_chat_demo.py             # load a profile + chat via the LLM
└── tools_demo.py                    # load profile + build ToolRegistry from it
```

Original layout had one flat `profiles/` directory and a single `errors.py`. Both consolidated on earlier passes; the agent-bundle split is the 2026-04-24 change.

### Dependencies added

Added to `requirements.txt`:
```
pyyaml>=6.0
pydantic>=2.0
```

(`pydantic` was previously transitive via `anthropic`; now declared explicitly so we own the lower bound.)

## Testing strategy

No I/O mocking needed — pytest's `tmp_path` fixture gives each test a fresh temp directory. Tests write small YAML files to `tmp_path` and call `load_profile(tmp_path / "x.yaml")`.

**Model tests** (direct construction, no YAML):
- Minimal required fields → valid profile, nested blocks default-populated.
- Each range validator: below/at/above bounds → test boundary behavior.
- Unknown field on any model → `ValidationError` (pydantic).
- Empty string on `id`/`name`/`traits`/`backstory`/`initial_plan` → rejected.

**Loader tests** (file-based, `tmp_path`):
- Valid YAML → equal to the expected `AgentProfile`.
- Missing file → `ConfigFileNotFoundError`.
- Malformed YAML → `ConfigParseError`.
- Top level is a list/scalar, not a mapping → `ConfigParseError`.
- Missing top-level `agent:` → `ConfigParseError`.
- Valid YAML shape but `age: "twenty-eight"` → `ConfigValidationError` (pydantic tried coercion and failed).
- `ConfigValidationError.__cause__` is a `pydantic.ValidationError` (so callers can inspect details).

## Integration with later modules

`AgentProfile` is consumed by:
- **Cognitive loop:** `profile.cognitive.max_steps_per_cycle`, `reflection_threshold`, `importance_threshold`, `planning_horizon`.
- **Context manager:** `profile.memory.max_working_memory_tokens`.
- **Memory retriever:** `profile.memory.retrieval_top_k`, `recency_weight`, `importance_weight`, `relevance_weight`.
- **Agent orchestrator:** `profile.id`, `name`, `traits`, `backstory` for the system prompt.

None of these modules exist yet. This module's only downstream contract is: _"we will hand you a validated `AgentProfile` with these fields and these types."_

## Open questions

None at spec-approval time. Design choices confirmed with the user on 2026-04-22:
1. Pydantic v2 over plain dataclasses — confirmed.
2. Strict validation (`extra="forbid"`) — chosen for clear feedback on typos.
3. YAML under `DefenseAgent/config/` (ships with the repo), loader accepts any path.
4. Single `agent:` top-level key — namespacing for future top-level sections.
