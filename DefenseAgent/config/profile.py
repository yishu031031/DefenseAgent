from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationError, model_validator


class ConfigError(Exception):
    """Base class for every error raised while loading agent configuration."""


class ConfigFileNotFoundError(ConfigError):
    """Raised when the profile YAML path does not point to a readable file."""


class ConfigParseError(ConfigError):
    """Raised when the file exists but cannot be parsed as the expected YAML structure."""


class ConfigValidationError(ConfigError):
    """Raised when the YAML parses but fails AgentProfile schema validation (pydantic.ValidationError chained)."""


_STRICT_MODEL_CONFIG = ConfigDict(extra="forbid")


def _drop_none_for_keys(data: Any, keys: tuple[str, ...]) -> Any:
    """In a `mode='before'` validator, drop entries whose value is None for the listed keys. This makes a blank YAML key (parsed as None) behave like an omitted key — the schema's default kicks in instead of pydantic crashing on `string_type`."""
    if not isinstance(data, dict):
        return data
    return {k: v for k, v in data.items() if not (k in keys and v is None)}


class CognitiveConfig(BaseModel):
    """Knobs that control the agent's cognitive loop (reflection threshold, plan horizon, etc.)."""

    model_config = _STRICT_MODEL_CONFIG

    max_steps_per_cycle: int = Field(ge=1, default=10)
    reflection_threshold: int = Field(ge=1, default=5)
    importance_threshold: float = Field(ge=1, le=10, default=7)
    planning_horizon: str = Field(min_length=1, default="1 day")

    @model_validator(mode="before")
    @classmethod
    def _coerce_blank_yaml_keys(cls, data: Any) -> Any:
        """Treat `planning_horizon:` (blank YAML key → None) as 'omitted' so the default fires."""
        return _drop_none_for_keys(data, ("planning_horizon",))


class MemoryConfig(BaseModel):
    """Memory subsystem configuration matching ms-agent's mem0-backed scheme: storage, search, compaction, ingestion knobs."""

    model_config = _STRICT_MODEL_CONFIG

    storage_path: str | None = None
    history_mode: str = Field(default="add", pattern=r"^(add|overwrite)$")
    is_retrieve: bool = True
    search_limit: int = Field(ge=1, default=10)
    ignore_roles: list[str] = Field(default_factory=lambda: ["tool", "system"])
    ignore_fields: list[str] = Field(default_factory=lambda: ["reasoning_content"])

    context_limit: int = Field(ge=1024, default=128_000)
    prune_protect: int = Field(ge=0, default=40_000)
    prune_minimum: int = Field(ge=0, default=20_000)
    reserved_buffer: int = Field(ge=0, default=20_000)
    enable_summary: bool = True


class RAGConfig(BaseModel):
    """Optional RAG (retrieval-augmented generation) backend configuration. Disabled by default; mirrors ms-agent's `LlamaIndexRAG` knobs. Embedding fields (`embedding`, `embedding_api_key`, `embedding_base_url`, `embedding_dims`) follow the same per-field profile-then-env fallback as `LLMConfig` — set them in the profile to override .env, leave them blank to inherit the `EMBEDDING_*` block from .env."""

    model_config = _STRICT_MODEL_CONFIG

    enabled: bool = False
    documents_dir: str | None = None
    storage_dir: str | None = None
    embedding_provider: str = Field(default="openai", pattern=r"^(openai|huggingface)$")
    embedding: str | None = None
    embedding_api_key: str | None = None
    embedding_base_url: str | None = None
    embedding_dims: int | None = Field(default=None, ge=1)
    chunk_size: int = Field(ge=1, default=512)
    chunk_overlap: int = Field(ge=0, default=50)
    retrieve_only: bool = True
    top_k: int = Field(ge=1, default=5)
    score_threshold: float = Field(ge=0.0, le=1.0, default=0.0)
    use_huggingface: bool = False


class MCPServerConfig(BaseModel):
    """One MCP server entry in the agent's tools config. Mirrors ms-agent's `mcpServers` dict shape: stdio servers set `command`+`args`; remote servers set `url` and (optionally) `transport` (`sse` / `websocket` / defaults to `streamable_http`). Per-server `include`/`exclude` filters are passed through to the underlying client. Empty values in `env` are interpolated from the process environment at connect time."""

    model_config = _STRICT_MODEL_CONFIG

    name: str | None = None

    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] | None = None
    cwd: str | None = None

    transport: str | None = Field(
        default=None,
        pattern=r"^(stdio|sse|websocket|streamable_http)$",
    )
    url: str | None = None
    headers: dict[str, str] | None = None
    timeout: float | None = Field(default=None, ge=0)
    sse_read_timeout: float | None = Field(default=None, ge=0)

    include: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_command_xor_url(self) -> "MCPServerConfig":
        """Each server must specify exactly one of `command` (stdio) or `url` (network). `include` and `exclude` are mutually exclusive."""
        if bool(self.command) == bool(self.url):
            raise ValueError(
                "MCP server config requires exactly one of `command` or `url`"
            )
        if self.include and self.exclude:
            raise ValueError(
                "MCP server config: set either `include` or `exclude`, not both"
            )
        return self


class ToolsConfig(BaseModel):
    """Per-agent tool registrations: skill directories (paths), MCP server launch configs, and Python entry-point strings (`module.path:function_name`) resolved via `importlib` at agent construction. The Python-entry path imports arbitrary code — only list entry points you trust."""

    model_config = _STRICT_MODEL_CONFIG

    skills: list[str] = Field(default_factory=list)
    mcp: list[MCPServerConfig] = Field(default_factory=list)
    python: list[str] = Field(default_factory=list)
    allow_skill_execution: bool = False
    skill_execution_timeout: int = Field(ge=1, default=300)


class PromptConfig(BaseModel):
    """User-authored system prompt for the agent. `system` wins over `path`; both are optional and fall back to the auto-built identity block."""

    model_config = _STRICT_MODEL_CONFIG

    system: str | None = None
    path: str | None = None
    extra_instructions: str | None = None


class LLMConfig(BaseModel):
    """Per-agent LLM overrides. Each field is optional; the resolver fills missing values from .env (`AGENT_LAB_LLM_PROVIDER`, `<PROVIDER>_API_KEY`, `<PROVIDER>_BASE_URL`, `<PROVIDER>_MODEL`, with the cross-provider `LLM_*` tier as the second fallback). Putting `api_key` in YAML is convenient for demos but *not* recommended for shared profiles — leave it blank in the file and let .env supply it."""

    model_config = _STRICT_MODEL_CONFIG

    provider: str | None = None
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None


class AgentProfile(BaseModel):
    """Module 2's unified facade: the validated agent identity plus nested cognitive, memory, and tools configs."""

    model_config = _STRICT_MODEL_CONFIG

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    age: int | None = Field(default=None, ge=0)
    traits: str = ""
    backstory: str = ""
    initial_plan: str = ""
    cognitive: CognitiveConfig = Field(default_factory=CognitiveConfig)

    @model_validator(mode="before")
    @classmethod
    def _coerce_blank_yaml_keys(cls, data: Any) -> Any:
        """Treat blank YAML keys (`traits:`, `backstory:`, `initial_plan:` parsed as None) as 'omitted' so the schema defaults fire instead of pydantic rejecting None as a non-string."""
        return _drop_none_for_keys(data, ("traits", "backstory", "initial_plan"))
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    prompt: PromptConfig = Field(default_factory=PromptConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)

    _source_path: Path | None = PrivateAttr(default=None)

    @property
    def source_path(self) -> Path | None:
        """Absolute path of the YAML this profile was loaded from; None when built in-memory."""
        return self._source_path

    @property
    def source_dir(self) -> Path | None:
        """Directory containing the loaded profile; the anchor for resolving relative tool paths."""
        if self._source_path is None:
            return None
        return self._source_path.parent

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AgentProfile":
        """Load and validate a profile from `path`; stores the resolved path on the instance for later path resolution."""
        file_path = Path(path)
        if not file_path.is_file():
            raise ConfigFileNotFoundError(f"profile file not found: {file_path}")

        raw_text = file_path.read_text(encoding="utf-8")
        try:
            data: Any = yaml.safe_load(raw_text)
        except yaml.YAMLError as e:
            raise ConfigParseError(f"invalid YAML in {file_path}: {e}") from e

        if not isinstance(data, dict):
            raise ConfigParseError(
                f"expected top-level mapping in {file_path}, "
                f"got {type(data).__name__}"
            )

        if "agent" not in data:
            raise ConfigParseError(f"missing top-level 'agent:' key in {file_path}")

        agent_data = data["agent"]
        if not isinstance(agent_data, dict):
            raise ConfigParseError(
                f"'agent:' value must be a mapping in {file_path}, "
                f"got {type(agent_data).__name__}"
            )

        try:
            profile = cls.model_validate(agent_data)
        except ValidationError as e:
            raise ConfigValidationError(
                f"profile at {file_path} failed schema validation:\n{e}"
            ) from e

        profile._source_path = file_path.resolve()
        return profile
