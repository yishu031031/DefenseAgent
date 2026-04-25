from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationError


class ConfigError(Exception):
    """Base class for every error raised while loading agent configuration."""


class ConfigFileNotFoundError(ConfigError):
    """Raised when the profile YAML path does not point to a readable file."""


class ConfigParseError(ConfigError):
    """Raised when the file exists but cannot be parsed as the expected YAML structure."""


class ConfigValidationError(ConfigError):
    """Raised when the YAML parses but fails AgentProfile schema validation (pydantic.ValidationError chained)."""


_STRICT_MODEL_CONFIG = ConfigDict(extra="forbid")


class CognitiveConfig(BaseModel):
    """Knobs that control the agent's cognitive loop (reflection threshold, plan horizon, etc.)."""

    model_config = _STRICT_MODEL_CONFIG

    max_steps_per_cycle: int = Field(ge=1, default=10)
    reflection_threshold: int = Field(ge=1, default=5)
    importance_threshold: float = Field(ge=1, le=10, default=7)
    planning_horizon: str = Field(min_length=1, default="1 day")


class MemoryConfig(BaseModel):
    """Knobs for the memory subsystem (retrieval weights, top_k, token budget)."""

    model_config = _STRICT_MODEL_CONFIG

    max_working_memory_tokens: int = Field(ge=1, default=4000)
    retrieval_top_k: int = Field(ge=1, default=10)
    recency_weight: float = Field(ge=0, default=1.0)
    importance_weight: float = Field(ge=0, default=1.0)
    relevance_weight: float = Field(ge=0, default=1.0)


class MCPServerConfig(BaseModel):
    """Launch parameters for one stdio-based MCP server the agent should talk to."""

    model_config = _STRICT_MODEL_CONFIG

    command: str = Field(min_length=1)
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] | None = None
    cwd: str | None = None


class ToolsConfig(BaseModel):
    """Per-agent tool registrations: skill directories (paths) plus MCP server launch configs."""

    model_config = _STRICT_MODEL_CONFIG

    skills: list[str] = Field(default_factory=list)
    mcp: list[MCPServerConfig] = Field(default_factory=list)


class AgentProfile(BaseModel):
    """Module 2's unified facade: the validated agent identity plus nested cognitive, memory, and tools configs."""

    model_config = _STRICT_MODEL_CONFIG

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    age: int = Field(ge=0)
    traits: str = Field(min_length=1)
    backstory: str = Field(min_length=1)
    initial_plan: str = Field(min_length=1)
    cognitive: CognitiveConfig = Field(default_factory=CognitiveConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)

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
