from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal


ToolSource = Literal["python", "skill", "mcp"]

ToolHandler = Callable[[dict[str, Any]], Awaitable[str]]


@dataclass
class Tool:
    """One registered tool: name, description, JSON input schema, source, and async handler."""
    name: str
    description: str
    input_schema: dict[str, Any]
    source: ToolSource
    handler: ToolHandler
    metadata: dict[str, Any] = field(default_factory=dict)


class ToolError(Exception):
    """Base class for every error raised from the tools module."""


class ToolRegistrationError(ToolError):
    """Raised when a tool cannot be registered (name collision, bad signature, etc.)."""


class ToolNotFoundError(ToolError):
    """Raised when execute() is asked for a tool name that is not registered."""


class ToolExecutionError(ToolError):
    """Raised when a tool's handler fails (original chained via __cause__)."""


class SkillLoadError(ToolError):
    """Raised when a skill directory cannot be loaded (missing SKILL.md, bad frontmatter, etc.)."""
