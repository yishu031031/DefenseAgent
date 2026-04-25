import asyncio
import inspect
from pathlib import Path
from typing import Any, Awaitable, Callable

from DefenseAgent.config.profile import AgentProfile
from DefenseAgent.llm.types import Message, ToolCall
from DefenseAgent.tools.mcp import MCPClient
from DefenseAgent.tools.skill import Skill
from DefenseAgent.tools.types import (
    Tool,
    ToolError,
    ToolExecutionError,
    ToolHandler,
    ToolNotFoundError,
    ToolRegistrationError,
)


_PY_TYPE_TO_JSON: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


class ToolRegistry:
    """Module 6's unified facade; registers user-defined functions, skills, and MCP servers and dispatches LLM tool calls."""

    def __init__(self) -> None:
        """Start with an empty registry; use as an async context manager to auto-close MCP clients."""
        self._tools: dict[str, Tool] = {}
        self._mcp_clients: list[MCPClient] = []

    async def __aenter__(self) -> "ToolRegistry":
        """Return self for `async with ToolRegistry() as registry:` style lifecycle management."""
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        """Close every MCP client opened through this registry on exit."""
        await self.close()

    @classmethod
    async def from_profile(
        cls,
        profile: AgentProfile,
        *,
        base_dir: str | Path | None = None,
    ) -> "ToolRegistry":
        """Build a ToolRegistry from `profile.tools`, resolving skill paths against `base_dir` (defaults to the profile's directory)."""
        registry = cls()
        if base_dir is None:
            if profile.source_dir is None:
                raise ToolRegistrationError(
                    "profile has no source_dir; pass base_dir explicitly when "
                    "loading tools from an in-memory profile"
                )
            base = profile.source_dir
        else:
            base = Path(base_dir).resolve()
        for skill_ref in profile.tools.skills:
            skill_path = (base / skill_ref).resolve()
            registry.add_skill(skill_path)
        for mcp_cfg in profile.tools.mcp:
            await registry.add_mcp(
                command=mcp_cfg.command,
                args=mcp_cfg.args,
                env=mcp_cfg.env,
                cwd=mcp_cfg.cwd,
            )
        return registry

    def tool(
        self,
        func: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> Any:
        """Decorator that registers a Python callable as a tool; usable as `@registry.tool` or `@registry.tool(name=...)`."""
        def register(f: Callable[..., Any]) -> Callable[..., Any]:
            tool_name = name if name is not None else f.__name__
            doc = description if description is not None else (inspect.getdoc(f) or "")
            schema = _schema_from_signature(f)
            handler = _wrap_python_handler(f)
            self.register(
                Tool(
                    name=tool_name,
                    description=doc,
                    input_schema=schema,
                    source="python",
                    handler=handler,
                )
            )
            return f
        if func is None:
            return register
        return register(func)

    def add_skill(self, directory: str | Path) -> Tool:
        """Load an Anthropic Agent Skill from `directory` (must contain SKILL.md) and register it as one Tool."""
        skill = Skill(directory)
        tool = skill.to_tool()
        self.register(tool)
        return tool

    async def add_mcp(
        self,
        *,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> list[Tool]:
        """Connect to an MCP stdio server, discover its tools, register each, and return them."""
        client = MCPClient(command=command, args=args, env=env, cwd=cwd)
        self._mcp_clients.append(client)
        discovered = await client.enter()
        for tool in discovered:
            self.register(tool)
        return discovered

    def register(self, tool: Tool) -> None:
        """Add a pre-built Tool to the registry; raises ToolRegistrationError on name collision."""
        if tool.name in self._tools:
            raise ToolRegistrationError(f"tool name already registered: {tool.name!r}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        """Return the Tool registered under `name`; raises ToolNotFoundError when missing."""
        if name not in self._tools:
            raise ToolNotFoundError(f"no tool named {name!r}")
        return self._tools[name]

    def spec(self) -> list[dict[str, Any]]:
        """Return every registered tool as a canonical {name, description, input_schema} dict for the LLM."""
        specs: list[dict[str, Any]] = []
        for tool in self._tools.values():
            specs.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                }
            )
        return specs

    async def execute(self, tool_calls: list[ToolCall]) -> list[Message]:
        """Run every ToolCall concurrently; return one role='tool' Message per call (errors become error messages)."""
        if not tool_calls:
            return []
        coros = [self._execute_one(tc) for tc in tool_calls]
        return await asyncio.gather(*coros)

    async def _execute_one(self, tc: ToolCall) -> Message:
        """Look up the tool by name, run its handler, package the return or failure as a tool-role Message."""
        tool = self._tools.get(tc.name)
        if tool is None:
            return Message(
                role="tool",
                content=f"ToolNotFoundError: no tool named {tc.name!r}",
                tool_call_id=tc.id,
                name=tc.name,
            )
        try:
            content = await tool.handler(tc.arguments)
        except ToolError as e:
            content = f"{type(e).__name__}: {e}"
        except Exception as e:
            content = f"ToolExecutionError: {type(e).__name__}: {e}"
        return Message(
            role="tool",
            content=content,
            tool_call_id=tc.id,
            name=tc.name,
        )

    async def close(self) -> None:
        """Close every MCP client that was opened by this registry."""
        for client in self._mcp_clients:
            await client.close()
        self._mcp_clients.clear()

    def names(self) -> list[str]:
        """Return the registered tool names in insertion order."""
        return list(self._tools.keys())

    def __len__(self) -> int:
        """Return the number of registered tools."""
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        """Return True when a tool with the given name is registered."""
        return name in self._tools


def _schema_from_signature(func: Callable[..., Any]) -> dict[str, Any]:
    """Build a JSON-schema `input_schema` from `func`'s parameter annotations + defaults."""
    sig = inspect.signature(func)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for param_name, param in sig.parameters.items():
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if param.annotation is inspect.Parameter.empty:
            annotation: Any = str
        else:
            annotation = param.annotation
        if annotation in _PY_TYPE_TO_JSON:
            json_type = _PY_TYPE_TO_JSON[annotation]
        else:
            json_type = "string"
        properties[param_name] = {"type": json_type}
        if param.default is inspect.Parameter.empty:
            required.append(param_name)
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def _wrap_python_handler(func: Callable[..., Any]) -> ToolHandler:
    """Wrap a sync-or-async Python callable into the async (arguments -> str) handler shape."""
    async def handler(arguments: dict[str, Any]) -> str:
        if inspect.iscoroutinefunction(func):
            result = await func(**arguments)
        else:
            result = await asyncio.to_thread(lambda: func(**arguments))
        return _stringify(result)
    return handler


def _stringify(value: Any) -> str:
    """Return `value` as a string (passes strings through; `str()` everything else)."""
    if isinstance(value, str):
        return value
    return str(value)
