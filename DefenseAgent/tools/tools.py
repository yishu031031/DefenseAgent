import asyncio
import inspect
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from DefenseAgent.config.profile import AgentProfile, MCPServerConfig
from DefenseAgent.llm.types import Message, ToolCall
from DefenseAgent.skills import SkillLoader
from DefenseAgent.tools.mcp import MCPClient
from DefenseAgent.tools.types import (
    Tool,
    ToolError,
    ToolExecutionError,
    ToolHandler,
    ToolNotFoundError,
    ToolRegistrationError,
)


if TYPE_CHECKING:
    from DefenseAgent.skills.container import SkillContainer


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
        self._mcp_client: MCPClient | None = None
        self._skill_container: "SkillContainer | None" = None

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
        """Build a ToolRegistry from `profile.tools`, resolving skill paths against `base_dir` (defaults to the profile's directory). When `profile.tools.allow_skill_execution` is True, a `SkillContainer` is constructed (local subprocess mode, timeout from `profile.tools.skill_execution_timeout`) and every script bundled in a skill becomes an additional executable Tool. All MCP server entries get folded into a single multi-server client opened in one connect()."""
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
        container = None
        if profile.tools.allow_skill_execution and profile.tools.skills:
            from DefenseAgent.skills import SkillContainer
            container = SkillContainer(timeout=profile.tools.skill_execution_timeout)
            registry._skill_container = container
        for skill_ref in profile.tools.skills:
            skill_path = (base / skill_ref).resolve()
            registry.add_skill(skill_path, container=container)
        if profile.tools.mcp:
            await registry.add_mcp_servers(profile.tools.mcp)
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

    def add_skill(
        self,
        directory: str | Path,
        *,
        container: "SkillContainer | None" = None,
    ) -> list[Tool]:
        """Load every skill rooted at `directory` via a fresh SkillLoader (parity with ms-agent: a `directory/SKILL.md` file is loaded as a single skill, otherwise every immediate subdirectory containing SKILL.md is walked). Each loaded SkillSchema becomes a read-only Tool. When `container` is provided, every script in `schema.scripts` additionally becomes an executable Tool dispatched through the container. Idempotent — already-registered tool names are skipped. Raises ToolRegistrationError when ms-agent's loader finds nothing usable."""
        loader = SkillLoader()
        loader.load_skills(str(directory))
        tools = loader.to_tools(container=container)
        if not tools:
            raise ToolRegistrationError(
                f"no skills loaded from {directory!r} — directory missing SKILL.md "
                f"or every candidate failed to parse"
            )
        registered: list[Tool] = []
        for tool in tools:
            if tool.name in self._tools:
                continue
            self.register(tool)
            registered.append(tool)
        return registered

    async def add_mcp(
        self,
        *,
        name: str | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        url: str | None = None,
        transport: str | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
        sse_read_timeout: float | None = None,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> list[Tool]:
        """Connect a single MCP server (stdio if `command` set; sse/websocket/streamable_http if `url` set) and register every discovered tool. Convenience wrapper over `add_mcp_servers` for one-off launches."""
        cfg = MCPServerConfig(
            name=name,
            command=command,
            args=args or [],
            env=env,
            cwd=cwd,
            url=url,
            transport=transport,
            headers=headers,
            timeout=timeout,
            sse_read_timeout=sse_read_timeout,
            include=include or [],
            exclude=exclude or [],
        )
        return await self.add_mcp_servers([cfg])

    async def add_mcp_servers(self, servers: list[MCPServerConfig]) -> list[Tool]:
        """Translate a list of MCPServerConfig into the `mcpServers` dict shape, attach them to the (single) underlying multi-server client, connect, then register every discovered tool. Safe to call multiple times — each call extends the existing client via `add_mcp_config`."""
        if not servers:
            return []
        mcp_servers: dict[str, dict[str, Any]] = {}
        for cfg in servers:
            entry_name = cfg.name or _default_server_name(cfg, mcp_servers)
            mcp_servers[entry_name] = _server_config_to_mcp_dict(cfg)
        if self._mcp_client is None:
            self._mcp_client = MCPClient(mcp_config={"mcpServers": mcp_servers})
            await self._mcp_client.connect()
        else:
            await self._mcp_client.add_mcp_config({"mcpServers": mcp_servers})
        discovered = await self._mcp_client.discover_tools()
        new_tools: list[Tool] = []
        for tool in discovered:
            if tool.name in self._tools:
                continue
            self.register(tool)
            new_tools.append(tool)
        return new_tools

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

    def specs(self) -> list[dict[str, Any]]:
        """Return every registered tool as a canonical {name, description, input_schema} dict for the LLM."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in self._tools.values()
        ]

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
        """Close the underlying multi-server MCP client (if any)."""
        if self._mcp_client is not None:
            await self._mcp_client.cleanup()
            self._mcp_client = None

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
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        annotation: Any = (
            param.annotation if param.annotation is not inspect.Parameter.empty else str
        )
        properties[param_name] = {"type": _PY_TYPE_TO_JSON.get(annotation, "string")}
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
        return result if isinstance(result, str) else str(result)
    return handler


def _default_server_name(cfg: MCPServerConfig, taken: dict[str, Any]) -> str:
    """Pick a stable, human-readable server name when the config didn't supply one. Stdio servers default to the binary name (e.g. `uvx`); url servers default to the host. Collisions are disambiguated with a numeric suffix."""
    if cfg.command:
        base = Path(cfg.command).name or "stdio_server"
    elif cfg.url:
        base = cfg.url.split("//", 1)[-1].split("/", 1)[0] or "remote_server"
    else:
        base = "mcp_server"
    candidate = base
    suffix = 2
    while candidate in taken:
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


def _server_config_to_mcp_dict(cfg: MCPServerConfig) -> dict[str, Any]:
    """Translate one MCPServerConfig into the per-server dict ms-agent's MCPClient consumes. Drops None / empty-list fields so ms-agent's `connect_to_server(**server)` doesn't see unsupported kwargs."""
    out: dict[str, Any] = {}
    if cfg.command:
        out["command"] = cfg.command
        out["args"] = list(cfg.args)
        if cfg.env is not None:
            out["env"] = dict(cfg.env)
        if cfg.cwd is not None:
            out["cwd"] = cfg.cwd
    if cfg.url:
        out["url"] = cfg.url
        if cfg.headers is not None:
            out["headers"] = dict(cfg.headers)
    if cfg.transport:
        out["transport"] = cfg.transport
    if cfg.timeout is not None:
        out["timeout"] = cfg.timeout
    if cfg.sse_read_timeout is not None:
        out["sse_read_timeout"] = cfg.sse_read_timeout
    if cfg.include:
        out["include"] = list(cfg.include)
    if cfg.exclude:
        out["exclude"] = list(cfg.exclude)
    return out
