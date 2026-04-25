from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from DefenseAgent.tools.types import (
    Tool,
    ToolExecutionError,
    ToolHandler,
)


class MCPClient:
    """One MCP stdio-server connection; opens on enter(), closes on close(). Discovered tools are returned from enter()."""

    def __init__(
        self,
        *,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> None:
        """Record the stdio launch parameters; the session opens lazily when enter() is called."""
        self.command = command
        if args is None:
            self.args: list[str] = []
        else:
            self.args = list(args)
        self.env = env
        self.cwd = cwd
        self._stack = AsyncExitStack()
        self._session: ClientSession | None = None

    async def enter(self) -> list[Tool]:
        """Open the stdio connection + ClientSession, initialize, list tools, and return them wrapped as Tools."""
        params = StdioServerParameters(
            command=self.command,
            args=self.args,
            env=self.env,
            cwd=self.cwd,
        )
        transport = await self._stack.enter_async_context(stdio_client(params))
        read, write = transport[0], transport[1]
        session = await self._stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        self._session = session
        listed = await session.list_tools()
        tools: list[Tool] = []
        for t in listed.tools:
            tools.append(
                Tool(
                    name=t.name,
                    description=t.description or "",
                    input_schema=_normalize_schema(t.inputSchema),
                    source="mcp",
                    handler=self._make_handler(t.name),
                    metadata={"mcp_command": self.command},
                )
            )
        return tools

    async def close(self) -> None:
        """Close the ClientSession and the underlying stdio connection."""
        await self._stack.aclose()
        self._session = None

    def _make_handler(self, tool_name: str) -> ToolHandler:
        """Return an async handler that forwards arguments to this MCP session's call_tool."""
        async def handler(arguments: dict[str, Any]) -> str:
            if self._session is None:
                raise ToolExecutionError(
                    f"MCP session is not open; cannot call tool {tool_name!r}"
                )
            try:
                result = await self._session.call_tool(tool_name, arguments)
            except Exception as e:
                raise ToolExecutionError(
                    f"MCP call_tool({tool_name!r}) failed: {e}"
                ) from e
            if getattr(result, "isError", False):
                raise ToolExecutionError(
                    f"MCP tool {tool_name!r} reported error: {_render_mcp_content(result)}"
                )
            return _render_mcp_content(result)
        return handler


def _normalize_schema(schema: Any) -> dict[str, Any]:
    """Coerce an MCP Tool.inputSchema (dict-or-None) into a valid JSON-schema object."""
    if isinstance(schema, dict):
        return schema
    return {"type": "object", "properties": {}}


def _render_mcp_content(result: Any) -> str:
    """Flatten an MCP CallToolResult's content blocks into a single string for the LLM."""
    parts: list[str] = []
    content = getattr(result, "content", []) or []
    for block in content:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "\n".join(parts)
