from typing import Any

from ms_agent.tools.mcp_client import MCPClient as MsMCPClient

from DefenseAgent.tools.types import Tool, ToolExecutionError, ToolHandler


class MCPClient(MsMCPClient):
    """Multi-server MCP client; inherits ms-agent's transport machinery (stdio + sse + websocket + streamable-http) and exposes the discovered tools as DefenseAgent.tools.Tool records.

    Construct with `mcp_config={"mcpServers": {<name>: <server_cfg>, ...}}` and call `connect()` once. Each `<server_cfg>` is a dict with at minimum either `command` (stdio) or `url` (HTTP-style transports). Optional fields per server: `transport`, `args`, `env`, `cwd`, `headers`, `timeout`, `sse_read_timeout`, `include`, `exclude`. Empty `env` values are filled from the process environment by ms-agent's `connect()`.
    """

    def __init__(self, mcp_config: dict[str, Any] | None = None) -> None:
        """Build with an mcpServers config; the underlying ms-agent class accepts `config=None` for standalone use."""
        super().__init__(mcp_config=mcp_config, config=None)

    async def discover_tools(self) -> list[Tool]:
        """Return one DefenseAgent.Tool per (server, tool); should be called after `connect()`. Each handler delegates to ms-agent's `call_tool(server, name, args)` so transport details stay encapsulated."""
        servers = await self.get_tools()
        out: list[Tool] = []
        for server_name, server_tools in servers.items():
            for t in server_tools:
                out.append(
                    Tool(
                        name=t.tool_name,
                        description=t.description or "",
                        input_schema=_normalize_schema(t.parameters),
                        source="mcp",
                        handler=self._make_handler(server_name, t.tool_name),
                        metadata={"server": server_name},
                    )
                )
        return out

    def _make_handler(self, server_name: str, tool_name: str) -> ToolHandler:
        """Return an async handler that forwards arguments to this server's `call_tool` and returns a string. ms-agent's `call_tool` may return `{text, resources}` for tool calls that produce binary/resource content; we collapse that to the text part."""
        async def handler(arguments: dict[str, Any]) -> str:
            try:
                result = await self.call_tool(server_name, tool_name, arguments)
            except Exception as e:
                raise ToolExecutionError(
                    f"MCP call_tool({server_name}/{tool_name}) failed: {e}"
                ) from e
            if isinstance(result, dict):
                return str(result.get("text", ""))
            return result
        return handler


def _normalize_schema(schema: Any) -> dict[str, Any]:
    """Coerce an MCP tool's parameters/inputSchema (dict-or-None) into a valid JSON-schema object."""
    if isinstance(schema, dict):
        return schema
    return {"type": "object", "properties": {}}
