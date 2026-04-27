"""Tests for DefenseAgent.tools.mcp.MCPClient — the multi-server adapter that subclasses ms-agent's MCPClient.

We patch the ms-agent class's `connect` and `get_tools` to keep the suite offline; the surface we care about is our `discover_tools()` translator + handler wiring.
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from DefenseAgent.tools import ToolExecutionError
from DefenseAgent.tools.mcp import MCPClient, _normalize_schema


def _ms_tool(*, name: str, server: str, description: str | None, params: dict | None):
    """Build a SimpleNamespace mimicking ms-agent's `Tool` dataclass (tool_name / server_name / description / parameters)."""
    return SimpleNamespace(
        tool_name=name,
        server_name=server,
        description=description,
        parameters=params,
    )


# ---------- discover_tools translates ms-agent records into our Tool dataclass ----------


def test_discover_tools_wraps_records_with_server_metadata() -> None:
    client = MCPClient(mcp_config={"mcpServers": {}})
    fake_get = AsyncMock(return_value={
        "fs": [
            _ms_tool(
                name="read_file", server="fs",
                description="Read a file.",
                params={"type": "object", "properties": {"path": {"type": "string"}}},
            ),
        ],
        "search": [
            _ms_tool(name="query", server="search", description=None, params=None),
        ],
    })

    async def run() -> list:
        with patch.object(MCPClient, "get_tools", fake_get):
            return await client.discover_tools()

    tools = asyncio.run(run())
    assert {t.name for t in tools} == {"read_file", "query"}
    fs_tool = next(t for t in tools if t.name == "read_file")
    assert fs_tool.description == "Read a file."
    assert fs_tool.input_schema["properties"]["path"]["type"] == "string"
    assert fs_tool.metadata == {"server": "fs"}
    assert fs_tool.source == "mcp"

    q = next(t for t in tools if t.name == "query")
    assert q.description == ""
    assert q.input_schema == {"type": "object", "properties": {}}
    assert q.metadata == {"server": "search"}


# ---------- handler delegates to ms-agent's call_tool, packaged for our LLM layer ----------


def test_handler_returns_string_from_call_tool() -> None:
    client = MCPClient(mcp_config={"mcpServers": {}})
    fake_get = AsyncMock(return_value={
        "fs": [_ms_tool(name="read", server="fs", description="x", params={})],
    })
    fake_call = AsyncMock(return_value="file contents")

    async def run() -> str:
        with patch.object(MCPClient, "get_tools", fake_get), \
             patch.object(MCPClient, "call_tool", fake_call):
            tools = await client.discover_tools()
            return await tools[0].handler({"path": "/etc/hosts"})

    out = asyncio.run(run())
    assert out == "file contents"
    fake_call.assert_awaited_once_with("fs", "read", {"path": "/etc/hosts"})


def test_handler_collapses_dict_result_to_text_field() -> None:
    """ms-agent returns `{text, resources}` for tool results that include resource blocks; the handler must keep only the text payload because the LLM expects a string."""
    client = MCPClient(mcp_config={"mcpServers": {}})
    fake_get = AsyncMock(return_value={
        "fs": [_ms_tool(name="read", server="fs", description="x", params={})],
    })
    fake_call = AsyncMock(return_value={"text": "joined text", "resources": [{"id": 1}]})

    async def run() -> str:
        with patch.object(MCPClient, "get_tools", fake_get), \
             patch.object(MCPClient, "call_tool", fake_call):
            tools = await client.discover_tools()
            return await tools[0].handler({})

    assert asyncio.run(run()) == "joined text"


def test_handler_wraps_call_tool_errors_in_tool_execution_error() -> None:
    client = MCPClient(mcp_config={"mcpServers": {}})
    fake_get = AsyncMock(return_value={
        "fs": [_ms_tool(name="read", server="fs", description="x", params={})],
    })
    fake_call = AsyncMock(side_effect=RuntimeError("boom"))

    async def run() -> None:
        with patch.object(MCPClient, "get_tools", fake_get), \
             patch.object(MCPClient, "call_tool", fake_call):
            tools = await client.discover_tools()
            with pytest.raises(ToolExecutionError) as e:
                await tools[0].handler({})
            assert "fs/read" in str(e.value)

    asyncio.run(run())


# ---------- mcp_config plumbing ----------


def test_constructor_accepts_mcp_config() -> None:
    """ms-agent's MCPClient stores `mcpServers` under `self.mcp_config`; verify our subclass forwards correctly."""
    cfg = {
        "mcpServers": {
            "fs": {"command": "uvx", "args": ["mcp-server-filesystem", "/tmp"]},
            "search": {"transport": "sse", "url": "https://mcp.example.com/sse"},
        }
    }
    client = MCPClient(mcp_config=cfg)
    assert client.mcp_config["mcpServers"]["fs"]["command"] == "uvx"
    assert client.mcp_config["mcpServers"]["search"]["url"] == "https://mcp.example.com/sse"


def test_constructor_with_none_yields_empty_servers_dict() -> None:
    client = MCPClient(mcp_config=None)
    assert client.mcp_config == {"mcpServers": {}}


# ---------- helpers ----------


def test_normalize_schema_passes_through_dicts() -> None:
    schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
    assert _normalize_schema(schema) is schema


def test_normalize_schema_replaces_none_with_empty_object() -> None:
    assert _normalize_schema(None) == {"type": "object", "properties": {}}
