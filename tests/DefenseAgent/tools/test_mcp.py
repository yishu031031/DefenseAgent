"""Tests for DefenseAgent.tools.mcp.

The real MCP SDK needs a subprocess + stdio streams, so we patch
`stdio_client` and `ClientSession` at the module boundary and drive the
adapter with a fake session that records calls.
"""
import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from DefenseAgent.tools import ToolExecutionError
from DefenseAgent.tools.mcp import MCPClient, _render_mcp_content


class _FakeListToolsResult:
    """Mimics mcp's ListToolsResult: object with a `.tools` list."""

    def __init__(self, tools: list) -> None:
        """Hold the list of tool records returned by list_tools()."""
        self.tools = tools


class _FakeCallToolResult:
    """Mimics mcp's CallToolResult: object with `.content` and `.isError`."""

    def __init__(self, content: list, *, is_error: bool = False) -> None:
        """Hold the text-content blocks and the error flag returned by call_tool()."""
        self.content = content
        self.isError = is_error


class _FakeSession:
    """Records initialize() / list_tools() / call_tool() calls for assertions."""

    def __init__(
        self,
        *,
        tools: list,
        call_results: dict[str, _FakeCallToolResult] | None = None,
    ) -> None:
        """Record the list_tools() payload and an optional per-tool-name response map for call_tool()."""
        self._tools = tools
        self._call_results = call_results or {}
        self.initialized = False
        self.call_log: list[tuple[str, dict]] = []

    async def initialize(self) -> None:
        """Mark the session as initialized."""
        self.initialized = True

    async def list_tools(self) -> _FakeListToolsResult:
        """Return the canned tool list."""
        return _FakeListToolsResult(self._tools)

    async def call_tool(self, name: str, arguments: dict) -> _FakeCallToolResult:
        """Record the call and return the canned response (raises if name is unmapped)."""
        self.call_log.append((name, arguments))
        if name not in self._call_results:
            raise RuntimeError(f"no canned result for {name!r}")
        return self._call_results[name]


def _patch_stdio_and_session(session: _FakeSession):
    """Return a context manager that patches stdio_client + ClientSession with fakes routing to `session`."""

    @asynccontextmanager
    async def fake_stdio_client(params):
        yield (object(), object())  # (read, write) placeholders

    @asynccontextmanager
    async def fake_client_session(read, write):
        yield session

    p1 = patch("DefenseAgent.tools.mcp.stdio_client", fake_stdio_client)
    p2 = patch("DefenseAgent.tools.mcp.ClientSession", fake_client_session)
    return p1, p2


def _make_tool_record(*, name: str, description: str | None, input_schema):
    """Build a SimpleNamespace mimicking mcp.types.Tool (name / description / inputSchema)."""
    return SimpleNamespace(
        name=name,
        description=description,
        inputSchema=input_schema,
    )


# ---------- enter() discovers + wraps tools ----------


def test_enter_discovers_tools_and_wraps_as_tool_records() -> None:
    tool_record = _make_tool_record(
        name="echo",
        description="Echo the input.",
        input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
    )
    session = _FakeSession(
        tools=[tool_record],
        call_results={"echo": _FakeCallToolResult([SimpleNamespace(text="hi")])},
    )
    p1, p2 = _patch_stdio_and_session(session)

    async def run() -> list:
        with p1, p2:
            client = MCPClient(command="fake", args=[])
            tools = await client.enter()
            await client.close()
            return tools

    tools = asyncio.run(run())
    assert session.initialized is True
    assert len(tools) == 1
    t = tools[0]
    assert t.name == "echo"
    assert t.description == "Echo the input."
    assert t.source == "mcp"
    assert t.input_schema["properties"]["text"] == {"type": "string"}
    assert t.metadata["mcp_command"] == "fake"


def test_enter_uses_empty_object_schema_when_input_schema_is_none() -> None:
    tool_record = _make_tool_record(name="ping", description=None, input_schema=None)
    session = _FakeSession(
        tools=[tool_record],
        call_results={"ping": _FakeCallToolResult([SimpleNamespace(text="pong")])},
    )
    p1, p2 = _patch_stdio_and_session(session)

    async def run() -> list:
        with p1, p2:
            client = MCPClient(command="fake")
            tools = await client.enter()
            await client.close()
            return tools

    tools = asyncio.run(run())
    assert tools[0].description == ""
    assert tools[0].input_schema == {"type": "object", "properties": {}}


# ---------- handler forwards to the MCP session ----------


def test_handler_forwards_arguments_and_returns_joined_text() -> None:
    tool_record = _make_tool_record(
        name="echo",
        description="echo",
        input_schema={"type": "object", "properties": {}},
    )
    session = _FakeSession(
        tools=[tool_record],
        call_results={
            "echo": _FakeCallToolResult(
                [SimpleNamespace(text="line 1"), SimpleNamespace(text="line 2")]
            )
        },
    )
    p1, p2 = _patch_stdio_and_session(session)

    async def run() -> str:
        with p1, p2:
            client = MCPClient(command="fake")
            tools = await client.enter()
            result = await tools[0].handler({"text": "greet"})
            await client.close()
            return result

    output = asyncio.run(run())
    assert output == "line 1\nline 2"
    assert session.call_log == [("echo", {"text": "greet"})]


def test_handler_raises_when_mcp_reports_is_error() -> None:
    tool_record = _make_tool_record(
        name="fail",
        description="",
        input_schema={"type": "object", "properties": {}},
    )
    session = _FakeSession(
        tools=[tool_record],
        call_results={
            "fail": _FakeCallToolResult(
                [SimpleNamespace(text="boom")], is_error=True
            )
        },
    )
    p1, p2 = _patch_stdio_and_session(session)

    async def run() -> None:
        with p1, p2:
            client = MCPClient(command="fake")
            tools = await client.enter()
            with pytest.raises(ToolExecutionError):
                await tools[0].handler({})
            await client.close()

    asyncio.run(run())


def test_handler_raises_after_close() -> None:
    tool_record = _make_tool_record(
        name="echo",
        description="",
        input_schema={"type": "object", "properties": {}},
    )
    session = _FakeSession(
        tools=[tool_record],
        call_results={"echo": _FakeCallToolResult([SimpleNamespace(text="x")])},
    )
    p1, p2 = _patch_stdio_and_session(session)

    async def run() -> None:
        with p1, p2:
            client = MCPClient(command="fake")
            tools = await client.enter()
            await client.close()
            with pytest.raises(ToolExecutionError):
                await tools[0].handler({})

    asyncio.run(run())


# ---------- helpers ----------


def test_render_mcp_content_joins_text_blocks() -> None:
    result = SimpleNamespace(
        content=[
            SimpleNamespace(text="alpha"),
            SimpleNamespace(text="beta"),
            SimpleNamespace(other=1),  # non-text block is skipped
        ]
    )
    assert _render_mcp_content(result) == "alpha\nbeta"


def test_render_mcp_content_handles_missing_content_field() -> None:
    assert _render_mcp_content(SimpleNamespace()) == ""
