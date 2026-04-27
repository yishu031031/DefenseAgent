"""Tests for DefenseAgent.tools.tools (ToolRegistry facade).

Groups:
  • @registry.tool decorator — with and without args, sig-derived schema.
  • add_skill — integrates Skill as a Tool, preserves progressive disclosure.
  • register — direct registration, collision detection.
  • spec() — emits canonical {name, description, input_schema} dicts.
  • execute() — dispatches ToolCalls, converts errors to role='tool' Messages.
"""
import asyncio
from pathlib import Path

import pytest

from DefenseAgent.llm.types import ToolCall
from DefenseAgent.tools import (
    Tool,
    ToolNotFoundError,
    ToolRegistrationError,
    ToolRegistry,
)


# ---------- decorator ----------


def test_tool_decorator_no_args_registers_function() -> None:
    registry = ToolRegistry()

    @registry.tool
    def add(a: int, b: int) -> int:
        """Sum two integers."""
        return a + b

    assert "add" in registry
    t = registry.get("add")
    assert t.source == "python"
    assert t.description == "Sum two integers."
    assert t.input_schema["type"] == "object"
    assert t.input_schema["properties"] == {
        "a": {"type": "integer"},
        "b": {"type": "integer"},
    }
    assert t.input_schema["required"] == ["a", "b"]


def test_tool_decorator_with_name_and_description() -> None:
    registry = ToolRegistry()

    @registry.tool(name="plus", description="Adds.")
    def f(a: int, b: int) -> int:
        return a + b

    assert "plus" in registry
    assert "f" not in registry
    assert registry.get("plus").description == "Adds."


def test_tool_decorator_handles_defaults_and_strings() -> None:
    registry = ToolRegistry()

    @registry.tool
    def greet(name: str, greeting: str = "Hello") -> str:
        """Greets a user."""
        return f"{greeting}, {name}!"

    schema = registry.get("greet").input_schema
    assert schema["properties"]["name"] == {"type": "string"}
    assert schema["properties"]["greeting"] == {"type": "string"}
    assert schema["required"] == ["name"]


def test_tool_decorator_async_function_is_awaited() -> None:
    registry = ToolRegistry()

    @registry.tool
    async def upper(text: str) -> str:
        """Async uppercase."""
        return text.upper()

    messages = asyncio.run(
        registry.execute([ToolCall(id="1", name="upper", arguments={"text": "hi"})])
    )
    assert len(messages) == 1
    assert messages[0].content == "HI"


def test_tool_decorator_registers_return_value_is_original_function() -> None:
    registry = ToolRegistry()

    @registry.tool
    def f(x: int) -> int:
        return x * 2

    # Decorator preserves the original callable:
    assert f(3) == 6


# ---------- register() + collisions ----------


def test_register_direct_tool_succeeds() -> None:
    registry = ToolRegistry()

    async def handler(args: dict) -> str:
        return "ok"

    tool = Tool(
        name="raw",
        description="",
        input_schema={"type": "object", "properties": {}},
        source="python",
        handler=handler,
    )
    registry.register(tool)
    assert "raw" in registry


def test_register_name_collision_raises() -> None:
    registry = ToolRegistry()

    @registry.tool
    def foo() -> str:
        """foo"""
        return "x"

    async def handler(args: dict) -> str:
        return "y"

    dup = Tool(
        name="foo", description="", input_schema={"type": "object", "properties": {}},
        source="python", handler=handler,
    )
    with pytest.raises(ToolRegistrationError):
        registry.register(dup)


def test_decorator_name_collision_raises() -> None:
    registry = ToolRegistry()

    @registry.tool
    def foo() -> str:
        """foo"""
        return "x"

    with pytest.raises(ToolRegistrationError):
        @registry.tool
        def foo() -> str:  # noqa: F811  — intentional duplicate
            """foo 2"""
            return "y"


def test_get_missing_tool_raises() -> None:
    registry = ToolRegistry()
    with pytest.raises(ToolNotFoundError):
        registry.get("no-such-tool")


def test_len_and_names_and_contains() -> None:
    registry = ToolRegistry()

    @registry.tool
    def a() -> str:
        """a"""
        return "a"

    @registry.tool
    def b() -> str:
        """b"""
        return "b"

    assert len(registry) == 2
    assert registry.names() == ["a", "b"]
    assert "a" in registry
    assert "b" in registry
    assert "c" not in registry


# ---------- spec() ----------


def test_spec_emits_name_description_and_input_schema() -> None:
    registry = ToolRegistry()

    @registry.tool
    def foo(x: int) -> int:
        """Doubles x."""
        return x * 2

    specs = registry.specs()
    assert len(specs) == 1
    entry = specs[0]
    assert set(entry.keys()) == {"name", "description", "input_schema"}
    assert entry["name"] == "foo"
    assert entry["description"] == "Doubles x."
    assert entry["input_schema"]["properties"]["x"] == {"type": "integer"}


def test_specs_order_matches_insertion() -> None:
    registry = ToolRegistry()

    @registry.tool
    def first() -> str:
        """1"""
        return "1"

    @registry.tool
    def second() -> str:
        """2"""
        return "2"

    specs = registry.specs()
    assert [e["name"] for e in specs] == ["first", "second"]


def test_add_skill_loads_directory_tree_of_skills(tmp_path: Path) -> None:
    """When `directory` itself has no SKILL.md, add_skill walks immediate subdirs and registers each (parity with ms-agent SkillLoader)."""
    root = tmp_path / "skills"
    for skill_name in ("alpha", "beta", "gamma"):
        d = root / skill_name
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(
            f"---\nname: {skill_name}\ndescription: skill {skill_name}\n---\n\nbody\n",
            encoding="utf-8",
        )
    (root / "not_a_skill").mkdir()
    (root / "not_a_skill" / "README.md").write_text("ignored", encoding="utf-8")

    registry = ToolRegistry()
    registered = registry.add_skill(root)
    assert sorted(t.name for t in registered) == ["alpha", "beta", "gamma"]
    assert sorted(registry.names()) == ["alpha", "beta", "gamma"]


def test_specs_for_skill_exposes_only_frontmatter(tmp_path: Path) -> None:
    skill_dir = tmp_path / "s"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: report\ndescription: Make reports.\n---\n\nBIG LAYER 2 BODY\n",
        encoding="utf-8",
    )
    registry = ToolRegistry()
    registry.add_skill(skill_dir)
    specs = registry.specs()
    assert len(specs) == 1
    assert specs[0]["name"] == "report"
    assert specs[0]["description"] == "Make reports."
    # Layer-2 body must NOT be in the initial specs:
    assert "BIG LAYER 2 BODY" not in specs[0]["description"]
    # The skill tool must expose the optional `file` arg in its schema:
    assert "file" in specs[0]["input_schema"]["properties"]


# ---------- execute() ----------


def test_execute_dispatches_python_tool() -> None:
    registry = ToolRegistry()

    @registry.tool
    def add(a: int, b: int) -> int:
        """sum"""
        return a + b

    calls = [ToolCall(id="c1", name="add", arguments={"a": 2, "b": 3})]
    messages = asyncio.run(registry.execute(calls))
    assert len(messages) == 1
    msg = messages[0]
    assert msg.role == "tool"
    assert msg.tool_call_id == "c1"
    assert msg.name == "add"
    assert msg.content == "5"


def test_execute_returns_error_message_for_unknown_tool() -> None:
    registry = ToolRegistry()
    calls = [ToolCall(id="c1", name="nope", arguments={})]
    messages = asyncio.run(registry.execute(calls))
    assert len(messages) == 1
    assert "ToolNotFoundError" in messages[0].content
    assert messages[0].tool_call_id == "c1"


def test_execute_catches_handler_exception() -> None:
    registry = ToolRegistry()

    @registry.tool
    def blow_up() -> str:
        """raises"""
        raise ValueError("boom")

    calls = [ToolCall(id="c1", name="blow_up", arguments={})]
    messages = asyncio.run(registry.execute(calls))
    assert messages[0].role == "tool"
    assert "ValueError" in messages[0].content
    assert "boom" in messages[0].content


def test_execute_empty_list_returns_empty() -> None:
    registry = ToolRegistry()
    assert asyncio.run(registry.execute([])) == []


def test_execute_dispatches_skill_layers(tmp_path: Path) -> None:
    skill_dir = tmp_path / "s"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: rpt\ndescription: r.\n---\n\nLAYER 2 CONTENT\n",
        encoding="utf-8",
    )
    (skill_dir / "scripts").mkdir()
    (skill_dir / "scripts" / "go.py").write_text("print('go')\n", encoding="utf-8")

    registry = ToolRegistry()
    registry.add_skill(skill_dir)

    # Layer-2: no file arg → body
    layer2 = asyncio.run(
        registry.execute([ToolCall(id="1", name="rpt", arguments={})])
    )[0]
    assert "LAYER 2 CONTENT" in layer2.content

    # Layer-3: file arg → file contents
    layer3 = asyncio.run(
        registry.execute(
            [ToolCall(id="2", name="rpt", arguments={"file": "scripts/go.py"})]
        )
    )[0]
    assert layer3.content == "print('go')\n"


def test_execute_concurrency_runs_in_parallel() -> None:
    registry = ToolRegistry()

    events: list[str] = []

    @registry.tool
    async def slow_a() -> str:
        """a"""
        await asyncio.sleep(0.05)
        events.append("a")
        return "a"

    @registry.tool
    async def slow_b() -> str:
        """b"""
        await asyncio.sleep(0.05)
        events.append("b")
        return "b"

    calls = [
        ToolCall(id="1", name="slow_a", arguments={}),
        ToolCall(id="2", name="slow_b", arguments={}),
    ]
    import time

    t0 = time.monotonic()
    results = asyncio.run(registry.execute(calls))
    elapsed = time.monotonic() - t0
    assert [m.content for m in results] == ["a", "b"]
    # Concurrent → should take ~0.05s, not 0.10s.
    assert elapsed < 0.09


# ---------- async context manager ----------


def test_registry_usable_as_async_context_manager() -> None:
    async def main() -> int:
        async with ToolRegistry() as registry:
            @registry.tool
            def f() -> str:
                """f"""
                return "f"
            return len(registry)

    assert asyncio.run(main()) == 1


# ---------- multi-server MCP wiring ----------


def test_add_mcp_servers_groups_into_one_client_and_registers_filtered_tools() -> None:
    """add_mcp_servers folds N profile entries into one mcpServers dict, calls connect once, and forwards include/exclude per server."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, patch

    from DefenseAgent.config.profile import MCPServerConfig
    from DefenseAgent.tools.mcp import MCPClient

    captured: dict[str, dict] = {}
    real_init = MCPClient.__init__

    def spy_init(self, mcp_config=None):
        real_init(self, mcp_config=mcp_config)
        captured["mcp_config"] = self.mcp_config

    fake_connect = AsyncMock(return_value=None)
    fake_get = AsyncMock(return_value={
        "fs": [SimpleNamespace(tool_name="read", server_name="fs", description="r", parameters={})],
        "search": [SimpleNamespace(tool_name="query", server_name="search", description="q", parameters={})],
    })
    fake_cleanup = AsyncMock(return_value=None)

    servers = [
        MCPServerConfig(
            name="fs",
            command="uvx",
            args=["mcp-server-filesystem", "/tmp"],
            env={"TOKEN": ""},  # empty value → ms-agent's connect interpolates from env
            include=["read"],
        ),
        MCPServerConfig(
            name="search",
            url="https://mcp.example.com/sse",
            transport="sse",
            headers={"Authorization": "Bearer x"},
        ),
    ]

    async def main() -> list[str]:
        registry = ToolRegistry()
        with (
            patch.object(MCPClient, "__init__", spy_init),
            patch.object(MCPClient, "connect", fake_connect),
            patch.object(MCPClient, "get_tools", fake_get),
            patch.object(MCPClient, "cleanup", fake_cleanup),
        ):
            await registry.add_mcp_servers(servers)
            names = registry.names()
            await registry.close()
            return names

    names = asyncio.run(main())
    assert sorted(names) == ["query", "read"]
    fake_connect.assert_awaited_once()
    fake_cleanup.assert_awaited_once()

    cfg = captured["mcp_config"]["mcpServers"]
    assert set(cfg.keys()) == {"fs", "search"}
    assert cfg["fs"]["command"] == "uvx"
    assert cfg["fs"]["env"] == {"TOKEN": ""}
    assert cfg["fs"]["include"] == ["read"]
    assert cfg["search"]["url"] == "https://mcp.example.com/sse"
    assert cfg["search"]["transport"] == "sse"
    assert cfg["search"]["headers"] == {"Authorization": "Bearer x"}
