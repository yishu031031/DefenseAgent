"""Tests for ToolRegistry.from_profile — skill + MCP registration driven by an AgentProfile.

Covers:
  • Skills declared in profile.tools.skills are resolved relative to the
    profile's directory and registered.
  • Empty tools section yields an empty registry.
  • Missing source_path + no base_dir raises ToolRegistrationError.
  • Explicit base_dir override works for in-memory profiles.
  • MCP entries from the profile are forwarded to add_mcp (verified via a
    patched stdio_client + ClientSession that records arguments).
  • A profile that points at the shipped maya agent bundle loads its real
    skill end-to-end.
"""
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from DefenseAgent.config import AgentProfile
from DefenseAgent.tools import ToolRegistry, ToolRegistrationError


from DefenseAgent.examples import EXAMPLE_PROFILE_PATH as _EXAMPLE_PROFILE


def _write_skill_dir(path: Path, *, name: str, description: str) -> Path:
    """Create a minimal SKILL.md-only skill directory at `path` and return the path."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\nbody\n",
        encoding="utf-8",
    )
    return path


def _write_profile(path: Path, body: str) -> Path:
    """Write a profile YAML at `path` and return it."""
    path.write_text(body, encoding="utf-8")
    return path


# ---------- skills ----------


def test_from_profile_registers_skills_relative_to_profile_dir(tmp_path: Path) -> None:
    agent_dir = tmp_path / "agents" / "ada"
    _write_skill_dir(agent_dir / "skills" / "tabular", name="tabular", description="t.")
    _write_profile(
        agent_dir / "profile.yaml",
        """\
agent:
  id: ada
  name: Ada
  age: 30
  traits: x
  backstory: y
  initial_plan: z
  tools:
    skills:
      - skills/tabular
""",
    )
    profile = AgentProfile.from_yaml(agent_dir / "profile.yaml")

    async def run() -> list[str]:
        async with await ToolRegistry.from_profile(profile) as registry:
            return registry.names()

    names = asyncio.run(run())
    assert names == ["tabular"]


def test_from_profile_empty_tools_section_yields_empty_registry(
    tmp_path: Path,
) -> None:
    agent_dir = tmp_path / "agents" / "empty"
    agent_dir.mkdir(parents=True)
    _write_profile(
        agent_dir / "profile.yaml",
        """\
agent:
  id: empty
  name: Empty
  age: 1
  traits: x
  backstory: y
  initial_plan: z
""",
    )
    profile = AgentProfile.from_yaml(agent_dir / "profile.yaml")

    async def run() -> int:
        async with await ToolRegistry.from_profile(profile) as registry:
            return len(registry)

    assert asyncio.run(run()) == 0


def test_from_profile_resolves_parent_relative_path(tmp_path: Path) -> None:
    # Simulates agents/ada/profile.yaml pointing at shared/skills/common
    _write_skill_dir(
        tmp_path / "shared" / "skills" / "common",
        name="common", description="shared skill.",
    )
    agent_dir = tmp_path / "agents" / "ada"
    agent_dir.mkdir(parents=True)
    _write_profile(
        agent_dir / "profile.yaml",
        """\
agent:
  id: ada
  name: Ada
  age: 30
  traits: x
  backstory: y
  initial_plan: z
  tools:
    skills:
      - ../../shared/skills/common
""",
    )
    profile = AgentProfile.from_yaml(agent_dir / "profile.yaml")

    async def run() -> list[str]:
        async with await ToolRegistry.from_profile(profile) as registry:
            return registry.names()

    assert asyncio.run(run()) == ["common"]


def test_from_profile_requires_source_dir_or_explicit_base(tmp_path: Path) -> None:
    profile = AgentProfile(
        id="x", name="X", age=1, traits="t", backstory="b", initial_plan="p",
    )

    async def run() -> None:
        await ToolRegistry.from_profile(profile)

    with pytest.raises(ToolRegistrationError):
        asyncio.run(run())


def test_from_profile_accepts_explicit_base_dir(tmp_path: Path) -> None:
    _write_skill_dir(
        tmp_path / "skills" / "inline",
        name="inline", description="d.",
    )
    profile = AgentProfile.model_validate(
        {
            "id": "x", "name": "X", "age": 1, "traits": "t",
            "backstory": "b", "initial_plan": "p",
            "tools": {"skills": ["skills/inline"]},
        }
    )

    async def run() -> list[str]:
        async with await ToolRegistry.from_profile(profile, base_dir=tmp_path) as r:
            return r.names()

    assert asyncio.run(run()) == ["inline"]


# ---------- MCP plumbing ----------


def test_from_profile_forwards_mcp_entries_to_add_mcp(tmp_path: Path) -> None:
    """MCP entries in the profile must reach the underlying multi-server MCPClient with the right launch params."""
    agent_dir = tmp_path / "agents" / "m"
    agent_dir.mkdir(parents=True)
    _write_profile(
        agent_dir / "profile.yaml",
        """\
agent:
  id: m
  name: M
  age: 1
  traits: t
  backstory: b
  initial_plan: p
  tools:
    mcp:
      - command: uvx
        args: [mcp-server-filesystem, /tmp]
        env:
          TOKEN: abc
""",
    )
    profile = AgentProfile.from_yaml(agent_dir / "profile.yaml")

    from unittest.mock import AsyncMock

    from DefenseAgent.tools.mcp import MCPClient

    fake_connect = AsyncMock(return_value=None)
    fake_get = AsyncMock(return_value={})
    fake_cleanup = AsyncMock(return_value=None)

    captured_configs: list[dict] = []
    real_init = MCPClient.__init__

    def spy_init(self, mcp_config=None):
        real_init(self, mcp_config=mcp_config)
        captured_configs.append(self.mcp_config)

    async def run() -> None:
        with (
            patch.object(MCPClient, "__init__", spy_init),
            patch.object(MCPClient, "connect", fake_connect),
            patch.object(MCPClient, "get_tools", fake_get),
            patch.object(MCPClient, "cleanup", fake_cleanup),
        ):
            async with await ToolRegistry.from_profile(profile):
                pass

    asyncio.run(run())
    fake_connect.assert_awaited_once()
    fake_cleanup.assert_awaited_once()
    assert len(captured_configs) == 1
    servers = captured_configs[0]["mcpServers"]
    assert len(servers) == 1
    (only_entry,) = servers.values()
    assert only_entry["command"] == "uvx"
    assert only_entry["args"] == ["mcp-server-filesystem", "/tmp"]
    assert only_entry["env"] == {"TOKEN": "abc"}


# ---------- skill execution opt-in ----------


def _write_profile(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def test_from_profile_with_allow_skill_execution_registers_executable_tools(tmp_path: Path) -> None:
    """When `tools.allow_skill_execution: true`, each script in a loaded skill becomes an additional executable Tool, named `'{skill_name}__{script_stem}'`."""
    agent_dir = tmp_path / "agents" / "calcbot"
    skills_dir = agent_dir / "skills" / "calc"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text(
        "---\nname: calc\ndescription: a calc skill\n---\n\nbody\n",
        encoding="utf-8",
    )
    (skills_dir / "scripts").mkdir()
    (skills_dir / "scripts" / "add.py").write_text("print('go')\n", encoding="utf-8")

    _write_profile(
        agent_dir / "profile.yaml",
        """\
agent:
  id: calcbot
  name: CalcBot
  age: 1
  traits: t
  backstory: b
  initial_plan: p
  tools:
    skills:
      - skills/calc
    allow_skill_execution: true
""",
    )
    profile = AgentProfile.from_yaml(agent_dir / "profile.yaml")

    async def run() -> list[str]:
        async with await ToolRegistry.from_profile(profile) as registry:
            return registry.names()

    names = asyncio.run(run())
    assert sorted(names) == ["calc", "calc__add"]


def test_from_profile_default_skips_skill_execution(tmp_path: Path) -> None:
    """Without `allow_skill_execution`, scripts inside a skill are NOT auto-promoted to tools — only the read-only skill tool is registered."""
    agent_dir = tmp_path / "agents" / "calcbot"
    skills_dir = agent_dir / "skills" / "calc"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text(
        "---\nname: calc\ndescription: a calc skill\n---\n\nbody\n",
        encoding="utf-8",
    )
    (skills_dir / "scripts").mkdir()
    (skills_dir / "scripts" / "add.py").write_text("print('go')\n", encoding="utf-8")

    _write_profile(
        agent_dir / "profile.yaml",
        """\
agent:
  id: calcbot
  name: CalcBot
  age: 1
  traits: t
  backstory: b
  initial_plan: p
  tools:
    skills:
      - skills/calc
""",
    )
    profile = AgentProfile.from_yaml(agent_dir / "profile.yaml")

    async def run() -> list[str]:
        async with await ToolRegistry.from_profile(profile) as registry:
            return registry.names()

    assert asyncio.run(run()) == ["calc"]


# ---------- end-to-end against the shipped Maya profile ----------


def test_from_profile_loads_real_maya_bundle() -> None:
    profile = AgentProfile.from_yaml(_EXAMPLE_PROFILE)

    async def run() -> list[str]:
        async with await ToolRegistry.from_profile(profile) as registry:
            return registry.names()

    names = asyncio.run(run())
    assert names == ["tabular-report"]
