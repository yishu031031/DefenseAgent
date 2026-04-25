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


_MAYA_PROFILE = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "agents" / "maya_rodriguez" / "profile.yaml"
)


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
    """MCP entries in the profile must reach MCPClient with the right launch params."""
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

    launched: list[dict] = []

    class _FakeSession:
        def __init__(self) -> None:
            self.initialized = False

        async def initialize(self) -> None:
            self.initialized = True

        async def list_tools(self):
            return SimpleNamespace(tools=[])

    fake_session = _FakeSession()

    @asynccontextmanager
    async def fake_stdio_client(params):
        launched.append(
            {
                "command": params.command,
                "args": list(params.args),
                "env": dict(params.env) if params.env else None,
            }
        )
        yield (object(), object())

    @asynccontextmanager
    async def fake_client_session(read, write):
        yield fake_session

    async def run() -> None:
        with (
            patch("DefenseAgent.tools.mcp.stdio_client", fake_stdio_client),
            patch("DefenseAgent.tools.mcp.ClientSession", fake_client_session),
        ):
            async with await ToolRegistry.from_profile(profile):
                pass

    asyncio.run(run())
    assert fake_session.initialized is True
    assert launched == [
        {
            "command": "uvx",
            "args": ["mcp-server-filesystem", "/tmp"],
            "env": {"TOKEN": "abc"},
        }
    ]


# ---------- end-to-end against the shipped Maya profile ----------


def test_from_profile_loads_real_maya_bundle() -> None:
    profile = AgentProfile.from_yaml(_MAYA_PROFILE)

    async def run() -> list[str]:
        async with await ToolRegistry.from_profile(profile) as registry:
            return registry.names()

    names = asyncio.run(run())
    assert names == ["tabular-report"]
