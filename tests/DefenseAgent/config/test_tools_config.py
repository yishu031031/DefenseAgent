"""Tests for the new per-agent tools section of AgentProfile + source_path tracking.

Covers:
  • AgentProfile.tools defaults to an empty ToolsConfig when omitted.
  • Skill paths and MCP server configs parse into their nested models.
  • source_path / source_dir expose the resolved YAML location after from_yaml.
  • In-memory profiles (not loaded from disk) have source_path == None.
  • MCPServerConfig rejects an empty command (extra="forbid" catches extras).
"""
from pathlib import Path

import pytest

from DefenseAgent.config import (
    AgentProfile,
    ConfigValidationError,
    MCPServerConfig,
    ToolsConfig,
)


_MINIMAL_YAML = """\
agent:
  id: "x1"
  name: "X"
  age: 1
  traits: "t"
  backstory: "b"
  initial_plan: "p"
"""

_WITH_TOOLS_YAML = """\
agent:
  id: "maya"
  name: "Maya"
  age: 20
  traits: "curious"
  backstory: "student"
  initial_plan: "study"
  tools:
    skills:
      - skills/tabular-report
      - ../shared/skills/common
    mcp:
      - command: uvx
        args: [mcp-server-filesystem, /tmp]
      - command: python
        args: [-m, my.module]
        env:
          API_KEY: secret
        cwd: /srv/mcp
"""


# ---------- defaults ----------


def test_tools_section_defaults_to_empty(tmp_path: Path) -> None:
    path = tmp_path / "p.yaml"
    path.write_text(_MINIMAL_YAML, encoding="utf-8")
    profile = AgentProfile.from_yaml(path)
    assert isinstance(profile.tools, ToolsConfig)
    assert profile.tools.skills == []
    assert profile.tools.mcp == []


# ---------- parsing ----------


def test_tools_section_parses_skills_and_mcp(tmp_path: Path) -> None:
    path = tmp_path / "p.yaml"
    path.write_text(_WITH_TOOLS_YAML, encoding="utf-8")
    profile = AgentProfile.from_yaml(path)

    assert profile.tools.skills == [
        "skills/tabular-report",
        "../shared/skills/common",
    ]

    assert len(profile.tools.mcp) == 2
    first = profile.tools.mcp[0]
    assert isinstance(first, MCPServerConfig)
    assert first.command == "uvx"
    assert first.args == ["mcp-server-filesystem", "/tmp"]
    assert first.env is None
    assert first.cwd is None

    second = profile.tools.mcp[1]
    assert second.command == "python"
    assert second.args == ["-m", "my.module"]
    assert second.env == {"API_KEY": "secret"}
    assert second.cwd == "/srv/mcp"


def test_unknown_key_under_tools_is_rejected(tmp_path: Path) -> None:
    bad = _MINIMAL_YAML + "  tools:\n    whatever: 1\n"
    path = tmp_path / "p.yaml"
    path.write_text(bad, encoding="utf-8")
    with pytest.raises(ConfigValidationError):
        AgentProfile.from_yaml(path)


def test_mcp_entry_requires_command(tmp_path: Path) -> None:
    bad = _MINIMAL_YAML + "  tools:\n    mcp:\n      - args: [x]\n"
    path = tmp_path / "p.yaml"
    path.write_text(bad, encoding="utf-8")
    with pytest.raises(ConfigValidationError):
        AgentProfile.from_yaml(path)


def test_mcp_entry_rejects_empty_command(tmp_path: Path) -> None:
    bad = _MINIMAL_YAML + '  tools:\n    mcp:\n      - command: ""\n'
    path = tmp_path / "p.yaml"
    path.write_text(bad, encoding="utf-8")
    with pytest.raises(ConfigValidationError):
        AgentProfile.from_yaml(path)


# ---------- multi-transport MCP entries ----------


def test_mcp_entry_accepts_url_only_with_sse_transport(tmp_path: Path) -> None:
    """url-mode entries set transport + url + headers; no command/args required."""
    yaml_text = _MINIMAL_YAML + (
        "  tools:\n"
        "    mcp:\n"
        "      - transport: sse\n"
        "        url: https://mcp.example.com/sse\n"
        "        headers:\n"
        "          Authorization: Bearer token\n"
        "        timeout: 30\n"
    )
    path = tmp_path / "p.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    profile = AgentProfile.from_yaml(path)
    (entry,) = profile.tools.mcp
    assert entry.transport == "sse"
    assert entry.url == "https://mcp.example.com/sse"
    assert entry.headers == {"Authorization": "Bearer token"}
    assert entry.timeout == 30
    assert entry.command is None


def test_mcp_entry_rejects_both_command_and_url(tmp_path: Path) -> None:
    """A server entry must be either stdio (command) or remote (url), never both."""
    bad = _MINIMAL_YAML + (
        "  tools:\n"
        "    mcp:\n"
        "      - command: uvx\n"
        "        args: [x]\n"
        "        url: https://example.com\n"
    )
    path = tmp_path / "p.yaml"
    path.write_text(bad, encoding="utf-8")
    with pytest.raises(ConfigValidationError):
        AgentProfile.from_yaml(path)


def test_mcp_entry_rejects_neither_command_nor_url(tmp_path: Path) -> None:
    bad = _MINIMAL_YAML + "  tools:\n    mcp:\n      - args: [x]\n"
    path = tmp_path / "p.yaml"
    path.write_text(bad, encoding="utf-8")
    with pytest.raises(ConfigValidationError):
        AgentProfile.from_yaml(path)


def test_mcp_entry_rejects_include_and_exclude_together(tmp_path: Path) -> None:
    bad = _MINIMAL_YAML + (
        "  tools:\n"
        "    mcp:\n"
        "      - command: uvx\n"
        "        args: [x]\n"
        "        include: [a]\n"
        "        exclude: [b]\n"
    )
    path = tmp_path / "p.yaml"
    path.write_text(bad, encoding="utf-8")
    with pytest.raises(ConfigValidationError):
        AgentProfile.from_yaml(path)


def test_mcp_entry_rejects_unknown_transport_value(tmp_path: Path) -> None:
    bad = _MINIMAL_YAML + (
        "  tools:\n"
        "    mcp:\n"
        "      - transport: bogus\n"
        "        url: https://example.com\n"
    )
    path = tmp_path / "p.yaml"
    path.write_text(bad, encoding="utf-8")
    with pytest.raises(ConfigValidationError):
        AgentProfile.from_yaml(path)


# ---------- source_path tracking ----------


def test_from_yaml_records_source_path(tmp_path: Path) -> None:
    path = tmp_path / "p.yaml"
    path.write_text(_MINIMAL_YAML, encoding="utf-8")
    profile = AgentProfile.from_yaml(path)
    assert profile.source_path == path.resolve()
    assert profile.source_dir == path.resolve().parent


def test_in_memory_profile_has_no_source_path() -> None:
    profile = AgentProfile(
        id="x", name="X", age=1, traits="t", backstory="b", initial_plan="p",
    )
    assert profile.source_path is None
    assert profile.source_dir is None


def test_shipped_example_profile_parses_with_tools() -> None:
    """The shipped DefenseAgent/examples/example_agent/profile.yaml must load with the new schema."""
    from DefenseAgent.examples import EXAMPLE_PROFILE_PATH

    profile = AgentProfile.from_yaml(EXAMPLE_PROFILE_PATH)
    assert profile.source_dir == EXAMPLE_PROFILE_PATH.resolve().parent
    assert profile.tools.skills == ["skills/tabular-report"]
    assert profile.tools.mcp == []
