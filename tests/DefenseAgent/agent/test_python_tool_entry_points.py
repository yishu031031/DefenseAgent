"""Tests for `profile.tools.python` — importlib-resolved Python entry points.

Each entry is a `'module.path:function_name'` string. The builder calls
`_resolve_python_entry_point(...)` which imports the module and returns the
callable; `ToolRegistry.tool(fn)` then registers it. The function's type
hints + docstring drive the tool's input schema + description, identical
to the in-code `@registry.tool` decorator path.
"""
import sys
import types
from pathlib import Path

import pytest

from DefenseAgent.agent._builder import _resolve_python_entry_point
from DefenseAgent.config import AgentProfile
from DefenseAgent.tools.types import ToolRegistrationError


# ---------- _resolve_python_entry_point: parsing + import + lookup ----------


def test_resolves_a_real_callable_from_a_module(monkeypatch):
    """Inject a fake module into sys.modules and verify the resolver returns its function unchanged."""
    mod = types.ModuleType("fake_tools_pkg")
    def my_tool(x: int) -> int:
        """Double x."""
        return x * 2
    mod.my_tool = my_tool
    monkeypatch.setitem(sys.modules, "fake_tools_pkg", mod)

    fn = _resolve_python_entry_point("fake_tools_pkg:my_tool")
    assert fn is my_tool
    assert fn(7) == 14


def test_supports_dotted_module_paths(monkeypatch):
    """`pkg.subpkg:fn` should resolve through nested module paths."""
    pkg = types.ModuleType("fake_pkg")
    pkg.__path__ = []  # mark as a package
    sub = types.ModuleType("fake_pkg.tools")
    def hello(name: str) -> str:
        """say hi"""
        return f"hi, {name}"
    sub.hello = hello
    monkeypatch.setitem(sys.modules, "fake_pkg", pkg)
    monkeypatch.setitem(sys.modules, "fake_pkg.tools", sub)

    fn = _resolve_python_entry_point("fake_pkg.tools:hello")
    assert fn("Nova") == "hi, Nova"


@pytest.mark.parametrize(
    "bad_entry",
    [
        "no_colon_at_all",
        "",
        ":missing_module",
        "missing_func:",
        ":",
    ],
)
def test_rejects_malformed_entry_points(bad_entry):
    with pytest.raises(ToolRegistrationError, match="must be"):
        _resolve_python_entry_point(bad_entry)


def test_rejects_unimportable_module():
    with pytest.raises(ToolRegistrationError, match="could not import module"):
        _resolve_python_entry_point("definitely.not.a.real.module:fn")


def test_rejects_missing_attribute(monkeypatch):
    mod = types.ModuleType("fake_empty_mod")
    monkeypatch.setitem(sys.modules, "fake_empty_mod", mod)
    with pytest.raises(ToolRegistrationError, match="has no attribute"):
        _resolve_python_entry_point("fake_empty_mod:nonexistent")


def test_rejects_non_callable_attribute(monkeypatch):
    mod = types.ModuleType("fake_const_mod")
    mod.SOME_CONST = 42
    monkeypatch.setitem(sys.modules, "fake_const_mod", mod)
    with pytest.raises(ToolRegistrationError, match="not a callable"):
        _resolve_python_entry_point("fake_const_mod:SOME_CONST")


# ---------- file-path form: relative/file.py:func ----------


def test_resolves_file_path_relative_to_base_dir(tmp_path: Path):
    """Entries that look like a file path (`*.py:func`) load via importlib.util.spec_from_file_location, with the path resolved against `base_dir`."""
    tool_file = tmp_path / "my_tools" / "calc.py"
    tool_file.parent.mkdir(parents=True)
    tool_file.write_text(
        "def adder(a: int, b: int) -> int:\n"
        '    """Add two ints."""\n'
        "    return a + b\n",
        encoding="utf-8",
    )
    fn = _resolve_python_entry_point("my_tools/calc.py:adder", base_dir=tmp_path)
    assert fn(2, 3) == 5


def test_file_path_entry_rejects_relative_path_without_base_dir(tmp_path: Path):
    """A relative file path needs `base_dir` (i.e. profile.source_dir) to anchor against; without it we surface an explicit error rather than silently falling back."""
    with pytest.raises(ToolRegistrationError, match="no source_dir"):
        _resolve_python_entry_point("my_tools/calc.py:fn", base_dir=None)


def test_file_path_entry_rejects_missing_file(tmp_path: Path):
    with pytest.raises(ToolRegistrationError, match="file not found"):
        _resolve_python_entry_point("missing.py:fn", base_dir=tmp_path)


def test_file_path_entry_surfaces_module_load_errors(tmp_path: Path):
    """If the target file raises at import time, the resolver wraps the error with file context."""
    tool_file = tmp_path / "broken.py"
    tool_file.write_text("raise RuntimeError('boom')\n", encoding="utf-8")
    with pytest.raises(ToolRegistrationError, match="failed to execute"):
        _resolve_python_entry_point("broken.py:fn", base_dir=tmp_path)


def test_file_path_entry_works_via_builder_against_example_agent(tmp_path: Path):
    """End-to-end: the shipped example bundle lists `python_tools/calc.py:calculator`. The builder must register it as a tool named `calculator`."""
    from DefenseAgent.agent._builder import build_components_sync
    from DefenseAgent.agent.config import AgentConfig
    from DefenseAgent.examples import EXAMPLE_PROFILE_PATH

    profile = AgentProfile.from_yaml(EXAMPLE_PROFILE_PATH)
    built = build_components_sync(
        AgentConfig(
            profile=profile, use_memory=False, use_reflection=False,
            use_compressor=False, load_env=False,
            llm=_DummyLLM(),
        )
    )
    assert "calculator" in built.tools


# ---------- end-to-end: profile.tools.python → ToolRegistry ----------


def test_profile_python_entries_become_tools_via_builder(monkeypatch, tmp_path: Path):
    """A profile that lists a Python entry point under `tools.python:` ends up with that callable as a registered tool — driven by `_builder.build_components_sync`."""
    mod = types.ModuleType("e2e_tool_mod")
    def double(value: int) -> int:
        """Return value times two."""
        return value * 2
    mod.double = double
    monkeypatch.setitem(sys.modules, "e2e_tool_mod", mod)

    yaml_text = (
        "agent:\n"
        '  id: "x"\n'
        '  name: "X"\n'
        "  age: 1\n"
        '  traits: "t"\n'
        '  backstory: "b"\n'
        '  initial_plan: "p"\n'
        "  tools:\n"
        "    python:\n"
        "      - e2e_tool_mod:double\n"
    )
    p = tmp_path / "profile.yaml"
    p.write_text(yaml_text, encoding="utf-8")
    profile = AgentProfile.from_yaml(p)

    from DefenseAgent.agent.config import AgentConfig
    from DefenseAgent.agent._builder import build_components_sync

    built = build_components_sync(
        AgentConfig(
            profile=profile, use_memory=False, use_reflection=False,
            use_compressor=False, load_env=False,
            llm=_DummyLLM(),
        )
    )
    assert "double" in built.tools


# Tiny stand-in so build_components_sync doesn't try to construct a real LLM.
class _DummyLLM:
    """Minimal LLM stub — just present enough so `build_components_sync` doesn't try to read .env or call from_profile."""
    def __init__(self): pass


# ---------- schema validation: tools.python must be a list of strings ----------


def test_profile_tools_python_defaults_to_empty_list(tmp_path: Path):
    yaml_text = (
        "agent:\n"
        '  id: "x"\n'
        '  name: "X"\n'
        "  age: 1\n"
        '  traits: "t"\n'
        '  backstory: "b"\n'
        '  initial_plan: "p"\n'
    )
    p = tmp_path / "profile.yaml"
    p.write_text(yaml_text, encoding="utf-8")
    profile = AgentProfile.from_yaml(p)
    assert profile.tools.python == []


def test_profile_tools_python_accepts_multiple_entries(tmp_path: Path):
    yaml_text = (
        "agent:\n"
        '  id: "x"\n'
        '  name: "X"\n'
        "  age: 1\n"
        '  traits: "t"\n'
        '  backstory: "b"\n'
        '  initial_plan: "p"\n'
        "  tools:\n"
        "    python:\n"
        "      - my_pkg.tools:calculator\n"
        "      - my_pkg.search:web_search\n"
    )
    p = tmp_path / "profile.yaml"
    p.write_text(yaml_text, encoding="utf-8")
    profile = AgentProfile.from_yaml(p)
    assert profile.tools.python == [
        "my_pkg.tools:calculator",
        "my_pkg.search:web_search",
    ]
