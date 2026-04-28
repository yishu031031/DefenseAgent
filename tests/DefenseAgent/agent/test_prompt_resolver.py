"""Tests for the system-prompt resolver on the Agent base class.

Covers `_identity_prompt` precedence (`profile.prompt.system` > `profile.prompt.path`
> auto-built default), `str.format()` substitution against profile fields, file
loading relative to `profile.source_dir`, missing-file fallback, bad-placeholder
fallback, and the `extra_instructions` append.
"""
from pathlib import Path

import pytest

from DefenseAgent.agent import ReActAgent
from DefenseAgent.config import AgentProfile
from DefenseAgent.tools import ToolRegistry

from tests.DefenseAgent.agent._support import ScriptedLLM, fake_memory, make_test_config


def _build_agent(profile: AgentProfile) -> ReActAgent:
    """Construct a ReActAgent with stubbed deps so we can exercise the prompt resolver in isolation."""
    return ReActAgent(make_test_config(
            profile=profile,
            llm=ScriptedLLM([]),
            memory=fake_memory(profile),
            tools=ToolRegistry(),
        ))


def _profile(**prompt_kwargs) -> AgentProfile:
    """Build a profile with the given prompt block overrides."""
    kwargs = dict(
        id="test_agent",
        name="Maya",
        age=20,
        traits="curious, persistent",
        backstory="A second-year CS student.",
        initial_plan="Review notes, attend lecture.",
    )
    if prompt_kwargs:
        kwargs["prompt"] = prompt_kwargs
    return AgentProfile(**kwargs)


def _profile_with_source(tmp_path: Path, **prompt_kwargs) -> AgentProfile:
    """Build a profile loaded from disk so `source_dir` is set, used for path-based prompt tests."""
    yaml = (
        "agent:\n"
        '  id: "agent_001"\n'
        '  name: "Maya"\n'
        "  age: 20\n"
        '  traits: "curious"\n'
        '  backstory: "A CS student."\n'
        '  initial_plan: "Review notes."\n'
    )
    if prompt_kwargs:
        yaml += "  prompt:\n"
        for k, v in prompt_kwargs.items():
            yaml += f'    {k}: "{v}"\n'
    p = tmp_path / "profile.yaml"
    p.write_text(yaml, encoding="utf-8")
    return AgentProfile.from_yaml(p)


# ---------- precedence ----------


def test_falls_back_to_default_when_neither_system_nor_path_set():
    profile = _profile()
    agent = _build_agent(profile)
    rendered = agent._identity_prompt()
    assert "Maya" in rendered
    assert "20" in rendered
    assert "curious" in rendered
    assert "second-year" in rendered  # from backstory


def test_inline_system_takes_precedence_over_path(tmp_path):
    (tmp_path / "system.md").write_text("FILE PROMPT for {name}", encoding="utf-8")
    profile = _profile_with_source(
        tmp_path,
        system="INLINE PROMPT for {name}",
        path="system.md",
    )
    agent = _build_agent(profile)
    assert agent._identity_prompt() == "INLINE PROMPT for Maya"


def test_path_used_when_no_inline_system(tmp_path):
    (tmp_path / "system.md").write_text(
        "You are {name}, age {age}.", encoding="utf-8"
    )
    profile = _profile_with_source(tmp_path, path="system.md")
    agent = _build_agent(profile)
    assert agent._identity_prompt() == "You are Maya, age 20."


def test_blank_system_string_is_treated_as_unset(tmp_path):
    """`system: '   '` should not block path lookup — whitespace-only is effectively empty."""
    (tmp_path / "system.md").write_text("FILE for {name}", encoding="utf-8")
    profile = _profile_with_source(tmp_path, system="   ", path="system.md")
    agent = _build_agent(profile)
    assert agent._identity_prompt() == "FILE for Maya"


# ---------- str.format() substitution ----------


def test_all_placeholders_substituted():
    template = (
        "id={id} name={name} age={age} traits={traits} "
        "backstory={backstory} plan={initial_plan}"
    )
    profile = _profile(system=template)
    agent = _build_agent(profile)
    rendered = agent._identity_prompt()
    assert "id=test_agent" in rendered
    assert "name=Maya" in rendered
    assert "age=20" in rendered
    assert "traits=curious, persistent" in rendered
    assert "backstory=A second-year CS student." in rendered
    assert "plan=Review notes, attend lecture." in rendered


def test_unknown_placeholder_falls_back_to_default():
    profile = _profile(system="hello {undefined_field}")
    agent = _build_agent(profile)
    rendered = agent._identity_prompt()
    # falls back to auto-built default — should contain identity, not the broken template
    assert "{undefined_field}" not in rendered
    assert "Maya" in rendered


def test_bad_format_syntax_falls_back_to_default():
    """A broken template (e.g. unclosed brace) must not crash the agent — fall back to the default identity."""
    profile = _profile(system="hello {name")
    agent = _build_agent(profile)
    rendered = agent._identity_prompt()
    assert "Maya" in rendered
    assert "{name" not in rendered


# ---------- path resolution edge cases ----------


def test_missing_prompt_file_falls_back_to_default(tmp_path):
    profile = _profile_with_source(tmp_path, path="does_not_exist.md")
    agent = _build_agent(profile)
    rendered = agent._identity_prompt()
    assert "Maya" in rendered  # default identity still rendered


def test_path_set_but_source_dir_unknown_falls_back():
    """An in-memory profile has no source_dir, so path-based prompts cannot resolve — must fall back."""
    profile = _profile(path="prompts/system.md")
    assert profile.source_dir is None
    agent = _build_agent(profile)
    rendered = agent._identity_prompt()
    assert "Maya" in rendered


# ---------- extra_instructions ----------


def test_extra_instructions_appended_to_default_identity():
    profile = _profile(extra_instructions="ALWAYS BE TERSE.")
    agent = _build_agent(profile)
    rendered = agent._identity_prompt()
    assert "Maya" in rendered
    assert rendered.rstrip().endswith("ALWAYS BE TERSE.")


def test_extra_instructions_appended_to_inline_system():
    profile = _profile(
        system="You are {name}.",
        extra_instructions="Be terse.",
    )
    agent = _build_agent(profile)
    assert agent._identity_prompt() == "You are Maya.\n\nBe terse."


def test_extra_instructions_blank_is_ignored():
    profile = _profile(system="You are {name}.", extra_instructions="   ")
    agent = _build_agent(profile)
    assert agent._identity_prompt() == "You are Maya."


# ---------- end-to-end via the shipped example_agent bundle ----------


from DefenseAgent.examples import EXAMPLE_PROFILE_PATH as _EXAMPLE_PROFILE


@pytest.mark.skipif(not _EXAMPLE_PROFILE.is_file(), reason="example_agent bundle not present")
def test_example_bundle_loads_path_prompt_with_substitution():
    """The shipped reference profile uses `prompt.path: prompts/system.md` plus `{name}`/`{age}` placeholders — verify substitution wires up end-to-end against the actual on-disk bundle."""
    profile = AgentProfile.from_yaml(_EXAMPLE_PROFILE)
    assert profile.prompt.path == "prompts/system.md"
    agent = _build_agent(profile)
    rendered = agent._identity_prompt()
    assert profile.name in rendered
    assert f"{profile.age}-year-old" in rendered
    assert "{name}" not in rendered  # placeholders all filled
