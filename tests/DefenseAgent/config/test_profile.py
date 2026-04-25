"""Tests for DefenseAgent.config.profile — the whole module in one file.

Covers:
  • Error hierarchy (subclassing + __cause__ preservation)
  • Pydantic model contract: CognitiveConfig, MemoryConfig, AgentProfile
  • AgentProfile.from_yaml happy paths
  • AgentProfile.from_yaml file-not-found / YAML-parse / schema-validation branches
  • yaml.safe_load defensive check
"""
from pathlib import Path

import pytest
from pydantic import ValidationError

from DefenseAgent.config import (
    AgentProfile,
    CognitiveConfig,
    ConfigError,
    ConfigFileNotFoundError,
    ConfigParseError,
    ConfigValidationError,
    MemoryConfig,
)


# ============================================================
# Error hierarchy
# ============================================================


def test_subclasses_inherit_from_config_error():
    assert issubclass(ConfigFileNotFoundError, ConfigError)
    assert issubclass(ConfigParseError, ConfigError)
    assert issubclass(ConfigValidationError, ConfigError)


def test_errors_carry_messages():
    assert "missing" in str(ConfigFileNotFoundError("missing file"))
    assert "bad" in str(ConfigParseError("bad yaml"))
    assert "schema" in str(ConfigValidationError("schema mismatch"))


def test_validation_error_preserves_cause():
    original = ValueError("field out of range")
    with pytest.raises(ConfigValidationError) as excinfo:
        try:
            raise original
        except ValueError as e:
            raise ConfigValidationError("wrapped") from e
    assert excinfo.value.__cause__ is original


# ============================================================
# Pydantic model contract (construct directly, no YAML)
# ============================================================


# ---- CognitiveConfig ----


def test_cognitive_config_defaults_when_empty():
    cfg = CognitiveConfig()
    assert cfg.max_steps_per_cycle == 10
    assert cfg.reflection_threshold == 5
    assert cfg.importance_threshold == 7
    assert cfg.planning_horizon == "1 day"


def test_cognitive_config_accepts_overrides():
    cfg = CognitiveConfig(
        max_steps_per_cycle=20, reflection_threshold=3,
        importance_threshold=5.5, planning_horizon="2 hours",
    )
    assert cfg.max_steps_per_cycle == 20
    assert cfg.reflection_threshold == 3
    assert cfg.importance_threshold == 5.5
    assert cfg.planning_horizon == "2 hours"


@pytest.mark.parametrize("field,bad_value", [
    ("max_steps_per_cycle", 0),
    ("reflection_threshold", 0),
    ("importance_threshold", 0.5),   # below min of 1
    ("importance_threshold", 11),    # above max of 10
])
def test_cognitive_config_rejects_out_of_range(field, bad_value):
    with pytest.raises(ValidationError):
        CognitiveConfig(**{field: bad_value})


def test_cognitive_config_rejects_unknown_key():
    with pytest.raises(ValidationError):
        CognitiveConfig(mystery_setting=42)


def test_cognitive_config_rejects_empty_planning_horizon():
    with pytest.raises(ValidationError):
        CognitiveConfig(planning_horizon="")


# ---- MemoryConfig ----


def test_memory_config_defaults_when_empty():
    cfg = MemoryConfig()
    assert cfg.max_working_memory_tokens == 4000
    assert cfg.retrieval_top_k == 10
    assert cfg.recency_weight == 1.0
    assert cfg.importance_weight == 1.0
    assert cfg.relevance_weight == 1.0


def test_memory_config_accepts_overrides():
    cfg = MemoryConfig(
        max_working_memory_tokens=8000, retrieval_top_k=5,
        recency_weight=0.5, importance_weight=2.0, relevance_weight=1.5,
    )
    assert cfg.max_working_memory_tokens == 8000
    assert cfg.retrieval_top_k == 5
    assert cfg.recency_weight == 0.5
    assert cfg.importance_weight == 2.0
    assert cfg.relevance_weight == 1.5


@pytest.mark.parametrize("field,bad_value", [
    ("max_working_memory_tokens", 0),
    ("retrieval_top_k", 0),
    ("recency_weight", -0.1),
    ("importance_weight", -1),
    ("relevance_weight", -2.5),
])
def test_memory_config_rejects_out_of_range(field, bad_value):
    with pytest.raises(ValidationError):
        MemoryConfig(**{field: bad_value})


def test_memory_config_allows_zero_weight():
    """Zero is a valid weight — user can disable one term of the retrieval score."""
    cfg = MemoryConfig(recency_weight=0.0, importance_weight=0.0, relevance_weight=0.0)
    assert cfg.recency_weight == 0.0


def test_memory_config_rejects_unknown_key():
    with pytest.raises(ValidationError):
        MemoryConfig(extra_knob=1)


# ---- AgentProfile (direct construction) ----


def _minimal_identity() -> dict:
    return {
        "id": "agent_001", "name": "Alice", "age": 28,
        "traits": "curious, methodical",
        "backstory": "A data scientist.",
        "initial_plan": "Review emails.",
    }


def test_agent_profile_minimal_fills_defaults():
    profile = AgentProfile(**_minimal_identity())
    assert profile.id == "agent_001"
    assert profile.name == "Alice"
    assert profile.age == 28
    # Nested blocks default-populated
    assert profile.cognitive.max_steps_per_cycle == 10
    assert profile.memory.retrieval_top_k == 10


def test_agent_profile_nested_overrides():
    profile = AgentProfile(
        **_minimal_identity(),
        cognitive={"max_steps_per_cycle": 3},
        memory={"retrieval_top_k": 2},
    )
    assert profile.cognitive.max_steps_per_cycle == 3
    assert profile.cognitive.reflection_threshold == 5  # other fields still default
    assert profile.memory.retrieval_top_k == 2


@pytest.mark.parametrize("missing_field", [
    "id", "name", "age", "traits", "backstory", "initial_plan",
])
def test_agent_profile_requires_identity_fields(missing_field):
    data = _minimal_identity()
    del data[missing_field]
    with pytest.raises(ValidationError):
        AgentProfile(**data)


@pytest.mark.parametrize("field", ["id", "name", "traits", "backstory", "initial_plan"])
def test_agent_profile_rejects_empty_string_identity(field):
    data = _minimal_identity()
    data[field] = ""
    with pytest.raises(ValidationError):
        AgentProfile(**data)


def test_agent_profile_rejects_negative_age():
    data = _minimal_identity()
    data["age"] = -1
    with pytest.raises(ValidationError):
        AgentProfile(**data)


def test_agent_profile_allows_age_zero():
    data = _minimal_identity()
    data["age"] = 0
    profile = AgentProfile(**data)
    assert profile.age == 0


def test_agent_profile_rejects_unknown_top_level_key():
    data = _minimal_identity()
    data["favorite_color"] = "blue"
    with pytest.raises(ValidationError):
        AgentProfile(**data)


def test_agent_profile_rejects_unknown_nested_key():
    data = _minimal_identity()
    data["cognitive"] = {"max_steps_per_cycle": 5, "bogus": True}
    with pytest.raises(ValidationError):
        AgentProfile(**data)


# ============================================================
# AgentProfile.from_yaml
# ============================================================

_MINIMAL_YAML = """\
agent:
  id: "agent_001"
  name: "Alice"
  age: 28
  traits: "curious"
  backstory: "A data scientist."
  initial_plan: "Review emails."
"""

_FULL_YAML = """\
agent:
  id: "agent_042"
  name: "Alice Chen"
  age: 28
  traits: "curious, methodical, empathetic"
  backstory: "A data scientist who recently moved to Lakeside."
  initial_plan: "Wake up, review emails, work on the data analysis project."
  cognitive:
    max_steps_per_cycle: 8
    reflection_threshold: 4
    importance_threshold: 7.5
    planning_horizon: "2 days"
  memory:
    max_working_memory_tokens: 8000
    retrieval_top_k: 15
    recency_weight: 2.0
    importance_weight: 0.5
    relevance_weight: 1.5
"""


def _write(tmp_path: Path, text: str, name: str = "profile.yaml") -> Path:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


# ---- happy path ----


def test_minimal_yaml_loads_with_defaults(tmp_path):
    profile = AgentProfile.from_yaml(_write(tmp_path, _MINIMAL_YAML))
    assert isinstance(profile, AgentProfile)
    assert profile.id == "agent_001"
    assert profile.name == "Alice"
    # Cognitive & memory default blocks:
    assert profile.cognitive.max_steps_per_cycle == 10
    assert profile.memory.retrieval_top_k == 10


def test_full_yaml_loads_all_fields(tmp_path):
    profile = AgentProfile.from_yaml(_write(tmp_path, _FULL_YAML))
    assert profile.id == "agent_042"
    assert profile.age == 28
    assert profile.cognitive.max_steps_per_cycle == 8
    assert profile.cognitive.planning_horizon == "2 days"
    assert profile.memory.max_working_memory_tokens == 8000
    assert profile.memory.recency_weight == 2.0


def test_from_yaml_accepts_string_path(tmp_path):
    profile = AgentProfile.from_yaml(str(_write(tmp_path, _MINIMAL_YAML)))
    assert profile.id == "agent_001"


# ---- file-not-found ----


def test_missing_file_raises_file_not_found(tmp_path):
    missing = tmp_path / "does_not_exist.yaml"
    with pytest.raises(ConfigFileNotFoundError) as e:
        AgentProfile.from_yaml(missing)
    assert str(missing) in str(e.value)


def test_directory_path_raises_file_not_found(tmp_path):
    """A directory is not a file; treat it as missing."""
    with pytest.raises(ConfigFileNotFoundError):
        AgentProfile.from_yaml(tmp_path)


# ---- YAML parse errors ----


def test_invalid_yaml_syntax_raises_parse_error(tmp_path):
    with pytest.raises(ConfigParseError):
        AgentProfile.from_yaml(_write(tmp_path, "agent:\n  id: [unclosed"))


def test_empty_file_raises_parse_error(tmp_path):
    """Empty YAML parses to None — not a mapping."""
    with pytest.raises(ConfigParseError):
        AgentProfile.from_yaml(_write(tmp_path, ""))


def test_top_level_list_raises_parse_error(tmp_path):
    with pytest.raises(ConfigParseError):
        AgentProfile.from_yaml(_write(tmp_path, "- a\n- b\n"))


def test_top_level_scalar_raises_parse_error(tmp_path):
    with pytest.raises(ConfigParseError):
        AgentProfile.from_yaml(_write(tmp_path, "just a string"))


def test_missing_agent_key_raises_parse_error(tmp_path):
    with pytest.raises(ConfigParseError) as e:
        AgentProfile.from_yaml(_write(tmp_path, "world:\n  name: Lakeside\n"))
    assert "agent" in str(e.value)


def test_agent_value_not_mapping_raises_parse_error(tmp_path):
    with pytest.raises(ConfigParseError):
        AgentProfile.from_yaml(_write(tmp_path, "agent: not-a-mapping\n"))


# ---- schema validation errors ----


def test_missing_required_identity_field_raises_validation_error(tmp_path):
    bad = _MINIMAL_YAML.replace('  id: "agent_001"\n', "")
    with pytest.raises(ConfigValidationError) as e:
        AgentProfile.from_yaml(_write(tmp_path, bad))
    assert isinstance(e.value.__cause__, ValidationError)


def test_wrong_type_raises_validation_error(tmp_path):
    bad = _MINIMAL_YAML.replace("age: 28", 'age: "twenty-eight"')
    with pytest.raises(ConfigValidationError):
        AgentProfile.from_yaml(_write(tmp_path, bad))


def test_out_of_range_raises_validation_error(tmp_path):
    bad = _FULL_YAML.replace("importance_threshold: 7.5", "importance_threshold: 42")
    with pytest.raises(ConfigValidationError):
        AgentProfile.from_yaml(_write(tmp_path, bad))


def test_unknown_key_raises_validation_error(tmp_path):
    bad = _MINIMAL_YAML + "  favorite_color: blue\n"
    with pytest.raises(ConfigValidationError):
        AgentProfile.from_yaml(_write(tmp_path, bad))


def test_unknown_nested_key_raises_validation_error(tmp_path):
    bad = _MINIMAL_YAML + "  cognitive:\n    max_steps_per_cycle: 5\n    mystery: 1\n"
    with pytest.raises(ConfigValidationError):
        AgentProfile.from_yaml(_write(tmp_path, bad))


# ---- defense against YAML code execution ----


def test_safe_load_rejects_python_object_tag(tmp_path):
    """yaml.safe_load must refuse !!python/object tags that could execute code."""
    malicious = "agent: !!python/object:os.system {}\n"
    with pytest.raises(ConfigParseError):
        AgentProfile.from_yaml(_write(tmp_path, malicious))
