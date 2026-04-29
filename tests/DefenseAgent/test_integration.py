"""Cross-module integration tests — config + LLM + ops + tools + memory compose cleanly.

Sections:
  • Config + LLM              — profile fields flow into adapter.chat()
  • Config + LLM + Logger     — wrap adapter calls with logger events
  • Config + Tools            — agent-bundle skill/MCP layout end-to-end
  • Config + Memory + Agent   — full-stack flow with mocked Mem0Memory
"""
import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from DefenseAgent.config import AgentProfile
from DefenseAgent.llm import LLMProviderError, LLMResponse, Message, TokenUsage
from DefenseAgent.llm.base import LLMAdapter
from DefenseAgent.llm.types import ToolCall
from DefenseAgent.ops import AgentLogger
from DefenseAgent.tools import ToolRegistry


def _example_profile_path() -> Path:
    from DefenseAgent.examples import EXAMPLE_PROFILE_PATH
    return EXAMPLE_PROFILE_PATH


def _read_log(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


class StubLLMAdapter(LLMAdapter):
    """LLMAdapter that records each chat() call and returns a canned response."""

    def __init__(self, canned: str = "OK."):
        self.calls: list[dict] = []
        self._canned = canned

    async def chat(self, messages, *, tools=None, temperature=0.7,
                   max_tokens=1024, system=None):
        self.calls.append({
            "messages": messages, "tools": tools,
            "temperature": temperature, "max_tokens": max_tokens,
            "system": system,
        })
        return LLMResponse(
            content=self._canned, tool_calls=[],
            usage=TokenUsage(50, 20, 70),
            stop_reason="end_turn", raw={},
        )


class StubErrorLLMAdapter(LLMAdapter):
    async def chat(self, messages, **kwargs):
        raise LLMProviderError(provider="stub", status_code=429, message="rate limited")


def _build_system_prompt(profile: AgentProfile) -> str:
    """Collapse identity fields into a system prompt — used by the LLM section."""
    return (
        f"You are {profile.name}, a {profile.age}-year-old.\n"
        f"Traits: {profile.traits}\n"
        f"Backstory: {profile.backstory.strip()}\n"
        f"Today's plan: {profile.initial_plan.strip()}\n"
        "Stay in character. Answer in first person and be concise."
    )


# ============================================================
# Config + LLM integration
# ============================================================


def test_shipped_example_profile_parses():
    """Regression guard: editing example_agent/profile.yaml must keep it valid."""
    profile = AgentProfile.from_yaml(_example_profile_path())
    assert profile.name == "Nova Patel"
    assert profile.age == 27
    assert "field engineer" in profile.backstory
    assert profile.cognitive.max_steps_per_cycle == 10
    assert profile.memory.search_limit == 10
    assert profile.memory.history_mode == "add"


async def test_example_profile_fields_reach_adapter_system_prompt():
    profile = AgentProfile.from_yaml(_example_profile_path())
    system = _build_system_prompt(profile)
    adapter = StubLLMAdapter(canned="The pipeline alerts cleared an hour ago.")

    resp = await adapter.chat(
        [Message(role="user", content="What have you been up to this afternoon?")],
        system=system, temperature=0.5, max_tokens=200,
    )

    call = adapter.calls[0]
    assert "Nova Patel" in call["system"]
    assert "27" in call["system"]
    assert "curious, methodical, candid" in call["system"]
    assert "field engineer" in call["system"]
    assert call["temperature"] == 0.5
    assert call["max_tokens"] == 200
    assert resp.content == "The pipeline alerts cleared an hour ago."


_INLINE_STUDENT_YAML = """\
agent:
  id: "student_test_001"
  name: "Test Student"
  age: 19
  traits: "focused, analytical"
  backstory: "A first-year physics major."
  initial_plan: "Finish problem set 4."
"""


async def test_inline_profile_composes_with_adapter(tmp_path):
    path = tmp_path / "student.yaml"
    path.write_text(_INLINE_STUDENT_YAML, encoding="utf-8")
    profile = AgentProfile.from_yaml(path)

    adapter = StubLLMAdapter(canned="Working on problem set 4.")
    resp = await adapter.chat(
        [Message(role="user", content="Hi")],
        system=_build_system_prompt(profile),
    )

    call = adapter.calls[0]
    assert "Test Student" in call["system"]
    assert "19" in call["system"]
    assert "physics" in call["system"]
    assert resp.content == "Working on problem set 4."


async def test_profile_defaults_survive_composition(tmp_path):
    """A minimal profile (no cognitive/memory override) still validates and composes with the adapter."""
    minimal = """\
agent:
  id: "mini"
  name: "Mini"
  age: 25
  traits: "terse"
  backstory: "A minimal test agent."
  initial_plan: "Do the thing."
"""
    path = tmp_path / "mini.yaml"
    path.write_text(minimal, encoding="utf-8")
    profile = AgentProfile.from_yaml(path)
    assert profile.cognitive.max_steps_per_cycle == 10
    assert profile.memory.search_limit == 10

    adapter = StubLLMAdapter()
    await adapter.chat(
        [Message(role="user", content="hi")],
        system=_build_system_prompt(profile),
    )
    assert "Mini" in adapter.calls[0]["system"]


# ============================================================
# Config + LLM + Logger integration
# ============================================================


async def test_logger_records_both_ends_of_a_chat_call(tmp_path):
    profile = AgentProfile.from_yaml(_example_profile_path())
    log_file = tmp_path / "maya.log"
    logger = AgentLogger.from_profile(
        profile, stream=None, log_file=log_file, level=logging.INFO,
    )
    adapter = StubLLMAdapter(canned="Hello!")

    logger.info("llm.request", "Calling model", messages_count=1, max_tokens=200)
    resp = await adapter.chat([Message(role="user", content="Say hi")], max_tokens=200)
    logger.info(
        "llm.response", "Model responded",
        stop_reason=resp.stop_reason, total_tokens=resp.usage.total_tokens,
    )

    records = _read_log(log_file)
    assert len(records) == 2
    req, res = records
    assert req["agent_id"] == "example_agent_001"
    assert res["agent_id"] == "example_agent_001"
    assert req["event_type"] == "llm.request"
    assert req["data"]["messages_count"] == 1
    assert req["data"]["max_tokens"] == 200
    assert res["event_type"] == "llm.response"
    assert res["data"]["stop_reason"] == "end_turn"
    assert res["data"]["total_tokens"] == 70


async def test_logger_records_provider_error_without_crashing(tmp_path):
    profile = AgentProfile.from_yaml(_example_profile_path())
    log_file = tmp_path / "maya.log"
    logger = AgentLogger.from_profile(
        profile, stream=None, log_file=log_file, level=logging.INFO,
    )
    adapter = StubErrorLLMAdapter()

    logger.info("llm.request", "Calling model")
    with pytest.raises(LLMProviderError):
        try:
            await adapter.chat([Message(role="user", content="hi")])
        except LLMProviderError as e:
            logger.error(
                "llm.error", "Provider failed",
                provider=e.provider, status_code=e.status_code,
            )
            raise

    records = _read_log(log_file)
    assert len(records) == 2
    assert records[0]["level"] == "INFO"
    assert records[1]["level"] == "ERROR"
    assert records[1]["event_type"] == "llm.error"
    assert records[1]["data"]["provider"] == "stub"
    assert records[1]["data"]["status_code"] == 429


# ============================================================
# Config + Tools integration (agent bundle end-to-end)
# ============================================================


async def test_tools_from_profile_loads_example_bundle_skill():
    """ToolRegistry.from_profile resolves skill paths relative to the profile's directory."""
    profile = AgentProfile.from_yaml(_example_profile_path())
    assert profile.tools.skills == ["skills/tabular-report"]

    async with await ToolRegistry.from_profile(profile) as registry:
        assert registry.names() == ["tabular-report"]

        specs = registry.specs()
        assert specs[0]["name"] == "tabular-report"
        assert "Render a list" in specs[0]["description"]
        assert "render_table" not in specs[0]["description"]
        assert "file" in specs[0]["input_schema"]["properties"]


async def test_tools_from_profile_serves_layer_2_body_from_disk():
    """Layer 2: an empty-args call returns the SKILL.md body verbatim."""
    profile = AgentProfile.from_yaml(_example_profile_path())
    async with await ToolRegistry.from_profile(profile) as registry:
        results = await registry.execute(
            [ToolCall(id="c1", name="tabular-report", arguments={})]
        )

    assert len(results) == 1
    msg = results[0]
    assert msg.role == "tool"
    assert msg.tool_call_id == "c1"
    assert "# Tabular Report" in msg.content
    assert "render_table" in msg.content


async def test_tools_from_profile_serves_layer_3_asset_from_bundle():
    """Layer 3: a `file` arg returns the contents of that asset inside the bundle."""
    profile = AgentProfile.from_yaml(_example_profile_path())
    async with await ToolRegistry.from_profile(profile) as registry:
        results = await registry.execute(
            [
                ToolCall(
                    id="c2",
                    name="tabular-report",
                    arguments={"file": "scripts/generate.py"},
                )
            ]
        )

    msg = results[0]
    assert "def render_table" in msg.content
    assert "Reference implementation" in msg.content


async def test_tools_from_profile_rejects_path_escape_as_tool_error():
    """Escape attempts become role='tool' error Messages, not exceptions."""
    profile = AgentProfile.from_yaml(_example_profile_path())
    async with await ToolRegistry.from_profile(profile) as registry:
        results = await registry.execute(
            [
                ToolCall(
                    id="c3",
                    name="tabular-report",
                    arguments={"file": "../../etc/passwd"},
                )
            ]
        )

    msg = results[0]
    assert msg.role == "tool"
    assert "SkillLoadError" in msg.content


# ============================================================
# Full-stack integration with mocked Mem0Memory
# ============================================================


async def test_full_stack_profile_memory_tools_compose(monkeypatch):
    """Agent bundle drives profile → Mem0Memory (mocked) → ToolRegistry → ReActAgent."""
    monkeypatch.setenv("AGENT_LAB_LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.example.com")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-chat")
    monkeypatch.setenv("EMBEDDING_API_KEY", "sk-test-emb")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://api.example.com")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("EMBEDDING_DIMS", "1536")

    profile = AgentProfile.from_yaml(_example_profile_path())

    from DefenseAgent.memory.mem0_memory import Mem0Memory
    from DefenseAgent.memory.orchestrator import MemoryOrchestrator
    from DefenseAgent.agent import ReActAgent

    fake_mem0 = MagicMock(name="mem0")
    fake_mem0.search.return_value = {"results": []}
    fake_mem0.get_all.return_value = {"results": []}
    fake_mem0.add.return_value = None

    with patch.object(
        Mem0Memory,
        "_init_memory_obj",
        return_value=fake_mem0,
    ):
        agent = await ReActAgent.from_profile(profile, load_env=False)

    try:
        # The agent has the right composed pieces.
        assert agent.profile is profile
        # P2: agent.memory is now the tier-aware orchestrator wrapping a
        # Mem0Memory persistent backend.
        assert isinstance(agent.memory, MemoryOrchestrator)
        assert isinstance(agent.memory.persistent, Mem0Memory)
        assert "tabular-report" in agent.tools
        assert agent.reflector is not None
    finally:
        await agent.close()
