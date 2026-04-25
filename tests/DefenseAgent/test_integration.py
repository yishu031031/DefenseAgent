"""Cross-module integration tests — config + LLM + ops + memory compose cleanly.

Split into three sections (one per module pairing):
  • Config + LLM              — profile fields flow into adapter.chat()
  • Config + LLM + Logger     — wrap adapter calls with logger events
  • Config + Memory + Logger  — wrap memory add/retrieve with logger events
"""
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from DefenseAgent.config import AgentProfile
from DefenseAgent.llm import LLMAdapter, LLMProviderError, LLMResponse, Message, TokenUsage
from DefenseAgent.llm.types import ToolCall
from DefenseAgent.memory import Memory
from DefenseAgent.memory.embedding import EmbeddingAdapter
from DefenseAgent.memory.retriever import MemoryRetriever
from DefenseAgent.memory.stream import MemoryStream
from DefenseAgent.ops import AgentLogger
from DefenseAgent.tools import ToolRegistry


# ---- shared fixtures / helpers ----

_NOW = datetime(2026, 4, 22, 18, 0, tzinfo=timezone.utc)


def _fixed_clock():
    return _NOW


def _maya_profile_path() -> Path:
    return (
        Path(__file__).resolve().parent.parent.parent
        / "agents"
        / "maya_rodriguez"
        / "profile.yaml"
    )


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


class StubEmbeddingAdapter(EmbeddingAdapter):
    """Keyword-aware stub; unknown text falls back to [0,0,0,1]."""

    def __init__(self, vectors: dict[str, list[float]]):
        self._vectors = vectors
        self.calls: list[str] = []

    async def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        return list(self._vectors.get(text, [0.0, 0.0, 0.0, 1.0]))

    async def embed_batch(self, texts):
        return [await self.embed(t) for t in texts]


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


def test_shipped_student_profile_parses():
    """Regression guard: editing maya_rodriguez.yaml must keep it valid."""
    profile = AgentProfile.from_yaml(_maya_profile_path())
    assert profile.name == "Maya Rodriguez"
    assert profile.age == 20
    assert "Computer Science" in profile.backstory
    assert profile.cognitive.max_steps_per_cycle == 8
    assert profile.memory.retrieval_top_k == 8
    assert profile.memory.relevance_weight == 1.5


async def test_student_profile_fields_reach_adapter_system_prompt():
    profile = AgentProfile.from_yaml(_maya_profile_path())
    system = _build_system_prompt(profile)
    adapter = StubLLMAdapter(canned="I just finished the data structures homework set.")

    resp = await adapter.chat(
        [Message(role="user", content="What have you been up to this afternoon?")],
        system=system, temperature=0.5, max_tokens=200,
    )

    call = adapter.calls[0]
    # Identity fields made it into the system prompt.
    assert "Maya Rodriguez" in call["system"]
    assert "20" in call["system"]
    assert "curious, persistent, collaborative" in call["system"]
    assert "Computer Science" in call["system"]
    # Caller-supplied overrides were honored.
    assert call["temperature"] == 0.5
    assert call["max_tokens"] == 200
    # Canonical response round-trips.
    assert resp.content == "I just finished the data structures homework set."
    assert resp.usage.total_tokens == 70


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
    """A minimal profile (no cognitive/memory override) still works end-to-end."""
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
    assert profile.memory.retrieval_top_k == 10

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
    profile = AgentProfile.from_yaml(_maya_profile_path())
    log_file = tmp_path / "maya.log"
    logger = AgentLogger.from_profile(
        profile, stream=None, log_file=log_file, level=logging.INFO,
    )
    adapter = StubLLMAdapter(canned="Hello!")

    # Wrap adapter.chat() with logger events — the pattern future modules follow.
    logger.info("llm.request", "Calling model", messages_count=1, max_tokens=200)
    resp = await adapter.chat([Message(role="user", content="Say hi")], max_tokens=200)
    logger.info(
        "llm.response", "Model responded",
        stop_reason=resp.stop_reason, total_tokens=resp.usage.total_tokens,
    )

    records = _read_log(log_file)
    assert len(records) == 2
    req, res = records
    # Both records carry Maya's id from the profile.
    assert req["agent_id"] == "student_maya_001"
    assert res["agent_id"] == "student_maya_001"
    assert req["event_type"] == "llm.request"
    assert req["data"]["messages_count"] == 1
    assert req["data"]["max_tokens"] == 200
    assert res["event_type"] == "llm.response"
    assert res["data"]["stop_reason"] == "end_turn"
    assert res["data"]["total_tokens"] == 70


async def test_logger_records_provider_error_without_crashing(tmp_path):
    profile = AgentProfile.from_yaml(_maya_profile_path())
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
# Config + Memory + Logger integration
# ============================================================


async def test_retriever_uses_maya_profile_weights_and_top_k():
    """Weights from MemoryConfig (Module 2) reach the retriever unchanged."""
    profile = AgentProfile.from_yaml(_maya_profile_path())
    assert profile.memory.retrieval_top_k == 8
    assert profile.memory.relevance_weight == 1.5

    vectors = {
        "how's the homework going?":        [0, 0, 1, 0],
        "reviewed binary trees in lecture": [1, 0, 0, 0],
        "spanish homework due friday":      [0, 1, 0, 0],
        "stuck on problem 3 at library":    [0, 0, 1, 0],
        "coffee":                           [0, 0, 0, 1],
    }
    adapter = StubEmbeddingAdapter(vectors)
    stream = MemoryStream(adapter, clock=_fixed_clock, dedup_threshold=None)
    retriever = MemoryRetriever(stream, adapter, profile.memory, clock=_fixed_clock)

    await stream.add("reviewed binary trees in lecture", kind="observation", importance=6.0)
    await stream.add("spanish homework due friday", kind="observation", importance=4.0)
    await stream.add("stuck on problem 3 at library", kind="observation", importance=8.0)
    await stream.add("coffee", kind="observation", importance=2.0)

    results = await retriever.retrieve("how's the homework going?")
    top = results[0]
    # The library/homework memory aligns with the query embedding AND has a
    # matching keyword; it should beat coffee / spanish easily.
    assert "library" in top.record.content or "homework" in top.record.content
    # top_k comes from profile, but the stream only has 4 records:
    assert len(results) == 4


async def test_logger_records_memory_events_with_agent_id(tmp_path):
    """Wrap add() and retrieve() with logger.info calls — Module 3 integration."""
    profile = AgentProfile.from_yaml(_maya_profile_path())
    log_file = tmp_path / "memory.log"
    logger = AgentLogger.from_profile(
        profile, stream=None, log_file=log_file, level=logging.INFO,
    )

    vectors = {
        "attended lecture": [1, 0, 0, 0],
        "how's class?":     [1, 0, 0, 0],
    }
    adapter = StubEmbeddingAdapter(vectors)
    stream = MemoryStream(adapter, clock=_fixed_clock, dedup_threshold=None)
    retriever = MemoryRetriever(stream, adapter, profile.memory, clock=_fixed_clock)

    # Write path
    logger.info("memory.add", "adding observation", kind="observation")
    record = await stream.add("attended lecture", importance=6.0)
    logger.info(
        "memory.added", "stored",
        record_id=record.id, kind=record.kind, importance=record.importance,
    )

    # Read path
    logger.info("memory.retrieve", "querying", query="how's class?")
    results = await retriever.retrieve("how's class?")
    logger.info(
        "memory.retrieved", "done",
        hits=len(results),
        top_score=results[0].score if results else None,
    )

    lines = _read_log(log_file)
    assert len(lines) == 4
    assert all(r["agent_id"] == "student_maya_001" for r in lines)
    assert [r["event_type"] for r in lines] == [
        "memory.add", "memory.added", "memory.retrieve", "memory.retrieved",
    ]


async def test_facts_survive_age_while_observations_decay():
    """Typed-schema integration: facts bypass recency decay end-to-end."""
    profile = AgentProfile.from_yaml(_maya_profile_path())
    # Make recency the dominant signal for this test.
    config = profile.memory.model_copy(
        update={"recency_weight": 1.0, "importance_weight": 0.0, "relevance_weight": 0.0},
    )
    vectors = {"q": [0, 0, 0, 1], "fact": [0, 0, 0, 1], "obs": [0, 0, 0, 1]}
    adapter = StubEmbeddingAdapter(vectors)
    stream = MemoryStream(adapter, clock=lambda: _NOW, dedup_threshold=None)

    # Back-date both records to 10 days ago.
    old_obs = await _seed_with_age(stream, "obs", kind="observation",
                                   importance=5.0, age_hours=24 * 10)
    old_fact = await _seed_with_age(stream, "fact", kind="fact",
                                    importance=5.0, age_hours=24 * 10)

    retriever = MemoryRetriever(stream, adapter, config, clock=lambda: _NOW)
    results = await retriever.retrieve("q")

    # Facts bypass recency decay → should beat observations when recency dominates.
    assert results[0].record.id == old_fact.id
    fact_scored = next(s for s in results if s.record.id == old_fact.id)
    obs_scored = next(s for s in results if s.record.id == old_obs.id)
    assert fact_scored.recency_score == 1.0
    assert obs_scored.recency_score < 0.1


async def _seed_with_age(stream, content, *, kind, importance, age_hours):
    """Add a record with a forged timestamp by temporarily shifting the stream clock."""
    original_clock = stream._clock
    stream._clock = lambda: _NOW - timedelta(hours=age_hours)
    try:
        return await stream.add(content, kind=kind, importance=importance)
    finally:
        stream._clock = original_clock


# ============================================================
# Config + Tools integration (agent bundle end-to-end)
# ============================================================


async def test_tools_from_profile_loads_maya_bundle_skill():
    """ToolRegistry.from_profile resolves skill paths relative to the profile's directory."""
    profile = AgentProfile.from_yaml(_maya_profile_path())
    assert profile.tools.skills == ["skills/tabular-report"]

    async with await ToolRegistry.from_profile(profile) as registry:
        assert registry.names() == ["tabular-report"]

        spec = registry.spec()
        # Layer 1: only metadata in the spec; body must NOT appear.
        assert spec[0]["name"] == "tabular-report"
        assert "Render a list" in spec[0]["description"]
        assert "render_table" not in spec[0]["description"]
        assert "file" in spec[0]["input_schema"]["properties"]


async def test_tools_from_profile_serves_layer_2_body_from_disk():
    """Layer 2: an empty-args call returns the SKILL.md body verbatim."""
    profile = AgentProfile.from_yaml(_maya_profile_path())
    async with await ToolRegistry.from_profile(profile) as registry:
        results = await registry.execute(
            [ToolCall(id="c1", name="tabular-report", arguments={})]
        )

    assert len(results) == 1
    msg = results[0]
    assert msg.role == "tool"
    assert msg.tool_call_id == "c1"
    # Layer-2 body is the SKILL.md content after the frontmatter.
    assert "# Tabular Report" in msg.content
    assert "render_table" in msg.content


async def test_tools_from_profile_serves_layer_3_asset_from_bundle():
    """Layer 3: a `file` arg returns the contents of that asset inside the bundle."""
    profile = AgentProfile.from_yaml(_maya_profile_path())
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
    profile = AgentProfile.from_yaml(_maya_profile_path())
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


async def test_full_stack_profile_memory_tools_compose():
    """Agent bundle drives profile + memory + tools in one flow:
    load profile → build Memory → store a memory → build ToolRegistry from same
    profile → execute the skill → remember() its result → recall() finds it."""
    profile = AgentProfile.from_yaml(_maya_profile_path())

    # Deterministic embeddings so the test stays offline.
    vectors = {
        "Today I read the tabular-report skill instructions.": [1, 0, 0, 0],
        "unrelated coffee observation":                         [0, 1, 0, 0],
        "how do I produce a markdown table?":                   [1, 0, 0, 0],
    }
    adapter = StubEmbeddingAdapter(vectors)
    memory = Memory(profile=profile, embedding_adapter=adapter, clock=_fixed_clock)

    async with await ToolRegistry.from_profile(profile) as registry:
        # 1. Tool registry is populated from profile.tools.
        assert "tabular-report" in registry

        # 2. Execute the skill (Layer 2) — pretend the agent just read it.
        [result] = await registry.execute(
            [ToolCall(id="tc", name="tabular-report", arguments={})]
        )
        assert "Tabular Report" in result.content

        # 3. Store an observation that the skill was consulted, plus a
        #    distractor that the retrieval should rank below it.
        await memory.remember(
            "Today I read the tabular-report skill instructions.",
            kind="observation",
            importance=7.0,
        )
        await memory.remember(
            "unrelated coffee observation",
            kind="observation",
            importance=2.0,
        )
        assert len(memory) == 2

        # 4. Retrieve for a query aligned with the skill memory; it should win.
        hits = await memory.recall("how do I produce a markdown table?")
        assert hits[0].record.content == (
            "Today I read the tabular-report skill instructions."
        )
        # Profile weights flowed through: retrieval_top_k == 8 caps results,
        # but we only stored two, so we get two.
        assert len(hits) == 2
