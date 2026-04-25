"""Tests for DefenseAgent.ops.logger.AgentLogger.

Every test uses io.StringIO for the stream and pytest's tmp_path for files.
Clock is injected with a fixed datetime so timestamps are deterministic.
No real stdout or clock is touched.
"""
import io
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pytest

from DefenseAgent.config.profile import AgentProfile
from DefenseAgent.ops.logger import AgentLogger


FIXED_DT = datetime(2026, 4, 22, 10, 15, 30, 480000, tzinfo=timezone.utc)
FIXED_ISO = "2026-04-22T10:15:30.480Z"


def _fixed_clock():
    return FIXED_DT


def _read_lines(stream: io.StringIO) -> list[dict]:
    """Parse each non-empty line of the stream as JSON."""
    return [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]


# ---------- construction & defaults ----------


def test_stores_agent_id():
    logger = AgentLogger("agent_42", stream=io.StringIO(), clock=_fixed_clock)
    assert logger.agent_id == "agent_42"


def test_default_level_is_info():
    stream = io.StringIO()
    logger = AgentLogger("a", stream=stream, clock=_fixed_clock)
    logger.debug("e", "m")    # should be filtered
    logger.info("e", "m")     # should pass
    assert len(_read_lines(stream)) == 1


# ---------- record shape ----------


def test_emits_one_line_per_call():
    stream = io.StringIO()
    logger = AgentLogger("a", stream=stream, clock=_fixed_clock)
    logger.info("e1", "m1")
    logger.info("e2", "m2")
    assert len(_read_lines(stream)) == 2


def test_record_has_exact_top_level_keys():
    stream = io.StringIO()
    logger = AgentLogger("a", stream=stream, clock=_fixed_clock)
    logger.info("e", "m", x=1)
    record = _read_lines(stream)[0]
    assert set(record.keys()) == {
        "timestamp",
        "agent_id",
        "level",
        "event_type",
        "message",
        "data",
    }


def test_record_values():
    stream = io.StringIO()
    logger = AgentLogger("student_001", stream=stream, clock=_fixed_clock)
    logger.info("llm.request", "Calling model", model="deepseek-chat", tokens=200)
    record = _read_lines(stream)[0]
    assert record["timestamp"] == FIXED_ISO
    assert record["agent_id"] == "student_001"
    assert record["level"] == "INFO"
    assert record["event_type"] == "llm.request"
    assert record["message"] == "Calling model"
    assert record["data"] == {"model": "deepseek-chat", "tokens": 200}


def test_data_is_empty_dict_when_no_kwargs():
    stream = io.StringIO()
    logger = AgentLogger("a", stream=stream, clock=_fixed_clock)
    logger.info("e", "m")
    record = _read_lines(stream)[0]
    assert record["data"] == {}


def test_lines_are_newline_terminated():
    stream = io.StringIO()
    logger = AgentLogger("a", stream=stream, clock=_fixed_clock)
    logger.info("e", "m")
    raw = stream.getvalue()
    assert raw.endswith("\n")


# ---------- level filtering + method-to-level mapping ----------


@pytest.mark.parametrize("method,level_name", [
    ("debug",    "DEBUG"),
    ("info",     "INFO"),
    ("warning",  "WARNING"),
    ("error",    "ERROR"),
    ("critical", "CRITICAL"),
])
def test_method_produces_matching_level_name(method, level_name):
    stream = io.StringIO()
    logger = AgentLogger(
        "a", stream=stream, level=logging.DEBUG, clock=_fixed_clock,
    )
    getattr(logger, method)("e", "m")
    assert _read_lines(stream)[0]["level"] == level_name


def test_below_threshold_is_dropped():
    stream = io.StringIO()
    logger = AgentLogger(
        "a", stream=stream, level=logging.WARNING, clock=_fixed_clock,
    )
    logger.debug("e", "m")
    logger.info("e", "m")
    logger.warning("e", "m")
    logger.error("e", "m")
    records = _read_lines(stream)
    assert [r["level"] for r in records] == ["WARNING", "ERROR"]


def test_debug_level_emits_everything():
    stream = io.StringIO()
    logger = AgentLogger(
        "a", stream=stream, level=logging.DEBUG, clock=_fixed_clock,
    )
    for m in ("debug", "info", "warning", "error", "critical"):
        getattr(logger, m)("e", "m")
    assert len(_read_lines(stream)) == 5


# ---------- sinks: stream, file, both, neither ----------


def test_stream_only_does_not_create_file(tmp_path):
    stream = io.StringIO()
    logger = AgentLogger("a", stream=stream, clock=_fixed_clock)
    logger.info("e", "m")
    assert stream.getvalue()
    assert not any(tmp_path.iterdir())


def test_file_only_writes_to_file(tmp_path):
    log_file = tmp_path / "agent.log"
    logger = AgentLogger(
        "a", stream=None, log_file=log_file, clock=_fixed_clock,
    )
    logger.info("e", "m")
    assert log_file.exists()
    record = json.loads(log_file.read_text().strip())
    assert record["event_type"] == "e"


def test_both_sinks_receive_identical_lines(tmp_path):
    stream = io.StringIO()
    log_file = tmp_path / "agent.log"
    logger = AgentLogger(
        "a", stream=stream, log_file=log_file, clock=_fixed_clock,
    )
    logger.info("e", "m", k=1)
    stream_lines = stream.getvalue().splitlines()
    file_lines = log_file.read_text().splitlines()
    assert stream_lines == file_lines
    assert len(stream_lines) == 1


def test_neither_sink_is_silent_noop(tmp_path):
    logger = AgentLogger("a", stream=None, log_file=None, clock=_fixed_clock)
    # Should not raise, should not produce output anywhere.
    logger.info("e", "m")


def test_file_is_appended_not_truncated(tmp_path):
    log_file = tmp_path / "agent.log"
    logger = AgentLogger(
        "a", stream=None, log_file=log_file, clock=_fixed_clock,
    )
    logger.info("first", "one")
    logger.info("second", "two")
    lines = log_file.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["event_type"] == "first"
    assert json.loads(lines[1])["event_type"] == "second"


def test_file_parent_dir_created(tmp_path):
    nested = tmp_path / "a" / "b" / "c" / "agent.log"
    logger = AgentLogger(
        "a", stream=None, log_file=nested, clock=_fixed_clock,
    )
    logger.info("e", "m")
    assert nested.is_file()


def test_file_accepts_string_path(tmp_path):
    str_path = str(tmp_path / "agent.log")
    logger = AgentLogger("a", stream=None, log_file=str_path, clock=_fixed_clock)
    logger.info("e", "m")
    assert Path(str_path).is_file()


# ---------- clock ----------


def test_injected_clock_sets_timestamp():
    stream = io.StringIO()
    logger = AgentLogger("a", stream=stream, clock=_fixed_clock)
    logger.info("e", "m")
    assert _read_lines(stream)[0]["timestamp"] == FIXED_ISO


def test_default_clock_produces_iso_utc_ms_with_z():
    stream = io.StringIO()
    logger = AgentLogger("a", stream=stream)  # no clock override
    logger.info("e", "m")
    ts = _read_lines(stream)[0]["timestamp"]
    # Shape: YYYY-MM-DDTHH:MM:SS.mmmZ
    assert ts.endswith("Z")
    assert len(ts) == 24  # 4-2-2 T 2:2:2 . 3 Z
    # Must parse back into datetime:
    parsed = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%fZ")
    assert parsed is not None


# ---------- reserved kwargs ----------


@pytest.mark.parametrize("reserved", [
    "agent_id",
    "level",
    "timestamp",
    "event_type",
    "message",
    "data",
])
def test_reserved_kwargs_raise_value_error(reserved):
    logger = AgentLogger("a", stream=io.StringIO(), clock=_fixed_clock)
    with pytest.raises(ValueError) as e:
        logger.info("e", "m", **{reserved: "oops"})
    assert reserved in str(e.value)


# ---------- non-JSON-safe data ----------


def test_data_with_path_serializes_via_str():
    stream = io.StringIO()
    logger = AgentLogger("a", stream=stream, clock=_fixed_clock)
    logger.info("e", "m", where=Path("/tmp/x"))
    record = _read_lines(stream)[0]
    assert "/tmp/x" in record["data"]["where"]


def test_data_with_datetime_does_not_crash():
    stream = io.StringIO()
    logger = AgentLogger("a", stream=stream, clock=_fixed_clock)
    logger.info("e", "m", when=datetime(2026, 4, 22, tzinfo=timezone.utc))
    record = _read_lines(stream)[0]
    assert "2026-04-22" in record["data"]["when"]


# ---------- I/O failure tolerance ----------


class _RaisingStream:
    def write(self, _):
        raise OSError("disk on fire")

    def flush(self):
        pass


def test_stream_write_failure_is_swallowed():
    logger = AgentLogger("a", stream=_RaisingStream(), clock=_fixed_clock)
    # Must not propagate.
    logger.info("e", "m")


def test_file_write_failure_is_swallowed(tmp_path, monkeypatch):
    # Simulate a broken open() by pointing log_file at an unwritable path.
    log_file = tmp_path / "agent.log"
    logger = AgentLogger(
        "a", stream=None, log_file=log_file, clock=_fixed_clock,
    )

    def _broken_open(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("builtins.open", _broken_open)
    # Must not propagate even though the file can't be written.
    logger.info("e", "m")


# ---------- from_profile ----------


def _minimal_profile(id_: str = "agent_42") -> AgentProfile:
    return AgentProfile(
        id=id_,
        name="Test",
        age=30,
        traits="stable",
        backstory="Here.",
        initial_plan="Do.",
    )


def test_from_profile_takes_id():
    logger = AgentLogger.from_profile(
        _minimal_profile("p_1"), stream=io.StringIO(), clock=_fixed_clock,
    )
    assert logger.agent_id == "p_1"


def test_from_profile_passes_kwargs_through(tmp_path):
    log_file = tmp_path / "agent.log"
    logger = AgentLogger.from_profile(
        _minimal_profile(),
        stream=None,
        log_file=log_file,
        level=logging.DEBUG,
        clock=_fixed_clock,
    )
    logger.debug("e", "m")  # DEBUG should emit since level was lowered
    assert log_file.exists()
    record = json.loads(log_file.read_text().strip())
    assert record["level"] == "DEBUG"


def test_from_profile_agent_id_appears_in_output(tmp_path):
    stream = io.StringIO()
    logger = AgentLogger.from_profile(
        _minimal_profile("student_maya_001"),
        stream=stream,
        clock=_fixed_clock,
    )
    logger.info("agent.started", "Maya wakes up")
    record = _read_lines(stream)[0]
    assert record["agent_id"] == "student_maya_001"
