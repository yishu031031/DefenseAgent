from __future__ import annotations

import json
import logging
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, TextIO

if TYPE_CHECKING:
    from DefenseAgent.config.profile import AgentProfile


_RESERVED_RECORD_KEYS = frozenset(
    {"agent_id", "level", "timestamp", "event_type", "message", "data"}
)


def _default_clock() -> datetime:
    """Return the current UTC time."""
    return datetime.now(timezone.utc)


def _format_timestamp(dt: datetime) -> str:
    """Render `dt` as ISO-8601 UTC with millisecond precision and a trailing 'Z'."""
    ms = dt.microsecond // 1000
    return dt.strftime("%Y-%m-%dT%H:%M:%S") + f".{ms:03d}Z"


class AgentLogger:
    """Module 3's unified facade: a per-agent JSON-lines logger that never raises on I/O failures."""

    def __init__(
        self,
        agent_id: str,
        *,
        log_file: str | Path | None = None,
        stream: TextIO | None = sys.stdout,
        level: int = logging.INFO,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """Bind the agent id and configure optional file / stream sinks, minimum level, and clock injection."""
        self.agent_id = agent_id
        self.level = level
        self.stream = stream
        self.log_file = Path(log_file) if log_file is not None else None
        self._clock = clock or _default_clock
        self._lock = threading.Lock()

        if self.log_file is not None:
            try:
                self.log_file.parent.mkdir(parents=True, exist_ok=True)
            except OSError:
                pass

    @classmethod
    def from_profile(cls, profile: "AgentProfile", **kwargs: Any) -> "AgentLogger":
        """Build an AgentLogger whose `agent_id` is sourced from `profile.id`; other kwargs pass through."""
        return cls(agent_id=profile.id, **kwargs)

    def debug(self, event_type: str, message: str, /, **data: Any) -> None:
        """Emit a DEBUG-level record (dropped if level above DEBUG)."""
        self.log(logging.DEBUG, event_type, message, **data)

    def info(self, event_type: str, message: str, /, **data: Any) -> None:
        """Emit an INFO-level record."""
        self.log(logging.INFO, event_type, message, **data)

    def warning(self, event_type: str, message: str, /, **data: Any) -> None:
        """Emit a WARNING-level record."""
        self.log(logging.WARNING, event_type, message, **data)

    def error(self, event_type: str, message: str, /, **data: Any) -> None:
        """Emit an ERROR-level record."""
        self.log(logging.ERROR, event_type, message, **data)

    def critical(self, event_type: str, message: str, /, **data: Any) -> None:
        """Emit a CRITICAL-level record."""
        self.log(logging.CRITICAL, event_type, message, **data)

    def log(
        self,
        level: int,
        event_type: str,
        message: str,
        /,
        **data: Any,
    ) -> None:
        """Emit one JSON line to both sinks; raises ValueError only on reserved-key collisions in `data`."""
        if level < self.level:
            return

        collision = _RESERVED_RECORD_KEYS.intersection(data.keys())
        if collision:
            raise ValueError(
                f"data kwargs may not reuse reserved record keys: "
                f"{sorted(collision)}. Pick a different name or nest them."
            )

        record = {
            "timestamp": _format_timestamp(self._clock()),
            "agent_id": self.agent_id,
            "level": logging.getLevelName(level),
            "event_type": event_type,
            "message": message,
            "data": dict(data),
        }
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"

        with self._lock:
            self._write_stream(line)
            self._write_file(line)

    def _write_stream(self, line: str) -> None:
        """Write `line` to the configured stream, swallowing any I/O error."""
        if self.stream is None:
            return
        try:
            self.stream.write(line)
            self.stream.flush()
        except Exception:
            pass

    def _write_file(self, line: str) -> None:
        """Append `line` to the configured file, swallowing any I/O error."""
        if self.log_file is None:
            return
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            pass
