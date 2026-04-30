"""DefenseAgent — agent framework with mem0-backed memory, reflection, RAG, and tools.

The recommended top-level entry points:

    from DefenseAgent import create_agent
    from DefenseAgent.examples import EXAMPLE_PROFILE_PATH

    agent = create_agent(EXAMPLE_PROFILE_PATH)
    result = await agent.run("Hello")

Or, when you need full control over the config:

    from DefenseAgent import AgentConfig, ReActAgent
    from DefenseAgent.examples import EXAMPLE_PROFILE_PATH

    config = AgentConfig(profile=EXAMPLE_PROFILE_PATH, tools=[my_func])
    agent = ReActAgent(config)
"""
import logging as _logging
import os as _os


def _silence_ms_agent_default_log_file() -> None:
    """Suppress ms-agent's default `<cwd>/ms_agent.log` file. Upstream's `ms_agent.utils.logger` runs `logger = get_logger()` at module-import time, which unconditionally instantiates a `FileHandler` pointing at `<cwd>/ms_agent.log` — so the moment any ms-agent submodule is imported, an empty file appears in the user's project root. We pre-import that module ourselves, drop the FileHandler, delete the empty file, and patch `add_file_handler_if_needed` so subsequent `get_logger()` calls don't re-add it. User-supplied non-default `log_file` paths are still honoured."""
    try:
        import ms_agent.utils.logger as _mslog
    except ImportError:
        return  # ms-agent not installed; nothing to suppress

    default_log_path = _os.path.join(_os.getcwd(), "ms_agent.log")
    removed_paths: list[str] = []
    for h in list(_mslog.logger.handlers):
        if isinstance(h, _logging.FileHandler):
            removed_paths.append(getattr(h, "baseFilename", ""))
            try:
                h.close()
            except Exception:  # noqa: BLE001 — best-effort cleanup
                pass
            _mslog.logger.removeHandler(h)

    for path in removed_paths:
        if path and _os.path.exists(path):
            try:
                # Only delete if empty, so we never clobber user-customised log content.
                if _os.path.getsize(path) == 0:
                    _os.remove(path)
            except OSError:
                pass

    _orig_add = _mslog.add_file_handler_if_needed

    def _patched_add(logger, log_file, file_mode, log_level):
        # Skip only the default cwd/ms_agent.log path; honour explicit user overrides.
        if log_file is None or log_file == default_log_path:
            return
        return _orig_add(logger, log_file, file_mode, log_level)

    _mslog.add_file_handler_if_needed = _patched_add


_silence_ms_agent_default_log_file()


from DefenseAgent._factory import create_agent
from DefenseAgent.agent import (
    AgentConfig,
    AgentError,
    AgentResult,
    AgentStep,
    AgentStepLimitError,
    BaseAgent,
    PlanAndSolveAgent,
    ReActAgent,
    SimpleAgent,
)
from DefenseAgent.config import AgentProfile

__version__ = "0.2.0"

__all__ = [
    "create_agent",
    "AgentConfig",
    "AgentProfile",
    "BaseAgent",
    "SimpleAgent",
    "ReActAgent",
    "PlanAndSolveAgent",
    "AgentResult",
    "AgentStep",
    "AgentError",
    "AgentStepLimitError",
    "__version__",
]
