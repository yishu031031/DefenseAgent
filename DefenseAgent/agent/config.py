"""Top-level config object for any DefenseAgent.

`AgentConfig` is the single argument every agent strategy accepts. It bundles
the agent's identity (the YAML profile or a pre-built `AgentProfile`) with
on/off switches for every optional subsystem (tools, memory, reflection, RAG,
context compressor, logger) and the per-strategy knobs (`memory_recall_top_k`,
`save_outcome`, `reflect_after_run`, ...).

Typical use:

    from DefenseAgent import AgentConfig, ReActAgent

    config = AgentConfig(
        profile="DefenseAgent/examples/example_agent/profile.yaml",
        tools=[calculator, web_search],   # plain Python functions
        use_memory=True,
        use_reflection=True,
        use_rag=True,
    )
    agent = ReActAgent(config)
    result = await agent.run("Hello")

Anything not configurable from `AgentConfig` (custom `LLM` adapters, MagicMock
test stubs, ...) can still be wired by passing the components as keyword args
to the agent constructor — see each agent's docstring for the legacy shape.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from DefenseAgent.config.profile import AgentProfile
from DefenseAgent.llm.llm import LLM
from DefenseAgent.memory import (
    ContextCompressor,
    Mem0Memory,
    MemoryBackendConfig,
    MemoryOrchestrator,
)
from DefenseAgent.ops import AgentLogger
from DefenseAgent.reflection import Reflector
from DefenseAgent.tools import ToolRegistry


@dataclass
class AgentConfig:
    """One config object that captures every choice an agent needs.

    Attributes
    ----------
    profile:
        The agent's identity. Either an already-loaded `AgentProfile` or a
        path (str/Path) to a YAML file that will be loaded via
        `AgentProfile.from_yaml`.

    dotenv_path, load_env:
        Where to load environment variables from. `load_env=True` (default)
        reads `.env` from the project root; `dotenv_path` overrides the path.
        Set `load_env=False` if env vars are already present in the process.

    use_tools:
        Register user tools (Python functions in `tools=`, skill bundles in
        `profile.tools.skills`, MCP servers in `profile.tools.mcp`). When
        False, only built-in agent tools (memory_recall, rag_search) may
        appear, and only if their respective subsystems are also enabled.

    use_memory:
        Build mem0-backed `Mem0Memory`, enable the `memory_recall`
        built-in tool, and persist outcomes/trajectories. When False, the
        agent runs stateless — no recall, no persistence, no reflection.

    use_reflection:
        Build a `Reflector` and run it after each `run()` (subject to its
        own `reflection_threshold`). Requires `use_memory=True`; ignored
        otherwise.

    use_rag:
        Tri-state. `True` forces RAG on (will raise if `profile.rag` is
        misconfigured); `False` forces RAG off; `None` (default) follows
        `profile.rag.enabled`. When on, the `rag_search` built-in tool is
        registered and `LlamaIndexRAG` is built lazily on the first `run()`.

    use_compressor:
        Build a `ContextCompressor` and chain it after memory in the
        condense_memory pipeline.

    use_logger:
        Build an `AgentLogger` writing to `<log_dir>/<profile.id>.log`.

    tools:
        Extra Python callables to register on the agent's `ToolRegistry`.
        Each function's signature + docstring become its JSON-schema spec
        (the same shape `ToolRegistry.tool` accepts).

    log_dir:
        Where to put the agent's log file. Defaults to `<profile.source_dir>/logs`
        when the profile was loaded from disk; required otherwise.

    storage_path:
        Memory storage directory. Defaults to `profile.memory.storage_path`,
        then `<profile.source_dir>/memory/`. Required when the profile was
        built in-memory and no `profile.memory.storage_path` is set.

    memory_recall_top_k:
        Default `top_k` for the agent-owned `memory_recall` tool when the
        LLM does not specify one. Set to 0 to suppress recall entirely.

    save_outcome:
        After each `run()`, write a (Q → A) pair to memory tagged
        `memory_type='outcome'` (or `'failure'` on error paths). Auto-disabled
        when `use_memory=False`.

    save_trajectory:
        Per ReAct tool turn, write a one-line summary of every (call → result)
        tagged `memory_type='trajectory'`. Auto-disabled when `use_memory=False`.

    reflect_after_run:
        Call `Reflector.maybe_reflect` after each `run()`. Auto-disabled
        when `use_reflection=False` or `use_memory=False`.

    extra_instructions:
        Free-form text appended to the agent's system prompt — tone, output
        format, hard rules, etc.

    max_substeps_per_step:
        `PlanAndSolveAgent` only — per-plan-step tool-call budget.

    max_steps:
        Default `max_steps` for `agent.run(task)` when the caller does not
        pass one. `None` falls back to `profile.cognitive.max_steps_per_cycle`.
    """

    profile: AgentProfile | str | Path

    # env loading
    dotenv_path: str | None = None
    load_env: bool = True

    # subsystem toggles
    use_tools: bool = True
    use_memory: bool = True
    use_reflection: bool = True
    use_rag: bool | None = None
    use_compressor: bool = True
    use_logger: bool = True

    # tool wiring
    tools: list[Callable[..., Any]] = field(default_factory=list)
    log_dir: str | Path | None = None
    storage_path: str | Path | None = None

    # behavior knobs
    memory_recall_top_k: int = 5
    save_outcome: bool = True
    save_trajectory: bool = True
    reflect_after_run: bool = True
    extra_instructions: str | None = None

    # PlanAndSolveAgent
    max_substeps_per_step: int = 3

    # default run cap
    max_steps: int | None = None

    # ---- Programmatic component injection ----
    # When any of these is given, the builder uses it as-is and skips the
    # env-driven construction path for that component. Lets SDK callers,
    # tests, and multi-LLM apps bypass .env entirely.
    llm: LLM | None = None
    # Accepts either a bare Mem0Memory (legacy) or a fully-built
    # MemoryOrchestrator (new). The builder wraps a Mem0Memory in an
    # orchestrator transparently, so downstream code always sees the facade.
    memory: Mem0Memory | MemoryOrchestrator | None = None
    tool_registry: ToolRegistry | None = None
    logger: AgentLogger | None = None
    reflector: Reflector | None = None       # bypass Reflector(memory, llm) construction
    compressor: ContextCompressor | None = None  # bypass auto-build of compressor
    rag: Any | None = None                   # pre-built LlamaIndexRAG (or compatible duck-type)

    # Programmatic mem0 backend — controls the *internal* LLM/embedder mem0
    # uses for fact extraction (separate from the agent's chat LLM above).
    # Only consulted when `memory` is None and `use_memory=True`; pure-code
    # construction of memory without ever touching .env.
    memory_backend: MemoryBackendConfig | None = None

    def resolved_profile(self) -> AgentProfile:
        """Return the `AgentProfile` instance, loading from YAML on demand."""
        if isinstance(self.profile, AgentProfile):
            return self.profile
        return AgentProfile.from_yaml(self.profile)
