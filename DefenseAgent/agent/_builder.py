"""Build the concrete component graph (LLM, memory, tools, ...) from an `AgentConfig`.

Splits cleanly into a sync phase (everything except MCP servers and
`LlamaIndexRAG`) and an async phase (`async_finish_setup`) that the agent
runs lazily on the first `run()` call. The sync phase is what makes
`agent = ReActAgent(config)` work without `await`.

Resolution priority for every component: an explicit `config.<name>` injection
wins; otherwise we fall back to building from `profile` + .env. This lets SDK
callers and tests construct components programmatically while preserving the
zero-config `AgentConfig(profile="...yaml")` path for local development.
"""
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from DefenseAgent.agent.config import AgentConfig
from DefenseAgent.config.profile import AgentProfile
from DefenseAgent.llm.llm import LLM
from DefenseAgent.memory import ContextCompressor, Mem0Memory, MemoryOrchestrator
from DefenseAgent.memory.working import WorkingMemory
from DefenseAgent.ops import AgentLogger
from DefenseAgent.reflection import Reflector
from DefenseAgent.tools import ToolRegistry
from DefenseAgent.tools.types import ToolRegistrationError


@dataclass
class BuiltComponents:
    """Bag of fully-wired modules, ready to hand to a `BaseAgent` subclass."""
    profile: AgentProfile
    llm: LLM
    # Tier-aware facade (P2). Wraps a Mem0Memory persistent backend; agents
    # interact through the orchestrator's `recall()` / `add_*()` API rather
    # than reaching into mem0 directly. Callers may still inject a bare
    # Mem0Memory via AgentConfig — the builder wraps it transparently.
    memory: MemoryOrchestrator | None
    tools: ToolRegistry
    reflector: Reflector | None
    compressor: ContextCompressor | None
    logger: AgentLogger | None
    rag: Any | None = None


def build_components_sync(config: AgentConfig) -> BuiltComponents:
    """Build everything that does not need `await`. MCP and RAG are deferred."""
    profile = config.resolved_profile()

    # LLM: prefer injection, else profile-then-env (per-field; profile wins where it's set).
    if config.llm is not None:
        llm = config.llm
    else:
        llm = LLM.from_profile(
            profile, dotenv_path=config.dotenv_path, load_env=config.load_env,
        )

    # Memory: three-tier priority — fully built > pure-code backend > env.
    # Whatever the source, we wrap a bare Mem0Memory in a MemoryOrchestrator
    # so downstream code (BaseAgent, Reflector) only sees the tier-aware
    # facade with all four lifecycle tiers reachable. An injected
    # MemoryOrchestrator passes through unchanged.
    if config.memory is not None:
        if isinstance(config.memory, MemoryOrchestrator):
            memory = config.memory
        else:
            memory = MemoryOrchestrator(
                profile,
                config.memory,
                working=WorkingMemory.from_profile(profile),
            )
    elif config.use_memory:
        if config.memory_backend is not None:
            persistent = Mem0Memory.create(
                profile,
                backend=config.memory_backend,
                storage_path=config.storage_path,
            )
        else:
            persistent = Mem0Memory(
                profile,
                dotenv_path=config.dotenv_path,
                load_env=False,
                storage_path=config.storage_path,
            )
        memory = MemoryOrchestrator(
            profile,
            persistent,
            working=WorkingMemory.from_profile(profile),
        )
    else:
        memory = None

    # Tools: prefer injection (caller manages everything); else build from
    # profile.skills + config.tools. MCP wiring still happens in
    # async_finish_setup, also gated on injection.
    if config.tool_registry is not None:
        tools = config.tool_registry
    else:
        tools = ToolRegistry()
        if config.use_tools:
            if profile.source_dir is not None:
                for skill_ref in profile.tools.skills:
                    tools.add_skill((profile.source_dir / skill_ref).resolve())
            _autoload_evolution_skills(tools, profile)
            for entry_point in profile.tools.python:
                tools.tool(_resolve_python_entry_point(entry_point, profile.source_dir))
            for fn in config.tools:
                tools.tool(fn)

    if config.reflector is not None:
        reflector = config.reflector
    elif config.use_reflection and memory is not None:
        reflector = Reflector(memory, llm)
    else:
        reflector = None

    if config.compressor is not None:
        compressor = config.compressor
    elif config.use_compressor:
        compressor = ContextCompressor(
            profile,
            load_env=False,
            storage_path=str(config.storage_path) if config.storage_path else None,
            backend=config.memory_backend,
        )
    else:
        compressor = None

    # Logger: prefer injection, else build at <log_dir>/<profile.id>.log when
    # use_logger=True.
    if config.logger is not None:
        logger = config.logger
    elif config.use_logger:
        logger = _build_logger(profile, config.log_dir)
    else:
        logger = None

    return BuiltComponents(
        profile=profile,
        llm=llm,
        memory=memory,
        tools=tools,
        reflector=reflector,
        compressor=compressor,
        logger=logger,
        rag=config.rag,
    )


async def async_finish_setup(
    config: AgentConfig,
    profile: AgentProfile,
    tools: ToolRegistry,
) -> Any | None:
    """Apply the parts that need `await`: register MCP servers and build RAG.

    Returns the (optional) RAG instance the caller should attach to the agent.
    Idempotency is the caller's job — invoke this at most once per agent.

    When `config.tool_registry` was injected the caller manages the tool
    surface; we skip MCP server registration to avoid clobbering their setup.
    Likewise when `config.rag` is pre-injected we skip the LlamaIndexRAG build.
    """
    if config.use_tools and config.tool_registry is None and profile.tools.mcp:
        await tools.add_mcp_servers(list(profile.tools.mcp))

    if config.rag is not None:
        return config.rag

    use_rag = config.use_rag if config.use_rag is not None else profile.rag.enabled
    if not use_rag:
        return None

    from DefenseAgent.rag.llama_index_rag import LlamaIndexRAG
    return await LlamaIndexRAG.from_profile(
        profile, load_env=False, dotenv_path=config.dotenv_path,
    )


def _resolve_python_entry_point(
    entry_point: str,
    base_dir: Path | None = None,
) -> Callable[..., Any]:
    """Resolve a `profile.tools.python` entry to a callable. Two accepted forms:

      * `'module.dotted.path:function_name'` — resolved via `importlib.import_module`. Module must be on `sys.path`.
      * `'relative/file.py:function_name'`  — resolved via `importlib.util.spec_from_file_location`, with the path resolved relative to `base_dir` (typically `profile.source_dir`). Best for tools bundled inside an agent's directory.

    Raises `ToolRegistrationError` on malformed strings, import failures, missing attributes, and non-callable resolutions. Both forms execute the target module's top level — only list entry points you trust.
    """
    if not isinstance(entry_point, str) or ":" not in entry_point:
        raise ToolRegistrationError(
            f"profile.tools.python entry must be 'module.path:func' or "
            f"'relative/file.py:func', got {entry_point!r}"
        )
    target, _, func_name = entry_point.rpartition(":")
    if not target or not func_name:
        raise ToolRegistrationError(
            f"profile.tools.python entry must be 'module.path:func' or "
            f"'relative/file.py:func', got {entry_point!r}"
        )
    if target.endswith(".py") or "/" in target or "\\" in target:
        module = _load_module_from_file(target, base_dir, entry_point)
    else:
        try:
            module = importlib.import_module(target)
        except ImportError as e:
            raise ToolRegistrationError(
                f"could not import module {target!r} for tool {entry_point!r}: {e}"
            ) from e
    fn = getattr(module, func_name, None)
    if fn is None:
        raise ToolRegistrationError(
            f"module {target!r} has no attribute {func_name!r}"
        )
    if not callable(fn):
        raise ToolRegistrationError(
            f"{entry_point!r} resolved to {type(fn).__name__}, not a callable"
        )
    return fn


def _load_module_from_file(
    rel_path: str,
    base_dir: Path | None,
    entry_point: str,
) -> Any:
    """Load a Python file as an anonymous module via `importlib.util.spec_from_file_location`. The path is resolved against `base_dir` when relative; absolute paths are used as-is. Raises ToolRegistrationError on missing files or import-time failures."""
    import importlib.util

    candidate = Path(rel_path)
    if not candidate.is_absolute():
        if base_dir is None:
            raise ToolRegistrationError(
                f"profile.tools.python entry {entry_point!r} uses a relative path "
                f"but the profile has no source_dir to resolve it against"
            )
        candidate = (base_dir / candidate).resolve()
    if not candidate.is_file():
        raise ToolRegistrationError(
            f"profile.tools.python file not found for {entry_point!r}: {candidate}"
        )
    module_name = f"_python_tool_{abs(hash(str(candidate)))}"
    spec = importlib.util.spec_from_file_location(module_name, candidate)
    if spec is None or spec.loader is None:
        raise ToolRegistrationError(
            f"could not build import spec for {entry_point!r} at {candidate}"
        )
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise ToolRegistrationError(
            f"failed to execute {entry_point!r} at {candidate}: {e}"
        ) from e
    return module


def _autoload_evolution_skills(
    tools: ToolRegistry,
    profile: AgentProfile,
) -> None:
    """Discover and register the framework-builtin / user / project skill layers configured by `profile.evolution`. Layer order is builtin → user → project; per-skill load failures are logged and skipped (never abort agent startup); missing layer directories are silent. Tool name collisions inside `ToolRegistry.add_skill` are also skipped — earlier-registered tools win, so a project-level skill of the same name as the loader's first hit is shadowed by the registry's idempotency rule (the loader-level "later wins" override is achieved here by registering the project layer last via the layer ordering, since the registry skips dupes silently)."""
    from DefenseAgent.skills import (
        SkillLoader,
        discover_skill_dirs,
    )

    dirs = discover_skill_dirs(profile.evolution)
    if not dirs:
        return
    # Reverse so that project (last in `dirs`) is registered first and wins
    # the registry's idempotent name-collision check, matching M3 precedence:
    # project > user > builtin.
    for skill_dir in reversed(dirs):
        if not skill_dir.exists():
            continue
        loader = SkillLoader()
        try:
            loader.load_dirs_tolerant([skill_dir])
        except Exception:  # noqa: BLE001 — never block agent startup
            continue
        layer_tools = loader.to_tools()
        for tool in layer_tools:
            if tool.name in tools._tools:
                continue
            tools.register(tool)


def _build_logger(
    profile: AgentProfile,
    log_dir: str | Path | None,
) -> AgentLogger | None:
    """Build an AgentLogger at `<log_dir>/<profile.id>.log`; returns None when no log dir can be resolved."""
    if log_dir is not None:
        resolved = Path(log_dir)
    elif profile.source_dir is not None:
        resolved = profile.source_dir / "logs"
    else:
        return None
    resolved.mkdir(parents=True, exist_ok=True)
    return AgentLogger.from_profile(
        profile,
        stream=None,
        log_file=resolved / f"{profile.id}.log",
    )
