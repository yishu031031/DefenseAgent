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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from DefenseAgent.agent.config import AgentConfig
from DefenseAgent.config.profile import AgentProfile
from DefenseAgent.llm.llm import LLM
from DefenseAgent.memory import ContextCompressor, DefaultMemory
from DefenseAgent.ops import AgentLogger
from DefenseAgent.reflection import Reflector
from DefenseAgent.tools import ToolRegistry


@dataclass
class BuiltComponents:
    """Bag of fully-wired modules, ready to hand to a `BaseAgent` subclass."""
    profile: AgentProfile
    llm: LLM
    memory: DefaultMemory | None
    tools: ToolRegistry
    reflector: Reflector | None
    compactor: ContextCompressor | None
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
    if config.memory is not None:
        memory = config.memory
    elif config.use_memory:
        if config.memory_backend is not None:
            memory = DefaultMemory.from_kwargs(
                profile,
                backend=config.memory_backend,
                storage_path=config.storage_path,
            )
        else:
            memory = DefaultMemory(
                profile,
                dotenv_path=config.dotenv_path,
                load_env=False,
                storage_path=config.storage_path,
            )
    else:
        memory = None

    # Tools: prefer injection (caller manages everything); else build from
    # profile.skills + config.tools. MCP wiring still happens in
    # async_finish_setup, also gated on injection.
    if config.tools_registry is not None:
        tools = config.tools_registry
    else:
        tools = ToolRegistry()
        if config.use_tools:
            if profile.source_dir is not None:
                for skill_ref in profile.tools.skills:
                    tools.add_skill((profile.source_dir / skill_ref).resolve())
            for fn in config.tools:
                tools.tool(fn)

    if config.reflector is not None:
        reflector = config.reflector
    elif config.use_reflection and memory is not None:
        reflector = Reflector(memory, llm)
    else:
        reflector = None

    if config.compactor is not None:
        compactor = config.compactor
    elif config.use_compactor:
        compactor = ContextCompressor(
            profile,
            load_env=False,
            storage_path=str(config.storage_path) if config.storage_path else None,
            backend=config.memory_backend,
        )
    else:
        compactor = None

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
        compactor=compactor,
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

    When `config.tools_registry` was injected the caller manages the tool
    surface; we skip MCP server registration to avoid clobbering their setup.
    Likewise when `config.rag` is pre-injected we skip the LlamaIndexRAG build.
    """
    if config.use_tools and config.tools_registry is None and profile.tools.mcp:
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
