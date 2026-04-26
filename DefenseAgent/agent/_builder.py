"""Build the concrete component graph (LLM, memory, tools, ...) from an `AgentConfig`.

Splits cleanly into a sync phase (everything except MCP servers and
`LlamaIndexRAG`) and an async phase (`async_finish_setup`) that the agent
runs lazily on the first `run()` call. The sync phase is what makes
`agent = ReActAgent(config)` work without `await`.
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


def build_components_sync(config: AgentConfig) -> BuiltComponents:
    """Build everything that does not need `await`. MCP and RAG are deferred."""
    profile = config.resolved_profile()
    llm = LLM.from_env(dotenv_path=config.dotenv_path, load_env=config.load_env)

    if config.use_memory:
        memory = DefaultMemory(
            profile,
            dotenv_path=config.dotenv_path,
            load_env=False,
            storage_path=config.storage_path,
        )
    else:
        memory = None

    tools = ToolRegistry()
    if config.use_tools:
        if profile.source_dir is not None:
            for skill_ref in profile.tools.skills:
                tools.add_skill((profile.source_dir / skill_ref).resolve())
        for fn in config.tools:
            tools.tool(fn)

    if config.use_reflection and memory is not None:
        reflector = Reflector(memory, llm)
    else:
        reflector = None

    if config.use_compactor:
        compactor = ContextCompressor(
            profile,
            load_env=False,
            storage_path=str(config.storage_path) if config.storage_path else None,
        )
    else:
        compactor = None
    logger = _build_logger(profile, config.log_dir) if config.use_logger else None

    return BuiltComponents(
        profile=profile,
        llm=llm,
        memory=memory,
        tools=tools,
        reflector=reflector,
        compactor=compactor,
        logger=logger,
    )


async def async_finish_setup(
    config: AgentConfig,
    profile: AgentProfile,
    tools: ToolRegistry,
) -> Any | None:
    """Apply the parts that need `await`: register MCP servers and build RAG.

    Returns the (optional) RAG instance the caller should attach to the agent.
    Idempotency is the caller's job — invoke this at most once per agent.
    """
    if config.use_tools:
        for mcp_cfg in profile.tools.mcp:
            await tools.add_mcp(
                command=mcp_cfg.command,
                args=mcp_cfg.args,
                env=mcp_cfg.env,
                cwd=mcp_cfg.cwd,
            )

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
