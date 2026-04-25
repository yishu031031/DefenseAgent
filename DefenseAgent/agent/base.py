from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal

from DefenseAgent.config.profile import AgentProfile
from DefenseAgent.llm.llm import LLM
from DefenseAgent.llm.types import Message, TokenUsage, ToolCall
from DefenseAgent.memory import ContextCompressor, DefaultMemory
from DefenseAgent.memory._bridge import record_memory_type
from DefenseAgent.memory.base import Memory as MemoryTool
from DefenseAgent.ops import AgentLogger
from DefenseAgent.reflection import Reflector
from DefenseAgent.tools import ToolRegistry


StepKind = Literal["plan", "tool_call", "tool_result", "answer"]

_AgentToolHandler = Callable[[dict[str, Any]], Awaitable[str]]


MEMORY_RECALL_TOOL_NAME = "memory_recall"

_MEMORY_RECALL_TOOL_SPEC: dict[str, Any] = {
    "name": MEMORY_RECALL_TOOL_NAME,
    "description": (
        "Search this agent's memory for records relevant to a query. Returns "
        "up to top_k records with their content and memory_type. Call this "
        "any time you need information from earlier sessions, stored facts, "
        "preferences, or past trajectory steps."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Semantic search query — the more specific the better.",
            },
            "top_k": {
                "type": "integer",
                "description": "Maximum number of records to return (default 5, max 20).",
            },
        },
        "required": ["query"],
    },
}

_OUTCOME_MEMORY_TYPE = "outcome"
FAILURE_MEMORY_TYPE = "failure"


@dataclass
class AgentStep:
    """One event emitted during a run: a plan, a tool call, a tool result, or the final answer."""
    index: int
    kind: StepKind
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[Message] = field(default_factory=list)
    usage: TokenUsage | None = None


@dataclass
class AgentResult:
    """Outcome of one `agent.run(task)` call: the answer, the full step trace, and aggregate token usage."""
    task: str
    final_answer: str
    steps: list[AgentStep]
    usage: TokenUsage
    stopped_reason: Literal["answered", "max_steps"] = "answered"


class AgentError(Exception):
    """Base class for every error raised from the agent module."""


class AgentStepLimitError(AgentError):
    """Raised when a run hits max_steps without producing a final answer."""


class BaseAgent(ABC):
    """Abstract base for every concrete agent strategy; composes profile + LLM + memory + tools + reflector and defines the `run(task)` contract. Mirrors ms-agent's `Agent` base shape."""

    def __init__(
        self,
        profile: AgentProfile,
        *,
        llm: LLM,
        memory: DefaultMemory,
        tools: ToolRegistry,
        reflector: Reflector | None = None,
        logger: AgentLogger | None = None,
        compactor: ContextCompressor | None = None,
        memory_tools: list[MemoryTool] | None = None,
    ) -> None:
        """Compose the modules; build the per-step memory chain (default order: [memory, compactor], skipping None)."""
        self.profile = profile
        self.llm = llm
        self.memory = memory
        self.tools = tools
        self.reflector = reflector
        self.logger = logger
        self.compactor = compactor
        if memory_tools is not None:
            self.memory_tools = list(memory_tools)
        else:
            self.memory_tools = [memory] + ([compactor] if compactor is not None else [])
        self._agent_tools: dict[str, _AgentToolHandler] = {
            MEMORY_RECALL_TOOL_NAME: self._handle_memory_recall,
        }

    @classmethod
    async def from_profile(
        cls,
        profile: AgentProfile,
        *,
        log_dir: str | Path | None = None,
        dotenv_path: str | None = None,
        load_env: bool = True,
        **kwargs: Any,
    ) -> "BaseAgent":
        """Build a fully-wired agent from a profile + .env; extra kwargs forward to the subclass `__init__`."""
        llm = LLM.from_env(dotenv_path=dotenv_path, load_env=load_env)
        memory = DefaultMemory.from_profile(
            profile, dotenv_path=dotenv_path, load_env=False,
        )
        compactor = ContextCompressor(profile, load_env=False)
        tools = await ToolRegistry.from_profile(profile)
        reflector = Reflector(memory, llm)
        logger = _build_logger(profile, log_dir)
        return cls(
            profile,
            llm=llm,
            memory=memory,
            tools=tools,
            reflector=reflector,
            logger=logger,
            compactor=compactor,
            **kwargs,
        )

    @abstractmethod
    async def run(
        self,
        task: str,
        *,
        max_steps: int | None = None,
    ) -> AgentResult:
        """Execute one `task` end to end; must respect `max_steps` (defaults to `profile.cognitive.max_steps_per_cycle`)."""

    async def close(self) -> None:
        """Close underlying MCP clients (mem0 storage is auto-managed)."""
        await self.tools.close()

    async def __aenter__(self) -> "BaseAgent":
        """Enter: return self so `async with BaseAgent.from_profile(...) as agent:` works cleanly."""
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        """Exit: close every long-lived resource the agent opened."""
        await self.close()

    def _resolve_max_steps(self, override: int | None) -> int:
        """Pick the caller's override if given, else `profile.cognitive.max_steps_per_cycle`."""
        return override if override is not None else self.profile.cognitive.max_steps_per_cycle

    async def _recall_memories(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Return mem0 records relevant to `query`, capped at `top_k`; returns [] when top_k<=0 or memory raises."""
        if top_k <= 0:
            return []
        try:
            return self.memory.search_records(query, limit=top_k)
        except Exception as e:
            self._log("warn", "agent.memory_search_failed", str(e), query=query)
            return []

    def _identity_prompt(self) -> str:
        """Resolve the system identity prompt; falls back to the auto-built default if no template is configured or substitution fails."""
        template = self._resolve_prompt_template()
        if template is None:
            base = self._default_identity_prompt()
        else:
            try:
                base = template.format(**self._prompt_format_args())
            except (KeyError, IndexError, ValueError) as e:
                self._log(
                    "warn",
                    "agent.prompt_format_failed",
                    "prompt template substitution failed; falling back to default identity",
                    error=repr(e),
                )
                base = self._default_identity_prompt()
        extra = (self.profile.prompt.extra_instructions or "").strip()
        if extra:
            return f"{base}\n\n{extra}"
        return base

    def _default_identity_prompt(self) -> str:
        """Auto-built identity block used when no `prompt.system` / `prompt.path` is configured."""
        p = self.profile
        return (
            f"You are {p.name}, a {p.age}-year-old.\n"
            f"Traits: {p.traits}\n"
            f"Backstory: {p.backstory.strip()}\n"
            f"Today's plan: {p.initial_plan.strip()}"
        )

    def _resolve_prompt_template(self) -> str | None:
        """Pick the user's authored prompt — inline `system` first, then a file at `path` (relative to profile.source_dir). Returns None if neither is set."""
        prompt = self.profile.prompt
        if prompt.system and prompt.system.strip():
            return prompt.system
        if prompt.path and self.profile.source_dir is not None:
            file_path = (self.profile.source_dir / prompt.path).resolve()
            if file_path.is_file():
                return file_path.read_text(encoding="utf-8")
            self._log(
                "warn",
                "agent.prompt_file_missing",
                f"profile.prompt.path={prompt.path!r} did not resolve to a readable file",
            )
        return None

    def _prompt_format_args(self) -> dict[str, Any]:
        """Build the kwargs dict that fills `{placeholders}` inside a user-authored prompt template."""
        p = self.profile
        return {
            "id": p.id,
            "name": p.name,
            "age": p.age,
            "traits": p.traits,
            "backstory": p.backstory.strip(),
            "initial_plan": p.initial_plan.strip(),
        }

    def _memory_block(self, records: list[dict[str, Any]]) -> str:
        """Render mem0 records as a bullet list; returns "" when records is empty."""
        if not records:
            return ""
        lines = [
            f"- [{record_memory_type(r) or 'memory'}] {r.get('memory', '')}"
            for r in records
        ]
        return "Relevant memories:\n" + "\n".join(lines)

    async def _persist_outcome(
        self,
        task: str,
        answer: str,
        *,
        memory_type: str = _OUTCOME_MEMORY_TYPE,
    ) -> None:
        """Append the Q→A pair to mem0 tagged with `memory_type` (default='outcome', failures override to 'failure')."""
        message = Message(role="user", content=f"Q: {task}\nA: {answer}")
        try:
            await self.memory.add([message], memory_type=memory_type)
        except Exception as e:
            self._log("warn", "agent.persist_outcome_failed", str(e))

    def _log(self, level: str, event_type: str, message: str, **data: Any) -> None:
        """Emit a structured log event; no-op when no logger is wired. Accepts 'warn' as an alias for 'warning'."""
        if self.logger is None:
            return
        fn = getattr(self.logger, "warning" if level == "warn" else level, None)
        if fn is not None:
            fn(event_type, message, **data)

    def _combined_tool_specs(self) -> list[dict[str, Any]] | None:
        """Return user tool specs followed by Agent-owned tool specs; returns None only if both are empty."""
        user_specs = self.tools.spec()
        builtin_specs: list[dict[str, Any]] = []
        if MEMORY_RECALL_TOOL_NAME in self._agent_tools:
            builtin_specs.append(_MEMORY_RECALL_TOOL_SPEC)
        combined = user_specs + builtin_specs
        return combined or None

    async def _dispatch_tool_calls(
        self, tool_calls: list[ToolCall]
    ) -> list[Message]:
        """Route Agent-owned calls to built-in handlers; forward everything else to `self.tools.execute`; preserves input order."""
        results: list[Message | None] = [None] * len(tool_calls)
        user_calls_with_index: list[tuple[int, ToolCall]] = []

        for i, tc in enumerate(tool_calls):
            handler = self._agent_tools.get(tc.name)
            if handler is None:
                user_calls_with_index.append((i, tc))
                continue
            try:
                content = await handler(tc.arguments)
            except Exception as e:
                content = f"{type(e).__name__}: {e}"
            results[i] = Message(
                role="tool",
                content=content,
                tool_call_id=tc.id,
                name=tc.name,
            )

        if user_calls_with_index:
            user_calls = [tc for _, tc in user_calls_with_index]
            user_results = await self.tools.execute(user_calls)
            for (i, _), msg in zip(user_calls_with_index, user_results):
                results[i] = msg

        return [r for r in results if r is not None]

    async def _handle_memory_recall(self, arguments: dict[str, Any]) -> str:
        """Agent-owned handler for the `memory_recall` tool; renders mem0 hits as a bullet list or a diagnostic string."""
        raw_query = arguments.get("query", "")
        query = raw_query.strip() if isinstance(raw_query, str) else ""
        if not query:
            return "(memory_recall called with empty query)"

        raw_top_k = arguments.get("top_k", 5)
        try:
            top_k = int(raw_top_k)
        except (TypeError, ValueError):
            top_k = 5
        top_k = max(1, min(top_k, 20))

        try:
            hits = self.memory.search_records(query, limit=top_k)
        except Exception as e:
            return f"(memory_recall failed: {type(e).__name__}: {e})"
        if not hits:
            return f"(no memories matched query={query!r})"
        return "\n".join(
            f"- [{record_memory_type(h) or 'memory'}] {h.get('memory', '')}"
            for h in hits
        )

    async def _condense_memory(self, messages: list[Message]) -> list[Message]:
        """Pipeline `messages` through every memory tool in order — same shape as ms-agent's LLMAgent.condense_memory; injection + compaction live in this single hop."""
        for tool in self.memory_tools:
            try:
                messages = await tool.run(messages)
            except Exception as e:
                self._log(
                    "warn",
                    "agent.condense_memory_failed",
                    f"memory tool {type(tool).__name__} failed; skipping",
                    error=repr(e),
                )
        return messages

    async def _run_reflection_safely(self) -> None:
        """Trigger threshold-gated reflection; never raises — failures must not mask run outcome."""
        if self.reflector is None:
            return
        try:
            await self.reflector.check_and_reflect()
        except Exception as e:
            self._log(
                "warn",
                "agent.reflect_failed",
                "reflection raised after run; swallowed",
                error=repr(e),
            )


def add_usage(a: TokenUsage, b: TokenUsage) -> TokenUsage:
    """Field-wise sum of two TokenUsage records — shared by both strategies to aggregate per-call totals."""
    return TokenUsage(
        prompt_tokens=a.prompt_tokens + b.prompt_tokens,
        completion_tokens=a.completion_tokens + b.completion_tokens,
        total_tokens=a.total_tokens + b.total_tokens,
    )


def truncate(text: str, max_len: int) -> str:
    """Return `text` unchanged if short enough; otherwise cut to `max_len` characters with a trailing `...`."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


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
