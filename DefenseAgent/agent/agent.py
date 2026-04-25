from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal

from DefenseAgent.config.profile import AgentProfile
from DefenseAgent.llm.llm import LLM
from DefenseAgent.llm.types import Message, TokenUsage, ToolCall
from DefenseAgent.memory import Memory, ScoredMemory
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
        "up to top_k scored records with their kind, importance, and content. "
        "Call this any time you need information from earlier sessions, stored "
        "facts, preferences, or past trajectory steps — retrieval is not "
        "limited to the single upfront prime in the system prompt."
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


class Agent(ABC):
    """Module 7's unified facade; composes the 6 earlier modules into a runnable agent. Concrete subclasses own the loop shape."""

    def __init__(
        self,
        profile: AgentProfile,
        *,
        llm: LLM,
        memory: Memory,
        tools: ToolRegistry,
        reflector: Reflector | None = None,
        logger: AgentLogger | None = None,
    ) -> None:
        """Hold the composed modules and register Agent-owned built-in tools (e.g., memory_recall)."""
        self.profile = profile
        self.llm = llm
        self.memory = memory
        self.tools = tools
        self.reflector = reflector
        self.logger = logger
        self._agent_tools: dict[str, _AgentToolHandler] = {}
        self._register_builtin_tools()

    def _register_builtin_tools(self) -> None:
        """Register Agent-owned tools (currently: memory_recall). Subclasses can override to add more."""
        self._agent_tools[MEMORY_RECALL_TOOL_NAME] = self._handle_memory_recall

    @classmethod
    async def from_profile(
        cls,
        profile: AgentProfile,
        *,
        persist_memory: bool = True,
        log_dir: str | Path | None = None,
        dotenv_path: str | None = None,
        load_env: bool = True,
        **kwargs: Any,
    ) -> "Agent":
        """Build a fully-wired agent from a profile + .env; extra kwargs forward to the subclass `__init__`."""
        llm = LLM.from_env(dotenv_path=dotenv_path, load_env=load_env)
        if persist_memory and profile.source_dir is not None:
            memory = Memory.from_profile(
                profile, dotenv_path=dotenv_path, load_env=False,
            )
        else:
            memory = Memory.from_env(
                profile, dotenv_path=dotenv_path, load_env=False,
            )
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
        """Close underlying MCP clients and the Memory SQLite connection."""
        await self.tools.close()
        self.memory.close()

    async def __aenter__(self) -> "Agent":
        """Enter: return self so `async with Agent.from_profile(...) as agent:` works cleanly."""
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        """Exit: close every long-lived resource the agent opened."""
        await self.close()

    # --- shared helpers subclasses use ---

    def _resolve_max_steps(self, override: int | None) -> int:
        """Pick the caller's override if given, else `profile.cognitive.max_steps_per_cycle`."""
        if override is not None:
            return override
        return self.profile.cognitive.max_steps_per_cycle

    async def _recall_memories(self, query: str, top_k: int) -> list[ScoredMemory]:
        """Return the top-k memories for `query`, or an empty list if top_k<=0."""
        if top_k <= 0:
            return []
        return await self.memory.recall(query, top_k=top_k)

    def _identity_prompt(self) -> str:
        """Render the identity block (name, age, traits, backstory, initial plan) used by every agent type."""
        p = self.profile
        return (
            f"You are {p.name}, a {p.age}-year-old.\n"
            f"Traits: {p.traits}\n"
            f"Backstory: {p.backstory.strip()}\n"
            f"Today's plan: {p.initial_plan.strip()}"
        )

    def _memory_block(self, memories: list[ScoredMemory]) -> str:
        """Render recalled memories as a bullet list; returns "" when `memories` is empty."""
        if not memories:
            return ""
        lines = [f"- [{m.record.kind}] {m.record.content}" for m in memories]
        return "Relevant memories:\n" + "\n".join(lines)

    async def _persist_outcome(
        self,
        task: str,
        answer: str,
        *,
        importance: float = 5.0,
    ) -> None:
        """Append the Q→A pair as an observation; `importance` defaults to 5.0 but failures bump it to 6.0."""
        await self.memory.remember(
            f"Q: {task}\nA: {answer}",
            kind="observation",
            importance=importance,
        )

    async def _maybe_reflect(self) -> None:
        """Trigger threshold-gated reflection; no-op when no Reflector is wired."""
        if self.reflector is not None:
            await self.reflector.check_and_reflect()

    def _log(self, level: str, event_type: str, message: str, **data: Any) -> None:
        """Emit a structured log event at the given level; no-op when no logger is wired."""
        if self.logger is None:
            return
        fn = getattr(self.logger, level)
        fn(event_type, message, **data)

    # --- unified tool spec + dispatch (user tools + agent built-ins) ---

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
        """Agent-owned handler for the `memory_recall` tool; renders hits as a simple bullet list or a diagnostic string."""
        raw_query = arguments.get("query", "")
        query = raw_query.strip() if isinstance(raw_query, str) else ""
        if not query:
            return "(memory_recall called with empty query)"

        raw_top_k = arguments.get("top_k", 5)
        try:
            top_k = int(raw_top_k)
        except (TypeError, ValueError):
            top_k = 5
        if top_k < 1:
            top_k = 1
        elif top_k > 20:
            top_k = 20

        hits = await self.memory.recall(query, top_k=top_k)
        if not hits:
            return f"(no memories matched query={query!r})"

        lines: list[str] = []
        for hit in hits:
            record = hit.record
            lines.append(
                f"- [{record.kind} imp={record.importance:.1f}] {record.content}"
            )
        return "\n".join(lines)

    async def _run_reflection_safely(self) -> None:
        """Trigger reflection and log any failure; never raise — reflection errors must not mask the run outcome."""
        if self.reflector is None:
            return
        try:
            await self._maybe_reflect()
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
    """Build an AgentLogger at `<log_dir>/<profile.id>.log`; return None if log_dir is blank or unresolvable."""
    if log_dir is None:
        if profile.source_dir is None:
            return None
        resolved = profile.source_dir / "logs"
    else:
        resolved = Path(log_dir)
    resolved.mkdir(parents=True, exist_ok=True)
    return AgentLogger.from_profile(
        profile,
        stream=None,
        log_file=resolved / f"{profile.id}.log",
    )
