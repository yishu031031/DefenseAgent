import base64
import mimetypes
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal

from DefenseAgent.agent.config import AgentConfig
from DefenseAgent.config.profile import AgentProfile
from DefenseAgent.llm.llm import LLM
from DefenseAgent.llm.types import Message, TokenUsage, ToolCall
from DefenseAgent.memory import ContextCompressor, Mem0Memory, MemoryOrchestrator
from DefenseAgent.memory._bridge import record_memory_type
from DefenseAgent.memory.base import Memory as MemoryTool
from DefenseAgent.ops import AgentLogger
from DefenseAgent.reflection import Reflector
from DefenseAgent.tools import ToolRegistry


StepKind = Literal["plan", "tool_call", "tool_result", "answer"]

_AgentToolHandler = Callable[[dict[str, Any]], Awaitable[str]]


MEMORY_RECALL_TOOL_NAME = "memory_recall"
RAG_SEARCH_TOOL_NAME = "rag_search"
RAG_GET_RESOURCE_TOOL_NAME = "rag_get_resource"

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

_RAG_SEARCH_TOOL_SPEC: dict[str, Any] = {
    "name": RAG_SEARCH_TOOL_NAME,
    "description": (
        "Search this agent's static reference knowledge base (textbooks, "
        "manuals, character lore, world docs) for passages relevant to a "
        "query. Distinct from `memory_recall` which searches dynamic "
        "experiential memory. Use this when you need facts grounded in "
        "documents you've been given.\n\n"
        "Hits may include inline `<resource_info>RID</resource_info>` markers "
        "and a follow-up `• resource [RID] (kind) \"caption\"` listing for "
        "embedded images / tables. Pass an RID to `rag_get_resource` to "
        "fetch the full content of one resource."
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
                "description": "Maximum number of passages to return (default 5, max 20).",
            },
        },
        "required": ["query"],
    },
}

_RAG_GET_RESOURCE_TOOL_SPEC: dict[str, Any] = {
    "name": RAG_GET_RESOURCE_TOOL_NAME,
    "description": (
        "Fetch the full content of a resource (image / table / custom kind) "
        "referenced in a previous `rag_search` hit. Pass the resource_id "
        "from a `<resource_info>RID</resource_info>` marker or the "
        "`• resource [RID]` listing.\n\n"
        "Returns a renderer-formatted string. Built-in renderers cover:\n"
        "  - tables: full markdown content (no truncation)\n"
        "  - images: file path + mime + size (host application is "
        "    responsible for displaying / passing to a vision model)\n"
        "Custom kinds use whatever renderer the SDK caller registered."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "resource_id": {
                "type": "string",
                "description": "The ID inside a <resource_info>...</resource_info> marker.",
            },
        },
        "required": ["resource_id"],
    },
}


_RESOURCE_INFO_RE = re.compile(r"<resource_info>[^<]+</resource_info>")

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
        memory: Mem0Memory | MemoryOrchestrator | None,
        tools: ToolRegistry,
        reflector: Reflector | None = None,
        logger: AgentLogger | None = None,
        compressor: ContextCompressor | None = None,
        rag: Any | None = None,
        memory_tools: list[MemoryTool] | None = None,
    ) -> None:
        """Compose the modules; build the per-step memory chain (default order: [memory, compressor], skipping None). When `rag` is provided, the `rag_search` built-in tool is registered alongside `memory_recall`. `memory=None` runs the agent stateless: no memory_recall tool, no persistence, no condense_memory."""
        self.profile = profile
        self.llm = llm
        self.memory = memory
        self.tools = tools
        self.reflector = reflector
        self.logger = logger
        self.compressor = compressor
        self.rag = rag
        if memory_tools is not None:
            self.memory_tools = list(memory_tools)
        else:
            chain: list[MemoryTool] = []
            if memory is not None:
                chain.append(memory)
            if compressor is not None:
                chain.append(compressor)
            self.memory_tools = chain
        self._agent_tools: dict[str, _AgentToolHandler] = {}
        if memory is not None:
            self._agent_tools[MEMORY_RECALL_TOOL_NAME] = self._handle_memory_recall
        if rag is not None:
            self._register_rag_tools()
        self._config: AgentConfig | None = None
        self._async_setup_done: bool = False

    def _register_rag_tools(self) -> None:
        """Register both rag_search and rag_get_resource handlers; called from __init__ and _ensure_async_setup."""
        self._agent_tools[RAG_SEARCH_TOOL_NAME] = self._handle_rag_search
        self._agent_tools[RAG_GET_RESOURCE_TOOL_NAME] = self._handle_rag_get_resource

    @classmethod
    async def from_profile(
        cls,
        profile: AgentProfile,
        *,
        log_dir: str | Path | None = None,
        dotenv_path: str | None = None,
        load_env: bool = True,
        **agent_config_kwargs: Any,
    ) -> "BaseAgent":
        """Build a fully-wired agent from a profile + .env. Equivalent to:

            config = AgentConfig(profile=profile, log_dir=log_dir, ...)
            agent = cls(config)
            await agent._ensure_async_setup()    # eager MCP/RAG setup

        Kept as a convenience for the original .env-driven workflow; new code
        should construct an `AgentConfig` directly. Will be deprecated in v0.2.
        """
        config = AgentConfig(
            profile=profile,
            log_dir=log_dir,
            dotenv_path=dotenv_path,
            load_env=load_env,
            **agent_config_kwargs,
        )
        agent = cls(config)
        await agent._ensure_async_setup()
        return agent

    @abstractmethod
    async def run(
        self,
        task: str,
        *,
        max_steps: int | None = None,
        images: "list[str | Path] | None" = None,
    ) -> AgentResult:
        """Execute one `task` end to end; must respect `max_steps` (defaults to `profile.cognitive.max_steps_per_cycle`). When `images` is provided, the user task is sent as an OpenAI-style multimodal message (`[{type:text}, {type:image_url}, ...]`); only the OpenAI-compatible LLM adapter currently consumes the list form. Each image may be a local file path (read + base64-encoded into a `data:` URL with the inferred MIME type), an `http(s)://` URL (passed through), or an existing `data:` URL."""

    async def close(self) -> None:
        """Close underlying MCP clients (mem0 storage is auto-managed)."""
        await self.tools.close()

    async def _ensure_async_setup(self) -> None:
        """Apply config-driven async wiring (MCP servers + RAG) on the first run().

        Sync construction (`agent = ReActAgent(config)`) cannot await, so MCP
        servers and `LlamaIndexRAG.from_profile` are deferred to here. No-op
        when the agent was built via the legacy keyword path or has already
        been set up.
        """
        if self._async_setup_done or self._config is None:
            return
        self._async_setup_done = True
        from DefenseAgent.agent._builder import async_finish_setup
        rag = await async_finish_setup(self._config, self.profile, self.tools)
        if rag is not None and self.rag is None:
            self.rag = rag
            self._register_rag_tools()

    async def __aenter__(self) -> "BaseAgent":
        """Enter: return self so `async with BaseAgent.from_profile(...) as agent:` works cleanly."""
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        """Exit: close every long-lived resource the agent opened."""
        await self.close()

    def _resolve_max_steps(self, override: int | None) -> int:
        """Pick the caller's override if given, else `config.max_steps`, else `profile.cognitive.max_steps_per_cycle`."""
        if override is not None:
            return override
        if self._config is not None and self._config.max_steps is not None:
            return self._config.max_steps
        return self.profile.cognitive.max_steps_per_cycle

    async def _recall_memories(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Return mem0 records relevant to `query`, capped at `top_k`; returns [] when memory is disabled, top_k<=0, or memory raises."""
        if self.memory is None or top_k <= 0:
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
        """Auto-built identity block used when no `prompt.system` / `prompt.path` is configured. Optional fields (age, traits, backstory, initial_plan) are skipped when empty/None so the resulting prompt has no awkward blank slots."""
        p = self.profile
        opener = f"You are {p.name}, a {p.age}-year-old." if p.age is not None else f"You are {p.name}."
        parts = [opener]
        if p.traits.strip():
            parts.append(f"Traits: {p.traits.strip()}")
        if p.backstory.strip():
            parts.append(f"Backstory: {p.backstory.strip()}")
        if p.initial_plan.strip():
            parts.append(f"Today's plan: {p.initial_plan.strip()}")
        return "\n".join(parts)

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
        """Build the kwargs dict that fills `{placeholders}` inside a user-authored prompt template. Optional fields render as empty strings when unset, so a template referencing `{age}` / `{traits}` / `{backstory}` / `{initial_plan}` does not crash on minimal profiles."""
        p = self.profile
        return {
            "id": p.id,
            "name": p.name,
            "age": "" if p.age is None else p.age,
            "traits": p.traits.strip(),
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

    def _build_user_message(
        self,
        task: str,
        images: "list[str | Path] | None" = None,
    ) -> Message:
        """Build the initial user message for `run(task, images=...)`. Returns a plain text Message when `images` is None or empty (current behaviour); otherwise returns a Message whose `content` is an OpenAI-style content-block list (`[{type:text,text:task}, {type:image_url,image_url:{url:...}}, ...]`). Each image may be a local file path (read + base64-encoded into a `data:` URL via the inferred MIME type), an `http(s)://` URL (passed through), or a pre-built `data:` URL."""
        if not images:
            return Message(role="user", content=task)
        blocks: list[dict[str, Any]] = [{"type": "text", "text": task}]
        for image in images:
            blocks.append(
                {"type": "image_url", "image_url": {"url": _resolve_image_url(image)}}
            )
        return Message(role="user", content=blocks)

    async def _save_outcome(
        self,
        task: str,
        answer: str,
        *,
        memory_type: str = _OUTCOME_MEMORY_TYPE,
    ) -> None:
        """Append the Q→A pair to mem0 tagged with `memory_type` (default='outcome', failures override to 'failure'). No-op when memory is disabled."""
        if self.memory is None:
            return
        message = Message(role="user", content=f"Q: {task}\nA: {answer}")
        try:
            await self.memory.add([message], memory_type=memory_type)
        except Exception as e:
            self._log("warn", "agent.save_outcome_failed", str(e))

    def _log(self, level: str, event_type: str, message: str, **data: Any) -> None:
        """Emit a structured log event; no-op when no logger is wired. Accepts 'warn' as an alias for 'warning'."""
        if self.logger is None:
            return
        fn = getattr(self.logger, "warning" if level == "warn" else level, None)
        if fn is not None:
            fn(event_type, message, **data)

    def _combined_tool_specs(self) -> list[dict[str, Any]] | None:
        """Return user tool specs followed by Agent-owned tool specs; returns None only if both are empty."""
        user_specs = self.tools.specs()
        builtin_specs: list[dict[str, Any]] = []
        if MEMORY_RECALL_TOOL_NAME in self._agent_tools:
            builtin_specs.append(_MEMORY_RECALL_TOOL_SPEC)
        if RAG_SEARCH_TOOL_NAME in self._agent_tools:
            builtin_specs.append(_RAG_SEARCH_TOOL_SPEC)
        if RAG_GET_RESOURCE_TOOL_NAME in self._agent_tools:
            builtin_specs.append(_RAG_GET_RESOURCE_TOOL_SPEC)
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
        """Agent-owned handler for the `memory_recall` tool. Renders hits as
        `- [tier/memory_type] content` bullets so the LLM sees both the
        lifecycle tier (when present) and the finer memory_type label. Goes
        through `self.memory.search_records()`, which on a MemoryOrchestrator
        is a back-compat shim that runs hybrid scoring inside the persistent
        backend; on a bare Mem0Memory it's the legacy vector-only path."""
        if self.memory is None:
            return "(memory_recall unavailable: memory subsystem disabled)"

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
        return "\n".join(_format_recall_hit(h) for h in hits)

    async def _handle_rag_search(self, arguments: dict[str, Any]) -> str:
        """Agent-owned handler for the `rag_search` tool; renders RAG passages + per-hit resource list, or a diagnostic string.

        Per-hit rendering is delegated to `_format_rag_hit`, which subclasses
        can override to customize the LLM-facing output without touching the
        retrieval / validation logic here.
        """
        if self.rag is None:
            return "(rag_search unavailable: no RAG backend configured)"

        raw_query = arguments.get("query", "")
        query = raw_query.strip() if isinstance(raw_query, str) else ""
        if not query:
            return "(rag_search called with empty query)"

        raw_top_k = arguments.get("top_k", self.profile.rag.top_k)
        try:
            top_k = int(raw_top_k)
        except (TypeError, ValueError):
            top_k = self.profile.rag.top_k
        top_k = max(1, min(top_k, 20))

        threshold = self.profile.rag.score_threshold
        try:
            hits = await self.rag.retrieve(
                query, limit=top_k, score_threshold=threshold,
            )
        except Exception as e:
            return f"(rag_search failed: {type(e).__name__}: {e})"
        if not hits:
            return f"(no documents matched query={query!r})"
        return "\n".join(self._format_rag_hit(h) for h in hits)

    def _format_rag_hit(self, hit: dict[str, Any]) -> str:
        """Render one RAG hit (text + resource manifest) for the LLM.

        Default format:
            - [score=0.91] <truncated text with <resource_info> markers preserved>
              • resource [RID] (kind) "caption"
              • resource [RID] (kind)
              ...

        Override in a `BaseAgent` subclass to customize the format; the
        retrieval + validation logic in `_handle_rag_search` stays untouched.
        """
        score = float(hit.get("score", 0.0))
        text = hit.get("text", "")
        truncated = truncate_preserving_markers(text, 500)
        lines = [f"- [score={score:.2f}] {truncated}"]
        meta = hit.get("metadata") or {}
        rids = meta.get("resource_ids") or []
        kinds = meta.get("resource_kinds") or []
        captions = meta.get("resource_captions") or []
        for i, rid in enumerate(rids):
            kind = kinds[i] if i < len(kinds) else "?"
            caption = captions[i] if i < len(captions) else ""
            cap_part = f' "{caption}"' if caption else ""
            lines.append(f"  • resource [{rid}] ({kind}){cap_part}")
        return "\n".join(lines)

    async def _handle_rag_get_resource(self, arguments: dict[str, Any]) -> str:
        """Agent-owned handler for `rag_get_resource`. Thin shell — actual rendering lives in `LlamaIndexRAG.render_resource()` (which dispatches to the registered ResourceRenderer for the resource's kind)."""
        if self.rag is None:
            return "(rag_get_resource unavailable: no RAG backend configured)"
        raw_rid = arguments.get("resource_id", "")
        rid = raw_rid.strip() if isinstance(raw_rid, str) else ""
        if not rid:
            return "(rag_get_resource called with empty resource_id)"
        try:
            return await self.rag.render_resource(rid)
        except Exception as e:  # noqa: BLE001 - diagnostic for the LLM
            return f"(rag_get_resource failed: {type(e).__name__}: {e})"

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
            await self.reflector.maybe_reflect()
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


def truncate_preserving_markers(text: str, max_len: int) -> str:
    """Truncate but never split a `<resource_info>...</resource_info>` marker mid-token.

    Finds the last complete marker that ends within `max_len` and truncates
    just after it. Falls back to `truncate(text, max_len)` when no markers fit
    inside the budget. Used by `_format_rag_hit` so the LLM always sees
    well-formed resource references.
    """
    if len(text) <= max_len:
        return text
    last_safe_end = 0
    for m in _RESOURCE_INFO_RE.finditer(text):
        if m.end() <= max_len:
            last_safe_end = m.end()
        else:
            break
    if last_safe_end > 0:
        return text[:last_safe_end] + ("..." if last_safe_end < len(text) else "")
    return truncate(text, max_len)


_DEFAULT_IMAGE_MIME = "image/png"


def _resolve_image_url(image: "str | Path") -> str:
    """Turn an image input into the URL string OpenAI's `image_url` block expects. Pass-through for `http(s)://` and `data:` URLs; for any other string or Path, treat as a local file, read its bytes, base64-encode, and emit a `data:<mime>;base64,...` URL with the MIME type inferred from the extension (defaulting to `image/png` when the extension is unknown). Raises `FileNotFoundError` if a local-file input doesn't exist."""
    if isinstance(image, str) and (
        image.startswith("http://")
        or image.startswith("https://")
        or image.startswith("data:")
    ):
        return image
    path = Path(image)
    if not path.is_file():
        raise FileNotFoundError(f"image file not found: {path}")
    mime, _ = mimetypes.guess_type(str(path))
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime or _DEFAULT_IMAGE_MIME};base64,{encoded}"


def _format_recall_hit(record: dict[str, Any]) -> str:
    """Render a single mem0 record as a `- [tier/memory_type] content` bullet
    for the memory_recall tool result. Tier comes from metadata (set by P0+
    writers); memory_type is the legacy finer label. When neither is present
    we fall back to a generic `memory` label so legacy records still render
    sanely. Lifecycle tier first because it's the more important signal for
    the LLM — a procedural SOP weighs differently than an episodic trace."""
    metadata = record.get("metadata") or {}
    tier = metadata.get("tier")
    memory_type = record_memory_type(record)
    label_parts: list[str] = []
    if tier:
        label_parts.append(str(tier))
    if memory_type:
        label_parts.append(str(memory_type))
    label = "/".join(label_parts) if label_parts else "memory"
    return f"- [{label}] {record.get('memory', '')}"

