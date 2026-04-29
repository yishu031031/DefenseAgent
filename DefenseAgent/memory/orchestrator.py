"""MemoryOrchestrator — tier-aware facade over the persistent (mem0) backend
and (eventually) the in-memory Working layer.

Inspired by Hello-Agents' MemoryManager: a single entry point that routes
writes to the right tier (Working / Episodic / Semantic / Procedural) and
fans reads across tiers with hybrid scoring. This module is the layer most
DefenseAgent code should depend on; `Mem0Memory` becomes an implementation
detail of the persistent tiers.

P2 ships with the persistent path complete and a stub Working slot. P4 will
plug in a real WorkingMemory; until then, write/read calls targeting
WORKING raise a clear error so misuse is loud, not silent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from DefenseAgent.config.profile import AgentProfile
from DefenseAgent.llm.types import Message
from DefenseAgent.memory._bridge import MemoryBackendConfig
from DefenseAgent.memory.mem0_memory import Mem0Memory
from DefenseAgent.memory.types import MemoryItem, MemoryTier


@runtime_checkable
class WorkingMemoryProtocol(Protocol):
    """Minimal contract the orchestrator needs from the Working tier (filled
    in by P4). Synchronous on purpose — the Working layer lives in memory and
    has no I/O. Items are typed so cross-tier merging in `recall()` works
    uniformly."""

    def add(self, item: MemoryItem) -> None: ...

    def search(self, query: str, *, limit: int) -> list[MemoryItem]: ...

    def clear(self) -> None: ...


class MemoryOrchestrator:
    """Single entry point for tier-aware memory operations.

    Writes route by tier:
      - WORKING                                → `self.working` (in-memory, P4)
      - EPISODIC / SEMANTIC / PROCEDURAL       → `self.persistent` (mem0)

    Reads (`recall`) default to hybrid scoring inside the persistent backend
    (similarity + recency + importance + frequency, weights from
    `profile.memory.scoring`). When `tier` is None the call spans every
    persistent tier; when set, it narrows to that tier only.
    """

    def __init__(
        self,
        profile: AgentProfile,
        persistent: Mem0Memory,
        *,
        working: WorkingMemoryProtocol | None = None,
    ) -> None:
        self.profile = profile
        self.persistent = persistent
        self.working = working

    @classmethod
    def from_profile(
        cls,
        profile: AgentProfile,
        *,
        user_id: str = "default_user",
        agent_id: str | None = None,
        run_id: str = "default_run",
        storage_path: str | Path | None = None,
        load_env: bool = True,
        dotenv_path: str | None = None,
        backend: MemoryBackendConfig | None = None,
        working: WorkingMemoryProtocol | None = None,
        with_working: bool = True,
    ) -> "MemoryOrchestrator":
        """Build a Mem0-backed persistent layer from `profile` and wrap it.
        Mirrors Mem0Memory's construction args so existing call sites can
        switch to the orchestrator with minimal churn.

        When `working` is None and `with_working=True` (default), the
        orchestrator auto-creates a `WorkingMemory` from
        `profile.memory.tier_limits`. Pass `with_working=False` to opt out
        (useful in tests that want to assert "no working layer" behavior)."""
        persistent = Mem0Memory(
            profile,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            storage_path=storage_path,
            load_env=load_env,
            dotenv_path=dotenv_path,
            backend=backend,
        )
        if working is None and with_working:
            from DefenseAgent.memory.working import WorkingMemory
            working = WorkingMemory.from_profile(profile)
        return cls(profile, persistent, working=working)

    # ------------------------------------------------------------------ chain

    async def run(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> list[Message]:
        """Compose with `BaseAgent._condense_memory`'s pipeline (the chain that
        also runs the ContextCompressor). Delegates to the persistent layer's
        `run()`, which in DefenseAgent is a passthrough — memory access happens
        through the explicit `memory_recall` tool, not via system-prompt
        injection. Keeping the shim lets the orchestrator drop into the
        existing `memory_tools` slot without changing the agent loop."""
        return await self.persistent.run(messages, **kwargs)

    # ------------------------------------------------------------------ writes

    async def add(
        self,
        messages: list[Message],
        *,
        tier: MemoryTier | str = MemoryTier.EPISODIC,
        memory_type: str | None = None,
        importance: float | None = None,
        source_run_id: str | None = None,
        extra: dict[str, Any] | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
    ) -> None:
        """Route a write to the tier's backend. Default tier is EPISODIC, the
        tier most agent-loop traces belong in. Importance defaults from
        `profile.memory.default_importance` when the caller doesn't supply one
        — the orchestrator owns this default so individual call sites don't
        have to know it."""
        resolved_tier = self._resolve_tier(tier)
        resolved_importance = (
            importance if importance is not None
            else self.profile.memory.default_importance
        )

        if resolved_tier is MemoryTier.WORKING:
            self._require_working_layer("add")
            assert self.working is not None  # narrowed by _require_working_layer
            content = _messages_to_content(messages)
            self.working.add(
                MemoryItem(
                    content=content,
                    tier=MemoryTier.WORKING,
                    memory_type=memory_type,
                    importance=resolved_importance,
                    source_run_id=source_run_id,
                    extra=extra or {},
                )
            )
            return

        await self.persistent.add(
            messages,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            memory_type=memory_type,
            tier=resolved_tier,
            importance=resolved_importance,
            source_run_id=source_run_id,
            extra=extra,
        )

    async def add_episodic(self, messages: list[Message], **kwargs: Any) -> None:
        """Convenience: write to the EPISODIC tier (raw events / trajectories)."""
        await self.add(messages, tier=MemoryTier.EPISODIC, **kwargs)

    async def add_semantic(self, messages: list[Message], **kwargs: Any) -> None:
        """Convenience: write to the SEMANTIC tier (distilled facts /
        reflections / lessons)."""
        await self.add(messages, tier=MemoryTier.SEMANTIC, **kwargs)

    async def add_procedural(self, messages: list[Message], **kwargs: Any) -> None:
        """Convenience: write to the PROCEDURAL tier (attack patterns / SOPs /
        workflows). Procedural records get extra retention by default."""
        await self.add(messages, tier=MemoryTier.PROCEDURAL, **kwargs)

    # ------------------------------------------------------------------- reads

    def recall(
        self,
        query: str,
        *,
        limit: int | None = None,
        tier: MemoryTier | str | None = None,
        memory_type: str | None = None,
        scoring: str = "hybrid",
        candidate_multiplier: int = 3,
    ) -> list[MemoryItem]:
        """Tier-aware retrieval returning typed MemoryItems.

        `tier=None` (default) searches every persistent tier and merges results
        by hybrid score. `tier=WORKING` queries the working layer only.
        Otherwise the call narrows to that single persistent tier.

        `scoring='hybrid'` is the new default — same score formula
        `Mem0Memory.search_records` exposes — so the orchestrator's API
        consciously diverges from the legacy mem0 vector-only behavior. Pass
        `scoring='vector'` to opt back into raw cosine ordering.
        """
        if not query:
            return []

        if tier is not None:
            resolved_tier = self._resolve_tier(tier)
            if resolved_tier is MemoryTier.WORKING:
                self._require_working_layer("recall")
                assert self.working is not None
                effective_limit = limit if limit is not None else self.persistent.search_limit
                return self.working.search(query, limit=effective_limit)
            return self.persistent.search_items(
                query,
                limit=limit,
                memory_type=memory_type,
                tier=resolved_tier,
                scoring=scoring,
                candidate_multiplier=candidate_multiplier,
            )

        # tier=None: search every persistent tier in one call (mem0 doesn't
        # need separate queries per tier — the tier filter is metadata-only).
        # Then merge in any working-layer hits at the front, since they're
        # session-fresh and almost always relevant.
        persistent_hits = self.persistent.search_items(
            query,
            limit=limit,
            memory_type=memory_type,
            tier=None,
            scoring=scoring,
            candidate_multiplier=candidate_multiplier,
        )
        if self.working is None:
            return persistent_hits
        effective_limit = limit if limit is not None else self.persistent.search_limit
        working_hits = self.working.search(query, limit=effective_limit)
        # Working items take precedence in the final list (session context is
        # nearly always more relevant than older persistent records); persistent
        # hits fill the remaining slots in their already-ranked order.
        merged = list(working_hits)
        merged.extend(persistent_hits)
        return merged[:effective_limit] if effective_limit else merged

    def search_records(
        self,
        query: str,
        *,
        limit: int | None = None,
        memory_type: str | None = None,
        tier: MemoryTier | str | None = None,
        scoring: str = "hybrid",
    ) -> list[dict[str, Any]]:
        """Back-compat shim: delegate to the persistent backend with hybrid
        scoring as the new default. Lets existing callers (notably the agent's
        `_handle_memory_recall` tool) keep their `.search_records()` call site
        while transparently picking up tier-aware ranking."""
        return self.persistent.search_records(
            query,
            limit=limit,
            memory_type=memory_type,
            tier=tier,
            scoring=scoring,
        )

    def get_all(
        self,
        *,
        memory_type: str | None = None,
        tier: MemoryTier | str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Back-compat shim mirroring `Mem0Memory.get_all` (used by the
        Reflector to enumerate unreflected records). Working-tier records are
        not enumerated here — they live outside mem0 and are session-scoped."""
        return self.persistent.get_all(
            memory_type=memory_type,
            tier=tier,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
        )

    def get_items(
        self,
        *,
        memory_type: str | None = None,
        tier: MemoryTier | str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
    ) -> list[MemoryItem]:
        """Typed counterpart to `get_all` — same persistent-only scope."""
        return self.persistent.get_items(
            memory_type=memory_type,
            tier=tier,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
        )

    # --------------------------------------------------------------- internals

    @staticmethod
    def _resolve_tier(tier: MemoryTier | str) -> MemoryTier:
        """Coerce a string tier (handy from YAML / tool args) into the enum,
        raising a clear error on unknown values rather than silently routing
        the write into the wrong place."""
        if isinstance(tier, MemoryTier):
            return tier
        try:
            return MemoryTier(tier)
        except ValueError as e:
            valid = ", ".join(t.value for t in MemoryTier)
            raise ValueError(
                f"unknown memory tier {tier!r}; expected one of: {valid}"
            ) from e

    def _require_working_layer(self, op: str) -> None:
        """The Working tier needs a plugged-in WorkingMemory (P4). Until that
        ships, any operation targeting WORKING fails loudly so misconfigured
        callers don't think their write succeeded."""
        if self.working is None:
            raise RuntimeError(
                f"MemoryOrchestrator.{op}() targeting WORKING tier requires a "
                "WorkingMemory implementation; pass `working=...` to the "
                "constructor (P4 will provide a default in-memory backend)."
            )


def _messages_to_content(messages: list[Message]) -> str:
    """Flatten a list of Messages into a single text blob for the working
    layer's keyword search. The working layer is intentionally simple — it
    doesn't run embeddings, so concatenating role-tagged content is the
    cheapest representation that still preserves who-said-what."""
    parts: list[str] = []
    for m in messages:
        role = (m.role or "").strip()
        content = (m.content or "").strip()
        if not content:
            continue
        parts.append(f"[{role}] {content}" if role else content)
    return "\n".join(parts)
