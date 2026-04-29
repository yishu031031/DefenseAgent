import contextvars
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from ms_agent.memory.default_memory import DefaultMemory as MsDefaultMemory

from DefenseAgent.config.profile import AgentProfile
from DefenseAgent.llm.types import Message
from DefenseAgent.memory._bridge import (
    MemoryBackendConfig,
    messages_ours_to_theirs,
    messages_theirs_to_ours,
    profile_to_dictconfig,
    record_memory_type,
    record_tier as _record_tier,
)
from DefenseAgent.memory.scoring import rank_items
from DefenseAgent.memory.types import MemoryItem, MemoryTier


# Per-coroutine slot used to ferry tier-aware metadata (tier, importance,
# source_run_id, extra) from `Mem0Memory.add()` through ms-agent's
# super().add() — which has an explicit signature and won't accept new kwargs —
# down to `_Mem0AddProxy.add()` which writes them into mem0's metadata dict.
# A ContextVar (not an instance attr) keeps this safe under asyncio.gather:
# each task gets its own context copy, so concurrent add() calls on the same
# Mem0Memory don't bleed metadata into each other.
_PENDING_METADATA: contextvars.ContextVar[dict[str, Any] | None] = (
    contextvars.ContextVar("_defense_agent_mem0_pending_metadata", default=None)
)


class _Mem0AddProxy:
    """Wraps a mem0 client so `add()` routes `memory_type` and any pending
    tier-aware metadata (set via the module-level `_PENDING_METADATA` ContextVar
    by `Mem0Memory.add()`) into `metadata`, and pins `infer=False`. Every other
    attribute proxies through to the underlying client unchanged. Lets
    DefenseAgent enforce mem0-level conventions without monkey-patching the
    live `mem0.Memory.add` attribute (which would clobber MagicMock attributes
    in tests)."""

    def __init__(self, underlying: Any) -> None:
        """Capture the wrapped mem0 client without touching its attributes."""
        object.__setattr__(self, "_underlying", underlying)

    def add(self, messages: Any, **kwargs: Any) -> Any:
        """Merge any caller-supplied `metadata=`, the per-coroutine pending
        metadata slot, and the legacy `memory_type=` kwarg into a single
        metadata dict; default `infer=False`; then delegate to the underlying
        client's `add`. mem0's native `memory_type` kwarg only accepts
        'procedural_memory', so DefenseAgent's custom labels live in metadata."""
        metadata = dict(kwargs.pop("metadata", {}) or {})
        pending = _PENDING_METADATA.get()
        if pending:
            # Caller-supplied `metadata=` wins on conflicts so explicit kwargs
            # always beat the implicit ContextVar channel.
            for k, v in pending.items():
                metadata.setdefault(k, v)
            _PENDING_METADATA.set(None)
        memory_type_arg = kwargs.pop("memory_type", None)
        if memory_type_arg and "memory_type" not in metadata:
            metadata["memory_type"] = memory_type_arg
        if metadata:
            kwargs["metadata"] = metadata
        kwargs.setdefault("infer", False)
        return self._underlying.add(messages, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Forward any attribute we don't override to the wrapped client (search, get_all, delete, _create_memory, etc.)."""
        return getattr(self._underlying, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Forward attribute assignments (e.g. ms-agent setting `_telemetry_vector_store`) to the underlying client so its internal state stays consistent."""
        if name == "_underlying":
            object.__setattr__(self, name, value)
        else:
            setattr(self._underlying, name, value)


class Mem0Memory(MsDefaultMemory):
    """Mem0-backed memory: inherits ms-agent's `DefaultMemory`, takes our `AgentProfile`, and converts at the `Message` boundary on `run()` / `add()`."""

    def __init__(
        self,
        profile: AgentProfile,
        *,
        user_id: str = "default_user",
        agent_id: str | None = None,
        run_id: str = "default_run",
        storage_path: str | Path | None = None,
        load_env: bool = True,
        dotenv_path: str | None = None,
        backend: MemoryBackendConfig | None = None,
    ) -> None:
        """Build the ms-agent DictConfig from `profile` (+ optional backend kwargs or .env), ensure the storage dir exists, then defer to ms-agent's `DefaultMemory.__init__`. When `backend=` is given, env vars are not consulted; otherwise the legacy .env path is used."""
        # Skip dotenv when the caller already supplied a programmatic backend —
        # they don't want env-var influence at all.
        if load_env and backend is None:
            load_dotenv(dotenv_path, override=False)
        config = profile_to_dictconfig(
            profile,
            backend=backend,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            storage_path=storage_path,
        )
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        super().__init__(config)
        self._wrap_mem0_add()
        self.profile = profile

    def _init_memory_obj(self):
        """Build mem0.Memory directly from the DictConfig's `mem0_config` block, bypassing ms-agent's hardcoded service-URL translation so custom embedder/LLM endpoints (OpenRouter, vLLM, etc.) are honoured."""
        try:
            import mem0
        except ImportError as e:
            raise ImportError(
                "DefenseAgent's memory subsystem requires the optional `mem0ai` package. "
                "Install it with: `pip install 'defense-agent[memory]'` "
                "(or pass `use_memory=False` to AgentConfig if you don't need persistent memory)."
            ) from e
        from omegaconf import OmegaConf
        cfg = OmegaConf.to_container(self.config.mem0_config, resolve=True)
        return mem0.Memory.from_config(cfg)

    def _wrap_mem0_add(self) -> None:
        """Replace `self.memory` with a thin proxy whose `add()` enforces DefenseAgent's mem0-level conventions no matter who calls it (including ms-agent's inherited `add_single`): custom `memory_type` values are tucked into `metadata` (mem0's native `memory_type` kwarg only accepts 'procedural_memory'), and `infer=False` is the default so trajectories are stored verbatim. Wrapping here lets us delegate the full add()/add_single() pipeline to ms-agent (block hashing, history_mode, cache persistence) without duplicating its ~50 lines. Every other attribute (search, get_all, delete, ...) is proxied through unchanged via `__getattr__`."""
        if self.memory is None:
            return
        self.memory = _Mem0AddProxy(self.memory)

    async def run(self, messages: list[Message], **kwargs: Any) -> list[Message]:
        """Passthrough in DefenseAgent — ms-agent's run() injects retrieved memories as a `system` message, which collides with `BaseAgent._build_system_prompt()` already passing identity via the LLM's `system=` kwarg. The LLM accesses memory explicitly through the built-in `memory_recall` tool instead."""
        return messages

    async def add(
        self,
        messages: list[Message],
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        memory_type: str | None = None,
        tier: MemoryTier | str | None = None,
        importance: float | None = None,
        source_run_id: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Convert at the Message boundary, then defer to ms-agent's `add()` so its block-hash dedup (`_analyze_messages`) and `history_mode` (add vs overwrite, sourced from `profile.memory.history_mode`) actually take effect. DefenseAgent's mem0-level tweaks — routing custom `memory_type` through `metadata`, attaching tier/importance/source_run_id/extra via the per-coroutine `_PENDING_METADATA` ContextVar consumed by `_Mem0AddProxy.add`, and pinning `infer=False` for verbatim storage — are applied transparently. ignore_roles filtering is handled by ms-agent's inherited `parse_messages()` per block."""
        ms_messages = messages_ours_to_theirs(messages)
        pending = self._build_pending_metadata(
            tier=tier,
            importance=importance,
            source_run_id=source_run_id,
            extra=extra,
        )
        token = _PENDING_METADATA.set(pending) if pending else None
        try:
            await super().add(
                ms_messages,
                user_id=user_id,
                agent_id=agent_id,
                run_id=run_id,
                memory_type=memory_type,
            )
        finally:
            if token is not None:
                _PENDING_METADATA.reset(token)

    @staticmethod
    def _build_pending_metadata(
        *,
        tier: MemoryTier | str | None,
        importance: float | None,
        source_run_id: str | None,
        extra: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Bundle the tier-aware add() kwargs into a flat dict for the proxy
        to merge into mem0's metadata. None-valued fields are dropped so legacy
        callers (which never set these) produce an empty dict and skip the
        ContextVar dance entirely."""
        payload: dict[str, Any] = {}
        if tier is not None:
            payload["tier"] = tier.value if isinstance(tier, MemoryTier) else tier
        if importance is not None:
            imp = float(importance)
            if not 0.0 <= imp <= 1.0:
                raise ValueError(f"importance must be in [0, 1]; got {importance!r}")
            payload["importance"] = imp
        if source_run_id is not None:
            payload["source_run_id"] = source_run_id
        if extra:
            for k, v in extra.items():
                payload.setdefault(k, v)
        return payload

    async def add_item(
        self,
        messages: list[Message],
        item: MemoryItem,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
    ) -> None:
        """Write `messages` tagged with all of `item`'s non-content fields.
        Convenience for callers that already have a MemoryItem template
        (e.g. the orchestrator routing a write into a specific tier)."""
        await self.add(
            messages,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            memory_type=item.memory_type,
            tier=item.tier,
            importance=item.importance,
            source_run_id=item.source_run_id,
            extra=item.extra or None,
        )

    def search_texts(self, query: str, meta_infos: list[dict[str, Any]] | None = None) -> list[str]:
        """Return mem0 hits as plain text strings (sibling to `search_records`, which returns full record dicts).

        Overrides ms-agent's `search()` shape so inherited callers (notably ms-agent's `run()`) keep working — `search` is kept as an alias below.
        """
        if not query:
            return []
        if meta_infos is None:
            meta_infos = [{}]
        out: list[str] = []
        for info in meta_infos:
            filters = {
                "user_id": info.get("user_id") or self.user_id,
                "agent_id": info.get("agent_id") or self.agent_id,
                "run_id": info.get("run_id") or self.run_id,
            }
            limit = info.get("limit", self.search_limit)
            response = self.memory.search(query, filters=filters, limit=limit)
            results = response.get("results", []) if isinstance(response, dict) else []
            out.extend(r.get("memory", "") for r in results)
        return out

    def search_records(
        self,
        query: str,
        *,
        limit: int | None = None,
        memory_type: str | None = None,
        tier: MemoryTier | str | None = None,
        scoring: str = "vector",
        candidate_multiplier: int = 3,
    ) -> list[dict[str, Any]]:
        """Return mem0 records as dicts (the ms-agent `search()` collapses to str — this preserves the full record for DefenseAgent callers). `tier` narrows results to a single lifecycle tier; `memory_type` to a finer label inside a tier.

        `scoring="vector"` (default) keeps the legacy behavior: mem0 returns its top-K by cosine similarity and we hand them back unchanged. `scoring="hybrid"` re-ranks using `memory.scoring.hybrid_score`, which combines similarity with recency, importance, and access frequency (weights from `profile.memory.scoring`). To give the re-rank room to surface a low-similarity-but-recent/important hit, the hybrid path fetches `limit * candidate_multiplier` candidates from mem0 before sorting and clipping back to `limit`. Each returned record gains a `_hybrid_score` field so callers can inspect the rank.
        """
        if not query:
            return []
        if scoring not in ("vector", "hybrid"):
            raise ValueError(
                f"scoring must be 'vector' or 'hybrid'; got {scoring!r}"
            )
        target_limit = limit if limit is not None else self.search_limit
        fetch_limit = (
            target_limit * max(1, candidate_multiplier)
            if scoring == "hybrid"
            else target_limit
        )
        filters = {
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "run_id": self.run_id,
        }
        response = self.memory.search(
            query,
            filters=filters,
            limit=fetch_limit,
        )
        results = response.get("results", []) if isinstance(response, dict) else []
        if memory_type is not None:
            results = [r for r in results if record_memory_type(r) == memory_type]
        if tier is not None:
            tier_value = tier.value if isinstance(tier, MemoryTier) else tier
            results = [r for r in results if _record_tier(r) == tier_value]
        if scoring == "hybrid":
            results = self._hybrid_rerank(results, limit=target_limit)
        else:
            results = results[:target_limit]
        return results

    def _hybrid_rerank(
        self,
        records: list[dict[str, Any]],
        *,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Decode candidates into MemoryItems, run hybrid scoring with the
        profile's weight config, then re-attach `_hybrid_score` to the original
        record dicts and return them in the new order. We keep the dict shape
        on output so callers that already consume `search_records` don't have
        to switch types — `search_items` is the typed entry point for new code."""
        weights = self.profile.memory.scoring
        decoded = [(MemoryItem.from_record(r), r) for r in records]
        scored = [
            (item, float(record.get("score") or 0.0))
            for item, record in decoded
        ]
        ranked = rank_items(scored, weights=weights, limit=limit)
        # Map item identity (by record id when present, else content) back to
        # the original record dict so we can preserve mem0's other fields
        # (created_at, score, ...) and just attach the new hybrid score.
        record_by_id = {
            id(item): record for item, record in decoded
        }
        out: list[dict[str, Any]] = []
        for item, hybrid in ranked:
            record = record_by_id.get(id(item))
            if record is None:
                continue
            new_record = dict(record)
            new_record["_hybrid_score"] = hybrid
            out.append(new_record)
        return out

    def search_items(
        self,
        query: str,
        *,
        limit: int | None = None,
        memory_type: str | None = None,
        tier: MemoryTier | str | None = None,
        scoring: str = "vector",
        candidate_multiplier: int = 3,
    ) -> list[MemoryItem]:
        """Same query path as `search_records` (including the `scoring` switch
        and tier/type filters) but decodes each hit into a MemoryItem so
        callers operate on typed data (tier / importance / access_count)
        without reaching into raw mem0 dicts."""
        records = self.search_records(
            query,
            limit=limit,
            memory_type=memory_type,
            tier=tier,
            scoring=scoring,
            candidate_multiplier=candidate_multiplier,
        )
        return [MemoryItem.from_record(r) for r in records]

    def get_all(
        self,
        *,
        memory_type: str | None = None,
        tier: MemoryTier | str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Pull every record under the (user_id, agent_id, run_id) tuple via mem0's `filters=` API, optionally narrowed by `memory_type` and/or `tier`."""
        filters = {
            "user_id": user_id or self.user_id,
            "agent_id": agent_id or self.agent_id,
            "run_id": run_id or self.run_id,
        }
        response = self.memory.get_all(filters=filters)
        results = response.get("results", []) if isinstance(response, dict) else []
        if memory_type is not None:
            results = [r for r in results if record_memory_type(r) == memory_type]
        if tier is not None:
            tier_value = tier.value if isinstance(tier, MemoryTier) else tier
            results = [r for r in results if _record_tier(r) == tier_value]
        return results

    def get_items(
        self,
        *,
        memory_type: str | None = None,
        tier: MemoryTier | str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
    ) -> list[MemoryItem]:
        """Same as `get_all` but returns typed MemoryItems."""
        records = self.get_all(
            memory_type=memory_type, tier=tier,
            user_id=user_id, agent_id=agent_id, run_id=run_id,
        )
        return [MemoryItem.from_record(r) for r in records]

    # Keep `search` as an alias so ms-agent's inherited `run()` path keeps
    # working (it calls `self.search(...)` with the legacy shape).
    search = search_texts

    @classmethod
    def from_profile(
        cls,
        profile: AgentProfile,
        **kwargs: Any,
    ) -> "Mem0Memory":
        """Convenience constructor matching the rest of DefenseAgent's `from_profile` pattern."""
        return cls(profile, **kwargs)

    @classmethod
    def create(
        cls,
        profile: AgentProfile,
        *,
        backend: MemoryBackendConfig,
        storage_path: str | Path | None = None,
        user_id: str = "default_user",
        agent_id: str | None = None,
        run_id: str = "default_run",
    ) -> "Mem0Memory":
        """Build mem0-backed memory from explicit args — no .env required.

        Use this from SDK code, tests, or multi-tenant servers where each
        memory instance may need different LLM / embedder credentials. The
        `from_profile` / `__init__` paths still resolve mem0 config from .env
        for backward compatibility.
        """
        return cls(
            profile,
            backend=backend,
            storage_path=storage_path,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            load_env=False,
        )


