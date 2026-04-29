from datetime import datetime
from typing import Any, Callable

from DefenseAgent.llm import LLM
from DefenseAgent.llm.types import Message
from DefenseAgent.memory import Mem0Memory, MemoryOrchestrator, MemoryTier
from DefenseAgent.memory._bridge import record_memory_type
from DefenseAgent.ops.logger import _default_clock
from DefenseAgent.reflection.scorer import ImportanceScorer
from DefenseAgent.reflection.synthesizer import InsightSynthesizer


_REFLECTION_MEMORY_TYPE = "reflection"


class Reflector:
    """Module 5's facade: ImportanceScorer + InsightSynthesizer over the
    memory backend; reflections land in the SEMANTIC tier (the lifecycle
    bucket Hello-Agents reserves for "distilled facts / lessons / insights")
    tagged `memory_type=reflection`. Accepts either a bare Mem0Memory or a
    MemoryOrchestrator — both have `.profile`, `.add()`, `.get_all()`."""

    def __init__(
        self,
        memory: Mem0Memory | MemoryOrchestrator,
        llm: LLM,
        *,
        scorer: ImportanceScorer | None = None,
        synthesizer: InsightSynthesizer | None = None,
        num_insights: int = 3,
        reflection_importance: float = 8.0,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """Wire the scorer + synthesizer; reflection records are written into
        the SEMANTIC tier with `memory_type=reflection`. `reflection_importance`
        is on the legacy 1-10 scale; it's normalized to [0, 1] before being
        stored on the MemoryItem so the new scoring layer can use it directly."""
        self.memory = memory
        self.llm = llm
        self.scorer = scorer or ImportanceScorer(llm)
        self.synthesizer = synthesizer or InsightSynthesizer(llm, num_insights=num_insights)
        self.reflection_importance = reflection_importance
        self._clock = clock or _default_clock
        self._last_reflection_time: datetime | None = None

    async def score_importance(self, content: str) -> float:
        """Delegate to the configured ImportanceScorer (LLM-based 1-10 rating)."""
        return await self.scorer.score(content)

    @property
    def unreflected_count(self) -> int:
        """Count non-reflection mem0 records added since the last reflection cutoff."""
        return len(self._get_unreflected_records())

    async def maybe_reflect(self) -> list[dict[str, Any]]:
        """Reflect only when unreflected_count >= profile.cognitive.reflection_threshold; otherwise no-op."""
        threshold = self.memory.profile.cognitive.reflection_threshold
        if self.unreflected_count < threshold:
            return []
        return await self.reflect_now()

    async def reflect_now(self) -> list[dict[str, Any]]:
        """Force a reflection cycle: synthesize insights from unreflected
        records and write each into the SEMANTIC tier with the configured
        reflection importance. The 1-10 reflection_importance is normalized to
        [0, 1] before storage; clamped on either end so a misconfigured value
        can't break the MemoryItem invariant."""
        recent = self._get_unreflected_records()
        if not recent:
            return []
        insights = await self.synthesizer.synthesize(recent)
        normalized_importance = max(0.0, min(1.0, self.reflection_importance / 10.0))
        stored: list[dict[str, Any]] = []
        for insight in insights:
            await self.memory.add(
                [Message(role="user", content=insight)],
                memory_type=_REFLECTION_MEMORY_TYPE,
                tier=MemoryTier.SEMANTIC,
                importance=normalized_importance,
            )
            stored.append({
                "memory": insight,
                "memory_type": _REFLECTION_MEMORY_TYPE,
                "importance": self.reflection_importance,
                "tier": MemoryTier.SEMANTIC.value,
            })
        self._last_reflection_time = self._clock()
        return stored

    def _get_unreflected_records(self) -> list[dict[str, Any]]:
        """Return mem0 records whose memory_type is not 'reflection'."""
        return [
            r for r in self.memory.get_all()
            if record_memory_type(r) != _REFLECTION_MEMORY_TYPE
        ]


