from datetime import datetime
from typing import Any, Callable

from DefenseAgent.llm import LLM
from DefenseAgent.llm.types import Message
from DefenseAgent.memory import DefaultMemory
from DefenseAgent.memory._bridge import record_memory_type
from DefenseAgent.ops.logger import _default_clock
from DefenseAgent.reflection.scorer import ImportanceScorer
from DefenseAgent.reflection.synthesizer import InsightSynthesizer


_REFLECTION_MEMORY_TYPE = "reflection"


class Reflector:
    """Module 5's facade: ImportanceScorer + InsightSynthesizer over a mem0-backed DefaultMemory; tags reflections via memory_type."""

    def __init__(
        self,
        memory: DefaultMemory,
        llm: LLM,
        *,
        scorer: ImportanceScorer | None = None,
        synthesizer: InsightSynthesizer | None = None,
        num_insights: int = 3,
        reflection_importance: float = 8.0,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """Wire the scorer + synthesizer; reflection records are written back to mem0 with memory_type=reflection."""
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

    async def check_and_reflect(self) -> list[dict[str, Any]]:
        """Trigger reflection only when unreflected_count >= profile.cognitive.reflection_threshold."""
        threshold = self.memory.profile.cognitive.reflection_threshold
        if self.unreflected_count < threshold:
            return []
        return await self.reflect_now()

    async def reflect_now(self) -> list[dict[str, Any]]:
        """Force a reflection cycle: synthesize insights from unreflected records, write each back tagged memory_type=reflection."""
        recent = self._get_unreflected_records()
        if not recent:
            return []
        insights = await self.synthesizer.synthesize(recent)
        stored: list[dict[str, Any]] = []
        for insight in insights:
            await self.memory.add(
                [Message(role="user", content=insight)],
                memory_type=_REFLECTION_MEMORY_TYPE,
            )
            stored.append({
                "memory": insight,
                "memory_type": _REFLECTION_MEMORY_TYPE,
                "importance": self.reflection_importance,
            })
        self._last_reflection_time = self._clock()
        return stored

    def _get_unreflected_records(self) -> list[dict[str, Any]]:
        """Return mem0 records whose memory_type is not 'reflection'."""
        return [
            r for r in self.memory.get_all()
            if record_memory_type(r) != _REFLECTION_MEMORY_TYPE
        ]


