from datetime import datetime, timezone
from typing import Callable

from DefenseAgent.llm import LLM
from DefenseAgent.memory import Memory, MemoryRecord
from DefenseAgent.reflection.scorer import ImportanceScorer
from DefenseAgent.reflection.synthesizer import InsightSynthesizer


def _default_clock() -> datetime:
    """Return the current UTC time."""
    return datetime.now(timezone.utc)


class Reflector:
    """Module 5's unified facade: compose ImportanceScorer + InsightSynthesizer over a Memory."""

    def __init__(
        self,
        memory: Memory,
        llm: LLM,
        *,
        scorer: ImportanceScorer | None = None,
        synthesizer: InsightSynthesizer | None = None,
        num_insights: int = 3,
        reflection_importance: float = 8.0,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """Construct a Reflector; caller may inject custom scorer/synthesizer for extensibility."""
        self.memory = memory
        self.llm = llm
        if scorer is None:
            self.scorer = ImportanceScorer(llm)
        else:
            self.scorer = scorer
        if synthesizer is None:
            self.synthesizer = InsightSynthesizer(llm, num_insights=num_insights)
        else:
            self.synthesizer = synthesizer
        self.reflection_importance = reflection_importance
        if clock is None:
            self._clock = _default_clock
        else:
            self._clock = clock
        self._last_reflection_time: datetime | None = None

    async def score_importance(self, content: str) -> float:
        """Delegate to the configured ImportanceScorer."""
        return await self.scorer.score(content)

    @property
    def unreflected_count(self) -> int:
        """Count of non-reflection records added since the last reflection."""
        return len(self._get_unreflected_records())

    async def check_and_reflect(self) -> list[MemoryRecord]:
        """Reflect only when unreflected_count has reached profile.cognitive.reflection_threshold."""
        threshold = self.memory.profile.cognitive.reflection_threshold
        if self.unreflected_count < threshold:
            return []
        return await self.reflect_now()

    async def reflect_now(self) -> list[MemoryRecord]:
        """Force a reflection over all unreflected records; always advances the cutoff."""
        recent = self._get_unreflected_records()
        if not recent:
            return []
        insights = await self.synthesizer.synthesize(recent)
        stored: list[MemoryRecord] = []
        for insight in insights:
            record = await self.memory.remember(
                insight,
                kind="reflection",
                importance=self.reflection_importance,
            )
            stored.append(record)
        self._last_reflection_time = self._clock()
        return stored

    def _get_unreflected_records(self) -> list[MemoryRecord]:
        """Return non-reflection records whose timestamp is strictly after the last reflection cutoff."""
        cutoff = self._last_reflection_time
        result: list[MemoryRecord] = []
        for r in self.memory.stream.get_all():
            if r.kind == "reflection":
                continue
            if cutoff is not None and r.timestamp <= cutoff:
                continue
            result.append(r)
        return result
