import re

from DefenseAgent.llm import LLM, Message
from DefenseAgent.memory import MemoryRecord


_REFLECTION_PROMPT = """\
Recent memories (chronological):
{memory_list}

Given the memories above, produce exactly {n} high-level insights about
patterns, lessons, or deeper observations. Each insight should be a
single clear sentence. Return one per line. No numbering, no bullets,
no empty lines."""


_BULLET_PREFIX_RE = re.compile(r"^\s*(?:\d+[\.)]\s*|[-*•]\s*)")


def parse_reflection_response(text: str, n: int) -> list[str]:
    """Strip bullet/number prefixes, drop empties, cap at `n` insights."""
    if not text:
        return []
    cleaned: list[str] = []
    for raw in text.splitlines():
        line = _BULLET_PREFIX_RE.sub("", raw).strip()
        if line:
            cleaned.append(line)
    return cleaned[:n]


def format_memories_for_prompt(records: list[MemoryRecord]) -> str:
    """Render records chronologically as `- [kind, imp=X.Y] content` lines."""
    chronological = sorted(records, key=lambda r: r.timestamp)
    lines: list[str] = []
    for r in chronological:
        lines.append(f"- [{r.kind}, imp={r.importance:.1f}] {r.content}")
    return "\n".join(lines)


class InsightSynthesizer:
    """Ask an LLM to synthesize high-level insights from a list of memory records."""

    def __init__(
        self,
        llm: LLM,
        *,
        num_insights: int = 3,
        temperature: float = 0.5,
        max_tokens: int = 512,
    ) -> None:
        """Wrap an LLM with synthesis defaults; `num_insights` is how many insights to request."""
        self.llm = llm
        self.num_insights = num_insights
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def synthesize(self, records: list[MemoryRecord]) -> list[str]:
        """Return up to `num_insights` insight strings; empty list when `records` is empty or output unparseable."""
        if not records:
            return []
        prompt = _REFLECTION_PROMPT.format(
            memory_list=format_memories_for_prompt(records),
            n=self.num_insights,
        )
        response = await self.llm.chat(
            [Message(role="user", content=prompt)],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return parse_reflection_response(response.content, self.num_insights)
