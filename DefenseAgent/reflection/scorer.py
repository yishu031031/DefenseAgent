import re

from DefenseAgent.llm import LLM, Message


_IMPORTANCE_PROMPT = """\
On a scale of 1 to 10, where 1 is purely mundane (e.g., brushing teeth,
making the bed) and 10 is extremely poignant (e.g., a breakup, a college
acceptance), rate the likely poignancy of the following memory.

Memory: {content}

Respond with a single integer between 1 and 10. No explanation."""


_INT_RE = re.compile(r"\d+")


def parse_importance_response(text: str, default: float = 5.0) -> float:
    """Return the first integer in `text` clipped to [1.0, 10.0]; `default` if none found."""
    if text is None:
        text = ""
    match = _INT_RE.search(text)
    if match is None:
        return default
    value = int(match.group())
    if value < 1:
        value = 1
    elif value > 10:
        value = 10
    return float(value)


class ImportanceScorer:
    """Rate how poignant a piece of content is, on a 1–10 scale, via an LLM."""

    def __init__(
        self,
        llm: LLM,
        *,
        temperature: float = 0.0,
        max_tokens: int = 16,
        default_score: float = 5.0,
    ) -> None:
        """Wrap an LLM with scoring defaults; `default_score` is used when parsing fails."""
        self.llm = llm
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.default_score = default_score

    async def score(self, content: str) -> float:
        """Return the LLM's 1–10 rating for `content`, or the configured default on parse failure."""
        prompt = _IMPORTANCE_PROMPT.format(content=content)
        response = await self.llm.chat(
            [Message(role="user", content=prompt)],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return parse_importance_response(response.content, default=self.default_score)
