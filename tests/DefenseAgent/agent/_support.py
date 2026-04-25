"""Shared stubs for the agent test suite: a scripted LLM and a zero-vector embedding adapter."""
from DefenseAgent.config import AgentProfile
from DefenseAgent.llm.types import LLMResponse, TokenUsage, ToolCall
from DefenseAgent.memory.embedding import EmbeddingAdapter


class ScriptedLLM:
    """LLM stub that plays back a pre-built list of LLMResponse objects; records every chat() call for assertions."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        """Store the scripted responses (copied) and the empty call log."""
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def chat(
        self,
        messages,
        *,
        system=None,
        tools=None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Record the call, pop and return the next scripted response; raises when the script is exhausted."""
        self.calls.append(
            {
                "messages": list(messages),
                "system": system,
                "tools": tools,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        if not self._responses:
            raise AssertionError(
                "ScriptedLLM ran out of responses — check the test's expected call count."
            )
        return self._responses.pop(0)


class ZeroEmbedder(EmbeddingAdapter):
    """Embedding stub that returns a constant [0.0] vector for every text; used so Memory works offline."""

    async def embed(self, text: str) -> list[float]:
        """Return a single-dim zero vector for any `text`."""
        return [0.0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return one zero vector per input text in order."""
        return [[0.0] for _ in texts]


def make_profile(max_steps: int = 10) -> AgentProfile:
    """Build a minimal AgentProfile suitable for agent-loop tests."""
    return AgentProfile(
        id="test_agent",
        name="Tester",
        age=25,
        traits="focused, terse",
        backstory="A test fixture.",
        initial_plan="Run tests.",
        cognitive={"max_steps_per_cycle": max_steps},  # type: ignore[arg-type]
    )


def resp(content: str = "", tool_calls: list[ToolCall] | None = None) -> LLMResponse:
    """Build a ready-to-enqueue LLMResponse with realistic-looking TokenUsage."""
    calls = list(tool_calls) if tool_calls else []
    return LLMResponse(
        content=content,
        tool_calls=calls,
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        stop_reason="tool_use" if calls else "end_turn",
        raw={},
    )
