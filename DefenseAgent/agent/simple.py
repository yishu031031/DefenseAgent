from typing import Any

from DefenseAgent.agent.base import (
    AgentResult,
    AgentStep,
    BaseAgent,
    FAILURE_MEMORY_TYPE,
)
from DefenseAgent.config.profile import AgentProfile
from DefenseAgent.llm.llm import LLM
from DefenseAgent.llm.types import Message
from DefenseAgent.memory import ContextCompressor, DefaultMemory
from DefenseAgent.ops import AgentLogger
from DefenseAgent.reflection import Reflector
from DefenseAgent.tools import ToolRegistry


class SimpleAgent(BaseAgent):
    """Single-turn agent — one LLM call per `run()`, no tool loop. Persona, memory condensation, outcome persistence and post-run reflection still apply."""

    def __init__(
        self,
        profile: AgentProfile,
        *,
        llm: LLM,
        memory: DefaultMemory,
        tools: ToolRegistry,
        reflector: Reflector | None = None,
        logger: AgentLogger | None = None,
        compactor: ContextCompressor | None = None,
        rag: Any | None = None,
        persist_outcome: bool = True,
        reflect_after_run: bool = True,
        extra_instructions: str | None = None,
    ) -> None:
        """Wire the base modules; SimpleAgent ignores tools at the LLM call site (kept on the registry for memory_recall/rag_search and uniformity with other agents)."""
        super().__init__(
            profile,
            llm=llm,
            memory=memory,
            tools=tools,
            reflector=reflector,
            logger=logger,
            compactor=compactor,
            rag=rag,
        )
        self.persist_outcome = persist_outcome
        self.reflect_after_run = reflect_after_run
        self.extra_instructions = extra_instructions

    async def run(
        self,
        task: str,
        *,
        max_steps: int | None = None,
    ) -> AgentResult:
        """One LLM turn: condense memory → chat → record the answer; never raises AgentStepLimitError because there is no loop. `max_steps` is accepted for interface uniformity but ignored."""
        self._log("info", "agent.run.start", "starting Simple run", task=task)

        system_prompt = self._build_system_prompt()
        messages: list[Message] = [Message(role="user", content=task)]

        try:
            messages = await self._condense_memory(messages)
            response = await self.llm.chat(messages, system=system_prompt)
            step = AgentStep(
                index=0,
                kind="answer",
                content=response.content,
                usage=response.usage,
            )
            self._log(
                "info", "agent.answer", "LLM produced final answer",
                total_tokens=response.usage.total_tokens,
            )
            if self.persist_outcome:
                await self._persist_outcome(task, response.content)
            return AgentResult(
                task=task,
                final_answer=response.content,
                steps=[step],
                usage=response.usage,
            )
        except Exception as e:
            if self.persist_outcome:
                await self._persist_outcome(
                    task,
                    f"FAILED: {type(e).__name__}: {e}",
                    memory_type=FAILURE_MEMORY_TYPE,
                )
            raise
        finally:
            if self.reflect_after_run:
                await self._run_reflection_safely()

    def _build_system_prompt(self) -> str:
        """Identity prompt plus optional `extra_instructions` from the constructor."""
        identity = self._identity_prompt()
        if self.extra_instructions:
            return f"{identity}\n\n{self.extra_instructions}"
        return identity
