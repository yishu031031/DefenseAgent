from DefenseAgent.agent._builder import build_components_sync
from DefenseAgent.agent.base import (
    AgentResult,
    AgentStep,
    BaseAgent,
    FAILURE_MEMORY_TYPE,
)
from DefenseAgent.agent.config import AgentConfig
from DefenseAgent.llm.types import Message


class SimpleAgent(BaseAgent):
    """Single-turn agent — one LLM call per `run()`, no tool loop. Persona, memory condensation, outcome persistence and post-run reflection still apply.

    Constructed from an `AgentConfig`:

        config = AgentConfig(profile="agents/maya.yaml")
        agent = SimpleAgent(config)

    Inject pre-built components (mocks, custom adapters) via the `llm`,
    `memory`, `tools_registry`, `reflector`, `rag`, `logger` fields on
    `AgentConfig`.
    """

    def __init__(self, config: AgentConfig) -> None:
        """Build the agent from an `AgentConfig` — the only supported construction path."""
        built = build_components_sync(config)
        super().__init__(
            built.profile,
            llm=built.llm,
            memory=built.memory,
            tools=built.tools,
            reflector=built.reflector,
            logger=built.logger,
            compactor=built.compactor,
            rag=built.rag,
        )
        self._config = config
        self.persist_outcome = config.persist_outcome and config.use_memory
        self.reflect_after_run = (
            config.reflect_after_run and config.use_reflection and config.use_memory
        )
        self.extra_instructions = config.extra_instructions

    async def run(
        self,
        task: str,
        *,
        max_steps: int | None = None,
    ) -> AgentResult:
        """One LLM turn: condense memory → chat → record the answer; never raises AgentStepLimitError because there is no loop. `max_steps` is accepted for interface uniformity but ignored."""
        await self._ensure_async_setup()
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
