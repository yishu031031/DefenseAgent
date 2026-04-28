from pathlib import Path

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

        config = AgentConfig(profile="DefenseAgent/examples/example_agent/profile.yaml")
        agent = SimpleAgent(config)

    Inject pre-built components (mocks, custom adapters) via the `llm`,
    `memory`, `tool_registry`, `reflector`, `rag`, `logger` fields on
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
            compressor=built.compressor,
            rag=built.rag,
        )
        self._config = config
        self.save_outcome = config.save_outcome and config.use_memory
        self.reflect_after_run = (
            config.reflect_after_run and config.use_reflection and config.use_memory
        )
        self.extra_instructions = config.extra_instructions

    async def run(
        self,
        task: str,
        *,
        max_steps: int | None = None,
        images: "list[str | Path] | None" = None,
    ) -> AgentResult:
        """One LLM turn: condense memory → chat → record the answer; never raises AgentStepLimitError because there is no loop. `max_steps` is accepted for interface uniformity but ignored. When `images` is provided, the user turn becomes an OpenAI-style multimodal message (see `BaseAgent._build_user_message`)."""
        await self._ensure_async_setup()
        self._log("info", "agent.run.start", "starting Simple run", task=task)

        system_prompt = self._build_system_prompt()
        messages: list[Message] = [self._build_user_message(task, images)]

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
            if self.save_outcome:
                await self._save_outcome(task, response.content)
            return AgentResult(
                task=task,
                final_answer=response.content,
                steps=[step],
                usage=response.usage,
            )
        except Exception as e:
            if self.save_outcome:
                await self._save_outcome(
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
