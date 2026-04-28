import re
from pathlib import Path

from DefenseAgent.agent._builder import build_components_sync
from DefenseAgent.agent.base import (
    AgentError,
    AgentResult,
    AgentStep,
    BaseAgent,
    FAILURE_MEMORY_TYPE,
    add_usage,
    truncate,
)
from DefenseAgent.agent.config import AgentConfig
from DefenseAgent.llm.types import Message, TokenUsage


_PLAN_PROMPT_TEMPLATE = """\
Task: {task}

Break the task into 2–5 concrete, ordered steps that together will solve it.
Return one step per line, prefixed with "1. ", "2. ", etc. No preamble, no
explanation, and no extra blank lines."""


_EXEC_INSTRUCTIONS = (
    "You are executing ONE step of a larger plan. You may call tools via the "
    "function-calling interface. When you have a concise result for this step, "
    "reply in plain text and stop calling tools."
)


_SYNTHESIS_PROMPT_TEMPLATE = """\
Original task: {task}

You executed the following plan:

{plan_with_results}

Write the final answer to the original task in 2–4 sentences. Do not restate
the plan; just give the user-facing answer."""


_STEP_LINE_RE = re.compile(r"^\s*\d+[\.)]\s*(.+?)\s*$")


class PlanAndSolveAgent(BaseAgent):
    """Wang et al. 2023 — plan the task into discrete steps, execute each with tools, then synthesize the final answer.

    Constructed from an `AgentConfig`:

        config = AgentConfig(profile="DefenseAgent/examples/example_agent/profile.yaml")
        agent = PlanAndSolveAgent(config)

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
        self.memory_recall_top_k = config.memory_recall_top_k
        self.max_substeps_per_step = config.max_substeps_per_step
        self.save_outcome = config.save_outcome and config.use_memory
        self.reflect_after_run = (
            config.reflect_after_run and config.use_reflection and config.use_memory
        )

    async def run(
        self,
        task: str,
        *,
        max_steps: int | None = None,
        images: "list[str | Path] | None" = None,
    ) -> AgentResult:
        """Three phases: (1) plan, (2) execute each step with a short tool loop, (3) synthesize. Reflection fires on every exit path via finally. When `images` is provided, Phase 1's planning user-turn and every Phase 2 execute-step user-turn become OpenAI-style multimodal messages, so each phase that re-references the original task can still see the image content (Phase 3 synthesises from per-step text outputs and stays text-only)."""
        await self._ensure_async_setup()
        cap = self._resolve_max_steps(max_steps)
        self._log(
            "info", "agent.run.start", "starting Plan-and-Solve run",
            task=task, max_steps=cap,
        )

        try:
            identity = self._identity_prompt()

            steps: list[AgentStep] = []
            total = TokenUsage(0, 0, 0)

            # Phase 1 — plan.
            plan_messages = [
                self._build_user_message(_PLAN_PROMPT_TEMPLATE.format(task=task), images),
            ]
            plan_messages = await self._condense_memory(plan_messages)
            plan_response = await self.llm.chat(
                plan_messages,
                system=identity,
            )
            total = add_usage(total, plan_response.usage)
            plan = _parse_plan(plan_response.content)
            if not plan:
                raise AgentError(
                    f"planning response did not produce any steps:\n{plan_response.content!r}"
                )
            if len(plan) > cap:
                plan = plan[:cap]
            steps.append(
                AgentStep(
                    index=0, kind="plan",
                    content="\n".join(plan), usage=plan_response.usage,
                )
            )
            self._log("info", "agent.plan", "plan produced", step_count=len(plan))

            # Phase 2 — execute each step.
            exec_system = _join_blocks(identity, _EXEC_INSTRUCTIONS)
            tool_specs = self._combined_tool_specs()
            step_outputs: list[str] = []
            for plan_index, plan_step in enumerate(plan):
                step_answer, step_usage = await self._execute_plan_step(
                    plan_step=plan_step,
                    original_task=task,
                    exec_system=exec_system,
                    tool_specs=tool_specs,
                    step_index=plan_index + 1,
                    all_steps=steps,
                    images=images,
                )
                total = add_usage(total, step_usage)
                step_outputs.append(step_answer)

            # Phase 3 — synthesize.
            plan_with_results = "\n".join(
                f"{i + 1}. {plan[i]}\n   → {step_outputs[i]}"
                for i in range(len(plan))
            )
            synthesis_messages = [
                Message(
                    role="user",
                    content=_SYNTHESIS_PROMPT_TEMPLATE.format(
                        task=task, plan_with_results=plan_with_results,
                    ),
                )
            ]
            synthesis_messages = await self._condense_memory(synthesis_messages)
            synthesis_response = await self.llm.chat(
                synthesis_messages,
                system=identity,
            )
            total = add_usage(total, synthesis_response.usage)
            steps.append(
                AgentStep(
                    index=len(steps),
                    kind="answer",
                    content=synthesis_response.content,
                    usage=synthesis_response.usage,
                )
            )
            self._log(
                "info", "agent.answer", "Plan-and-Solve synthesized final answer",
                total_tokens=total.total_tokens,
            )

            if self.save_outcome:
                await self._save_outcome(task, synthesis_response.content)

            return AgentResult(
                task=task,
                final_answer=synthesis_response.content,
                steps=steps,
                usage=total,
            )
        except AgentError as e:
            if self.save_outcome:
                await self._save_outcome(
                    task,
                    f"FAILED: {truncate(str(e), 200)}",
                    memory_type=FAILURE_MEMORY_TYPE,
                )
            raise
        finally:
            if self.reflect_after_run:
                await self._run_reflection_safely()

    async def _execute_plan_step(
        self,
        *,
        plan_step: str,
        original_task: str,
        exec_system: str,
        tool_specs: list[dict] | None,
        step_index: int,
        all_steps: list[AgentStep],
        images: "list[str | Path] | None" = None,
    ) -> tuple[str, TokenUsage]:
        """Run a short ReAct-style sub-loop for ONE planned step; returns (step_answer, sub-loop usage). When `images` is set, the per-step user-turn carries the same multimodal content as the original task so each step can re-inspect the image content."""
        messages: list[Message] = [
            self._build_user_message(
                f"Original task: {original_task}\nExecute ONLY this step: {plan_step}",
                images,
            )
        ]
        sub_total = TokenUsage(0, 0, 0)

        for _ in range(self.max_substeps_per_step):
            messages = await self._condense_memory(messages)
            response = await self.llm.chat(
                messages, system=exec_system, tools=tool_specs,
            )
            sub_total = add_usage(sub_total, response.usage)

            if response.tool_calls:
                messages.append(
                    Message(
                        role="assistant",
                        content=response.content or "",
                        tool_calls=list(response.tool_calls),
                    )
                )
                all_steps.append(
                    AgentStep(
                        index=len(all_steps),
                        kind="tool_call",
                        content=response.content or "",
                        tool_calls=list(response.tool_calls),
                        usage=response.usage,
                    )
                )
                tool_results = await self._dispatch_tool_calls(response.tool_calls)
                messages.extend(tool_results)
                all_steps.append(
                    AgentStep(
                        index=len(all_steps),
                        kind="tool_result",
                        tool_results=list(tool_results),
                    )
                )
                continue

            return response.content, sub_total

        # Ran out of sub-steps: let the caller proceed with whatever the last
        # assistant content was, or a diagnostic string if none.
        fallback = f"(step {step_index} incomplete: tool loop exceeded)"
        return fallback, sub_total


def _parse_plan(text: str) -> list[str]:
    """Extract numbered steps from the LLM's planning response; tolerates extra whitespace and both '1.' and '1)' styles."""
    if not text:
        return []
    out: list[str] = []
    for line in text.splitlines():
        match = _STEP_LINE_RE.match(line)
        if match:
            body = match.group(1).strip()
            if body:
                out.append(body)
    return out


def _join_blocks(*blocks: str) -> str:
    """Join non-empty string blocks with blank lines (drops any empty or whitespace-only block)."""
    return "\n\n".join(b for b in blocks if b and b.strip())
