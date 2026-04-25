import json
from typing import Any

from DefenseAgent.agent.agent import (
    Agent,
    AgentResult,
    AgentStep,
    AgentStepLimitError,
    add_usage,
    truncate,
)
from DefenseAgent.config.profile import AgentProfile
from DefenseAgent.llm.llm import LLM
from DefenseAgent.llm.types import Message, TokenUsage, ToolCall
from DefenseAgent.memory import Memory
from DefenseAgent.ops import AgentLogger
from DefenseAgent.reflection import Reflector
from DefenseAgent.tools import ToolRegistry


_REACT_INSTRUCTIONS = (
    "You have access to tools — including `memory_recall` for searching your "
    "own memory. Call tools whenever they'd sharpen your answer; query memory "
    "any time you suspect a prior fact, preference, plan, or trajectory step "
    "is relevant. Reply in plain text (and stop calling tools) only when you "
    "have enough information to answer."
)

_FAILURE_OUTCOME_IMPORTANCE = 6.0


class ReActAgent(Agent):
    """Yao et al. 2022 — interleaved reasoning + acting. Memory is live-queryable via a tool; each step persists one consolidated trajectory record; reflection and outcome marking fire on both success and failure paths."""

    def __init__(
        self,
        profile: AgentProfile,
        *,
        llm: LLM,
        memory: Memory,
        tools: ToolRegistry,
        reflector: Reflector | None = None,
        logger: AgentLogger | None = None,
        memory_recall_top_k: int = 5,
        persist_outcome: bool = True,
        persist_trajectory: bool = True,
        reflect_after_run: bool = True,
        extra_instructions: str | None = None,
        trajectory_importance: float = 5.0,
    ) -> None:
        """Wire the base modules plus ReAct knobs; default `trajectory_importance=5.0` so past attempts rank alongside organic observations."""
        super().__init__(
            profile,
            llm=llm,
            memory=memory,
            tools=tools,
            reflector=reflector,
            logger=logger,
        )
        self.memory_recall_top_k = memory_recall_top_k
        self.persist_outcome = persist_outcome
        self.persist_trajectory = persist_trajectory
        self.reflect_after_run = reflect_after_run
        self.extra_instructions = extra_instructions
        self.trajectory_importance = trajectory_importance

    async def run(
        self,
        task: str,
        *,
        max_steps: int | None = None,
    ) -> AgentResult:
        """LLM-call loop: dispatch tool calls (user tools + built-in memory_recall) until a plain-text answer or max_steps. Success persists the answer; failure persists a `FAILED:` marker; both paths reflect in finally."""
        cap = self._resolve_max_steps(max_steps)
        self._log("info", "agent.run.start", "starting ReAct run", task=task, max_steps=cap)

        system_prompt = await self._build_system_prompt(task)
        messages: list[Message] = [Message(role="user", content=task)]
        steps: list[AgentStep] = []
        total = TokenUsage(0, 0, 0)
        tool_specs = self._combined_tool_specs()

        try:
            for i in range(cap):
                response = await self.llm.chat(
                    messages, system=system_prompt, tools=tool_specs,
                )
                total = add_usage(total, response.usage)

                if response.tool_calls:
                    await self._handle_tool_turn(
                        step_index=i,
                        task=task,
                        response=response,
                        messages=messages,
                        steps=steps,
                    )
                    continue

                steps.append(
                    AgentStep(
                        index=i,
                        kind="answer",
                        content=response.content,
                        usage=response.usage,
                    )
                )
                self._log(
                    "info",
                    "agent.answer",
                    "LLM produced final answer",
                    step=i,
                    total_tokens=total.total_tokens,
                )
                if self.persist_outcome:
                    await self._persist_outcome(task, response.content)
                return AgentResult(
                    task=task,
                    final_answer=response.content,
                    steps=steps,
                    usage=total,
                )

            self._log(
                "warn",
                "agent.max_steps",
                "ReAct exhausted max_steps without a final answer",
                max_steps=cap,
            )
            raise AgentStepLimitError(
                f"ReAct exceeded max_steps={cap} without producing a final answer"
            )
        except AgentStepLimitError:
            if self.persist_outcome:
                await self._persist_outcome(
                    task,
                    f"FAILED: exceeded max_steps={cap}",
                    importance=_FAILURE_OUTCOME_IMPORTANCE,
                )
            raise
        finally:
            if self.reflect_after_run:
                await self._run_reflection_safely()

    async def _handle_tool_turn(
        self,
        *,
        step_index: int,
        task: str,
        response: Any,
        messages: list[Message],
        steps: list[AgentStep],
    ) -> None:
        """Append the assistant message, dispatch the tool calls, append results, record both steps, and persist one consolidated trajectory entry for the step."""
        tool_calls = list(response.tool_calls)
        messages.append(
            Message(
                role="assistant",
                content=response.content or "",
                tool_calls=tool_calls,
            )
        )
        steps.append(
            AgentStep(
                index=step_index,
                kind="tool_call",
                content=response.content or "",
                tool_calls=tool_calls,
                usage=response.usage,
            )
        )
        self._log(
            "info",
            "agent.tool_call",
            "LLM requested tools",
            step=step_index,
            tool_names=[tc.name for tc in tool_calls],
        )

        tool_results = await self._dispatch_tool_calls(tool_calls)
        messages.extend(tool_results)
        steps.append(
            AgentStep(
                index=step_index,
                kind="tool_result",
                tool_results=list(tool_results),
            )
        )

        if self.persist_trajectory:
            await self._persist_trajectory(
                task=task,
                step_index=step_index,
                tool_calls=tool_calls,
                tool_results=tool_results,
            )

    async def _persist_trajectory(
        self,
        *,
        task: str,
        step_index: int,
        tool_calls: list[ToolCall],
        tool_results: list[Message],
    ) -> None:
        """Write ONE observation per step summarizing every (call → result) pair — minimizes embedding calls and gives each entry step-level coherence."""
        pair_parts: list[str] = []
        for tc, tr in zip(tool_calls, tool_results):
            args_preview = _preview_json(tc.arguments)
            result_preview = truncate(tr.content or "", 100)
            pair_parts.append(f"{tc.name}({args_preview}) → {result_preview}")
        calls_summary = "; ".join(pair_parts)

        content = f"Trajectory step {step_index}: {calls_summary}"
        await self.memory.remember(
            content,
            kind="observation",
            importance=self.trajectory_importance,
            metadata={
                "trajectory": True,
                "task": truncate(task, 120),
                "tool_names": [tc.name for tc in tool_calls],
                "step": step_index,
            },
        )

    async def _build_system_prompt(self, task: str) -> str:
        """Identity + upfront memory prime + ReAct instructions (+ optional user extras), joined with blank lines."""
        memories = await self._recall_memories(task, self.memory_recall_top_k)
        parts: list[str] = [self._identity_prompt()]
        memory_block = self._memory_block(memories)
        if memory_block:
            parts.append(memory_block)
        parts.append(_REACT_INSTRUCTIONS)
        if self.extra_instructions:
            parts.append(self.extra_instructions)
        return "\n\n".join(parts)


def _preview_json(value: dict[str, Any], *, max_len: int = 80) -> str:
    """Render a tool-args dict compactly for trajectory storage; truncates with ellipsis past max_len."""
    try:
        rendered = json.dumps(value, default=str, ensure_ascii=False, sort_keys=True)
    except Exception:
        rendered = str(value)
    return truncate(rendered, max_len)
