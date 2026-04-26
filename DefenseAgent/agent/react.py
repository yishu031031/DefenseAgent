import json
from typing import Any

from DefenseAgent.agent.base import (
    AgentResult,
    AgentStep,
    AgentStepLimitError,
    BaseAgent,
    FAILURE_MEMORY_TYPE,
    add_usage,
    truncate,
)
from DefenseAgent.config.profile import AgentProfile
from DefenseAgent.llm.llm import LLM
from DefenseAgent.llm.types import Message, TokenUsage, ToolCall
from DefenseAgent.memory import ContextCompressor, DefaultMemory
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

_REACT_RAG_INSTRUCTIONS = (
    "You also have `rag_search` for static reference documents (textbooks, "
    "manuals, lore). Use it when a question would benefit from grounded facts "
    "from your knowledge base, distinct from your experiential memory."
)

_TRAJECTORY_MEMORY_TYPE = "trajectory"


class ReActAgent(BaseAgent):
    """Yao et al. 2022 — interleaved reasoning + acting. Memory is mem0-backed; trajectories and outcomes get tagged via memory_type for later filtering."""

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
        memory_recall_top_k: int = 5,
        persist_outcome: bool = True,
        persist_trajectory: bool = True,
        reflect_after_run: bool = True,
        extra_instructions: str | None = None,
    ) -> None:
        """Wire the base modules plus ReAct knobs; trajectory/outcome/failure are distinguished by mem0's memory_type tag."""
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
        self.memory_recall_top_k = memory_recall_top_k
        self.persist_outcome = persist_outcome
        self.persist_trajectory = persist_trajectory
        self.reflect_after_run = reflect_after_run
        self.extra_instructions = extra_instructions

    async def run(
        self,
        task: str,
        *,
        max_steps: int | None = None,
    ) -> AgentResult:
        """LLM-call loop: dispatch tool calls (user tools + built-in memory_recall) until a plain-text answer or max_steps. Both success and failure paths persist + reflect."""
        cap = self._resolve_max_steps(max_steps)
        self._log("info", "agent.run.start", "starting ReAct run", task=task, max_steps=cap)

        system_prompt = self._build_system_prompt()
        messages: list[Message] = [Message(role="user", content=task)]
        steps: list[AgentStep] = []
        total = TokenUsage(0, 0, 0)
        tool_specs = self._combined_tool_specs()

        try:
            for i in range(cap):
                messages = await self._condense_memory(messages)
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
                    memory_type=FAILURE_MEMORY_TYPE,
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
        """Append the assistant message, dispatch the tool calls, append results, record both steps, and persist a consolidated trajectory entry for the step."""
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
        """Write ONE memory per step summarizing every (call → result) pair, tagged memory_type='trajectory' so the LLM can recall past attempts."""
        pair_parts: list[str] = []
        for tc, tr in zip(tool_calls, tool_results):
            args_preview = _preview_json(tc.arguments)
            result_preview = truncate(tr.content or "", 100)
            pair_parts.append(f"{tc.name}({args_preview}) → {result_preview}")
        calls_summary = "; ".join(pair_parts)
        content = (
            f"Trajectory step {step_index} for task {truncate(task, 80)!r}: "
            f"{calls_summary}"
        )
        try:
            await self.memory.add(
                [Message(role="user", content=content)],
                memory_type=_TRAJECTORY_MEMORY_TYPE,
            )
        except Exception as e:
            self._log("warn", "agent.persist_trajectory_failed", str(e))

    def _build_system_prompt(self) -> str:
        """Static system prompt: identity + ReAct instructions (+ rag tip when wired + optional user extras). Memory injection is now handled by `_condense_memory` on every loop turn."""
        parts: list[str] = [self._identity_prompt()]
        parts.append(_REACT_INSTRUCTIONS)
        if self.rag is not None:
            parts.append(_REACT_RAG_INSTRUCTIONS)
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
