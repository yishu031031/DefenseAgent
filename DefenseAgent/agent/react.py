import json
from pathlib import Path
from typing import Any

from DefenseAgent.agent._builder import build_components_sync
from DefenseAgent.agent.base import (
    AgentResult,
    AgentStep,
    AgentStepLimitError,
    BaseAgent,
    FAILURE_MEMORY_TYPE,
    add_usage,
    truncate,
)
from DefenseAgent.agent.config import AgentConfig
from DefenseAgent.llm.types import Message, TokenUsage, ToolCall


_REACT_INSTRUCTIONS = (
    "You have access to tools — including `memory_recall` for searching your "
    "own memory. Call tools whenever they'd sharpen your answer; query memory "
    "any time you suspect a prior fact, preference, plan, or trajectory step "
    "is relevant. Reply in plain text (and stop calling tools) only when you "
    "have enough information to answer."
)

_REACT_RAG_INSTRUCTIONS = (
    "You also have `rag_search` for static reference documents (textbooks, "
    "manuals, lore, reports). Use it when a question would benefit from "
    "grounded facts from your knowledge base, distinct from your "
    "experiential memory.\n\n"
    "Each rag_search hit may contain inline `<resource_info>RID</resource_info>` "
    "markers and a follow-up `• resource [RID] (kind) \"caption\"` listing. "
    "When the user asks about a specific image, table, or figure cited in a "
    "hit — call `rag_get_resource` with that RID to fetch the full content "
    "(complete table markdown, image path + mime, or whatever the renderer "
    "for that kind produces)."
)

_TRAJECTORY_MEMORY_TYPE = "trajectory"


class ReActAgent(BaseAgent):
    """Yao et al. 2022 — interleaved reasoning + acting. Memory is mem0-backed; trajectories and outcomes get tagged via memory_type for later filtering.

    Constructed from an `AgentConfig`:

        config = AgentConfig(profile="DefenseAgent/examples/example_agent/profile.yaml")
        agent = ReActAgent(config)

    The agent builds its own LLM, memory, tools, reflector, compressor and
    logger from the profile + flags. MCP servers and RAG (when not pre-injected)
    are wired lazily on the first `run()` call — they need `await`.

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
        self.save_outcome = config.save_outcome and config.use_memory
        self.save_trajectory = config.save_trajectory and config.use_memory
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
        """LLM-call loop: dispatch tool calls (user tools + built-in memory_recall) until a plain-text answer or max_steps. Both success and failure paths persist + reflect. When `images` is provided, the initial user turn is sent as an OpenAI-style multimodal message (see `BaseAgent._build_user_message`)."""
        await self._ensure_async_setup()
        cap = self._resolve_max_steps(max_steps)
        self._log("info", "agent.run.start", "starting ReAct run", task=task, max_steps=cap)

        system_prompt = self._build_system_prompt()
        messages: list[Message] = [self._build_user_message(task, images)]
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
                if self.save_outcome:
                    await self._save_outcome(task, response.content)
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
            if self.save_outcome:
                await self._save_outcome(
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

        if self.save_trajectory:
            await self._save_trajectory(
                task=task,
                step_index=step_index,
                tool_calls=tool_calls,
                tool_results=tool_results,
            )

    async def _save_trajectory(
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
            self._log("warn", "agent.save_trajectory_failed", str(e))

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
