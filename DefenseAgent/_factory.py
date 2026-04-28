"""One-shot agent factory.

`create_agent(...)` is the SDK's front door — collapses the two-step
`config = AgentConfig(...); agent = ReActAgent(config)` into a single call
and accepts the most convenient input shape per use case.
"""
from pathlib import Path
from typing import Any, Literal

from DefenseAgent.agent import (
    AgentConfig,
    BaseAgent,
    PlanAndSolveAgent,
    ReActAgent,
    SimpleAgent,
)


Strategy = Literal["simple", "react", "plan_and_solve"]


_STRATEGIES: dict[str, type[BaseAgent]] = {
    "simple": SimpleAgent,
    "react": ReActAgent,
    "plan_and_solve": PlanAndSolveAgent,
}


def create_agent(
    config: AgentConfig | dict[str, Any] | str | Path,
    *,
    strategy: Strategy = "react",
) -> BaseAgent:
    """Build an agent in one call.

    Accepts whichever config shape is most convenient:

    * ``AgentConfig`` instance → used as-is.
    * ``dict`` → ``AgentConfig(**config)``.
    * ``str`` / ``Path`` → treated as a profile YAML path
      (``AgentConfig(profile=...)``).

    The ``strategy`` kwarg picks the concrete agent class (defaults to
    ``"react"`` — the most general-purpose). Equivalent to constructing
    ``ReActAgent(config)`` / ``SimpleAgent(config)`` /
    ``PlanAndSolveAgent(config)`` directly.

    Examples
    --------
    >>> from DefenseAgent import create_agent
    >>> from DefenseAgent.examples import EXAMPLE_PROFILE_PATH
    >>> agent = create_agent(EXAMPLE_PROFILE_PATH)
    >>> agent = create_agent({"profile": "...", "use_rag": True})
    >>> agent = create_agent(AgentConfig(profile="..."), strategy="plan_and_solve")
    """
    if isinstance(config, AgentConfig):
        resolved = config
    elif isinstance(config, (str, Path)):
        resolved = AgentConfig(profile=config)
    elif isinstance(config, dict):
        resolved = AgentConfig(**config)
    else:
        raise TypeError(
            f"create_agent: unsupported config type {type(config).__name__!r}; "
            "expected AgentConfig, dict, str, or Path"
        )

    try:
        agent_cls = _STRATEGIES[strategy]
    except KeyError:
        valid = ", ".join(sorted(_STRATEGIES))
        raise ValueError(
            f"create_agent: unknown strategy {strategy!r}; expected one of {valid}"
        ) from None

    return agent_cls(resolved)
