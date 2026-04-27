"""DefenseAgent — agent framework with mem0-backed memory, reflection, RAG, and tools.

The recommended top-level entry points:

    from DefenseAgent import create_agent

    agent = create_agent("agents/example_agent/profile.yaml")
    result = await agent.run("Hello")

Or, when you need full control over the config:

    from DefenseAgent import AgentConfig, ReActAgent

    config = AgentConfig(profile="agents/example_agent/profile.yaml", tools=[my_func])
    agent = ReActAgent(config)
"""
from DefenseAgent._factory import create_agent
from DefenseAgent.agent import (
    AgentConfig,
    AgentError,
    AgentResult,
    AgentStep,
    AgentStepLimitError,
    BaseAgent,
    PlanAndSolveAgent,
    ReActAgent,
    SimpleAgent,
)
from DefenseAgent.config import AgentProfile

__version__ = "0.1.0"

__all__ = [
    "create_agent",
    "AgentConfig",
    "AgentProfile",
    "BaseAgent",
    "SimpleAgent",
    "ReActAgent",
    "PlanAndSolveAgent",
    "AgentResult",
    "AgentStep",
    "AgentError",
    "AgentStepLimitError",
    "__version__",
]
