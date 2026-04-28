"""DefenseAgent — agent framework with mem0-backed memory, reflection, RAG, and tools.

The recommended top-level entry points:

    from DefenseAgent import create_agent
    from DefenseAgent.examples import EXAMPLE_PROFILE_PATH

    agent = create_agent(EXAMPLE_PROFILE_PATH)
    result = await agent.run("Hello")

Or, when you need full control over the config:

    from DefenseAgent import AgentConfig, ReActAgent
    from DefenseAgent.examples import EXAMPLE_PROFILE_PATH

    config = AgentConfig(profile=EXAMPLE_PROFILE_PATH, tools=[my_func])
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

__version__ = "0.1.3"

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
