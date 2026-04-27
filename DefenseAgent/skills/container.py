from pathlib import Path
from typing import Any, Optional, Union

from ms_agent.skill.container import (
    ExecutionInput,
    ExecutionOutput,
    ExecutionRecord,
    ExecutionSpec,
    ExecutionStatus,
    ExecutorType,
)
from ms_agent.skill.container import SkillContainer as MsSkillContainer


__all__ = [
    "SkillContainer",
    "ExecutionInput",
    "ExecutionOutput",
    "ExecutionRecord",
    "ExecutionSpec",
    "ExecutionStatus",
    "ExecutorType",
]


class SkillContainer(MsSkillContainer):
    """Local-mode subprocess executor for skill scripts. Inherits ms-agent's `SkillContainer` for the full executor surface (Python script, Python code, Python function, shell, JavaScript) — Docker isolation is bypassed by defaulting `use_sandbox=False`, which keeps execution as plain subprocesses with the inherited security pattern checks. Use this for trusted skill code; pin `use_sandbox=True` to opt back into Docker explicitly."""

    def __init__(
        self,
        *,
        workspace_dir: Optional[Union[str, Path]] = None,
        timeout: int = 300,
        enable_security_check: bool = True,
        use_sandbox: bool = False,
        **kwargs: Any,
    ) -> None:
        """Construct the container in local mode by default. `kwargs` is forwarded to ms-agent's `SkillContainer.__init__` so future fields (image, memory_limit, network_enabled) keep working without us listing each."""
        super().__init__(
            workspace_dir=workspace_dir,
            timeout=timeout,
            enable_security_check=enable_security_check,
            use_sandbox=use_sandbox,
            **kwargs,
        )
