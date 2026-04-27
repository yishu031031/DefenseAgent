"""Tests for DefenseAgent.skills.container.SkillContainer.

We inherit ms-agent's `SkillContainer` and run it in local subprocess mode (use_sandbox=False), so these tests exercise real subprocesses but stay offline. Heavier sandbox-mode behaviour (Docker/EnclaveSandbox) is out of scope; this file covers the local executor + dataclass surface.
"""
import asyncio
from pathlib import Path

import pytest

from DefenseAgent.skills import (
    ExecutionInput,
    ExecutionOutput,
    ExecutionStatus,
    ExecutorType,
    SkillContainer,
)


# ---------- construction ----------


def test_container_defaults_to_local_mode(tmp_path: Path) -> None:
    container = SkillContainer(workspace_dir=tmp_path)
    assert container.use_sandbox is False
    assert container.workspace_dir == tmp_path.resolve()
    assert (tmp_path / "outputs").is_dir()
    assert (tmp_path / "scripts").is_dir()


def test_container_can_be_forced_into_sandbox_mode(tmp_path: Path) -> None:
    """Default is local; passing use_sandbox=True keeps the door open for callers who actually want Docker isolation. We don't exercise the sandbox path here — just check the flag is respected."""
    container = SkillContainer(workspace_dir=tmp_path, use_sandbox=True)
    assert container.use_sandbox is True


# ---------- execute_python_script ----------


def test_execute_python_script_returns_stdout(tmp_path: Path) -> None:
    script = tmp_path / "echo.py"
    script.write_text("print('hello from skill')\n", encoding="utf-8")
    container = SkillContainer(workspace_dir=tmp_path / "ws")

    output = asyncio.run(container.execute_python_script(script))
    assert isinstance(output, ExecutionOutput)
    assert output.exit_code == 0
    assert "hello from skill" in output.stdout


def test_execute_python_script_records_failure_when_exit_nonzero(tmp_path: Path) -> None:
    script = tmp_path / "fail.py"
    script.write_text("import sys; sys.exit(7)\n", encoding="utf-8")
    container = SkillContainer(workspace_dir=tmp_path / "ws")

    output = asyncio.run(container.execute_python_script(script))
    assert output.exit_code == 7
    assert container.spec.records[-1].status == ExecutionStatus.FAILED


def test_execute_python_script_blocks_dangerous_pattern(tmp_path: Path) -> None:
    script = tmp_path / "bad.py"
    script.write_text("import os\nos.system('echo nope')\n", encoding="utf-8")
    container = SkillContainer(workspace_dir=tmp_path / "ws")

    output = asyncio.run(container.execute_python_script(script))
    assert output.exit_code == -1
    assert "Security check failed" in output.stderr
    assert container.spec.records[-1].status == ExecutionStatus.SECURITY_BLOCKED


def test_execute_python_script_can_disable_security_check(tmp_path: Path) -> None:
    script = tmp_path / "ok.py"
    script.write_text("print('passthrough')\n", encoding="utf-8")
    container = SkillContainer(workspace_dir=tmp_path / "ws", enable_security_check=False)

    output = asyncio.run(container.execute_python_script(script))
    assert output.exit_code == 0


# ---------- execute via the unified entrypoint ----------


def test_execute_dispatches_python_script(tmp_path: Path) -> None:
    script = tmp_path / "hello.py"
    script.write_text("print('via execute()')\n", encoding="utf-8")
    container = SkillContainer(workspace_dir=tmp_path / "ws")

    output = asyncio.run(
        container.execute(
            executor_type=ExecutorType.PYTHON_SCRIPT,
            skill_id="test",
            script_path=script,
        )
    )
    assert "via execute()" in output.stdout


def test_execute_python_code_runs_inline(tmp_path: Path) -> None:
    container = SkillContainer(workspace_dir=tmp_path / "ws")
    output = asyncio.run(
        container.execute_python_code(
            code="print(2 + 2)\n",
            skill_id="test",
        )
    )
    assert output.exit_code == 0
    assert "4" in output.stdout


# ---------- execution spec accumulates ----------


def test_spec_accumulates_records_across_executions(tmp_path: Path) -> None:
    container = SkillContainer(workspace_dir=tmp_path / "ws")
    asyncio.run(container.execute_python_code(code="print(1)\n", skill_id="a"))
    asyncio.run(container.execute_python_code(code="print(2)\n", skill_id="b"))
    asyncio.run(container.execute_python_code(code="print(3)\n", skill_id="c"))

    assert len(container.spec.records) == 3
    assert [r.skill_id for r in container.spec.records] == ["a", "b", "c"]


# ---------- timeout ----------


def test_execute_python_code_respects_timeout(tmp_path: Path) -> None:
    """A script that sleeps longer than the configured timeout is killed; subprocess returns non-zero. We don't pin the exact ExecutionStatus because ms-agent's local executor distinguishes timeout via stderr text."""
    container = SkillContainer(workspace_dir=tmp_path / "ws", timeout=1)
    output = asyncio.run(
        container.execute_python_code(
            code="import time; time.sleep(5)\n",
            skill_id="slow",
        )
    )
    assert output.exit_code != 0


# ---------- ExecutionInput plumbing ----------


def test_execution_input_args_arrive_as_argv(tmp_path: Path) -> None:
    """The Python local executor injects `args` into `sys.argv[1:]` via its env_setup preamble."""
    script = tmp_path / "args.py"
    script.write_text("import sys; print('|'.join(sys.argv[1:]))\n", encoding="utf-8")
    container = SkillContainer(workspace_dir=tmp_path / "ws")

    output = asyncio.run(
        container.execute_python_script(
            script,
            input_spec=ExecutionInput(args=["alpha", "beta", "gamma"]),
        )
    )
    assert output.exit_code == 0
    assert "alpha|beta|gamma" in output.stdout
