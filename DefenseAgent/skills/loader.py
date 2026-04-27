import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ms_agent.skill.loader import SkillLoader as MsSkillLoader
from ms_agent.skill.schema import SkillFile, SkillSchema

from DefenseAgent.tools.types import SkillLoadError, Tool, ToolHandler


if TYPE_CHECKING:
    from DefenseAgent.skills.container import SkillContainer


_FRONTMATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)

_SKILL_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "file": {
            "type": "string",
            "description": (
                "Optional file inside the skill directory to fetch. Match by basename "
                "(e.g. 'generate.py') for files enumerated in the skill schema, or by "
                "POSIX-style relative path (e.g. 'scripts/generate.py'). Omit to "
                "receive the SKILL.md body — the skill's primary instructions."
            ),
        },
    },
}

_SCRIPT_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "args": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Positional command-line arguments forwarded to the script.",
        },
        "stdin": {
            "type": "string",
            "description": "Optional standard-input payload piped to the script.",
        },
        "timeout": {
            "type": "integer",
            "description": "Optional per-call timeout in seconds (overrides the container default).",
        },
    },
}


class SkillLoader(MsSkillLoader):
    """DefenseAgent's skill loader — inherits ms-agent's `SkillLoader` (load + scan + reload + registry) and adds `to_tools()` to expose every loaded skill as a `Tool` for the LLM. The runtime tool handler is generic: no `file` arg returns the SKILL.md body, a `file` arg returns that file's contents from anywhere inside the skill directory."""

    def to_tools(self, container: "SkillContainer | None" = None) -> list[Tool]:
        """Convert every skill in the registry into one read-only Tool (`name = schema.name`) plus, when `container` is provided, one executable Tool per script in `schema.scripts` (`name = '{schema.name}__{script_stem}'`). The read-only handler resolves `file` arguments against `schema.files` and `schema.skill_path`; the executable handler dispatches to the container's `execute_python_script` / `execute_shell` / `execute_javascript` based on the script's extension."""
        out: list[Tool] = []
        for schema in self.all_skills().values():
            out.append(_schema_to_tool(schema))
            if container is not None:
                for script in schema.scripts:
                    out.append(_script_to_executable_tool(schema, script, container))
        return out

    def all_skills(self) -> dict[str, SkillSchema]:
        """Return a shallow copy of the loaded-skills mapping (keys are `{skill_id}@{version}`). ms-agent calls this `get_all_skills()`; we keep both spellings for ergonomics."""
        return self.get_all_skills()


_EXECUTOR_FOR_SUFFIX: dict[str, str] = {
    ".py": "python_script",
    ".sh": "shell",
    ".bash": "shell",
    ".js": "javascript",
    ".mjs": "javascript",
}


def _script_to_executable_tool(
    schema: SkillSchema,
    script: SkillFile,
    container: "SkillContainer",
) -> Tool:
    """Wrap one script inside a skill as an executable Tool. The script's extension picks the executor: `.py` → python_script, `.sh`/`.bash` → shell, `.js`/`.mjs` → javascript. Tool name is `'{schema.name}__{script_stem}'`. The handler builds an `ExecutionInput` from the LLM-provided `args`/`stdin`/`timeout` and renders the resulting `ExecutionOutput` as a string the LLM can consume."""
    from DefenseAgent.skills.container import ExecutionInput, ExecutorType

    suffix = Path(script.path).suffix
    executor_name = _EXECUTOR_FOR_SUFFIX.get(suffix)
    if executor_name is None:
        raise SkillLoadError(
            f"unsupported script extension {suffix!r} for {script.name} "
            f"(supported: {sorted(_EXECUTOR_FOR_SUFFIX)})"
        )
    executor_type = ExecutorType(executor_name)
    script_stem = Path(script.name).stem
    tool_name = f"{schema.name}__{script_stem}"

    async def handler(arguments: dict[str, Any]) -> str:
        raw_args = arguments.get("args") or []
        if not isinstance(raw_args, list) or not all(isinstance(a, str) for a in raw_args):
            raise SkillLoadError("'args' must be a list of strings")
        stdin = arguments.get("stdin")
        if stdin is not None and not isinstance(stdin, str):
            raise SkillLoadError("'stdin' must be a string")
        original_timeout = container.timeout
        timeout_arg = arguments.get("timeout")
        if timeout_arg is not None:
            if not isinstance(timeout_arg, int) or timeout_arg <= 0:
                raise SkillLoadError("'timeout' must be a positive integer")
            container.timeout = timeout_arg
        try:
            input_spec = ExecutionInput(args=list(raw_args), stdin=stdin)
            if executor_type == ExecutorType.SHELL:
                output = await container.execute_shell(
                    command=[str(Path(script.path)), *raw_args],
                    skill_id=schema.skill_id,
                    input_spec=input_spec,
                )
            else:
                output = await container.execute(
                    executor_type=executor_type,
                    skill_id=schema.skill_id,
                    script_path=Path(script.path),
                    input_spec=input_spec,
                )
        finally:
            container.timeout = original_timeout
        return _render_execution_output(output)

    return Tool(
        name=tool_name,
        description=(
            f"Execute the {script.name!r} script bundled with the {schema.name!r} skill. "
            f"Provide CLI args via `args`, optional `stdin`, and an optional `timeout` "
            f"(seconds). Returns stdout/stderr and the process exit code."
        ),
        input_schema=_SCRIPT_INPUT_SCHEMA,
        source="skill",
        handler=handler,
        metadata={
            "skill_id": schema.skill_id,
            "version": schema.version,
            "skill_path": str(schema.skill_path),
            "script": script.name,
            "executor": executor_name,
        },
    )


def _render_execution_output(output: Any) -> str:
    """Format an `ExecutionOutput` for the LLM: `stdout` first, then a separator + `stderr` when present, with a trailing exit-code marker."""
    parts: list[str] = []
    stdout = (output.stdout or "").rstrip()
    stderr = (output.stderr or "").rstrip()
    if stdout:
        parts.append(stdout)
    if stderr:
        parts.append(f"--- stderr ---\n{stderr}")
    parts.append(f"[exit_code={output.exit_code} duration_ms={output.duration_ms:.0f}]")
    return "\n".join(parts)


def _schema_to_tool(schema: SkillSchema) -> Tool:
    """Wrap one ms-agent SkillSchema as a DefenseAgent Tool with progressive-disclosure behaviour."""
    return Tool(
        name=schema.name,
        description=schema.description,
        input_schema=_SKILL_INPUT_SCHEMA,
        source="skill",
        handler=_make_handler(schema),
        metadata={
            "skill_id": schema.skill_id,
            "version": schema.version,
            "skill_path": str(schema.skill_path),
        },
    )


def _make_handler(schema: SkillSchema) -> ToolHandler:
    """Build the async handler for one skill: dispatch on the optional `file` argument, otherwise return the SKILL.md body."""
    async def handler(arguments: dict[str, Any]) -> str:
        file_arg = arguments.get("file")
        if file_arg in (None, ""):
            return _strip_frontmatter(schema.content)
        if not isinstance(file_arg, str):
            raise SkillLoadError(
                f"'file' argument must be a string, got {type(file_arg).__name__}"
            )
        return _read_skill_file(schema, file_arg)
    return handler


def _read_skill_file(schema: SkillSchema, requested: str) -> str:
    """Resolve a `file` argument against the SkillSchema's enumerated files first (basename match), then by relative path inside `skill_path`. Rejects path-escape attempts."""
    if requested.startswith("/") or requested.startswith("\\"):
        raise SkillLoadError(f"absolute paths are not allowed: {requested!r}")

    if requested == "SKILL.md":
        return _strip_frontmatter(schema.content)

    by_name = schema.get_file_by_name(requested)
    if by_name is not None:
        return _safe_read(Path(by_name.path), root=schema.skill_path, label=requested)

    candidate = (schema.skill_path / requested).resolve()
    return _safe_read(candidate, root=schema.skill_path, label=requested)


def _safe_read(path: Path, *, root: Path, label: str) -> str:
    """Read `path`, but only if it actually lives under `root`; otherwise refuse."""
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as e:
        raise SkillLoadError(
            f"path escapes skill directory {root}: {label!r}"
        ) from e
    if not resolved.is_file():
        raise SkillLoadError(f"no such file in skill: {label}")
    return resolved.read_text(encoding="utf-8")


def _strip_frontmatter(content: str) -> str:
    """Drop the leading YAML frontmatter block from a SKILL.md file so handlers return just the body."""
    return _FRONTMATTER_RE.sub("", content, count=1).lstrip("\n")
