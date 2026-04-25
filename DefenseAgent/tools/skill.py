from pathlib import Path
from typing import Any

import yaml

from DefenseAgent.tools.types import SkillLoadError, Tool


_FRONTMATTER_DELIM = "---"

_SKILL_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "file": {
            "type": "string",
            "description": (
                "Optional POSIX-style relative path to an additional file inside the "
                "skill directory (Layer 3). Omit to load the skill's main instructions "
                "(Layer 2: the SKILL.md body)."
            ),
        },
    },
}


class Skill:
    """One Anthropic-style Agent Skill: a directory with SKILL.md + optional assets, served with progressive disclosure."""

    def __init__(self, directory: str | Path) -> None:
        """Resolve the skill root, parse SKILL.md frontmatter eagerly, remember the body for on-demand serving."""
        root = Path(directory).resolve()
        if not root.is_dir():
            raise SkillLoadError(f"skill path is not a directory: {root}")
        skill_md = root / "SKILL.md"
        if not skill_md.is_file():
            raise SkillLoadError(f"SKILL.md not found in {root}")
        raw = skill_md.read_text(encoding="utf-8")
        name, description, body = _parse_frontmatter(raw, source=str(skill_md))
        self.root = root
        self.name = name
        self.description = description
        self._body = body

    @property
    def body(self) -> str:
        """Return the Layer-2 payload (SKILL.md body after frontmatter)."""
        return self._body

    def read_file(self, relative_path: str) -> str:
        """Return a Layer-3 file's contents; rejects paths that escape the skill root."""
        return self._read_file(relative_path)

    def to_tool(self) -> Tool:
        """Wrap this skill as a Tool whose name/description are Layer 1, and whose handler serves Layer 2 or 3."""
        return Tool(
            name=self.name,
            description=self.description,
            input_schema=_SKILL_INPUT_SCHEMA,
            source="skill",
            handler=self._handle,
            metadata={"root": str(self.root)},
        )

    async def _handle(self, arguments: dict[str, Any]) -> str:
        """Dispatch on presence of a `file` arg: no file → Layer 2 body, file → Layer 3 contents."""
        file = arguments.get("file")
        if file is None or file == "":
            return self._body
        if not isinstance(file, str):
            raise SkillLoadError(f"'file' argument must be a string, got {type(file).__name__}")
        return self._read_file(file)

    def _read_file(self, relative_path: str) -> str:
        """Resolve `relative_path` inside the skill root and return its contents; raise on escape or miss."""
        if relative_path.startswith("/") or relative_path.startswith("\\"):
            raise SkillLoadError(f"absolute paths are not allowed: {relative_path!r}")
        target = (self.root / relative_path).resolve()
        try:
            target.relative_to(self.root)
        except ValueError:
            raise SkillLoadError(
                f"path escapes skill directory {self.root}: {relative_path!r}"
            )
        if target == self.root / "SKILL.md":
            return self._body
        if not target.is_file():
            raise SkillLoadError(
                f"no such file in skill {self.name!r}: {relative_path}"
            )
        return target.read_text(encoding="utf-8")


def _parse_frontmatter(text: str, *, source: str) -> tuple[str, str, str]:
    """Split a '---'-fenced YAML frontmatter block from the body; return (name, description, body) or raise."""
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != _FRONTMATTER_DELIM:
        raise SkillLoadError(f"{source}: must start with a '---' frontmatter delimiter")
    end_index = -1
    for i in range(1, len(lines)):
        if lines[i].strip() == _FRONTMATTER_DELIM:
            end_index = i
            break
    if end_index == -1:
        raise SkillLoadError(f"{source}: frontmatter is not closed with a second '---'")
    yaml_text = "".join(lines[1:end_index])
    body = "".join(lines[end_index + 1:]).lstrip("\n")
    try:
        data = yaml.safe_load(yaml_text)
    except yaml.YAMLError as e:
        raise SkillLoadError(f"{source}: invalid YAML frontmatter: {e}") from e
    if not isinstance(data, dict):
        raise SkillLoadError(f"{source}: frontmatter must be a mapping, got {type(data).__name__}")
    name = data.get("name")
    description = data.get("description")
    if not isinstance(name, str) or not name.strip():
        raise SkillLoadError(f"{source}: frontmatter missing non-empty 'name'")
    if not isinstance(description, str) or not description.strip():
        raise SkillLoadError(f"{source}: frontmatter missing non-empty 'description'")
    return name.strip(), description.strip(), body
