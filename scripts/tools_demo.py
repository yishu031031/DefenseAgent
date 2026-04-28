"""Tools module demo — agent bundles, skills, and progressive disclosure.

This script shows, end to end:
  1. Where an agent's tools belong in the repo (convention: agents/<id>/).
     Each agent gets its own folder containing profile.yaml + skills/.
  2. How to build a ToolRegistry from the agent's profile with
     ToolRegistry.from_profile(profile) — skill paths in the profile are
     resolved relative to the profile's directory, so every agent's
     configuration stays completely independent.
  3. The three layers of Anthropic-style progressive disclosure:
        Layer 1 — name + description (always in registry.specs()).
        Layer 2 — SKILL.md body (returned on an empty-args invocation).
        Layer 3 — any other file in the skill directory (returned when
                  the LLM re-invokes with {"file": "relative/path"}).
  4. How a user-defined Python function co-exists with profile-loaded
     skills in the same registry.

Usage (from project root, conda env active):
    python scripts/tools_demo.py

No LLM calls, no network, no subprocess — every interaction is simulated
locally by constructing ToolCall objects and calling registry.execute().
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from DefenseAgent.config import AgentProfile
from DefenseAgent.llm.types import ToolCall
from DefenseAgent.tools import ToolRegistry


from DefenseAgent.examples import EXAMPLE_AGENT_DIR, EXAMPLE_PROFILE_PATH as EXAMPLE_PROFILE
PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENTS_DIR = EXAMPLE_AGENT_DIR.parent


def _banner(title: str) -> None:
    """Print a wide visual divider so the layers are easy to tell apart."""
    line = "=" * 72
    print(f"\n{line}\n{title}\n{line}")


async def main() -> None:
    """Load the example profile, build its tool registry from it, and walk every layer."""
    _banner("Agent bundle layout")
    print(f"Repo root:    {PROJECT_ROOT}")
    print(f"examples/:    {AGENTS_DIR}")
    print(f"Agents:       {sorted(p.name for p in AGENTS_DIR.iterdir() if p.is_dir())}")
    print(f"Example profile: {EXAMPLE_PROFILE}")
    example_dir = EXAMPLE_PROFILE.parent
    print(f"Example dir contents: {sorted(p.name for p in example_dir.iterdir())}")

    _banner("1. Load the profile + build the registry from it")
    profile = AgentProfile.from_yaml(EXAMPLE_PROFILE)
    print(f"agent:        {profile.name}  (id={profile.id})")
    print(f"profile.tools.skills: {profile.tools.skills}")
    print(f"profile.tools.mcp:    {profile.tools.mcp}")

    async with await ToolRegistry.from_profile(profile) as registry:

        _banner("2. Add a user-defined Python tool after from_profile()")
        @registry.tool
        def square(x: int) -> int:
            """Return x squared."""
            return x * x

        print(f"registered tools: {registry.names()}")

        _banner("3. MCP example (would register a real subprocess if run)")
        print(
            "Add servers either in profile.yaml under tools.mcp, or at runtime:\n\n"
            "    await registry.add_mcp(\n"
            "        command=\"uvx\",\n"
            "        args=[\"mcp-server-filesystem\", \"/tmp\"],\n"
            "    )\n"
        )

        _banner("LAYER 1 — registry.specs() (what the LLM sees every turn)")
        for entry in registry.specs():
            print(f"\n• {entry['name']}")
            print(f"  description: {entry['description']}")
            print(f"  input_schema: {entry['input_schema']}")

        _banner("Python tool dispatches through the same execute() path")
        calls = [ToolCall(id="c1", name="square", arguments={"x": 7})]
        for msg in await registry.execute(calls):
            print(f"[{msg.name}] → {msg.content}")

        _banner("LAYER 2 — invoke skill with NO arguments → SKILL.md body")
        calls = [ToolCall(id="c2", name="tabular-report", arguments={})]
        layer2 = (await registry.execute(calls))[0]
        print(layer2.content)

        _banner(
            "LAYER 3 — invoke same skill with "
            "{'file': 'scripts/generate.py'}"
        )
        calls = [
            ToolCall(
                id="c3",
                name="tabular-report",
                arguments={"file": "scripts/generate.py"},
            )
        ]
        layer3_script = (await registry.execute(calls))[0]
        print(layer3_script.content)

        _banner("LAYER 3 — and again for templates/header.md")
        calls = [
            ToolCall(
                id="c4",
                name="tabular-report",
                arguments={"file": "templates/header.md"},
            )
        ]
        layer3_template = (await registry.execute(calls))[0]
        print(layer3_template.content)

        _banner("Guardrail — escape attempts are rejected as a tool error")
        calls = [
            ToolCall(
                id="c5",
                name="tabular-report",
                arguments={"file": "../../etc/passwd"},
            )
        ]
        rejected = (await registry.execute(calls))[0]
        print(rejected.content)


if __name__ == "__main__":
    asyncio.run(main())
