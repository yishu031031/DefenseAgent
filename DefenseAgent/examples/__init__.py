"""Reference profile bundles shipped inside the wheel."""
from pathlib import Path

EXAMPLE_AGENT_DIR: Path = Path(__file__).parent / "example_agent"
EXAMPLE_PROFILE_PATH: Path = EXAMPLE_AGENT_DIR / "profile.yaml"

__all__ = ["EXAMPLE_AGENT_DIR", "EXAMPLE_PROFILE_PATH"]
