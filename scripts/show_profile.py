"""Load the default agent profile and pretty-print the parsed model.

Usage (from project root, with the conda env active):
    python scripts/show_profile.py
    python scripts/show_profile.py path/to/other_profile.yaml
"""
import json
import sys
from pathlib import Path

# Allow running as a script without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from DefenseAgent.config import AgentProfile, ConfigError            # ← front-door class


DEFAULT_PATH = (
    Path(__file__).resolve().parent.parent
    / "agents" / "alice_chen" / "profile.yaml"
)


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH

    # Module 2 — AgentProfile.from_yaml is the canonical entry point.
    try:
        profile = AgentProfile.from_yaml(path)
    except ConfigError as e:
        print(f"[show_profile] failed to load {path}:")
        print(f"  {type(e).__name__}: {e}")
        return 1

    print(f"[show_profile] loaded {path}")
    print(f"[show_profile] model: {type(profile).__name__}")
    print("---")
    # model_dump + json gives a clean, indented view of all fields.
    print(json.dumps(profile.model_dump(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
