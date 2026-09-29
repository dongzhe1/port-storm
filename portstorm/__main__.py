"""Command-line entry point: python -m portstorm <command> [options]."""
from __future__ import annotations

import importlib
import sys

COMMANDS = {
    "download": "download", "fake": "fake", "calls": "calls", "storms": "storms",
    "covariates": "covariates", "fleet": "fleet", "fuel": "fuel", "zones": "zones",
    "tiers": "tiers", "sulfur": "sulfur", "gaps": "gaps", "resilience": "resilience",
    "elasticity": "elasticity", "window": "window", "cyport": "cyport",
    "archive": "archive",
}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print("usage: python -m portstorm <command> [options]\n\ncommands: "
              + ", ".join(COMMANDS))
        return 2
    cmd = sys.argv[1]
    module = importlib.import_module(f"portstorm.{COMMANDS[cmd]}")
    sys.argv = [f"portstorm {cmd}", *sys.argv[2:]]
    return module.main()


if __name__ == "__main__":
    sys.exit(main())
