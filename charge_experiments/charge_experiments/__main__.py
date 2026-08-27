"""``python -m charge_experiments <command> ...``"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    from charge_experiments.cli import main as cli_main

    return cli_main(argv)


if __name__ == "__main__":
    sys.exit(main())
