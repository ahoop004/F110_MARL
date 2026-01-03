#!/usr/bin/env python3
"""Thin dispatcher for training entrypoints.

Defaults to run_v2 unless an SB3-style argument is present.
Use --runner to force a specific entrypoint.
"""

from __future__ import annotations

import argparse
import sys


def _select_runner(argv: list[str]) -> tuple[str, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--runner", choices=["v2", "sb3"])
    args, remaining = parser.parse_known_args(argv)

    if args.runner:
        return args.runner, remaining

    if "--algo" in remaining:
        return "sb3", remaining

    return "v2", remaining


def main() -> None:
    runner, remaining = _select_runner(sys.argv[1:])

    if runner == "sb3":
        import run_sb3_baseline as target
        sys.argv = ["run_sb3_baseline.py", *remaining]
    else:
        import run_v2 as target
        sys.argv = ["run_v2.py", *remaining]

    target.main()


if __name__ == "__main__":
    main()
