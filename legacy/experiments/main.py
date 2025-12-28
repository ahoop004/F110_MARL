#!/usr/bin/env python3
"""Minimal wrapper around the experiments CLI."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

from experiments.cli import main as cli_main


def main() -> int:
    return cli_main()


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    sys.exit(main())
