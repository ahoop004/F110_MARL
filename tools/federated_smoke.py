#!/usr/bin/env python3
"""Launch a short federated TD3 run to sanity-check multi-client coordination."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        type=Path,
        default=Path("scenarios/gaplock_td3.yaml"),
        help="Scenario manifest to execute",
    )
    parser.add_argument("--repeat", type=int, default=2, help="Number of coordinated clients")
    parser.add_argument("--max-parallel", type=int, default=2, help="Maximum concurrent processes")
    parser.add_argument("--episodes", type=int, help="Override training episodes for the smoke run")
    parser.add_argument("--eval-episodes", type=int, help="Override evaluation episodes for the smoke run")
    parser.add_argument("--seed-base", type=int, default=12345, help="Base seed used when auto-seeding clients")
    parser.add_argument("--fed-root", type=Path, help="Directory used for federated round exchange")
    parser.add_argument("--keep-root", action="store_true", help="Preserve temporary federated directory")
    parser.add_argument("--skip-checkpoints", action="store_true", help="Disable checkpoint saves after sync")
    parser.add_argument("--dry-run", action="store_true", help="Show the command without executing it")
    return parser


def _build_command(args: argparse.Namespace, scenario: Path) -> List[str]:
    cmd: List[str] = [
        sys.executable,
        "run.py",
        "--scenario",
        str(scenario),
        "--repeat",
        str(max(args.repeat, 1)),
        "--max-parallel",
        str(max(args.max_parallel, 1)),
        "--auto-seed",
        "--seed-base",
        str(args.seed_base),
    ]
    if args.episodes is not None:
        cmd.extend(["--episodes", str(args.episodes)])
    if args.eval_episodes is not None:
        cmd.extend(["--eval-episodes", str(args.eval_episodes)])
    return cmd


def _collect_round_metrics(root: Path) -> Dict[str, int]:
    rounds: Dict[str, int] = {}
    for round_dir in sorted(root.glob("round_*/")):
        if not round_dir.is_dir():
            continue
        client_count = len(list(round_dir.glob("client_*.pt")))
        rounds[round_dir.name] = client_count
    return rounds


def _main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    scenario = args.scenario.expanduser().resolve()
    if not scenario.exists():
        parser.error(f"Scenario not found: {scenario}")

    cmd = _build_command(args, scenario)

    cleanup_temp = False
    temp_dir: Optional[tempfile.TemporaryDirectory[str]] = None
    if args.fed_root is not None:
        root_path = args.fed_root.expanduser().resolve()
        root_path.mkdir(parents=True, exist_ok=True)
    else:
        temp_dir = tempfile.TemporaryDirectory(prefix="fed_smoke_")
        root_path = Path(temp_dir.name)
        cleanup_temp = not args.keep_root

    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "offline")
    env.setdefault("FED_ROOT", str(root_path))
    if args.skip_checkpoints:
        env["FED_CHECKPOINT_AFTER_SYNC"] = "0"

    summary: Dict[str, object]
    if args.dry_run:
        summary = {
            "dry_run": True,
            "command": cmd,
            "env_overrides": {
                "FED_ROOT": env["FED_ROOT"],
                "FED_CHECKPOINT_AFTER_SYNC": env.get("FED_CHECKPOINT_AFTER_SYNC", "1"),
            },
        }
    else:
        try:
            subprocess.run(cmd, env=env, check=True)
        except subprocess.CalledProcessError as exc:
            if temp_dir and cleanup_temp:
                temp_dir.cleanup()
            raise RuntimeError(f"Smoke run failed with exit code {exc.returncode}") from exc

        round_metrics = _collect_round_metrics(root_path)
        summary = {
            "dry_run": False,
            "root": str(root_path),
            "rounds_completed": len(round_metrics),
            "round_client_counts": round_metrics,
            "clients": max(args.repeat, 1),
        }
        if not round_metrics:
            summary["warning"] = "No federated rounds detected"

    print(json.dumps(summary, indent=2))

    if temp_dir and cleanup_temp:
        temp_dir.cleanup()
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(_main())
