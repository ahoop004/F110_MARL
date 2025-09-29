"""Utility to launch `experiments/main.py` multiple times with the desired config."""

from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, Sequence


BASE_DIR = Path(__file__).resolve().parent


DEFAULT_CONFIGS: Dict[str, Path] = {
    "dqn": Path("configs/experiment_gaplock_dqn.yaml"),
    "ppo": Path("configs/experiment_gaplock_ppo.yaml"),
    "td3": Path("configs/experiment_gaplock_td3.yaml"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run experiments/main.py repeatedly with a chosen algorithm config.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--algo",
        choices=sorted(DEFAULT_CONFIGS.keys()),
        default="dqn",
        help="Selects the default config associated with an algorithm.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Explicit path to a config file (overrides --algo default).",
    )
    parser.add_argument(
        "--map",
        type=str,
        help="Map name forwarded to experiments/main.py (e.g., 'line_map').",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="How many consecutive runs to launch.",
    )
    parser.add_argument(
        "--auto-seed",
        action="store_true",
        help="If set, injects RUN_SEED per run with incremental offsets.",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        help="Base seed for the first run when --auto-seed is enabled (random if omitted).",
    )
    parser.add_argument(
        "--seed-step",
        type=int,
        default=1,
        help="Amount to increment the seed between runs when --auto-seed is enabled.",
    )
    parser.add_argument(
        "--wandb-prefix",
        type=str,
        default="",
        help="Optional prefix applied to generated W&B group/name strings.",
    )
    parser.add_argument(
        "--wandb-group-template",
        type=str,
        default="{prefix}{config_slug}",
        help="Template for W&B group; keys: prefix, algo, config, config_stem, config_slug, run_idx, total_runs.",
    )
    parser.add_argument(
        "--wandb-name-template",
        type=str,
        default="{group}-r{run_idx:02d}",
        help="Template for W&B run name; keys include group, prefix, algo, config, config_stem, run_idx, total_runs.",
    )
    parser.add_argument(
        "main_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded verbatim to experiments/main.py (prefix with --).",
    )
    return parser


def resolve_config(algo: str, override: Path | None, parser: argparse.ArgumentParser) -> Path:
    if override is not None:
        cfg_path = override
    else:
        cfg_path = DEFAULT_CONFIGS[algo]

    cfg_path = cfg_path if cfg_path.is_absolute() else (BASE_DIR / cfg_path)
    cfg_path = cfg_path.resolve()
    if not cfg_path.exists():
        parser.error(f"Config file not found: {cfg_path}")
    return cfg_path


def _normalize_prefix(prefix: str) -> str:
    if not prefix:
        return ""
    return prefix if prefix.endswith(('-', '/', '_')) else f"{prefix}-"


def _format_wandb_labels(
    algo: str,
    cfg_path: Path,
    run_idx: int,
    total: int,
    prefix: str,
    group_template: str,
    name_template: str,
) -> Dict[str, str]:
    context = {
        "prefix": prefix,
        "algo": algo,
        "config": cfg_path.name,
        "config_stem": cfg_path.stem,
        "config_slug": cfg_path.stem.replace("experiment_", ""),
        "run_idx": run_idx,
        "total_runs": total,
    }

    try:
        group = group_template.format(**context).strip()
    except KeyError as exc:
        raise ValueError(f"Unknown placeholder '{{{exc.args[0]}}}' in --wandb-group-template") from exc
    context["group"] = group

    try:
        name = name_template.format(**context).strip()
    except KeyError as exc:
        raise ValueError(f"Unknown placeholder '{{{exc.args[0]}}}' in --wandb-name-template") from exc

    labels: Dict[str, str] = {}
    if group:
        labels["WANDB_RUN_GROUP"] = group
    if name:
        labels["WANDB_NAME"] = name
    return labels


def run_once(
    cfg_path: Path,
    forwarded_args: Sequence[str],
    run_idx: int,
    total: int,
    env_overrides: Dict[str, str],
) -> int:
    cmd = [sys.executable, "experiments/main.py", *forwarded_args]
    env = os.environ.copy()
    env["F110_CONFIG"] = str(cfg_path)
    env.update(env_overrides)

    pretty_args = " ".join(forwarded_args) if forwarded_args else "(none)"
    seed_msg = f", RUN_SEED={env_overrides.get('RUN_SEED')}" if "RUN_SEED" in env_overrides else ""
    wandb_group = env_overrides.get("WANDB_RUN_GROUP")
    wandb_name = env_overrides.get("WANDB_NAME")
    wandb_msg = ""
    if wandb_group or wandb_name:
        wandb_msg = f", W&B group={wandb_group or '—'}, name={wandb_name or '—'}"
    print(
        f"[run.py] Launching run {run_idx}/{total} with config '{cfg_path}'{seed_msg}{wandb_msg} and args: {pretty_args}"
    )

    result = subprocess.run(cmd, env=env)
    return result.returncode


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.repeat < 1:
        parser.error("--repeat must be >= 1")
    if args.auto_seed and args.seed_step == 0:
        parser.error("--seed-step must be non-zero when --auto-seed is enabled")

    forwarded_args = []
    if args.map is not None:
        map_name = args.map.strip()
        if not map_name:
            parser.error("--map value cannot be empty")
        forwarded_args.extend(["--map", map_name])
    forwarded_args.extend(arg for arg in args.main_args if arg)
    cfg_path = resolve_config(args.algo, args.config, parser)
    prefix = _normalize_prefix(args.wandb_prefix)

    if args.auto_seed:
        base_seed = args.seed_base
        if base_seed is None:
            base_seed = random.randint(0, 2_147_483_647)
            print(f"[run.py] Auto seed enabled without --seed-base; using randomly chosen base {base_seed}.")
        seed_step = args.seed_step
    else:
        base_seed = None
        seed_step = 0

    for idx in range(1, args.repeat + 1):
        env_overrides: Dict[str, str] = {}
        if base_seed is not None:
            run_seed = base_seed + (idx - 1) * seed_step
            env_overrides["RUN_SEED"] = str(run_seed)

        try:
            wandb_labels = _format_wandb_labels(
                args.algo,
                cfg_path,
                idx,
                args.repeat,
                prefix,
                args.wandb_group_template,
                args.wandb_name_template,
            )
        except ValueError as exc:
            parser.error(str(exc))
        env_overrides.update(wandb_labels)

        exit_code = run_once(cfg_path, forwarded_args, idx, args.repeat, env_overrides)
        if exit_code != 0:
            print(f"[run.py] Run {idx} failed with exit code {exit_code}; aborting.")
            sys.exit(exit_code)

    print(f"[run.py] Completed {args.repeat} run(s) successfully.")


if __name__ == "__main__":
    main()
