"""Utility to launch `experiments/main.py` multiple times with the desired config."""

from __future__ import annotations

import argparse
import itertools
import os
import random
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml


BASE_DIR = Path(__file__).resolve().parent


DEFAULT_CONFIGS: Dict[str, Tuple[Path, str]] = {
    "dqn": (Path("configs/experiments.yaml"), "gaplock_dqn"),
    "ppo": (Path("configs/experiments.yaml"), "gaplock_ppo"),
    "rec_ppo": (Path("configs/experiments.yaml"), "gaplock_rec_ppo"),
    "td3": (Path("configs/experiments.yaml"), "gaplock_td3"),
    "sac": (Path("configs/experiments.yaml"), "gaplock_sac"),
    "dqn_starved": (Path("configs/experiments_starved.yaml"), "gaplock_dqn_starved"),
    "ppo_starved": (Path("configs/experiments_starved.yaml"), "gaplock_ppo_starved"),
    "td3_starved": (Path("configs/experiments_starved.yaml"), "gaplock_td3_starved"),
    "sac_starved": (Path("configs/experiments_starved.yaml"), "gaplock_sac_starved"),
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
        help="Selects the default experiment profile (algo→experiment mapping).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Explicit path to a config file (overrides --algo default).",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        help="Experiment name inside the config file (required when config contains multiple experiments).",
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
        "--grid",
        type=Path,
        help="YAML/JSON file describing a configuration grid to sweep over.",
    )
    parser.add_argument(
        "main_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded verbatim to experiments/main.py (prefix with --).",
    )
    return parser


def resolve_config(
    algo: str,
    override: Optional[Path],
    experiment: Optional[str],
    parser: argparse.ArgumentParser,
) -> Tuple[Path, Optional[str]]:
    if override is not None:
        cfg_path = override
        default_exp = None
    else:
        cfg_path, default_exp = DEFAULT_CONFIGS[algo]

    cfg_path = cfg_path.expanduser()
    if not cfg_path.is_absolute():
        cfg_path = (BASE_DIR / cfg_path).resolve()
    else:
        cfg_path = cfg_path.resolve()

    if not cfg_path.exists():
        parser.error(f"Config file not found: {cfg_path}")

    try:
        doc = yaml.safe_load(cfg_path.read_text()) or {}
    except Exception as exc:  # pragma: no cover - defensive
        parser.error(f"Failed to parse config {cfg_path}: {exc}")

    if not isinstance(doc, dict):
        parser.error(f"Configuration {cfg_path} must have a mapping as its root object")

    experiments_section = doc.get("experiments")
    selected_exp = experiment or default_exp
    if isinstance(experiments_section, dict):
        if not selected_exp:
            selected_exp = doc.get("default_experiment")
        if not selected_exp:
            parser.error(
                f"Config {cfg_path} defines multiple experiments; please supply --experiment"
            )
        if selected_exp not in experiments_section:
            parser.error(
                f"Experiment '{selected_exp}' not found in {cfg_path}. Available: {sorted(experiments_section)}"
            )
    else:
        if experiment:
            print(
                f"[run.py] Warning: --experiment provided but config {cfg_path} has no experiments section; ignoring.",
                file=sys.stderr,
            )
        selected_exp = None

    return cfg_path, selected_exp


def _normalize_prefix(prefix: str) -> str:
    if not prefix:
        return ""
    return prefix if prefix.endswith(('-', '/', '_')) else f"{prefix}-"


def _prune_option(args: List[str], option: str) -> List[str]:
    cleaned: List[str] = []
    skip_next = False
    prefix = f"{option}="
    for token in args:
        if skip_next:
            skip_next = False
            continue
        if token == option:
            skip_next = True
            continue
        if token.startswith(prefix):
            continue
        cleaned.append(token)
    return cleaned


def _format_wandb_labels(
    algo: str,
    cfg_path: Path,
    experiment: Optional[str],
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
        "experiment": experiment or "",
        "config_slug": "",
        "run_idx": run_idx,
        "total_runs": total,
    }

    slug = f"{cfg_path.stem}-{experiment}" if experiment else cfg_path.stem
    slug = slug.replace("/", "_").replace(" ", "_")
    context["config_slug"] = slug

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


def _args_to_base_spec(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "algo": args.algo,
        "config": args.config,
        "experiment": args.experiment,
        "map": args.map,
        "repeat": args.repeat,
        "auto_seed": args.auto_seed,
        "seed_base": args.seed_base,
        "seed_step": args.seed_step,
        "wandb_prefix": args.wandb_prefix,
        "wandb_group_template": args.wandb_group_template,
        "wandb_name_template": args.wandb_name_template,
        "main_args": list(args.main_args) if args.main_args else [],
    }


def _normalize_main_args(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    if isinstance(value, str):
        return shlex.split(value)
    return [str(value)]


def _expand_matrix(matrix: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not matrix:
        return [{}]

    keys = list(matrix.keys())
    value_lists: List[List[Any]] = []
    for key in keys:
        value = matrix[key]
        if isinstance(value, (list, tuple)):
            value_lists.append(list(value))
        else:
            value_lists.append([value])

    combos: List[Dict[str, Any]] = []
    for values in itertools.product(*value_lists):
        entry = {key: val for key, val in zip(keys, values)}
        combos.append(entry)
    return combos


def _load_grid_specs(path: Path) -> List[Dict[str, Any]]:
    try:
        data = yaml.safe_load(path.read_text())
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"Failed to parse grid file {path}: {exc}") from exc

    if data is None:
        raise ValueError(f"Grid file {path} is empty")
    if not isinstance(data, dict):
        raise ValueError(f"Grid file {path} must contain a mapping at the top level")

    base = data.get("base", {}) or {}
    matrix = data.get("matrix", {}) or {}
    runs = data.get("runs", []) or []

    specs: List[Dict[str, Any]] = []
    for combo in _expand_matrix(matrix):
        spec: Dict[str, Any] = {}
        spec.update(base)
        spec.update(combo)
        specs.append(spec)

    for entry in runs:
        if not isinstance(entry, dict):
            raise ValueError("Each entry in 'runs' must be a mapping")
        spec = {}
        spec.update(base)
        spec.update(entry)
        specs.append(spec)

    if not specs:
        specs.append(dict(base))
    return specs


def _merge_specs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for key, value in base.items():
        if isinstance(value, list):
            merged[key] = list(value)
        elif isinstance(value, dict):
            merged[key] = dict(value)
        else:
            merged[key] = value

    for key, value in override.items():
        if isinstance(value, list):
            merged[key] = list(value)
        elif isinstance(value, dict):
            merged[key] = dict(value)
        else:
            merged[key] = value
    return merged


def _resolve_bool(value: Any, name: str, default: bool, parser: argparse.ArgumentParser) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in {"1", "true", "yes", "on"}:
            return True
        if cleaned in {"0", "false", "no", "off"}:
            return False
        parser.error(f"Invalid boolean for {name!r}: {value!r}")
    return bool(value)


def _resolve_optional_int(value: Any, name: str, parser: argparse.ArgumentParser) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        parser.error(f"Invalid integer for {name!r}: {value!r}")


def _prepare_env_overrides(spec: Dict[str, Any], parser: argparse.ArgumentParser) -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    for key in ("env", "env_overrides"):
        block = spec.get(key)
        if block is None:
            continue
        if not isinstance(block, dict):
            parser.error(f"'{key}' must be a mapping if provided in a grid spec")
        for env_key, env_value in block.items():
            overrides[str(env_key)] = str(env_value)
    return overrides


def _execute_spec(
    parser: argparse.ArgumentParser,
    spec: Dict[str, Any],
    spec_index: int,
    total_specs: int,
) -> Tuple[bool, int, int]:
    algo = spec.get("algo")
    if not algo:
        parser.error("Each run specification must define an 'algo'")
    algo = str(algo)

    config_override = spec.get("config")
    if isinstance(config_override, Path):
        cfg_override_path = config_override
    elif config_override is not None:
        cfg_override_path = Path(str(config_override)).expanduser()
    else:
        cfg_override_path = None

    map_name = spec.get("map")
    forwarded_args: List[str] = []
    if map_name is not None:
        map_value = str(map_name).strip()
        if not map_value:
            parser.error("Map value in grid spec cannot be empty")
        forwarded_args.extend(["--map", map_value])
        map_display = map_value
    else:
        map_display = "(default map)"

    main_args = _normalize_main_args(spec.get("main_args"))
    forwarded_args.extend(main_args)
    extra_args = _normalize_main_args(spec.get("extra_main_args"))
    forwarded_args.extend(extra_args)

    spec_experiment = spec.get("experiment")
    if spec_experiment is not None:
        spec_experiment = str(spec_experiment).strip() or None

    cfg_path, experiment_name = resolve_config(algo, cfg_override_path, spec_experiment, parser)

    forwarded_args = _prune_option(forwarded_args, "--config")
    forwarded_args = _prune_option(forwarded_args, "--experiment")
    forwarded_args = ["--config", str(cfg_path)] + forwarded_args
    if experiment_name:
        forwarded_args = ["--experiment", experiment_name] + forwarded_args

    repeat_value = spec.get("repeat")
    repeat = _resolve_optional_int(repeat_value, "repeat", parser) or 1
    if repeat < 1:
        parser.error("Each run specification must have repeat >= 1")

    auto_seed = _resolve_bool(spec.get("auto_seed"), "auto_seed", False, parser)
    seed_step = _resolve_optional_int(spec.get("seed_step"), "seed_step", parser) or 1
    if auto_seed and seed_step == 0:
        parser.error("seed_step must be non-zero when auto_seed is enabled")

    base_seed_value = _resolve_optional_int(spec.get("seed_base"), "seed_base", parser)

    explicit_seeds = spec.get("seeds")
    if explicit_seeds is None and "seed" in spec:
        explicit_seeds = spec.get("seed")
    seeds: List[Optional[int]]
    if explicit_seeds is not None:
        if isinstance(explicit_seeds, (list, tuple)):
            seeds_iterable = explicit_seeds
        else:
            seeds_iterable = [explicit_seeds]
        seeds = []
        for value in seeds_iterable:
            converted = _resolve_optional_int(value, "seed", parser)
            if converted is None:
                parser.error("Seed values must be integers when provided")
            seeds.append(converted)
        run_count = len(seeds)
    elif auto_seed:
        if base_seed_value is None:
            base_seed_value = random.randint(0, 2_147_483_647)
            label = spec.get("label") or spec.get("name") or algo
            print(
                f"[run.py] Auto seed enabled without base for spec '{label}'; using randomly chosen base {base_seed_value}."
            )
        seeds = [base_seed_value + i * seed_step for i in range(repeat)]
        run_count = len(seeds)
    else:
        seeds = [None] * repeat
        run_count = repeat

    if run_count == 0:
        parser.error("Run specification must yield at least one execution")

    base_env_overrides = _prepare_env_overrides(spec, parser)

    label = spec.get("label") or spec.get("name") or algo
    # map_display already set in branch above
    experiment_display = experiment_name or "(default)"
    print(
        f"[run.py] Sweep spec {spec_index}/{total_specs}: {label} (algo={algo}, experiment={experiment_display}, map={map_display}, runs={run_count})"
    )

    prefix = _normalize_prefix(str(spec.get("wandb_prefix", "")))
    group_template = str(spec.get("wandb_group_template", "{prefix}{config_slug}"))
    name_template = str(spec.get("wandb_name_template", "{group}-r{run_idx:02d}"))

    executed = 0
    for run_idx, seed in enumerate(seeds, start=1):
        env_overrides = dict(base_env_overrides)
        if seed is not None:
            env_overrides["RUN_SEED"] = str(int(seed))
        try:
            wandb_labels = _format_wandb_labels(
                algo,
                cfg_path,
                experiment_name,
                run_idx,
                run_count,
                prefix,
                group_template,
                name_template,
            )
        except ValueError as exc:
            parser.error(str(exc))
        env_overrides.update(wandb_labels)
        if experiment_name:
            env_overrides.setdefault("F110_EXPERIMENT", experiment_name)

        exit_code = run_once(cfg_path, experiment_name, forwarded_args, run_idx, run_count, env_overrides)
        if exit_code != 0:
            print(
                f"[run.py] Spec '{label}' run {run_idx}/{run_count} failed with exit code {exit_code}; aborting."
            )
            return False, executed, exit_code
        executed += 1

    print(f"[run.py] Completed spec '{label}' ({executed} run(s)).")
    return True, executed, 0


def run_once(
    cfg_path: Path,
    experiment: Optional[str],
    forwarded_args: Sequence[str],
    run_idx: int,
    total: int,
    env_overrides: Dict[str, str],
) -> int:
    cmd = [sys.executable, "experiments/main.py", *forwarded_args]
    env = os.environ.copy()
    env["F110_CONFIG"] = str(cfg_path)
    if experiment:
        env["F110_EXPERIMENT"] = experiment
    env.update(env_overrides)

    pretty_args = " ".join(forwarded_args) if forwarded_args else "(none)"
    seed_msg = f", RUN_SEED={env_overrides.get('RUN_SEED')}" if "RUN_SEED" in env_overrides else ""
    wandb_group = env_overrides.get("WANDB_RUN_GROUP")
    wandb_name = env_overrides.get("WANDB_NAME")
    wandb_msg = ""
    if wandb_group or wandb_name:
        wandb_msg = f", W&B group={wandb_group or '—'}, name={wandb_name or '—'}"
    exp_msg = f" (experiment={experiment})" if experiment else ""
    print(
        f"[run.py] Launching run {run_idx}/{total} with config '{cfg_path}'{exp_msg}{seed_msg}{wandb_msg} and args: {pretty_args}"
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

    base_spec = _args_to_base_spec(args)

    specs: List[Dict[str, Any]] = []
    if args.grid:
        grid_path = args.grid.expanduser().resolve()
        if not grid_path.exists():
            parser.error(f"Grid file not found: {grid_path}")
        try:
            grid_specs = _load_grid_specs(grid_path)
        except ValueError as exc:
            parser.error(str(exc))
        for spec in grid_specs:
            specs.append(_merge_specs(base_spec, spec))
    else:
        specs.append(dict(base_spec))

    total_specs = len(specs)
    total_runs = 0
    for idx, spec in enumerate(specs, start=1):
        success, executed, exit_code = _execute_spec(parser, spec, idx, total_specs)
        total_runs += executed
        if not success:
            sys.exit(exit_code)

    print(f"[run.py] Completed {total_runs} run(s) across {total_specs} spec(s).")


if __name__ == "__main__":
    main()
