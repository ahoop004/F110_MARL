"""Command-line entrypoints for training and evaluation sessions."""
from __future__ import annotations

import argparse
import os
import random
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import yaml

from experiments import session as session_module
from f110x.utils.config import resolve_active_config_block
from f110x.utils.logger import ConsoleSink, Logger, WandbSink


@dataclass(frozen=True)
class RuntimeOptions:
    render: bool
    train_episodes: Optional[int]
    eval_episodes: Optional[int]
    config_path: Optional[Path]
    experiment: Optional[str]
    algo: Optional[str]
    mode: Optional[str]
    seed: Optional[int]


DEFAULT_CONFIGS: Dict[str, Tuple[str, str]] = {
    "dqn": ("scenarios/gaplock_dqn.yaml", "gaplock_dqn"),
}


def _coerce_positive_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    if value is None:
        return default
    try:
        result = int(value)
    except (TypeError, ValueError):
        return default
    return result if result > 0 else default


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    import numpy as np

    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _resolve_default_config(algo: Optional[str]) -> Tuple[Optional[Path], Optional[str]]:
    if not algo:
        return None, None
    key = str(algo).strip().lower()
    if key not in DEFAULT_CONFIGS:
        raise ValueError(
            f"Unknown algorithm preset '{algo}'. Available: {sorted(DEFAULT_CONFIGS)}"
        )
    cfg_path, exp = DEFAULT_CONFIGS[key]
    return Path(cfg_path), exp


def _load_document(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if key.startswith("_") or key in {"wandb_version"}:
            continue
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _prepare_config(
    cfg_path: Path,
    overrides: Mapping[str, Any],
    *,
    wandb_run: Optional[Any],
) -> Path:
    if not overrides and wandb_run is None:
        return cfg_path

    doc = _load_document(cfg_path)
    target_block, _ = resolve_active_config_block(doc)
    _deep_update(target_block, dict(overrides))

    if wandb_run is not None:
        cfg_updates = wandb_run.config.as_dict()
        if cfg_updates:
            _deep_update(target_block, cfg_updates)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml", mode="w", encoding="utf-8")
    with tmp:
        yaml.safe_dump(doc, tmp)
    return Path(tmp.name)


def _build_logger(run_suffix: Optional[str], wandb_run: Optional[Any]) -> Logger:
    sinks: List[Any] = [ConsoleSink()]
    if wandb_run is not None:
        sinks.append(WandbSink(wandb_run))
    logger = Logger(sinks)
    if wandb_run is not None and run_suffix:
        logger.update_context(wandb_run_name=run_suffix, run_suffix=run_suffix)
    return logger


def run_sessions(
    *,
    overrides: Mapping[str, Any],
    runtime: RuntimeOptions,
    wandb_enabled: bool,
    wandb_run_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    cfg_path: Optional[Path] = runtime.config_path
    experiment_override = runtime.experiment

    if runtime.algo and cfg_path is None:
        preset_path, preset_exp = _resolve_default_config(runtime.algo)
        cfg_path = preset_path
        if experiment_override is None:
            experiment_override = preset_exp

    if cfg_path is None:
        raise ValueError("No configuration provided; supply --config or an algorithm preset")

    cfg_path = cfg_path.expanduser().resolve()

    if runtime.seed is not None:
        _seed_everything(runtime.seed)

    wandb_run = None
    prepared_cfg: Path
    try:
        if wandb_enabled:
            import wandb

            wandb_run = wandb.init(**(wandb_run_kwargs or {}))
            if wandb_run is not None:
                os.environ["WANDB_RUN_ID"] = str(wandb_run.id)
                if getattr(wandb_run, "name", None):
                    os.environ.setdefault("WANDB_RUN_NAME", str(wandb_run.name))

        prepared_cfg = _prepare_config(cfg_path, overrides, wandb_run=wandb_run)

        doc = _load_document(prepared_cfg)
        active_block, _ = resolve_active_config_block(doc)
        main_block = dict(active_block.get("main", {}))
        resolved_mode = runtime.mode or str(main_block.get("mode", "train_eval")).lower()

        logger = _build_logger(os.environ.get("F110_RUN_SUFFIX"), wandb_run)
        logger.start({"config_path": str(prepared_cfg), "experiment": experiment_override})

        try:
            if resolved_mode in {"train", "train_eval"}:
                train_session = session_module.create_training_session(
                    prepared_cfg,
                    experiment=experiment_override,
                    logger=logger,
                )
                train_env = getattr(train_session.runner, "env", None)
                try:
                    if runtime.render:
                        train_session.enable_render()
                    fallback_train = _coerce_positive_int(main_block.get("train_episodes"), 10)
                    episodes = session_module.resolve_train_episodes(
                        train_session,
                        override=runtime.train_episodes,
                        fallback=fallback_train,
                    )
                    session_module.run_training(train_session, episodes=episodes)
                    train_session.save_final_model()
                finally:
                    if train_env is not None:
                        train_env.close()

            if resolved_mode in {"eval", "train_eval"}:
                eval_session = session_module.create_evaluation_session(
                    prepared_cfg,
                    experiment=experiment_override,
                    auto_load=True,
                    logger=logger,
                )
                eval_env = getattr(eval_session.runner, "env", None)
                try:
                    if runtime.render:
                        eval_session.enable_render()
                    fallback_eval = _coerce_positive_int(main_block.get("eval_episodes"), 5)
                    eval_episodes = session_module.resolve_eval_episodes(
                        eval_session,
                        override=runtime.eval_episodes,
                        fallback=fallback_eval,
                    )
                    session_module.run_evaluation(
                        eval_session,
                        episodes=eval_episodes,
                        force_render=runtime.render,
                    )
                finally:
                    if eval_env is not None:
                        eval_env.close()
        finally:
            logger.stop()
    finally:
        if wandb_run is not None:
            try:
                wandb_run.finish()
            except Exception:
                pass
        if cfg_path is not None and 'prepared_cfg' in locals() and prepared_cfg != cfg_path:
            try:
                prepared_cfg.unlink()
            except FileNotFoundError:
                pass


def build_main_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run training/evaluation sessions")
    parser.add_argument("--config", type=Path, help="Path to configuration or scenario file")
    parser.add_argument("--scenario", type=Path, help="Scenario manifest to materialize", default=None)
    parser.add_argument("--experiment", type=str, help="Experiment name to select", default=None)
    parser.add_argument("--algo", type=str, help="Preset algorithm name")
    parser.add_argument("--mode", choices=["train", "eval", "train_eval"], help="Override config mode")
    parser.add_argument("--episodes", type=int, help="Override training episodes")
    parser.add_argument("--eval-episodes", type=int, help="Override evaluation episodes")
    parser.add_argument("--collect-workers", type=int, help="Override main.collect_workers")
    parser.add_argument("--collect-prefetch", type=int, help="Override main.collect_prefetch")
    parser.add_argument("--collect-seed-stride", type=int, help="Override main.collect_seed_stride")
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--map", type=str, help="Override map name")
    parser.add_argument("--spawn-profile", type=str, help="Select a named spawn profile")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str)
    parser.add_argument("--wandb-entity", type=str)
    parser.add_argument("--wandb-group", type=str)
    parser.add_argument("--wandb-name", type=str)
    parser.add_argument("--wandb-tags", nargs="*", default=None)
    return parser


def _collect_wandb_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    if args.wandb_project:
        kwargs["project"] = args.wandb_project
    if args.wandb_entity:
        kwargs["entity"] = args.wandb_entity
    if args.wandb_group:
        kwargs["group"] = args.wandb_group
    if args.wandb_name:
        kwargs["name"] = args.wandb_name
    if args.wandb_tags:
        kwargs["tags"] = args.wandb_tags
    return kwargs


def _build_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    if args.map:
        env_override: Dict[str, Any] = {"map": args.map}
        if Path(args.map).suffix in {"", ".yaml", ".yml"} and not str(args.map).endswith((".yaml", ".yml")):
            env_override["map_yaml"] = f"{args.map}.yaml"
        overrides.setdefault("env", {}).update(env_override)
    if args.spawn_profile:
        overrides.setdefault("env", {})["spawn_profile"] = args.spawn_profile
    if args.mode:
        overrides.setdefault("main", {})["mode"] = args.mode
    if args.collect_workers is not None:
        overrides.setdefault("main", {})["collect_workers"] = args.collect_workers
    if args.collect_prefetch is not None:
        overrides.setdefault("main", {})["collect_prefetch"] = args.collect_prefetch
    if args.collect_seed_stride is not None:
        overrides.setdefault("main", {})["collect_seed_stride"] = args.collect_seed_stride
    return overrides


def _resolve_config_path(args: argparse.Namespace) -> Optional[Path]:
    if args.config:
        return args.config
    if args.scenario:
        return args.scenario
    env_value = os.environ.get("F110_CONFIG")
    if env_value:
        return Path(env_value)
    return None


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_main_parser()
    if argv is None:
        argv = sys.argv[1:]
    else:
        argv = list(argv)
    argv = [arg for arg in argv if arg != "--"]
    args = parser.parse_args(argv)

    config_path = _resolve_config_path(args)
    experiment = args.experiment or os.environ.get("F110_EXPERIMENT")

    overrides = _build_overrides(args)

    runtime = RuntimeOptions(
        render=bool(args.render),
        train_episodes=_coerce_positive_int(args.episodes),
        eval_episodes=_coerce_positive_int(args.eval_episodes),
        config_path=config_path,
        experiment=experiment,
        algo=args.algo,
        mode=args.mode,
        seed=args.seed,
    )

    wandb_kwargs = _collect_wandb_kwargs(args)

    try:
        run_sessions(
            overrides=overrides,
            runtime=runtime,
            wandb_enabled=bool(args.wandb),
            wandb_run_kwargs=wandb_kwargs if args.wandb else None,
        )
    except ValueError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    sys.exit(main())
