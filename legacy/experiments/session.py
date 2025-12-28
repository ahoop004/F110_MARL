"""Unified training/evaluation session helpers."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from f110x.engine.builder import build_runner_context
from f110x.runner.eval_runner import EvalRunner
from f110x.runner.train_runner import TrainRunner, TrainerUpdateHook
from f110x.utils.config import (
    DEFAULT_ENV_CONFIG_KEY,
    DEFAULT_ENV_EXPERIMENT_KEY,
    load_config,
)
from f110x.utils.config_models import ExperimentConfig
from f110x.utils.logger import Logger


DEFAULT_CONFIG_PATH = Path("scenarios/gaplock_dqn.yaml")


@dataclass
class TrainingSession:
    runner: TrainRunner
    config: ExperimentConfig
    config_path: Path
    experiment: Optional[str]

    @property
    def bundle_config(self) -> Dict[str, Any]:
        return dict(self.runner.primary_bundle.metadata.get("config", {}))

    def default_episodes(self, fallback: int = 10) -> int:
        return resolve_train_episodes(self, fallback=fallback)

    def enable_render(self) -> None:
        self.runner.context.render_interval = max(1, int(self.runner.context.render_interval or 1))
        self.runner.env.render_mode = "human"

    def save_final_model(self) -> Optional[Path]:
        controller = self.runner.primary_bundle.controller
        save_fn = getattr(controller, "save", None)
        if not callable(save_fn):
            return None
        self.runner.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        save_fn(str(self.runner.best_model_path))
        return self.runner.best_model_path


@dataclass
class EvaluationSession:
    runner: EvalRunner
    config: ExperimentConfig
    config_path: Path
    experiment: Optional[str]

    def enable_render(self) -> None:
        self.runner.env.render_mode = "human"

    def auto_load(self, checkpoint: Optional[Path | str] = None) -> Optional[Path]:
        return self.runner.load_checkpoint(checkpoint)

    def default_episodes(self, fallback: int = 5) -> int:
        return resolve_eval_episodes(self, fallback=fallback)


def _build_session(
    config: ExperimentConfig,
    *,
    config_path: Path,
    experiment: Optional[str],
    logger: Optional[Logger],
    run_type: str,
) -> TrainingSession | EvaluationSession:
    ensure_trainable = run_type == "train"
    prefer_algos = ("centerline", "ppo", "rec_ppo", "dqn") if run_type == "eval" else ("ppo", "rec_ppo")
    runner_ctx = build_runner_context(
        config,
        logger=logger,
        ensure_trainable=ensure_trainable,
        prefer_algorithms=prefer_algos,
    )
    try:
        runner_ctx.update_metadata(
            wandb_run_id=os.environ.get("WANDB_RUN_ID"),
            wandb_run_name=os.environ.get("WANDB_RUN_NAME"),
            run_suffix=os.environ.get("F110_RUN_SUFFIX"),
        )
    except Exception:
        pass
    if run_type == "train":
        runner = TrainRunner(runner_ctx)
        return TrainingSession(runner=runner, config=config, config_path=config_path, experiment=experiment)
    runner = EvalRunner(runner_ctx)
    return EvaluationSession(runner=runner, config=config, config_path=config_path, experiment=experiment)


def _load_config(
    cfg_path: Path | None,
    *,
    experiment: str | None,
    env_config_key: str,
    env_experiment_key: str,
) -> tuple[ExperimentConfig, Path, Optional[str]]:
    return load_config(
        cfg_path,
        default_path=DEFAULT_CONFIG_PATH,
        experiment=experiment,
        env_config_key=env_config_key,
        env_experiment_key=env_experiment_key,
    )


def create_training_session(
    cfg_path: Path | None = None,
    *,
    experiment: str | None = None,
    env_config_key: str = DEFAULT_ENV_CONFIG_KEY,
    env_experiment_key: str = DEFAULT_ENV_EXPERIMENT_KEY,
    logger: Optional[Logger] = None,
) -> TrainingSession:
    cfg, resolved_path, resolved_experiment = _load_config(
        cfg_path,
        experiment=experiment,
        env_config_key=env_config_key,
        env_experiment_key=env_experiment_key,
    )
    return _build_session(
        cfg,
        config_path=resolved_path,
        experiment=resolved_experiment,
        logger=logger,
        run_type="train",
    )


def create_evaluation_session(
    cfg_path: Path | None = None,
    *,
    experiment: str | None = None,
    env_config_key: str = DEFAULT_ENV_CONFIG_KEY,
    env_experiment_key: str = DEFAULT_ENV_EXPERIMENT_KEY,
    auto_load: bool = False,
    checkpoint: Optional[Path | str] = None,
    logger: Optional[Logger] = None,
) -> EvaluationSession:
    cfg, resolved_path, resolved_experiment = _load_config(
        cfg_path,
        experiment=experiment,
        env_config_key=env_config_key,
        env_experiment_key=env_experiment_key,
    )
    session = _build_session(
        cfg,
        config_path=resolved_path,
        experiment=resolved_experiment,
        logger=logger,
        run_type="eval",
    )
    if auto_load:
        session.auto_load(checkpoint)
    return session


def run_training(
    session: TrainingSession,
    *,
    episodes: int,
    update_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    trainer_update_hook: Optional[TrainerUpdateHook] = None,
) -> List[Dict[str, Any]]:
    return session.runner.run(
        episodes=int(episodes),
        update_callback=update_callback,
        trainer_update_hook=trainer_update_hook,
    )


def run_evaluation(
    session: EvaluationSession,
    *,
    episodes: int,
    force_render: bool = False,
    save_rollouts: Optional[bool] = None,
    rollout_dir: Optional[Path | str] = None,
    auto_load: bool = False,
    checkpoint_path: Optional[Path | str] = None,
) -> List[Dict[str, Any]]:
    return session.runner.run(
        episodes=int(episodes),
        force_render=force_render,
        save_rollouts=save_rollouts,
        rollout_dir=rollout_dir,
        auto_load=auto_load,
        checkpoint_path=checkpoint_path,
    )


def resolve_train_episodes(
    session: TrainingSession,
    *,
    override: Optional[int | str] = None,
    fallback: int = 10,
) -> int:
    if override is not None:
        coerced = _coerce_positive_int(override)
        if coerced is not None:
            return coerced

    candidates = [
        session.bundle_config.get("train_episodes"),
        session.config.main.get("train_episodes"),
        session.config.raw.get("train_episodes"),
    ]
    for candidate in candidates:
        episodes = _coerce_positive_int(candidate)
        if episodes is not None:
            return episodes
    return max(int(fallback), 1)


def resolve_eval_episodes(
    session: EvaluationSession,
    *,
    override: Optional[int | str] = None,
    fallback: int = 5,
) -> int:
    if override is not None:
        coerced = _coerce_positive_int(override)
        if coerced is not None:
            return coerced

    bundle_cfg = session.runner.primary_bundle.metadata.get("config", {}) if session.runner.primary_bundle else {}
    candidates = [
        bundle_cfg.get("eval_episodes"),
        session.config.main.get("eval_episodes"),
        session.config.raw.get("eval_episodes"),
    ]
    for candidate in candidates:
        episodes = _coerce_positive_int(candidate)
        if episodes is not None:
            return episodes
    return max(int(fallback), 1)


def _coerce_positive_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


__all__ = [
    "TrainingSession",
    "EvaluationSession",
    "create_training_session",
    "create_evaluation_session",
    "run_training",
    "run_evaluation",
    "resolve_train_episodes",
    "resolve_eval_episodes",
]
