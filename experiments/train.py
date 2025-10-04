"""Training entrypoint that delegates execution to the runner layer."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from f110x.engine.builder import build_runner_context
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
    """Encapsulates configuration + runner state for a training invocation."""

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


def create_training_session(
    cfg_path: Path | None = None,
    *,
    experiment: str | None = None,
    env_config_key: str = DEFAULT_ENV_CONFIG_KEY,
    env_experiment_key: str = DEFAULT_ENV_EXPERIMENT_KEY,
    logger: Optional[Logger] = None,
) -> TrainingSession:
    """Load configuration and assemble a :class:`TrainingSession`."""

    cfg, resolved_path, resolved_experiment = load_config(
        cfg_path,
        default_path=DEFAULT_CONFIG_PATH,
        experiment=experiment,
        env_config_key=env_config_key,
        env_experiment_key=env_experiment_key,
    )
    return build_training_session(
        cfg,
        config_path=resolved_path,
        experiment=resolved_experiment,
        logger=logger,
    )


def build_training_session(
    config: ExperimentConfig,
    *,
    config_path: Path,
    experiment: Optional[str],
    logger: Optional[Logger] = None,
) -> TrainingSession:
    """Compose a training runner for the supplied configuration."""

    runner_ctx = build_runner_context(config, logger=logger)
    runner = TrainRunner(runner_ctx)
    return TrainingSession(
        runner=runner,
        config=config,
        config_path=config_path,
        experiment=experiment,
    )


def run_training(
    session: TrainingSession,
    *,
    episodes: int,
    update_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    trainer_update_hook: Optional[TrainerUpdateHook] = None,
) -> List[Dict[str, Any]]:
    """Execute training episodes via the session runner."""

    return session.runner.run(
        episodes=int(episodes),
        update_callback=update_callback,
        trainer_update_hook=trainer_update_hook,
    )


def resolve_train_episodes(
    session: TrainingSession,
    *,
    override: Optional[int | str] = None,
    fallback: int = 10,
) -> int:
    """Determine how many training episodes to run."""

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


def _coerce_positive_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def main() -> None:  # pragma: no cover - CLI compatibility path
    session = create_training_session()
    episodes = session.default_episodes()
    run_training(session, episodes=episodes)
    saved = session.save_final_model()
    if saved is not None:
        print(f"[INFO] Saved final model to {saved}")


__all__ = [
    "TrainingSession",
    "create_training_session",
    "build_training_session",
    "run_training",
    "resolve_train_episodes",
]


if __name__ == "__main__":
    main()
