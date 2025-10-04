"""Evaluation entrypoint that leverages the new runner abstraction."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from f110x.engine.builder import build_runner_context
from f110x.runner.eval_runner import EvalRunner
from f110x.utils.config import (
    DEFAULT_ENV_CONFIG_KEY,
    DEFAULT_ENV_EXPERIMENT_KEY,
    load_config,
)
from f110x.utils.config_models import ExperimentConfig


DEFAULT_CONFIG_PATH = Path("scenarios/gaplock_dqn.yaml")


@dataclass
class EvaluationSession:
    """Wraps configuration and runner state for evaluation passes."""

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


def create_evaluation_session(
    cfg_path: Path | None = None,
    *,
    experiment: str | None = None,
    auto_load: bool = False,
    env_config_key: str = DEFAULT_ENV_CONFIG_KEY,
    env_experiment_key: str = DEFAULT_ENV_EXPERIMENT_KEY,
    checkpoint: Optional[Path | str] = None,
) -> EvaluationSession:
    """Load configuration and assemble an :class:`EvaluationSession`."""

    cfg, resolved_path, resolved_experiment = load_config(
        cfg_path,
        default_path=DEFAULT_CONFIG_PATH,
        experiment=experiment,
        env_config_key=env_config_key,
        env_experiment_key=env_experiment_key,
    )
    session = build_evaluation_session(cfg, config_path=resolved_path, experiment=resolved_experiment)
    if auto_load:
        session.auto_load(checkpoint)
    return session


def build_evaluation_session(
    config: ExperimentConfig,
    *,
    config_path: Path,
    experiment: Optional[str],
) -> EvaluationSession:
    """Compose an evaluation runner for the supplied configuration."""

    runner_ctx = build_runner_context(config)
    runner = EvalRunner(runner_ctx)
    return EvaluationSession(
        runner=runner,
        config=config,
        config_path=config_path,
        experiment=experiment,
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
    """Execute evaluation episodes via the session runner."""

    return session.runner.run(
        episodes=int(episodes),
        force_render=force_render,
        save_rollouts=save_rollouts,
        rollout_dir=rollout_dir,
        auto_load=auto_load,
        checkpoint_path=checkpoint_path,
    )


def resolve_eval_episodes(
    session: EvaluationSession,
    *,
    override: Optional[int | str] = None,
    fallback: int = 5,
) -> int:
    """Determine how many evaluation episodes to execute."""

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


def main() -> None:  # pragma: no cover - CLI compatibility path
    session = create_evaluation_session(auto_load=True)
    episodes = session.default_episodes()
    run_evaluation(session, episodes=episodes, force_render=False)


__all__ = [
    "EvaluationSession",
    "create_evaluation_session",
    "build_evaluation_session",
    "run_evaluation",
    "resolve_eval_episodes",
]


if __name__ == "__main__":
    main()
