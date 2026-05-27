"""Training hooks — called by trainers at step, episode, and update boundaries."""
from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING, Any, Deque, Dict, List, Optional

import numpy as np

from loggers.console import ConsoleLogger
from loggers.wandb_logger import WandbLogger

if TYPE_CHECKING:
    from env.types import TransitionRecord
    from training.curriculum import CurriculumManager

_log = logging.getLogger(__name__)


class TrainingHook:
    """Base class — all methods are no-ops by default."""

    def on_step(self, record: "TransitionRecord") -> None:
        """Called after each agent decision with the full transition record.

        Override to collect transitions for dataset logging, custom metrics,
        or any per-step side-effect.  Default is a no-op.
        """
        pass

    def on_episode_end(
        self,
        episode: int,
        reward: float,
        info: Dict,
        metrics: Dict[str, float],
    ) -> None:
        pass

    def on_update(self, metrics: Dict[str, float]) -> None:
        pass

    def on_training_end(self) -> None:
        pass


class ConsoleHook(TrainingHook):
    """Logs per-episode stats to the console."""

    def __init__(
        self,
        logger: ConsoleLogger,
        log_every: int = 1,
        summary_every: int = 25,
    ) -> None:
        self._log = logger
        self._log_every = max(1, log_every)
        self._summary_every = max(1, summary_every)
        self._rewards: Deque[float] = deque(maxlen=summary_every)
        self._outcomes: List[str] = []

    def on_episode_end(self, episode: int, reward: float, info: Dict, metrics: Dict) -> None:
        self._rewards.append(reward)
        outcome = info.get("outcome", "?") if isinstance(info, dict) else "?"
        self._outcomes.append(str(outcome))

        if episode % self._log_every == 0:
            mean_r = np.mean(self._rewards) if self._rewards else 0.0
            self._log.print_info(
                f"ep {episode:>6}  reward={reward:+.2f}  mean={mean_r:+.2f}  outcome={outcome}"
            )

        if episode % self._summary_every == 0 and self._outcomes:
            from collections import Counter
            counts = Counter(self._outcomes[-self._summary_every:])
            self._log.print_info(f"  outcomes (last {self._summary_every}): {dict(counts)}")


class WandbHook(TrainingHook):
    """Logs episode and update metrics to Weights & Biases."""

    def __init__(self, wandb_logger: WandbLogger) -> None:
        self._wandb = wandb_logger
        self._step = 0

    def on_episode_end(self, episode: int, reward: float, info: Dict, metrics: Dict) -> None:
        log = {"episode/reward": reward, "episode/number": episode}
        if isinstance(info, dict):
            outcome = info.get("outcome")
            if outcome:
                log["episode/outcome"] = str(outcome)
        log.update({f"episode/{k}": v for k, v in metrics.items()})
        if hasattr(self._wandb, "log"):
            self._wandb.log(log, step=episode)

    def on_update(self, metrics: Dict[str, float]) -> None:
        self._step += 1
        if hasattr(self._wandb, "log"):
            self._wandb.log(metrics, step=self._step)


class CurriculumHook(TrainingHook):
    """Updates a :class:`~training.curriculum.CurriculumManager` after each episode.

    Reads ``info["outcome"]`` (the string value set by the trainers via
    :func:`~metrics.outcomes.determine_outcome`) and forwards it to
    :meth:`~training.curriculum.CurriculumManager.on_episode_end`.  When the
    curriculum advances, logs a message and optionally emits the new phase
    metrics to a W&B logger.

    Parameters
    ----------
    manager:
        The :class:`~training.curriculum.CurriculumManager` to update.
    wandb_logger:
        Optional W&B logger.  If supplied, curriculum summary metrics are
        logged after every episode.
    """

    def __init__(
        self,
        manager: "CurriculumManager",
        wandb_logger: Optional[Any] = None,
    ) -> None:
        self._manager = manager
        self._wandb = wandb_logger

    @property
    def manager(self) -> "CurriculumManager":
        return self._manager

    def on_episode_end(self, episode: int, reward: float, info: Dict, metrics: Dict) -> None:
        outcome = info.get("outcome", "timeout") if isinstance(info, dict) else "timeout"
        advanced = self._manager.on_episode_end(outcome)

        if advanced:
            _log.info(
                "Curriculum: phase → %d ('%s') at episode %d (success_rate=%.2f)",
                self._manager.phase_index,
                self._manager.current_phase.name,
                episode,
                self._manager.success_rate,
            )

        if self._wandb is not None and hasattr(self._wandb, "log"):
            summary = self._manager.summary()
            self._wandb.log(summary, step=episode)


class CheckpointHook(TrainingHook):
    """Saves agent checkpoints periodically and on best reward."""

    def __init__(
        self,
        agent: Any,
        output_dir: str,
        save_every: int = 100,
    ) -> None:
        from pathlib import Path
        self._agent = agent
        self._dir = Path(output_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._save_every = max(1, save_every)
        self._best_reward = float("-inf")
        self._recent_rewards: Deque[float] = deque(maxlen=50)

    def on_episode_end(self, episode: int, reward: float, info: Dict, metrics: Dict) -> None:
        self._recent_rewards.append(reward)
        mean = float(np.mean(self._recent_rewards))

        if episode % self._save_every == 0:
            self._agent.save(str(self._dir / f"checkpoint_ep{episode:06d}.pt"))

        if mean > self._best_reward and len(self._recent_rewards) >= 10:
            self._best_reward = mean
            self._agent.save(str(self._dir / "best_model.pt"))
