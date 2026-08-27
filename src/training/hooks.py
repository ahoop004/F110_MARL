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
    from loggers.csv_logger import CSVLogger
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
        self._agent_outcomes: Dict[str, Deque[str]] = {}

    def on_episode_end(self, episode: int, reward: float, info: Dict, metrics: Dict) -> None:
        self._rewards.append(reward)
        outcome = info.get("outcome", "?") if isinstance(info, dict) else "?"
        self._outcomes.append(str(outcome))

        agent_outcomes = metrics.get("agent_outcomes") if isinstance(metrics, dict) else None
        if isinstance(agent_outcomes, dict):
            for aid, agent_outcome in agent_outcomes.items():
                self._agent_outcomes.setdefault(aid, deque(maxlen=self._summary_every)).append(
                    str(agent_outcome)
                )
        agent_terminal_reasons = (
            metrics.get("agent_terminal_reasons") if isinstance(metrics, dict) else None
        )

        if episode % self._log_every == 0:
            mean_r = np.mean(self._rewards) if self._rewards else 0.0
            agent_rewards = metrics.get("agent_rewards") if isinstance(metrics, dict) else None
            if isinstance(agent_rewards, dict) and len(agent_rewards) > 1:
                rewards_str = "  ".join(
                    f"{aid}={r:+.2f}" for aid, r in agent_rewards.items()
                )
                self._log.print_info(
                    f"ep {episode:>6}  mean={mean_r:+.2f}  outcome={outcome}  | {rewards_str}"
                )
                if isinstance(agent_terminal_reasons, dict):
                    terminal_str = "  ".join(
                        f"{aid}={reason or 'active'}"
                        for aid, reason in agent_terminal_reasons.items()
                    )
                    self._log.print_info(f"  terminal reasons: {terminal_str}")
                individual_rewards = metrics.get("agent_individual_rewards")
                if (
                    metrics.get("reward_mode") == "team_shared"
                    and isinstance(individual_rewards, dict)
                ):
                    individual_str = "  ".join(
                        f"{aid}={r:+.2f}" for aid, r in individual_rewards.items()
                    )
                    self._log.print_info(
                        f"  individual reward signals: {individual_str}"
                    )
            else:
                self._log.print_info(
                    f"ep {episode:>6}  reward={reward:+.2f}  mean={mean_r:+.2f}  outcome={outcome}"
                )

        if episode % self._summary_every == 0 and self._outcomes:
            from collections import Counter
            counts = Counter(self._outcomes[-self._summary_every:])
            self._log.print_info(f"  outcomes (last {self._summary_every}): {dict(counts)}")
            for aid, outcomes in self._agent_outcomes.items():
                agent_counts = Counter(outcomes)
                self._log.print_info(f"    {aid} outcomes: {dict(agent_counts)}")


class WandbHook(TrainingHook):
    """Logs episode and update metrics to Weights & Biases."""

    def __init__(self, wandb_logger: WandbLogger) -> None:
        self._wandb = wandb_logger
        self._update = 0
        self._episode_agents: set[str] = set()
        self._reward_components: Dict[str, Dict[str, float]] = {}
        self._map_id: Optional[str] = None

    def on_step(self, record: "TransitionRecord") -> None:
        aid = str(record.agent_id)
        self._episode_agents.add(aid)
        if record.map_id:
            self._map_id = str(record.map_id)
        agent_components = self._reward_components.setdefault(aid, {})
        for component, value in (record.reward_components or {}).items():
            try:
                component_value = float(value)
            except (TypeError, ValueError):
                continue
            agent_components[str(component)] = (
                agent_components.get(str(component), 0.0) + component_value
            )

    def on_episode_end(self, episode: int, reward: float, info: Dict, metrics: Dict) -> None:
        log = {"episode/reward": reward, "episode/number": episode}
        if isinstance(info, dict):
            outcome = info.get("outcome")
            if outcome:
                log["episode/outcome"] = str(outcome)
            map_id = info.get("map_bundle") or self._map_id
            if map_id:
                log["episode/map_bundle"] = str(map_id)

        episode_steps = metrics.get("episode_steps")
        if episode_steps is not None:
            log["episode/steps"] = episode_steps

        # Per-agent breakdown (MAPPO with >1 trainable agent) — flatten into
        # individual scalar/string keys rather than nested dicts.
        agent_rewards = metrics.get("agent_rewards")
        if isinstance(agent_rewards, dict):
            for aid, r in agent_rewards.items():
                log[f"episode/reward/{aid}"] = r
        agent_individual_rewards = metrics.get("agent_individual_rewards")
        if isinstance(agent_individual_rewards, dict):
            for aid, r in agent_individual_rewards.items():
                log[f"episode/individual_reward/{aid}"] = r
        agent_outcomes = metrics.get("agent_outcomes")
        if isinstance(agent_outcomes, dict):
            for aid, o in agent_outcomes.items():
                log[f"episode/outcome/{aid}"] = str(o)
        for metric_name in (
            "agent_terminal_reasons",
            "agent_finish_positions",
            "agent_lap_counts",
        ):
            values = metrics.get(metric_name)
            if isinstance(values, dict):
                label = metric_name.removeprefix("agent_")
                for aid, value in values.items():
                    if value is not None:
                        log[f"episode/{label}/{aid}"] = value

        terminal_reasons = metrics.get("agent_terminal_reasons")
        if isinstance(agent_outcomes, dict) and agent_outcomes:
            agent_count = len(agent_outcomes)
            finished = sum(str(value) == "finished" for value in agent_outcomes.values())
            log["episode/team/completion_rate"] = finished / agent_count
            log["episode/team/all_finished"] = float(finished == agent_count)
        if isinstance(terminal_reasons, dict) and terminal_reasons:
            agent_count = len(terminal_reasons)
            collisions = sum(
                str(value) == "collision" for value in terminal_reasons.values()
            )
            timeouts = sum(
                str(value) == "time_limit" for value in terminal_reasons.values()
            )
            log["episode/team/collision_rate"] = collisions / agent_count
            log["episode/team/timeout_rate"] = timeouts / agent_count

        # Persist per-agent and team-mean component totals for reward debugging.
        component_totals: Dict[str, float] = {}
        for aid, components in self._reward_components.items():
            for component, value in components.items():
                log[f"episode/reward_component/{component}/{aid}"] = value
                component_totals[component] = component_totals.get(component, 0.0) + value
        denominator = max(len(self._episode_agents), 1)
        for component, total in component_totals.items():
            log[f"episode/reward_component_mean/{component}"] = total / denominator

        self._wandb.log_metrics(log)
        self._episode_agents.clear()
        self._reward_components.clear()
        self._map_id = None

    def on_update(self, metrics: Dict[str, float]) -> None:
        self._update += 1
        self._wandb.log_metrics({"train/update": self._update, **metrics})


class CSVHook(TrainingHook):
    """Persist normal training metrics locally, independent of W&B."""

    def __init__(self, csv_logger: "CSVLogger") -> None:
        self._csv = csv_logger

    def on_episode_end(self, episode: int, reward: float, info: Dict, metrics: Dict) -> None:
        self._csv.log_training_episode(episode, reward, info, metrics)

    def on_training_end(self) -> None:
        self._csv.close()


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
        provenance: Optional[Dict[str, Any]] = None,
    ) -> None:
        from pathlib import Path
        self._agent = agent
        self._dir = Path(output_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._save_every = max(1, save_every)
        self._best_reward = float("-inf")
        self._recent_rewards: Deque[float] = deque(maxlen=50)
        self._provenance = dict(provenance or {})

    def _save(self, path: Any) -> None:
        """Save atomically enough to preserve the original agent checkpoint on error."""
        if not self._provenance:
            self._agent.save(str(path))
            return

        import torch
        from pathlib import Path
        from utils.torch_io import safe_load

        target = Path(path)
        unannotated = target.with_name(f".{target.name}.unannotated")
        annotated = target.with_name(f".{target.name}.annotated")
        try:
            self._agent.save(str(unannotated))
            checkpoint = safe_load(str(unannotated), map_location="cpu")
            if not isinstance(checkpoint, dict):
                raise TypeError("Agent checkpoint must be a dictionary to attach provenance.")
            checkpoint["provenance"] = dict(self._provenance)
            torch.save(checkpoint, annotated)
            annotated.replace(target)
        finally:
            unannotated.unlink(missing_ok=True)
            annotated.unlink(missing_ok=True)

    def on_episode_end(self, episode: int, reward: float, info: Dict, metrics: Dict) -> None:
        self._recent_rewards.append(reward)
        mean = float(np.mean(self._recent_rewards))

        if episode % self._save_every == 0:
            self._save(self._dir / f"checkpoint_ep{episode:06d}.pt")

        if mean > self._best_reward and len(self._recent_rewards) >= 10:
            self._best_reward = mean
            self._save(self._dir / "best_model.pt")
