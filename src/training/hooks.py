"""Training hooks — called by trainers at step, episode, and update boundaries."""
from __future__ import annotations

import logging
import json
import time
from collections import deque
from pathlib import Path
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

    requires_transition_record: Optional[bool] = None

    def on_episode_start(self, metadata: Dict) -> None:
        """Small reset provenance record, independent of transition capture."""
        pass

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

    def on_collector_progress(self, metrics: Dict) -> None:
        """Liveness telemetry; does not advance the optimizer or episode count."""
        pass

    def on_training_end(self) -> None:
        pass


def transition_record_hooks(hooks: List[Any]) -> List[Any]:
    """Return hooks that consume full per-agent transition records.

    Explicit ``requires_transition_record`` declarations take priority.  For
    custom hooks without a declaration, an overridden ``on_step`` method is
    treated as requiring the full stable record contract.
    """
    consumers: List[Any] = []
    for hook in hooks:
        explicit = getattr(hook, "requires_transition_record", None)
        if explicit is not None:
            if bool(explicit):
                consumers.append(hook)
            continue
        on_step = getattr(type(hook), "on_step", None)
        if on_step is not None and on_step is not TrainingHook.on_step:
            consumers.append(hook)
    return consumers


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
                lap_count = info.get("lap_count") if isinstance(info, dict) else None
                lap_time = metrics.get("lap_time_s")
                laps_str = str(lap_count) if lap_count is not None else "n/a"
                lap_time_str = f"{lap_time:.2f}s" if lap_time is not None else "n/a"
                self._log.print_info(
                    f"ep {episode:>6}  reward={reward:+.2f}  mean={mean_r:+.2f}  "
                    f"laps={laps_str}  lap_time={lap_time_str}  outcome={outcome}"
                )

        if episode % self._summary_every == 0 and self._outcomes:
            from collections import Counter
            counts = Counter(self._outcomes[-self._summary_every:])
            self._log.print_info(f"  outcomes (last {self._summary_every}): {dict(counts)}")
            for aid, outcomes in self._agent_outcomes.items():
                agent_counts = Counter(outcomes)
                self._log.print_info(f"    {aid} outcomes: {dict(agent_counts)}")


class MAPPOConsoleHook(TrainingHook):
    """Bounded completed-episode window; update-driven even during long races."""

    def __init__(self, logger, *, window=100, every_updates=10, diagnostic_every=100):
        if min(window, every_updates) < 1 or diagnostic_every < 0:
            raise ValueError("Monitoring window/cadence must be positive; diagnostics may be zero")
        self._log = logger
        self._races = deque(maxlen=window)
        self._every = every_updates
        self._diagnostic_every = diagnostic_every
        self._metrics = {}
        self._episodes = 0
        self._last_printed = None

    def on_episode_end(self, episode, reward, info, metrics):
        if "race_record" in metrics:
            self._races.append(metrics["race_record"])
            self._episodes += 1

    def on_update(self, metrics):
        self._metrics = dict(metrics)
        update = int(metrics.get("train/updates", 0))
        if update == 1 or update % self._every == 0:
            self._print()
        if self._diagnostic_every and update % self._diagnostic_every == 0:
            values = " ".join(f"{key}={value:.4g}" for key, value in metrics.items()
                              if isinstance(value, (float, int)) and any(
                                  token in key for token in ("loss", "entropy", "kl", "clip", "explained_variance")))
            if values:
                self._log.print_info("MAPPO diagnostics  " + values)

    def _print(self):
        m, rows = self._metrics, list(self._races)
        identity = (m.get("train/environment_steps"), self._episodes)
        if identity == self._last_printed:
            return
        self._last_printed = identity
        text = (f"MAPPO update={m.get('train/updates', 0)} "
                f"env_steps={m.get('train/environment_steps', 0)} "
                f"env_steps/s={m.get('perf/round_env_steps_per_second', 0):.1f} "
                f"completed_window={len(rows)} completed_total={self._episodes}")
        def mean(key):
            values = [r[key] for r in rows if r.get(key) is not None]
            return float(np.mean(values)) if values else None
        def number(key):
            value = mean(key)
            return "n/a" if value is None else f"{value:.2f}"
        if rows:
            text += f" return={number('training_return')}"
            if rows[-1]["race_mode"] == "continuous":
                text += (f" progress_laps={number('mean_net_progress_laps')} "
                         f"laps={number('mean_learner_laps')} duration_s={number('duration_s')} "
                         f"collision_dnfs={sum(r['own_collision_dnf_count'] for r in rows)} "
                         f"boundary_dnfs={sum(r['own_boundary_dnf_count'] for r in rows)}")
            else:
                for label, key in (("both_finished", "both_finished"), ("first_place", "first_place"),
                                   ("sweep", "sweep"), ("collision_dnf", "any_learner_collision_dnf")):
                    value = mean(key)
                    text += f" {label}=" + ("n/a" if value is None else f"{value:.1%}")
                for aid, agent in rows[-1]["agents"].items():
                    if agent["team"] == "trainable":
                        rate = np.mean([r["agents"][aid]["finished"] for r in rows])
                        text += f" {aid}_finished={rate:.1%}"
        self._log.print_info(text)

    def on_training_end(self):
        self._print()


class WandbHook(TrainingHook):
    """Logs episode and update metrics to Weights & Biases."""

    requires_transition_record = True

    def __init__(self, wandb_logger: Optional[WandbLogger]) -> None:
        self._wandb = wandb_logger
        self._update = 0
        self._episodes: Dict[Any, Dict[str, Any]] = {}

    def on_step(self, record: "TransitionRecord") -> None:
        worker_id = getattr(record, "info", {}).get("worker_id")
        state = self._episodes.setdefault(worker_id, {"agents": set(), "components": {}, "map_id": None})
        aid = str(record.agent_id)
        state["agents"].add(aid)
        if record.map_id:
            state["map_id"] = str(record.map_id)
        agent_components = state["components"].setdefault(aid, {})
        for component, value in (record.reward_components or {}).items():
            try:
                component_value = float(value)
            except (TypeError, ValueError):
                continue
            agent_components[str(component)] = (
                agent_components.get(str(component), 0.0) + component_value
            )

    def take_episode_state(self, worker_id=None) -> Dict[str, Any]:
        """Drain accumulated metrics for local logging or transfer from a worker."""
        return self._episodes.pop(worker_id, {"agents": set(), "components": {}, "map_id": None})

    def on_episode_end(self, episode: int, reward: float, info: Dict, metrics: Dict) -> None:
        log = {"episode/reward": reward, "episode/number": episode}
        worker_id = info.get("worker_id") if isinstance(info, dict) else None
        state = self.take_episode_state(worker_id)
        if type(self) is WandbHook and isinstance(info, dict):
            state = info.get("_wandb_episode_state", state)
        if isinstance(info, dict):
            for key in ("worker_id", "worker_seed", "worker_episode"):
                if key in info:
                    log[f"episode/{key}"] = info[key]
            outcome = info.get("outcome")
            if outcome:
                log["episode/outcome"] = str(outcome)
            map_id = info.get("map_bundle") or state["map_id"]
            if map_id:
                log["episode/map_bundle"] = str(map_id)

        episode_steps = metrics.get("episode_steps")
        if episode_steps is not None:
            log["episode/steps"] = episode_steps
        lap_count = info.get("lap_count") if isinstance(info, dict) else None
        if lap_count is not None:
            log["episode/lap_count"] = lap_count
        lap_time = metrics.get("lap_time_s")
        if lap_time is not None:
            log["episode/lap_time_s"] = lap_time

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
        for aid, components in state["components"].items():
            for component, value in components.items():
                log[f"episode/reward_component/{component}/{aid}"] = value
                component_totals[component] = component_totals.get(component, 0.0) + value
        denominator = max(len(state["agents"]), 1)
        for component, total in component_totals.items():
            log[f"episode/reward_component_mean/{component}"] = total / denominator

        self._wandb.log_metrics(log)

    def on_collector_progress(self, metrics: Dict) -> None:
        self._wandb.log_metrics(metrics)

    def on_update(self, metrics: Dict[str, float]) -> None:
        self._update += 1
        self._wandb.log_metrics({"train/update": self._update, **metrics})


class CSVHook(TrainingHook):
    """Persist normal training metrics locally, independent of W&B."""

    def __init__(self, csv_logger: "CSVLogger") -> None:
        self._csv = csv_logger

    def on_episode_end(self, episode: int, reward: float, info: Dict, metrics: Dict) -> None:
        self._csv.log_training_episode(episode, reward, info, metrics)

    def on_update(self, metrics: Dict[str, float]) -> None:
        self._csv.log_update(metrics)

    def on_collector_progress(self, metrics: Dict) -> None:
        self._csv.log_collector_progress(metrics)

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
        save_best_training_reward: bool = True,
        save_every_steps: Optional[int] = None,
        save_final: bool = False,
    ) -> None:
        from pathlib import Path
        self._agent = agent
        self._dir = Path(output_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._save_every = max(1, save_every)
        self._best_reward = float("-inf")
        self._recent_rewards: Deque[float] = deque(maxlen=50)
        self._provenance = dict(provenance or {})
        self._save_best_training_reward = bool(save_best_training_reward)
        self._save_final = bool(save_final)
        if save_every_steps is not None and (isinstance(save_every_steps, bool)
                or not isinstance(save_every_steps, int) or save_every_steps <= 0):
            raise ValueError("save_every_steps must be a positive integer")
        self._save_every_steps = save_every_steps
        self._next_save_step = save_every_steps
        self._environment_steps = 0

    def on_update(self, metrics: Dict[str, float]) -> None:
        self._environment_steps = int(metrics.get("train/environment_steps", self._environment_steps))
        if self._next_save_step is not None and self._environment_steps >= self._next_save_step:
            self._save(self._dir / f"checkpoint_step{self._environment_steps:09d}.pt",
                       metadata={"environment_steps": self._environment_steps})
            self._next_save_step = (self._environment_steps // self._save_every_steps + 1) * self._save_every_steps

    def on_training_end(self) -> None:
        if self._save_every_steps is not None or self._save_final:
            self._save(self._dir / "final_model.pt",
                       metadata={"environment_steps": self._environment_steps})

    def _save(self, path: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Save atomically enough to preserve the original agent checkpoint on error."""
        if not self._provenance and not metadata:
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
            if self._provenance:
                checkpoint["provenance"] = dict(self._provenance)
            if metadata:
                checkpoint.update(metadata)
            torch.save(checkpoint, annotated)
            annotated.replace(target)
        finally:
            unannotated.unlink(missing_ok=True)
            annotated.unlink(missing_ok=True)

    def on_episode_end(self, episode: int, reward: float, info: Dict, metrics: Dict) -> None:
        if self._save_every_steps is not None:
            return
        self._recent_rewards.append(reward)
        mean = float(np.mean(self._recent_rewards))

        if episode % self._save_every == 0:
            self._save(self._dir / f"checkpoint_ep{episode:06d}.pt")

        if (
            self._save_best_training_reward
            and mean > self._best_reward
            and len(self._recent_rewards) >= 10
        ):
            self._best_reward = mean
            self._save(self._dir / "best_model.pt")


class EvaluationCheckpointHook(CheckpointHook):
    """Select ``best_model.pt`` using deterministic racing outcomes.

    Completion is always first. The default then ranks collision avoidance,
    absolute progress, and finish speed. The opt-in completion_progress strategy
    ranks earned net progress before collision avoidance until every race is
    completed, then ranks safety and finish time. Neither uses reward.
    """

    def __init__(
        self,
        agent: Any,
        output_dir: str,
        evaluator: Any,
        evaluate_every: int,
        provenance: Optional[Dict[str, Any]] = None,
        console: Optional[ConsoleLogger] = None,
        wandb_logger: Optional[WandbLogger] = None,
        selection_strategy: str = "completion_safety",
        evaluate_every_steps: Optional[int] = None,
    ) -> None:
        if selection_strategy not in {"map_curriculum", "completion_safety", "completion_progress", "lap_time", "team_completion", "team_combined", "team_first_place", "team_sweep", "team_combined_penalties"}:
            raise ValueError(f"Unknown checkpoint selection strategy: {selection_strategy!r}")
        self._selection_strategy = selection_strategy
        super().__init__(
            agent=agent,
            output_dir=output_dir,
            save_every=2**63 - 1,
            provenance=provenance,
            save_best_training_reward=False,
        )
        self._evaluator = evaluator
        self._evaluate_every = max(1, int(evaluate_every))
        if evaluate_every_steps is not None and (isinstance(evaluate_every_steps, bool)
                or not isinstance(evaluate_every_steps, int) or evaluate_every_steps <= 0):
            raise ValueError("evaluate_every_steps must be a positive integer")
        self._evaluate_every_steps = evaluate_every_steps
        self._next_evaluation_step = evaluate_every_steps
        self._console = console
        self._wandb = wandb_logger
        self._best_score: Optional[tuple[float, float, float, float]] = None
        self._history_path = self._dir / "evaluation_history.jsonl"
        self._evaluation_count = 0
        self.evaluation_seconds = 0.0
        self._policy_version = 0
        if console is not None and selection_strategy.startswith("team_"):
            console.print_info("Checkpoint priority: " + self.selection_priority(selection_strategy))

    @staticmethod
    def selection_priority(strategy):
        objective = {"team_completion": "at-least-one-finished rate",
                     "team_combined": "rank score", "team_combined_penalties": "rank plus penalty score",
                     "team_first_place": "first-place rate", "team_sweep": "sweep rate"}.get(strategy, strategy)
        return (f"both-finished rate > {objective} > fewer learner collision DNFs > "
                "clean finish time if both-finished=100%, otherwise earned net progress")

    @staticmethod
    def selection_score(summary: Dict[str, Any], strategy: str = "completion_safety") -> tuple[float, float, float, float]:
        if strategy == "map_curriculum":
            return tuple(summary["curriculum_selection_score"])
        if strategy.startswith("team_"):
            objective_key = {"team_completion": "team_completion_rate",
                             "team_combined": "team_rank_score",
                             "team_combined_penalties": "team_rank_penalty_score",
                             "team_first_place": "team_first_place",
                             "team_sweep": "team_sweep"}[strategy]
            complete = float(summary["team_both_finished_rate"])
            finish = summary.get("mean_clean_finish_time_s")
            if complete == 1.0:
                tie_break = -float(finish) if finish is not None else float("-inf")
            else:
                tie_break = round(float(summary.get("mean_net_progress") or 0.0), 6)
            return (complete,
                    float(summary[objective_key]),
                    -float(summary["team_collision_rate"]),
                    tie_break)
        if strategy == "lap_time":
            fastest = summary.get("fastest_valid_lap_s")
            error = summary.get("offtrack_error_m_s_per_lap")
            return (float(summary.get("completion_rate", 0.0)),
                    float(summary.get("valid_laps", 0)),
                    -float(fastest) if fastest is not None else float("-inf"),
                    -float(error) if error is not None else float("-inf"))
        completion = float(summary.get("completion_rate", 0.0))
        collision = float(summary.get("collision_rate", 1.0))
        progress = float(summary.get("mean_progress", 0.0))
        finish_steps = summary.get("mean_finish_steps")
        finish_speed_score = (
            -float(finish_steps) if finish_steps is not None else float("-inf")
        )
        if strategy == "completion_progress":
            net_progress = summary.get("mean_net_progress")
            if net_progress is None or not np.isfinite(net_progress):
                raise ValueError("completion_progress selection requires finite centerline progress deltas in every evaluation episode.")
            if completion == 1.0:
                # Successful episodes end just past the line. Their numerical
                # overshoot must not outrank safety or a faster race time.
                return (completion, 0.0, -collision, finish_speed_score)
            # Ignore sub-millionth-lap numerical jitter when selecting a model.
            return (completion, round(float(net_progress), 6), -collision, finish_speed_score)
        if strategy != "completion_safety":
            raise ValueError(f"Unknown checkpoint selection strategy: {strategy!r}")
        return (completion, -collision, progress, finish_speed_score)

    def on_episode_end(self, episode: int, reward: float, info: Dict, metrics: Dict) -> None:
        if self._evaluate_every_steps is not None:
            return
        if (episode + 1) % self._evaluate_every != 0:
            return
        self._evaluate_checkpoint(episode + 1)

    def on_update(self, metrics: Dict[str, float]) -> None:
        self._environment_steps = int(metrics.get("train/environment_steps", self._environment_steps))
        self._policy_version = int(metrics.get("train/updates", self._policy_version + 1))
        if self._next_evaluation_step is not None and self._environment_steps >= self._next_evaluation_step:
            self._evaluate_checkpoint(None)
            self._next_evaluation_step = (self._environment_steps // self._evaluate_every_steps + 1) * self._evaluate_every_steps

    def _evaluate_checkpoint(self, completed_episodes: Optional[int]) -> None:
        recording = getattr(self._evaluator, 'recording', None)
        context = {}
        if recording is not None:
            import hashlib
            evaluation_id = f'eval{self._evaluation_count+1:06d}'
            checkpoint, digest = None, None
            if recording.writer.can_record(self._environment_steps):
                checkpoint = (self._dir/'evaluation_checkpoints'/f'{evaluation_id}.pt').resolve()
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                self._save(checkpoint, metadata=dict(environment_steps=self._environment_steps,
                    policy_version=self._policy_version))
                digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            context = dict(evaluation_id=evaluation_id, environment_steps=self._environment_steps,
                policy_version=self._policy_version, checkpoint=str(checkpoint) if checkpoint else None,
                checkpoint_sha256=digest)
            recording.begin(context)
        started = time.perf_counter()
        summary = dict(self._evaluator.evaluate())
        summary["evaluation_seconds"] = time.perf_counter() - started
        self.evaluation_seconds += summary["evaluation_seconds"]
        summary["evaluation_seconds_total"] = self.evaluation_seconds
        summary.update(context)
        self._evaluation_count += 1
        run_id = self._provenance.get("run_id", self._dir.name)
        for index, row in enumerate(summary.get("episode_results", [])):
            row.update(run_id=run_id, environment_id="evaluation",
                       episode_id=f"{run_id}_eval{self._evaluation_count:06d}_ep{index:06d}",
                       policy_version=self._policy_version,
                       reported_at_environment_steps=self._environment_steps)
        score = self.selection_score(summary, self._selection_strategy)
        is_best = self._best_score is None or score > self._best_score
        record = {
            "training_episode": completed_episodes,
            "run_id": run_id,
            "policy_version": self._policy_version,
            "environment_steps": self._environment_steps,
            "selection_strategy": self._selection_strategy,
            "selection_priority": (self.selection_priority(self._selection_strategy)
                                   if self._selection_strategy.startswith("team_") else None),
            "selection_score": [
                value if np.isfinite(value) else None for value in score
            ],
            "is_best": is_best,
            **summary,
        }
        with self._history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

        if self._wandb is not None:
            scalar_metrics = {
                f"eval/{key}": value
                for key, value in summary.items()
                if isinstance(value, (int, float)) and value is not None
            }
            if completed_episodes is not None:
                scalar_metrics["eval/training_episode"] = completed_episodes
            scalar_metrics["eval/environment_steps"] = self._environment_steps
            self._wandb.log_metrics(scalar_metrics)

        if self._console is not None and self._selection_strategy.startswith("team_"):
            finish = summary.get("mean_clean_finish_time_s")
            finish_text = "n/a" if finish is None else f"{finish:.2f}s"
            self._console.print_info(
                f"checkpoint eval races={summary.get('episodes', 0)} env_steps={self._environment_steps} "
                f"both_finished={summary.get('team_both_finished_rate', 0):.1%} "
                f"first_place={summary.get('team_first_place', 0):.1%} "
                f"sweep={summary.get('team_sweep', 0):.1%} rank={summary.get('team_rank_score', 0):.3f} "
                f"clean_finish={finish_text} finishers={summary.get('clean_finish_count', 0)} "
                f"checkpoint={'saved best' if is_best else 'kept previous'}")
        elif self._console is not None:
            finish = summary.get("mean_clean_finish_time_s")
            finish_text = "n/a" if finish is None else f"{float(finish):.1f}"
            progress_key = "mean_net_progress" if self._selection_strategy == "completion_progress" else "mean_progress"
            self._console.print_info(
                "checkpoint eval  "
                f"episode={completed_episodes}  steps={self._environment_steps}  "
                f"completion={float(summary.get('completion_rate', 0.0)):.1%}  "
                f"collision={float(summary.get('collision_rate', 0.0)):.1%}  "
                f"timeout={float(summary.get('timeout_rate', 0.0)):.1%}  "
                f"{progress_key}={float(summary.get(progress_key, 0.0)):.3f}  "
                f"clean_finish_s={finish_text}  best={is_best}"
            )

        if is_best:
            self._best_score = score
            self._save(
                self._dir / "best_model.pt",
                metadata={"checkpoint_selection": record},
            )


class PhysicsEpisodeHook(TrainingHook):
    """Write sampled episode physics through the shared provenance logger."""
    requires_transition_record = True

    def __init__(self, output_dir, *, transition_records=True) -> None:
        from core.provenance import PhysicsEpisodeLog
        self.requires_transition_record = transition_records
        self._log = PhysicsEpisodeLog(output_dir)
        self.path = self._log.path

    def on_episode_start(self, metadata) -> None:
        self._log.write(metadata['episode_id'], metadata['map_id'], metadata.get('physics'))

    def on_step(self, record) -> None:
        self._log.write(record.episode_id, record.map_id, record.info.get('physics'))
