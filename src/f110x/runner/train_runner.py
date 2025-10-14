"""Training runner orchestrating engine rollouts and trainer updates."""
from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional

from f110x.engine.rollout import (
    BestReturnTracker,
    IdleTerminationTracker,
    build_trajectory_buffers,
    run_episode,
)
from f110x.runner.context import RunnerContext
from f110x.runner.rollout_helpers import build_rollout_hooks
from f110x.trainer.base import Trainer
from f110x.utils.builders import AgentBundle, AgentTeam
from f110x.utils.logger import Logger
from f110x.utils.output import resolve_output_dir, resolve_output_file


TrainerUpdateHook = Callable[[str, Trainer, Optional[Dict[str, Any]]], None]


@dataclass
class TrainRunner:
    """Compose trainers, environments, and reward helpers for training loops."""

    context: RunnerContext
    best_model_path: Path = field(init=False)
    checkpoint_dir: Path = field(init=False)
    trainer_map: Dict[str, Trainer] = field(init=False)
    _primary_bundle: AgentBundle = field(init=False)
    _primary_trainer: Optional[Trainer] = field(init=False, default=None)
    _logger: Logger = field(init=False)

    def __post_init__(self) -> None:  # noqa: D401 - behaviour described in class docstring
        self._ensure_primary_agent()
        self._primary_bundle = self.context.primary_bundle
        try:
            self._primary_trainer = self.context.primary_trainer
        except RuntimeError:
            self._primary_trainer = None

        self.trainer_map = dict(self.context.trainer_map)
        self._configure_output_paths()
        self._logger = self.context.logger
        self._trainer_stats: Dict[str, Dict[str, Any]] = {
            trainer_id: {} for trainer_id in self.trainer_map
        }

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------
    @property
    def team(self) -> AgentTeam:
        return self.context.team

    @property
    def env(self):
        return self.context.env

    @property
    def primary_agent_id(self) -> str:
        return self.context.primary_agent_id or self._primary_bundle.agent_id

    @property
    def primary_bundle(self) -> AgentBundle:
        return self._primary_bundle

    @property
    def primary_trainer(self) -> Optional[Trainer]:
        return self._primary_trainer

    @property
    def trainable_agent_ids(self) -> List[str]:
        return list(self.context.trainable_agent_ids)

    @property
    def opponent_agent_ids(self) -> List[str]:
        primary_id = self.primary_agent_id
        return [bundle.agent_id for bundle in self.team.agents if bundle.agent_id != primary_id]

    @property
    def roster_metadata(self) -> Dict[str, Any]:
        return {
            "agent_ids": [bundle.agent_id for bundle in self.team.agents],
            "roles": dict(self.team.roles),
            "trainable": list(self.context.trainable_agent_ids),
        }

    def run(
        self,
        *,
        episodes: int,
        update_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        trainer_update_hook: Optional[TrainerUpdateHook] = None,
        update_start: int = 0,
    ) -> List[Dict[str, Any]]:
        env = self.env
        team = self.team
        trainer_map = self.trainer_map
        results: List[Dict[str, Any]] = []

        primary_id = self.primary_agent_id
        primary_bundle = self.primary_bundle
        attacker_id = team.primary_role("attacker")
        defender_id = team.primary_role("defender")

        primary_cfg = primary_bundle.metadata.get("config", {})
        recent_window = max(1, int(primary_cfg.get("rolling_avg_window", 10)))
        best_tracker = BestReturnTracker(recent_window)
        recent_returns: Deque[float] = deque(maxlen=recent_window)
        recent_success: Deque[float] = deque(maxlen=recent_window)

        reward_cfg = self.context.reward_cfg
        truncation_penalty = self._resolve_reward_value(reward_cfg, "truncation_penalty")
        params_block = reward_cfg.get("params") if isinstance(reward_cfg.get("params"), dict) else {}

        def _extract_reward_param(name: str, default: float) -> float:
            value = reward_cfg.get(name)
            if value is None and isinstance(params_block, dict):
                value = params_block.get(name)
            if value is None:
                return float(default)
            try:
                return float(value)
            except (TypeError, ValueError):
                return float(default)

        idle_speed_threshold = _extract_reward_param("idle_speed_threshold", 0.4)
        idle_patience_steps = int(round(_extract_reward_param("idle_patience_steps", 200)))
        idle_tracker = IdleTerminationTracker(
            idle_speed_threshold,
            idle_patience_steps,
            agent_ids=self.trainable_agent_ids,
        )
        reward_sharing_cfg = reward_cfg.get("shared_reward")

        trajectory_buffers, off_policy_ids = build_trajectory_buffers(team, trainer_map)

        hooks = build_rollout_hooks(
            self.context,
            team,
            env,
            deterministic=False,
        )
        reward_factory = hooks.reward_factory
        compute_actions = hooks.compute_actions
        prepare_next = hooks.prepare_next_observation
        reset_env = hooks.reset_fn

        def on_offpolicy_flush(agent_id: str, trainer: Trainer, buffer) -> None:
            for _ in range(buffer.updates_per_step):
                stats = trainer.update()
                self._record_trainer_stats(agent_id, stats)
                if trainer_update_hook:
                    trainer_update_hook(agent_id, trainer, stats)

        if self.context.render_interval:
            def should_render(ep_index: int, _step: int) -> bool:
                return (ep_index + 1) % self.context.render_interval == 0
        else:
            should_render = None

        agent_ids = list(env.possible_agents)
        update_after = max(1, int(self.context.update_after or 1))
        _ = update_start  # kept for compatibility with legacy callers

        logger = self._logger
        total_episodes = int(episodes)
        logger.start({
            "mode": "train",
            "primary_agent": primary_id,
            "train/episodes_total": total_episodes,
        })
        logger.update_context(
            mode="train",
            primary_agent=primary_id,
            **{"train/episodes_total": total_episodes},
        )

        for episode_idx in range(int(episodes)):
            rollout = run_episode(
                env=env,
                team=team,
                trainer_map=trainer_map,
                trajectory_buffers=trajectory_buffers,
                reward_wrapper_factory=reward_factory,
                compute_actions=compute_actions,
                prepare_next_observation=prepare_next,
                idle_tracker=idle_tracker,
                episode_index=episode_idx,
                reset_fn=reset_env,
                agent_ids=agent_ids,
                render_condition=should_render,
                on_offpolicy_flush=on_offpolicy_flush,
                reward_sharing=reward_sharing_cfg,
            )

            returns = dict(rollout.returns)
            reward_breakdown = dict(rollout.reward_breakdown)

            if truncation_penalty:
                for agent_id, truncated in rollout.truncations.items():
                    if not truncated:
                        continue
                    agent_breakdown = reward_breakdown.setdefault(agent_id, {})
                    if "truncation_penalty" in agent_breakdown:
                        continue
                    returns[agent_id] = returns.get(agent_id, 0.0) + truncation_penalty
                    agent_breakdown["truncation_penalty"] = (
                        agent_breakdown.get("truncation_penalty", 0.0) + truncation_penalty
                    )

            if rollout.idle_triggered:
                logger.info(
                    "Idle stop triggered",
                    extra={
                        "episode": episode_idx + 1,
                        "steps": rollout.steps,
                        "idle_patience_steps": idle_patience_steps,
                        "idle_speed_threshold": idle_speed_threshold,
                    },
                )

            epsilon_val = self._resolve_primary_epsilon()

            defender_crashed: Optional[bool] = None
            defender_survival_steps: Optional[int] = None
            if defender_id is not None:
                defender_step = rollout.collision_steps.get(defender_id, -1)
                defender_crashed = defender_step >= 0
                defender_survival_steps = (
                    int(defender_step) if defender_crashed else rollout.steps
                )

            attacker_crashed: Optional[bool] = None
            if attacker_id is not None:
                attacker_step = rollout.collision_steps.get(attacker_id, -1)
                attacker_crashed = attacker_step >= 0

            success: Optional[bool] = None
            if defender_crashed is not None and attacker_crashed is not None:
                success = defender_crashed and not attacker_crashed
            elif defender_crashed is not None:
                success = defender_crashed
            elif attacker_crashed is not None:
                success = not attacker_crashed

            collisions_total = int(sum(rollout.collisions.values()))
            episode_record: Dict[str, Any] = {
                "episode": episode_idx + 1,
                "steps": rollout.steps,
                "cause": rollout.cause,
                "reward_task": rollout.reward_task,
                "reward_mode": rollout.reward_mode,
                "returns": returns,
                "reward_breakdown": reward_breakdown,
                "success": success,
                "collisions_total": collisions_total,
                "idle_truncated": rollout.idle_triggered,
            }

            if defender_crashed is not None:
                episode_record["defender_crashed"] = defender_crashed
            if attacker_crashed is not None:
                episode_record["attacker_crashed"] = attacker_crashed
            if defender_survival_steps is not None:
                episode_record["defender_survival_steps"] = defender_survival_steps

            if rollout.spawn_points:
                episode_record["spawn_points"] = dict(rollout.spawn_points)
            if rollout.spawn_option is not None:
                episode_record["spawn_option"] = rollout.spawn_option
            if epsilon_val is not None:
                episode_record["epsilon"] = epsilon_val

            for aid in agent_ids:
                episode_record[f"collision_count_{aid}"] = int(rollout.collisions.get(aid, 0))
                step_val = rollout.collision_steps.get(aid, -1)
                if step_val >= 0:
                    episode_record[f"collision_step_{aid}"] = int(step_val)
                episode_record[f"avg_speed_{aid}"] = float(rollout.average_speeds.get(aid, 0.0))

            primary_return = float(returns.get(primary_id, 0.0))
            recent_returns.append(primary_return)
            rolling_return = float(sum(recent_returns) / len(recent_returns))

            if success is not None:
                recent_success.append(1.0 if success else 0.0)
            success_rate = (
                float(sum(recent_success) / len(recent_success)) if recent_success else None
            )

            collision_rate = float(collisions_total) / max(float(rollout.steps), 1.0)
            best_average = best_tracker.best if best_tracker.best != float("-inf") else None
            buffer_fraction = None
            if primary_id:
                buffer_fraction = self._trainer_stats.get(primary_id, {}).get("buffer_fraction")
            if buffer_fraction is None:
                for stats_snapshot in self._trainer_stats.values():
                    candidate = stats_snapshot.get("buffer_fraction")
                    if candidate is not None:
                        buffer_fraction = candidate
                        break

            metrics: Dict[str, Any] = {
                "train/episode": float(episode_idx + 1),
                "train/episodes_total": float(total_episodes),
                "train/steps": float(rollout.steps),
                "train/return": primary_return,
                "train/return_mean": rolling_return,
                "train/collisions": float(collisions_total),
                "train/collision_rate": collision_rate,
                "train/idle": bool(rollout.idle_triggered),
                "train/reward_task": rollout.reward_task,
            }
            if primary_id:
                metrics["train/primary_agent"] = primary_id
            if success is not None:
                metrics["train/success"] = bool(success)
            if success_rate is not None:
                metrics["train/success_rate"] = success_rate
            if epsilon_val is not None:
                metrics["train/epsilon"] = float(epsilon_val)
            if attacker_crashed is not None:
                metrics["train/attacker_crashed"] = bool(attacker_crashed)
            if defender_crashed is not None:
                metrics["train/defender_crashed"] = bool(defender_crashed)
            if defender_survival_steps is not None:
                metrics["train/defender_survival_steps"] = float(defender_survival_steps)
            if best_average is not None:
                metrics["train/return_best"] = float(best_average)
            metrics["train/return_window"] = float(recent_returns.maxlen)
            if buffer_fraction is not None:
                metrics["train/buffer_fraction"] = float(buffer_fraction)
                episode_record["buffer_fraction"] = float(buffer_fraction)

            for aid, value in returns.items():
                metrics[f"train/agent/{aid}/return"] = float(value)
            for aid, count in rollout.collisions.items():
                metrics[f"train/agent/{aid}/collisions"] = float(count)
            for aid, speed in rollout.average_speeds.items():
                metrics[f"train/agent/{aid}/avg_speed"] = float(speed)
            for aid, step_val in rollout.collision_steps.items():
                if step_val >= 0:
                    metrics[f"train/agent/{aid}/collision_step"] = float(step_val)
            for aid, breakdown in reward_breakdown.items():
                for name, value in breakdown.items():
                    metrics[f"train/reward/{aid}/{name}"] = float(value)

            logger.log_metrics("train", metrics, step=episode_idx + 1)

            results.append(episode_record)

            if update_callback:
                payload: Dict[str, Any] = {
                    "train/episode": float(episode_idx + 1),
                    "train/return": primary_return,
                }
                for aid, value in returns.items():
                    payload[f"train/agent/{aid}/return"] = float(value)
                update_callback(payload)

            if (episode_idx + 1) % update_after == 0:
                for trainer_id, trainer in trainer_map.items():
                    if trainer_id in off_policy_ids:
                        continue
                    stats = trainer.update()
                    self._record_trainer_stats(trainer_id, stats)
                    if trainer_update_hook:
                        trainer_update_hook(trainer_id, trainer, stats)

            ppo_return = returns.get(primary_id, 0.0)
            new_best_avg = best_tracker.observe(ppo_return)
            if new_best_avg is not None:
                saved = self._save_primary_model()
                if saved:
                    logger.info(
                        "New best model checkpoint",
                        extra={
                            "episode": episode_idx + 1,
                            "avg_return": float(new_best_avg),
                            "path": str(self.best_model_path),
                        },
                    )

        for trainer_id, trainer in trainer_map.items():
            buffer = trajectory_buffers.get(trainer_id)
            flushed = False
            if buffer is not None:
                flushed = buffer.flush()
            if buffer is not None and trainer_id in off_policy_ids:
                if flushed:
                    for _ in range(buffer.updates_per_step):
                        stats = trainer.update()
                        self._record_trainer_stats(trainer_id, stats)
                        if trainer_update_hook:
                            trainer_update_hook(trainer_id, trainer, stats)
                continue
            stats = trainer.update()
            self._record_trainer_stats(trainer_id, stats)
            if trainer_update_hook:
                trainer_update_hook(trainer_id, trainer, stats)

        self.context.trainer_map = dict(trainer_map)
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _record_trainer_stats(
        self,
        trainer_id: str,
        stats: Optional[Dict[str, Any]],
    ) -> None:
        if not stats:
            return
        snapshot = self._trainer_stats.setdefault(trainer_id, {})
        snapshot.update(stats)

    def _ensure_primary_agent(self) -> None:
        if self.context.primary_agent_id:
            return
        candidates = list(self.context.trainable_agent_ids)
        if not candidates and self.context.team.agents:
            candidates = [self.context.team.agents[0].agent_id]
        if not candidates:
            raise RuntimeError("RunnerContext does not expose any agents to select as primary")
        self.context.set_primary_agent(candidates[0])

    def _configure_output_paths(self) -> None:
        output_root = self.context.output_root
        output_root.mkdir(parents=True, exist_ok=True)
        self.context.cfg.main.schema.output_root = str(output_root)

        bundle_cfg = dict(self._primary_bundle.metadata.get("config", {}))
        if not bundle_cfg and self._primary_bundle.algo.lower() == "ppo":
            bundle_cfg = self.context.cfg.ppo.to_dict()

        save_dir_value = bundle_cfg.get("save_dir", "checkpoints")
        self.checkpoint_dir = resolve_output_dir(save_dir_value, output_root)
        bundle_cfg["save_dir"] = str(self.checkpoint_dir)

        checkpoint_name = bundle_cfg.get(
            "checkpoint_name",
            f"{self._primary_bundle.algo.lower()}_best.pt",
        )
        run_suffix = os.environ.get("RUN_ITER") or os.environ.get("RUN_SEED")
        safe_suffix = self._slugify_suffix(run_suffix)
        if safe_suffix:
            base_name = Path(checkpoint_name)
            checkpoint_name = f"{base_name.stem}_{safe_suffix}{base_name.suffix}"
        bundle_cfg["checkpoint_name"] = checkpoint_name
        self.best_model_path = self.checkpoint_dir / checkpoint_name
        self._primary_bundle.metadata["config"] = bundle_cfg

        main_checkpoint = self.context.cfg.main.checkpoint
        if main_checkpoint:
            resolved = resolve_output_file(main_checkpoint, output_root)
            self.context.cfg.main.schema.checkpoint = str(resolved)

    def _resolve_primary_epsilon(self) -> Optional[float]:
        trainer = self.primary_trainer
        if trainer is None:
            return None
        accessor = getattr(trainer, "epsilon", None)
        if callable(accessor):
            try:
                value = accessor()
            except Exception:  # pragma: no cover - defensive guard around custom trainers
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
        return None

    def _save_primary_model(self) -> bool:
        controller = self._primary_bundle.controller
        save_fn = getattr(controller, "save", None)
        if not callable(save_fn):
            return False
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        save_fn(str(self.best_model_path))
        return True

    @staticmethod
    def _slugify_suffix(value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in str(value))
        cleaned = cleaned.strip("-")
        return cleaned or None

    @staticmethod
    def _resolve_reward_value(cfg: Dict[str, Any], key: str) -> float:
        """Resolve reward configuration values, honouring nested params."""

        raw = cfg.get(key)
        if raw is not None:
            try:
                return float(raw)
            except (TypeError, ValueError):
                pass

        params = cfg.get("params")
        if isinstance(params, dict):
            raw = params.get(key)
            if raw is not None:
                try:
                    return float(raw)
                except (TypeError, ValueError):
                    pass

        return 0.0


__all__ = ["TrainRunner", "TrainerUpdateHook"]
