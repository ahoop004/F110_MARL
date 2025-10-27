"""Training runner orchestrating engine rollouts and trainer updates."""
from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Mapping

from f110x.engine.rollout import (
    BestReturnTracker,
    IdleTerminationTracker,
    build_trajectory_buffers,
    run_episode,
)
from f110x.runner.context import RunnerContext
from f110x.runner.eval_runner import EvalRunner
from f110x.runner.rollout_helpers import build_rollout_hooks
from f110x.trainer.base import Trainer
from f110x.utils.builders import AgentBundle, AgentTeam
from f110x.utils.logger import Logger
from f110x.utils.output import resolve_output_dir, resolve_output_file
from f110x.utils.start_pose import StartPoseOption


TrainerUpdateHook = Callable[[str, Trainer, Optional[Dict[str, Any]]], None]


class SpawnCurriculumManager:
    """Gate random spawn usage based on recent success rate."""

    def __init__(self, context: RunnerContext, config: Dict[str, Any], logger: Logger) -> None:
        self.context = context
        self.logger = logger
        self.config = dict(config)

        self.success_window = max(1, int(self.config.get("success_window", self.config.get("window", 200))))
        self.activation_samples = max(1, int(self.config.get("activation_samples", self.success_window)))
        self.min_episode = max(0, int(self.config.get("min_episode", 0)))
        self.enable_rate = float(self.config.get("enable_rate", self.config.get("threshold", 0.7)))
        self.enable_patience = max(1, int(self.config.get("enable_patience", self.config.get("confirm_patience", 3))))
        self.disable_rate = float(self.config.get("disable_rate", self.config.get("revert_rate", 0.5)))
        self.disable_patience = max(1, int(self.config.get("disable_patience", self.config.get("revert_patience", 2))))
        self.cooldown = max(0, int(self.config.get("cooldown", 0)))
        self.persist = bool(self.config.get("persist", False))
        self.start_enabled = bool(self.config.get("start_enabled", False))

        self._success_history: Deque[float] = deque(maxlen=self.success_window)
        self._structured_history: Deque[float] = deque(maxlen=self.success_window)
        self._random_history: Deque[float] = deque(maxlen=self.success_window)
        self._enable_streak = 0
        self._disable_streak = 0
        self._last_toggle_episode: Optional[int] = None

        all_options = list(context.start_pose_options or [])
        self._all_options: List[StartPoseOption] = list(all_options)
        self._random_options: List[StartPoseOption] = [opt for opt in all_options if self._is_random_option(opt)]
        random_ids = {id(opt) for opt in self._random_options}
        self._baseline_options: List[StartPoseOption] = [opt for opt in all_options if id(opt) not in random_ids]
        if not self._baseline_options and all_options:
            self._baseline_options = list(all_options)

        self.random_capable = bool(self._random_options)
        initial_state = self.start_enabled and self.random_capable
        self.random_enabled = False
        self._apply_state(initial_state, force=True)

    @staticmethod
    def _is_random_option(option: StartPoseOption) -> bool:
        metadata = getattr(option, "metadata", {}) or {}
        if not isinstance(metadata, dict):
            return False
        if metadata.get("spawn_random_pool"):
            return True
        option_id = metadata.get("spawn_option_id")
        if isinstance(option_id, str) and option_id.strip().lower() == "spawn_random":
            return True
        return False

    @property
    def window(self) -> int:
        return self.success_window

    @property
    def success_rate(self) -> Optional[float]:
        if not self._success_history:
            return None
        return float(sum(self._success_history) / len(self._success_history))

    @staticmethod
    def _rate(history: Deque[float]) -> Optional[float]:
        if not history:
            return None
        return float(sum(history) / len(history))

    def _apply_state(self, enable: bool, *, force: bool = False) -> bool:
        target = bool(enable and self.random_capable)
        previous = self.random_enabled
        self.random_enabled = target
        state_changed = previous != target
        if state_changed:
            if self.random_enabled:
                self._random_history.clear()
            else:
                self._structured_history.clear()
        if target:
            selected = list(self._random_options or self._baseline_options or self._all_options)
        else:
            selected = list(self._baseline_options or self._random_options or self._all_options)

        self.context.start_pose_options = selected or None
        return state_changed and not force

    def observe(self, episode: int, success: Optional[bool]) -> Dict[str, Any]:
        if success is not None:
            value = 1.0 if success else 0.0
            self._success_history.append(value)
            if self.random_enabled:
                self._random_history.append(value)
            else:
                self._structured_history.append(value)

        success_rate = self.success_rate
        structured_rate = self._rate(self._structured_history)
        random_rate = self._rate(self._random_history)
        stage_success_rate = random_rate if self.random_enabled else structured_rate
        history_ready = len(self._success_history) >= self.activation_samples
        changed = False

        if not self.random_capable:
            return {
                "changed": False,
                "enabled": self.random_enabled,
                "success_rate": success_rate,
                "stage_success_rate": stage_success_rate,
                "structured_success_rate": structured_rate,
                "random_success_rate": random_rate,
                "stage": "random" if self.random_enabled else "structured",
            }

        if not self.random_enabled:
            if episode < self.min_episode or not history_ready or success_rate is None:
                self._enable_streak = 0
            else:
                if success_rate >= self.enable_rate:
                    self._enable_streak += 1
                else:
                    self._enable_streak = 0

                if self._enable_streak >= self.enable_patience:
                    changed = self._apply_state(True)
                    self._last_toggle_episode = episode
                    self._enable_streak = 0
                    self._disable_streak = 0
        else:
            if not self.persist and history_ready and success_rate is not None:
                if success_rate < self.disable_rate:
                    self._disable_streak += 1
                else:
                    self._disable_streak = 0

                cooldown_ready = (
                    self.cooldown <= 0
                    or self._last_toggle_episode is None
                    or (episode - self._last_toggle_episode) >= self.cooldown
                )
                if cooldown_ready and self._disable_streak >= self.disable_patience:
                    changed = self._apply_state(False)
                    self._last_toggle_episode = episode
                    self._disable_streak = 0
                    self._enable_streak = 0

        if changed:
            self.logger.info(
                "Spawn curriculum transition",
                extra={
                    "episode": episode,
                    "stage": "random" if self.random_enabled else "structured",
                    "success_rate": float(success_rate) if success_rate is not None else None,
                },
            )

        return {
            "changed": changed,
            "enabled": self.random_enabled,
            "success_rate": success_rate,
            "stage_success_rate": stage_success_rate,
            "structured_success_rate": structured_rate,
            "random_success_rate": random_rate,
            "stage": "random" if self.random_enabled else "structured",
        }


class DefenderCurriculumManager:
    """Schedules defender heuristic parameters using attacker outcomes."""

    def __init__(
        self,
        *,
        controller: Any,
        config: Mapping[str, Any],
        logger: Logger,
    ) -> None:
        self.controller = controller
        self.logger = logger
        self.config = dict(config or {})

        self.success_window = max(1, int(self.config.get("success_window", self.config.get("window", 200))))
        self.activation_samples = max(1, int(self.config.get("activation_samples", self.success_window)))
        self.min_episode = max(0, int(self.config.get("min_episode", 0)))
        self.cooldown = max(0, int(self.config.get("cooldown", 0)))
        self.persist = bool(self.config.get("persist", False))
        self.enable_rate_default = float(self.config.get("enable_rate", 0.7))
        self.disable_rate_default = float(self.config.get("disable_rate", 0.5))
        self.enable_patience_default = max(1, int(self.config.get("enable_patience", 3)))
        self.disable_patience_default = max(1, int(self.config.get("disable_patience", 2)))

        raw_stages = self.config.get("stages") or []
        stages: List[Dict[str, Any]] = []
        for idx, stage in enumerate(raw_stages):
            if isinstance(stage, Mapping):
                entry = dict(stage)
            else:
                continue
            entry.setdefault("name", f"stage_{idx}")
            params = entry.get("params")
            if params is not None and not isinstance(params, Mapping):
                raise TypeError("Defender curriculum stage 'params' must be a mapping")
            stages.append(entry)
        if not stages:
            stages = [{"name": "baseline"}]
        baseline_params = stages[0].get("params")
        if not isinstance(baseline_params, Mapping) or not baseline_params:
            stages[0]["params"] = self._capture_params()
        else:
            stages[0]["params"] = dict(baseline_params)

        self.stages = stages
        self.history: Deque[float] = deque(maxlen=self.success_window)
        self.stage_histories: Dict[int, Deque[float]] = {}
        self.current_stage_index = 0
        self._last_change_episode: Optional[int] = None
        self._promote_streak = 0
        self._regress_streak = 0
        self._apply_stage(0, episode=0, initial=True)

    @property
    def window(self) -> int:
        return self.success_window

    @property
    def success_rate(self) -> Optional[float]:
        if not self.history:
            return None
        return float(sum(self.history) / len(self.history))

    def _stage_history(self, index: int) -> Deque[float]:
        history = self.stage_histories.get(index)
        if history is None:
            history = deque(maxlen=self.success_window)
            self.stage_histories[index] = history
        return history

    @staticmethod
    def _rate(history: Deque[float]) -> Optional[float]:
        if not history:
            return None
        return float(sum(history) / len(history))

    def _capture_params(self) -> Dict[str, Any]:
        exporter = getattr(self.controller, "export_config", None)
        if callable(exporter):
            snapshot = exporter()
        else:
            keys = ("max_speed", "min_speed", "steering_gain", "bubble_radius", "steer_smooth")
            snapshot = {key: getattr(self.controller, key) for key in keys if hasattr(self.controller, key)}
        return self._sanitize_params(snapshot)

    @staticmethod
    def _sanitize_params(params: Mapping[str, Any]) -> Dict[str, Any]:
        cleaned: Dict[str, Any] = {}
        for key, value in params.items():
            if isinstance(value, (int, float)):
                cleaned[key] = float(value)
            else:
                cleaned[key] = value
        return cleaned

    def _apply_stage(self, index: int, *, episode: int, initial: bool = False) -> None:
        index = max(0, min(index, len(self.stages) - 1))
        stage = self.stages[index]
        params = stage.get("params")
        if isinstance(params, Mapping):
            self.controller.apply_config(params)
        applied = self._capture_params()
        stage["_applied_params"] = applied
        stage_name = stage.get("name", f"stage_{index}")
        previous_index = getattr(self, "current_stage_index", 0)
        self.current_stage_index = index
        self._promote_streak = 0
        self._regress_streak = 0
        if initial:
            self._last_change_episode = None
            return
        self._stage_history(index).clear()
        self._last_change_episode = episode
        if index != previous_index:
            self.logger.info(
                "Defender curriculum transition",
                extra={
                    "episode": episode,
                    "stage": stage_name,
                    "stage_index": index,
                    "defender_params": applied,
                },
            )

    def _cooldown_ready(self, episode: int) -> bool:
        if self.cooldown <= 0:
            return True
        if self._last_change_episode is None:
            return True
        return (episode - self._last_change_episode) >= self.cooldown

    def observe(self, episode: int, success: Optional[bool]) -> Dict[str, Any]:
        if success is not None:
            value = 1.0 if success else 0.0
            self.history.append(value)
            self._stage_history(self.current_stage_index).append(value)

        success_rate = self.success_rate
        history_ready = len(self.history) >= self.activation_samples
        changed = False

        if len(self.stages) <= 1:
            return self._state(changed, success_rate)

        if not history_ready or success_rate is None or episode < self.min_episode:
            self._promote_streak = 0
            self._regress_streak = 0
            return self._state(changed, success_rate)

        if self.current_stage_index < len(self.stages) - 1:
            next_stage_index = self.current_stage_index + 1
            next_stage = self.stages[next_stage_index]
            enable_rate = float(next_stage.get("enable_rate", self.enable_rate_default))
            enable_patience = max(1, int(next_stage.get("enable_patience", self.enable_patience_default)))
            if success_rate >= enable_rate:
                self._promote_streak += 1
            else:
                self._promote_streak = 0
            if self._promote_streak >= enable_patience and self._cooldown_ready(episode):
                self._apply_stage(next_stage_index, episode=episode)
                changed = True

        if not self.persist and self.current_stage_index > 0:
            current_stage = self.stages[self.current_stage_index]
            disable_rate = float(current_stage.get("disable_rate", self.disable_rate_default))
            disable_patience = max(1, int(current_stage.get("disable_patience", self.disable_patience_default)))
            if success_rate < disable_rate:
                self._regress_streak += 1
            else:
                self._regress_streak = 0
            if self._regress_streak >= disable_patience and self._cooldown_ready(episode):
                self._apply_stage(self.current_stage_index - 1, episode=episode)
                changed = True

        return self._state(changed, success_rate)

    def _state(self, changed: bool, success_rate: Optional[float]) -> Dict[str, Any]:
        stage = self.stages[self.current_stage_index]
        stage_history = self.stage_histories.get(self.current_stage_index)
        return {
            "changed": changed,
            "stage_index": self.current_stage_index,
            "stage": stage.get("name", f"stage_{self.current_stage_index}"),
            "success_rate": success_rate,
            "stage_success_rate": self._rate(stage_history) if stage_history else None,
            "params": dict(stage.get("_applied_params", {})),
        }


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
    _eval_interval: int = field(init=False, default=0)
    _eval_episodes: int = field(init=False, default=1)
    _eval_runner: Optional[EvalRunner] = field(init=False, default=None)

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
        interval_value = getattr(self.context, "eval_interval", 0)
        try:
            interval_int = int(interval_value)
        except (TypeError, ValueError):
            interval_int = 0
        self._eval_interval = max(interval_int, 0)
        episodes_value = getattr(self.context, "eval_episodes", 1)
        try:
            episodes_int = int(episodes_value)
        except (TypeError, ValueError):
            episodes_int = 1
        self._eval_episodes = max(episodes_int, 1)
        self._eval_runner = None

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

        logger = self._logger

        spawn_manager: Optional[SpawnCurriculumManager] = None
        spawn_cfg = self.context.cfg.env.get("spawn_curriculum", {})
        if isinstance(spawn_cfg, dict) and spawn_cfg:
            spawn_manager = SpawnCurriculumManager(
                context=self.context,
                config=spawn_cfg,
                logger=logger,
            )

        primary_id = self.primary_agent_id
        primary_bundle = self.primary_bundle
        roster = team.roster

        attacker_id = team.primary_role("attacker")
        if attacker_id is None:
            attacker_id = primary_id or (roster.agent_ids[0] if roster.agent_ids else None)

        defender_id = team.primary_role("defender")
        if defender_id is None and attacker_id is not None:
            defender_id = roster.first_other(attacker_id)
        if defender_id is None:
            for bundle in team.agents:
                if attacker_id is None or bundle.agent_id != attacker_id:
                    defender_id = bundle.agent_id
                    break

        defender_manager: Optional[DefenderCurriculumManager] = None
        if defender_id:
            defender_bundle = getattr(team, "by_id", {}).get(defender_id)
            if defender_bundle is not None:
                curriculum_cfg = defender_bundle.metadata.get("policy_curriculum", {})
                controller = getattr(defender_bundle, "controller", None)
                if curriculum_cfg and hasattr(controller, "apply_config"):
                    defender_manager = DefenderCurriculumManager(
                        controller=controller,
                        config=curriculum_cfg,
                        logger=logger,
                    )

        primary_cfg = primary_bundle.metadata.get("config", {})
        recent_window = max(1, int(primary_cfg.get("rolling_avg_window", 10)))
        if spawn_manager is not None:
            recent_window = max(recent_window, spawn_manager.window)
        if defender_manager is not None:
            recent_window = max(recent_window, defender_manager.window)
        best_tracker = BestReturnTracker(recent_window)
        recent_returns: Deque[float] = deque(maxlen=recent_window)
        recent_success: Deque[float] = deque(maxlen=recent_window)
        total_successes: int = 0

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

            assisted_success: Optional[bool] = None
            if success and attacker_id is not None:
                attacker_components = reward_breakdown.get(attacker_id, {})
                success_reward_val = float(attacker_components.get("success_reward", 0.0) or 0.0)
                assisted_success = success_reward_val > 0.0
                if not assisted_success:
                    logger.log_event(
                        "debug",
                        "Defender crash without explicit success reward",
                        extra={
                            "episode": episode_idx + 1,
                            "steps": rollout.steps,
                            "spawn_option": rollout.spawn_option,
                            "spawn_points": rollout.spawn_points,
                        },
                    )

            if not success and attacker_id is not None:
                attacker_components = reward_breakdown.get(attacker_id, {})
                success_reward_val = float(attacker_components.get("success_reward", 0.0) or 0.0)
                if success_reward_val > 0.0:
                    success = True
                    assisted_success = True

            if success:
                total_successes += 1
                logger.info(
                    "Gaplock success recorded",
                    extra={
                        "episode": episode_idx + 1,
                        "steps": rollout.steps,
                        "spawn_option": rollout.spawn_option,
                        "spawn_points": rollout.spawn_points,
                    },
                )

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
                "assisted_success": assisted_success,
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

            spawn_state: Optional[Dict[str, Any]] = None
            if spawn_manager is not None:
                spawn_state = spawn_manager.observe(episode_idx + 1, success)
            defender_state: Optional[Dict[str, Any]] = None
            if defender_manager is not None:
                defender_state = defender_manager.observe(episode_idx + 1, success)

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
                episode_record["success"] = bool(success)
            if assisted_success is not None:
                metrics["train/assisted_success"] = bool(assisted_success)
            if success_rate is not None:
                metrics["train/success_rate"] = success_rate
            metrics["train/success_total"] = float(total_successes)
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
            if spawn_state is not None:
                metrics["train/random_spawn_enabled"] = bool(spawn_state.get("enabled", False))
                stage_value = spawn_state.get("stage")
                if stage_value is not None:
                    metrics["train/spawn_stage"] = stage_value
                    episode_record["spawn_stage"] = stage_value
                spawn_rate = spawn_state.get("success_rate")
                if spawn_rate is not None:
                    metrics["train/spawn_success_rate"] = float(spawn_rate)
                    episode_record["spawn_success_rate"] = float(spawn_rate)
                stage_rate = spawn_state.get("stage_success_rate")
                if stage_rate is not None:
                    metrics["train/spawn_stage_success_rate"] = float(stage_rate)
                    episode_record["spawn_stage_success_rate"] = float(stage_rate)
                structured_rate = spawn_state.get("structured_success_rate")
                if structured_rate is not None:
                    metrics["train/spawn_structured_success_rate"] = float(structured_rate)
                random_rate = spawn_state.get("random_success_rate")
                if random_rate is not None:
                    metrics["train/spawn_random_success_rate"] = float(random_rate)
                episode_record["random_spawn_enabled"] = bool(spawn_state.get("enabled", False))
            if defender_state is not None:
                stage_name = defender_state.get("stage")
                if stage_name is not None:
                    metrics["train/defender_stage"] = stage_name
                    episode_record["defender_stage"] = stage_name
                stage_index = defender_state.get("stage_index")
                if stage_index is not None:
                    metrics["train/defender_stage_index"] = float(stage_index)
                    episode_record["defender_stage_index"] = int(stage_index)
                defender_success = defender_state.get("success_rate")
                if defender_success is not None:
                    metrics["train/defender_success_rate"] = float(defender_success)
                    episode_record["defender_success_rate"] = float(defender_success)
                defender_stage_success = defender_state.get("stage_success_rate")
                if defender_stage_success is not None:
                    metrics["train/defender_stage_success_rate"] = float(defender_stage_success)
                    episode_record["defender_stage_success_rate"] = float(defender_stage_success)
                params_snapshot = defender_state.get("params") or {}
                if params_snapshot:
                    episode_record["defender_params"] = dict(params_snapshot)
                    for key, value in params_snapshot.items():
                        if isinstance(value, (int, float)):
                            metrics[f"train/defender_param/{key}"] = float(value)
            episode_record["success_total"] = total_successes

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
            publish_metrics = getattr(env, "update_render_metrics", None)
            if callable(publish_metrics):
                try:
                    publish_metrics("train", metrics, step=episode_idx + 1)
                except Exception:
                    pass

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

            for trainer in trainer_map.values():
                reset_noise = getattr(trainer, "reset_noise_schedule", None)
                if callable(reset_noise):
                    reset_noise()

            if self._eval_interval and (episode_idx + 1) % self._eval_interval == 0:
                self._run_periodic_evaluation(
                    episode_idx + 1,
                    trainer_map,
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
        checkpoint_name = self._apply_run_suffix(checkpoint_name)
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
        current_name = self.best_model_path.name
        suffixed = self._apply_run_suffix(current_name)
        if suffixed != current_name:
            self.best_model_path = self.checkpoint_dir / suffixed
            bundle_cfg = dict(self._primary_bundle.metadata.get("config", {}))
            bundle_cfg["checkpoint_name"] = suffixed
            self._primary_bundle.metadata["config"] = bundle_cfg
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        save_fn(str(self.best_model_path))
        return True

    def _run_periodic_evaluation(
        self,
        train_episode: int,
        trainer_map: Dict[str, Trainer],
    ) -> None:
        if self._eval_episodes <= 0:
            return
        self.context.trainer_map = dict(trainer_map)
        if self._eval_runner is None:
            try:
                self._eval_runner = EvalRunner(self.context)
            except Exception as exc:
                self._logger.warning(
                    "Failed to initialise periodic evaluator",
                    extra={
                        "episode": train_episode,
                        "error": str(exc),
                    },
                )
                self._eval_runner = None
                return
        else:
            self._eval_runner.trainer_map = dict(trainer_map)
            self._eval_runner.context.trainer_map = dict(trainer_map)

        self._logger.info(
            "Starting periodic evaluation",
            extra={
                "train_episode": train_episode,
                "eval_interval": self._eval_interval,
                "eval_episodes": self._eval_episodes,
            },
        )
        try:
            self.team.reset_actions()
            self._eval_runner.run(
                episodes=self._eval_episodes,
                auto_load=False,
                force_render=False,
            )
        except Exception as exc:
            self._logger.warning(
                "Periodic evaluation failed",
                extra={
                    "train_episode": train_episode,
                    "error": str(exc),
                },
            )
        finally:
            self.team.reset_actions()
            self._logger.update_context(mode="train")

    @staticmethod
    def _slugify_suffix(value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in str(value))
        cleaned = cleaned.strip("-")
        return cleaned or None

    def _resolve_run_suffix(self) -> Optional[str]:
        candidates = [
            os.environ.get("F110_RUN_SUFFIX"),
            os.environ.get("WANDB_RUN_ID"),
            os.environ.get("WANDB_RUN_NAME"),
            os.environ.get("WANDB_RUN_PATH"),
            os.environ.get("RUN_ITER"),
            os.environ.get("RUN_SEED"),
        ]
        meta = getattr(self.context, "metadata", {})
        if isinstance(meta, dict):
            candidates.extend([
                meta.get("wandb_run_id"),
                meta.get("wandb_run_name"),
                meta.get("run_suffix"),
            ])
        for candidate in candidates:
            slug = self._slugify_suffix(candidate)
            if slug:
                return slug
        return None

    def _apply_run_suffix(self, checkpoint_name: str) -> str:
        suffix = self._resolve_run_suffix()
        if not suffix:
            return checkpoint_name
        base = Path(checkpoint_name)
        stem = base.stem
        if stem.endswith(f"_{suffix}"):
            return base.name
        return f"{stem}_{suffix}{base.suffix}"

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
