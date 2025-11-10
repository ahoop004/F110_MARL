"""Evaluation runner built around the shared engine rollout helpers."""
from __future__ import annotations

import pickle
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import numpy as np
import os

from f110x.engine.rollout import IdleTerminationTracker, collect_trajectory, run_episode
from f110x.runner.context import RunnerContext
from f110x.runner.rollout_helpers import build_rollout_hooks
from f110x.trainer.base import Trainer
from f110x.utils.builders import AgentBundle, AgentTeam
from f110x.utils.logger import Logger
from f110x.utils.output import resolve_output_dir, resolve_output_file


@dataclass
class EvalRunner:
    """Execute deterministic evaluation episodes for a prepared runner context."""

    context: RunnerContext
    trainer_map: Dict[str, Trainer] = field(init=False)
    default_checkpoint_path: Optional[Path] = field(init=False, default=None)
    explicit_checkpoint_path: Optional[Path] = field(init=False, default=None)
    rollout_dir: Optional[Path] = field(init=False, default=None)
    save_rollouts_default: bool = field(init=False, default=False)
    _primary_bundle: Optional[AgentBundle] = field(init=False, default=None)
    _logger: Logger = field(init=False)

    def __post_init__(self) -> None:  # noqa: D401 - behaviour captured in class docstring
        self._ensure_primary_agent()
        self.trainer_map = dict(self.context.trainer_map)
        self._primary_bundle = self._resolve_primary_bundle()
        self._configure_paths()
        self._logger = self.context.logger

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
    def primary_agent_id(self) -> Optional[str]:
        if self.context.primary_agent_id:
            return self.context.primary_agent_id
        return self._primary_bundle.agent_id if self._primary_bundle else None

    @property
    def primary_bundle(self) -> Optional[AgentBundle]:
        return self._primary_bundle

    @property
    def trainable_agent_ids(self) -> List[str]:
        return list(self.context.trainable_agent_ids)

    @property
    def opponent_agent_ids(self) -> List[str]:
        primary_id = self.primary_agent_id
        if primary_id is None:
            return [bundle.agent_id for bundle in self.team.agents]
        return [bundle.agent_id for bundle in self.team.agents if bundle.agent_id != primary_id]

    @property
    def roster_metadata(self) -> Dict[str, Any]:
        return {
            "agent_ids": [bundle.agent_id for bundle in self.team.agents],
            "roles": dict(self.team.roles),
            "trainable": list(self.context.trainable_agent_ids),
        }

    def load_checkpoint(self, checkpoint_path: Optional[Path | str] = None) -> Optional[Path]:
        bundle = self.primary_bundle
        if bundle is None:
            return None
        candidate = self._resolve_checkpoint_path(checkpoint_path)
        if candidate is None:
            return None
        logger = self._logger
        if not candidate.exists():
            logger.warning(
                "Checkpoint unavailable",
                extra={"path": str(candidate)},
            )
            return None
        controller = bundle.controller
        load_fn = getattr(controller, "load", None)
        if not callable(load_fn):
            logger.warning(
                "Controller does not implement load()",
                extra={
                    "agent_id": bundle.agent_id,
                    "path": str(candidate),
                },
            )
            return None
        load_fn(str(candidate))
        logger.info(
            "Loaded evaluation checkpoint",
            extra={
                "agent_id": bundle.agent_id,
                "path": str(candidate),
            },
        )
        return candidate

    def run(
        self,
        *,
        episodes: int,
        auto_load: bool = False,
        checkpoint_path: Optional[Path | str] = None,
        force_render: bool = False,
        save_rollouts: Optional[bool] = None,
        rollout_dir: Optional[Path | str] = None,
    ) -> List[Dict[str, Any]]:
        if auto_load:
            self.load_checkpoint(checkpoint_path)

        env = self.env
        team = self.team
        trainer_map = self.trainer_map
        results: List[Dict[str, Any]] = []

        primary_id = self.primary_agent_id
        attacker_id = team.primary_role("attacker")
        defender_id = team.primary_role("defender")
        agent_ids = list(env.possible_agents)

        logger = self._logger
        total_episodes = int(episodes)
        logger.start(
            {
                "mode": "eval",
                "primary_agent": primary_id,
                "eval/episodes_total": total_episodes,
            }
        )
        logger.update_context(
            mode="eval",
            primary_agent=primary_id,
            **{"eval/episodes_total": total_episodes},
        )
        eval_window = max(1, min(total_episodes, 10))
        recent_returns: Deque[float] = deque(maxlen=eval_window)
        recent_success: Deque[float] = deque(maxlen=eval_window)

        render_enabled = force_render or str(self.context.cfg.env.get("render_mode", "")).lower() == "human"
        reward_cfg = self.context.reward_cfg
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

        idle_speed_threshold = _extract_reward_param("idle_speed_threshold", 0.0)
        idle_patience_steps = int(round(_extract_reward_param("idle_patience_steps", 0.0)))
        idle_tracker = IdleTerminationTracker(
            idle_speed_threshold,
            idle_patience_steps,
            agent_ids=self.trainable_agent_ids,
        )
        reward_sharing_cfg = self.context.reward_cfg.get("shared_reward")

        hooks = build_rollout_hooks(
            self.context,
            team,
            env,
            deterministic=True,
        )
        reward_factory = hooks.reward_factory
        compute_actions = hooks.compute_actions
        prepare_next = hooks.prepare_next_observation
        reset_env = hooks.reset_fn

        save_rollouts_flag = self.save_rollouts_default if save_rollouts is None else bool(save_rollouts)
        rollout_dir_path = self._ensure_rollout_dir(rollout_dir) if rollout_dir else self.rollout_dir
        if save_rollouts_flag and rollout_dir_path is None:
            rollout_dir_path = resolve_output_dir("eval_rollouts", self.context.output_root)
        if rollout_dir_path is not None and save_rollouts_flag:
            rollout_dir_path.mkdir(parents=True, exist_ok=True)

        for ep_index in range(int(episodes)):
            trace_buffer: Optional[List[Any]] = [] if save_rollouts_flag and rollout_dir_path else None

            if render_enabled:
                def render_condition(_episode: int, _step: int) -> bool:
                    return True
            else:
                render_condition = None

            rollout = run_episode(
                env=env,
                team=team,
                trainer_map=trainer_map,
                trajectory_buffers={},
                reward_wrapper_factory=reward_factory,
                compute_actions=compute_actions,
                prepare_next_observation=prepare_next,
                idle_tracker=idle_tracker,
                episode_index=ep_index,
                reset_fn=reset_env,
                agent_ids=agent_ids,
                render_condition=render_condition,
                trace_buffer=trace_buffer,
                reward_sharing=reward_sharing_cfg,
            )

            returns = dict(rollout.returns)
            reward_breakdown = {
                aid: dict(components) for aid, components in rollout.reward_breakdown.items()
            }
            finish_line_hits = dict(rollout.finish_line_hits or {})
            collision_total = int(sum(rollout.collisions.values()))
            collision_steps = {
                aid: (step if step >= 0 else None)
                for aid, step in rollout.collision_steps.items()
            }

            lap_counts = self._extract_lap_counts(env, agent_ids)

            defender_crashed: Optional[bool] = None
            defender_crash_step: Optional[int] = None
            if defender_id is not None:
                defender_step = collision_steps.get(defender_id)
                defender_crashed = defender_step is not None
                if defender_step is not None:
                    defender_crash_step = int(defender_step)

            attacker_crashed: Optional[bool] = None
            attacker_crash_step: Optional[int] = None
            if attacker_id is not None:
                attacker_step = collision_steps.get(attacker_id)
                attacker_crashed = attacker_step is not None
                if attacker_step is not None:
                    attacker_crash_step = int(attacker_step)

            success: Optional[bool] = None
            if defender_crashed is not None and attacker_crashed is not None:
                success = defender_crashed and not attacker_crashed
            elif defender_crashed is not None:
                success = defender_crashed
            elif attacker_crashed is not None:
                success = not attacker_crashed

            assisted_success: Optional[bool] = None
            record_finish_agent: Optional[str] = None
            if success and attacker_id is not None:
                attacker_components = reward_breakdown.get(attacker_id, {})
                success_reward_val = float(attacker_components.get("success_reward", 0.0) or 0.0)
                assisted_success = success_reward_val > 0.0

            if not success and attacker_id is not None:
                attacker_components = reward_breakdown.get(attacker_id, {})
                success_reward_val = float(attacker_components.get("success_reward", 0.0) or 0.0)
                if success_reward_val > 0.0:
                    success = True
                    assisted_success = True
            if success is None:
                finish_agent = defender_id or primary_id or (agent_ids[0] if agent_ids else None)
                if finish_agent and finish_line_hits.get(finish_agent):
                    success = True
                    assisted_success = False
                    record_finish_agent = finish_agent

            record: Dict[str, Any] = {
                "episode": ep_index + 1,
                "steps": rollout.steps,
                "cause": rollout.cause,
                "returns": returns,
                "collision_total": collision_total,
            }
            if success is not None:
                record["success"] = success
            record["assisted_success"] = assisted_success
            record["reward_breakdown"] = reward_breakdown
            record["finish_line_hits"] = finish_line_hits
            if record_finish_agent:
                record["finish_line_agent"] = record_finish_agent
            if rollout.spawn_points:
                record["spawn_points"] = dict(rollout.spawn_points)
            if rollout.spawn_option is not None:
                record["spawn_option"] = rollout.spawn_option
            for aid, value in returns.items():
                record[f"return_{aid}"] = value
            for aid, count in rollout.collisions.items():
                record[f"collision_count_{aid}"] = count
            for aid, count in lap_counts.items():
                record[f"lap_count_{aid}"] = count
            for aid, step_val in collision_steps.items():
                if step_val is not None:
                    record[f"collision_step_{aid}"] = int(step_val)
            for aid in agent_ids:
                record[f"avg_speed_{aid}"] = float(rollout.average_speeds.get(aid, 0.0))

            if attacker_id and attacker_id in rollout.average_speeds:
                record["avg_speed_attacker"] = float(rollout.average_speeds.get(attacker_id, 0.0))
            if defender_id and defender_id in rollout.average_speeds:
                record["avg_speed_defender"] = float(rollout.average_speeds.get(defender_id, 0.0))
            if defender_crashed is not None:
                record["defender_crashed"] = defender_crashed
            if defender_crash_step is not None:
                record["defender_crash_step"] = defender_crash_step
                record["defender_survival_steps"] = defender_crash_step
            elif defender_id is not None:
                record["defender_survival_steps"] = rollout.steps
            if attacker_crashed is not None:
                record["attacker_crashed"] = attacker_crashed
            if attacker_crash_step is not None:
                record["attacker_crash_step"] = attacker_crash_step

            defender_survival_steps_value = record.get("defender_survival_steps")

            if trace_buffer is not None and rollout_dir_path is not None:
                transformed = collect_trajectory(trace_buffer, transform=self._trace_transform)
                rollout_path = rollout_dir_path / f"episode_{ep_index + 1:03d}.pkl"
                with rollout_path.open("wb") as handle:
                    pickle.dump({"trajectory": transformed, "metrics": record}, handle)
                record["rollout_path"] = str(rollout_path)

            primary_return = float(returns.get(primary_id, 0.0))
            recent_returns.append(primary_return)
            mean_return = float(sum(recent_returns) / len(recent_returns))

            if success is not None:
                recent_success.append(1.0 if success else 0.0)
            success_rate = (
                float(sum(recent_success) / len(recent_success)) if recent_success else None
            )

            collision_rate = float(collision_total) / max(float(rollout.steps), 1.0)

            metrics: Dict[str, Any] = {
                "eval/episode": float(ep_index + 1),
                "eval/episodes_total": float(total_episodes),
                "eval/steps": float(rollout.steps),
                "eval/return": primary_return,
                "eval/return_mean": mean_return,
                "eval/collisions": float(collision_total),
                "eval/collision_rate": collision_rate,
                "eval/cause": rollout.cause,
            }
            if primary_id:
                metrics["eval/primary_agent"] = primary_id
            if success is not None:
                metrics["eval/success"] = bool(success)
            if assisted_success is not None:
                metrics["eval/assisted_success"] = bool(assisted_success)
            if success_rate is not None:
                metrics["eval/success_rate"] = success_rate
            if defender_crashed is not None:
                metrics["eval/defender_crashed"] = bool(defender_crashed)
            if attacker_crashed is not None:
                metrics["eval/attacker_crashed"] = bool(attacker_crashed)
            if defender_survival_steps_value is not None:
                metrics["eval/defender_survival_steps"] = float(defender_survival_steps_value)
            metrics["eval/return_window"] = float(recent_returns.maxlen or len(recent_returns))

            for aid, value in returns.items():
                metrics[f"eval/agent/{aid}/return"] = float(value)
            for aid, count in rollout.collisions.items():
                metrics[f"eval/agent/{aid}/collisions"] = float(count)
            for aid, speed in rollout.average_speeds.items():
                metrics[f"eval/agent/{aid}/avg_speed"] = float(speed)
            for aid, step_val in collision_steps.items():
                if step_val is not None:
                    metrics[f"eval/agent/{aid}/collision_step"] = float(step_val)
            for aid, count in lap_counts.items():
                metrics[f"eval/agent/{aid}/lap_count"] = float(count)
            for aid, breakdown in reward_breakdown.items():
                for name, value in breakdown.items():
                    metrics[f"eval/reward/{aid}/{name}"] = float(value)
            if finish_line_hits:
                metrics["eval/finish_line_any"] = float(any(finish_line_hits.values()))
                for aid, hit in finish_line_hits.items():
                    metrics[f"eval/finish_line_hit/{aid}"] = 1.0 if hit else 0.0

            logger.log_metrics("eval", metrics, step=ep_index + 1)
            publish_metrics = getattr(env, "update_render_metrics", None)
            if callable(publish_metrics):
                try:
                    publish_metrics("eval", metrics, step=ep_index + 1)
                except Exception:
                    pass

            results.append(record)

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_primary_agent(self) -> None:
        if self.context.primary_agent_id:
            return
        candidates = list(self.context.trainable_agent_ids)
        if not candidates:
            candidates = [bundle.agent_id for bundle in self.context.team.agents]
        if not candidates:
            raise RuntimeError("RunnerContext does not expose any agents for evaluation")
        self.context.set_primary_agent(candidates[0])

    def _resolve_primary_bundle(self) -> Optional[AgentBundle]:
        try:
            return self.context.primary_bundle
        except RuntimeError:
            return None

    def _configure_paths(self) -> None:
        output_root = self.context.output_root
        output_root.mkdir(parents=True, exist_ok=True)
        self.context.cfg.main.schema.output_root = str(output_root)

        bundle = self._primary_bundle
        bundle_cfg: Dict[str, Any] = dict(bundle.metadata.get("config", {})) if bundle else {}
        if bundle and (not bundle_cfg) and bundle.algo.lower() == "ppo":
            bundle_cfg = self.context.cfg.ppo.to_dict()

        save_dir_value = bundle_cfg.get("save_dir", "checkpoints")
        checkpoint_dir = resolve_output_dir(save_dir_value, output_root)
        bundle_cfg["save_dir"] = str(checkpoint_dir)

        checkpoint_name = bundle_cfg.get(
            "checkpoint_name",
            f"{bundle.algo.lower()}_best.pt" if bundle else "model.pt",
        )
        checkpoint_name = self._apply_run_suffix(checkpoint_name)
        self.default_checkpoint_path = checkpoint_dir / checkpoint_name if bundle else None

        explicit_raw = self.context.cfg.main.checkpoint
        if explicit_raw:
            candidate = Path(explicit_raw).expanduser()
            if not candidate.is_absolute():
                candidate = resolve_output_file(explicit_raw, output_root)
            self.explicit_checkpoint_path = candidate
            self.context.cfg.main.schema.checkpoint = str(candidate)
        else:
            self.explicit_checkpoint_path = None

        save_rollouts_flag = bool(self.context.cfg.main.get("save_eval_rollouts", False))
        self.save_rollouts_default = save_rollouts_flag
        if save_rollouts_flag:
            rollout_dir_value = self.context.cfg.main.get("eval_rollout_dir", "eval_rollouts")
            self.rollout_dir = resolve_output_dir(rollout_dir_value, output_root)
            self.context.cfg.main.schema.extras["eval_rollout_dir"] = str(self.rollout_dir)
        else:
            self.rollout_dir = None

        if bundle is not None:
            bundle_cfg["checkpoint_name"] = checkpoint_name
            bundle.metadata["config"] = bundle_cfg

    def _resolve_run_suffix(self) -> Optional[str]:
        meta = getattr(self.context, "metadata", {})
        candidates = [
            os.environ.get("F110_RUN_SUFFIX"),
        ]
        if isinstance(meta, dict):
            candidates.append(meta.get("run_suffix"))
        candidates.extend([
            os.environ.get("WANDB_NAME"),
            os.environ.get("WANDB_RUN_NAME"),
        ])
        if isinstance(meta, dict):
            candidates.append(meta.get("wandb_run_name"))
        candidates.extend([
            os.environ.get("RUN_CONFIG_HASH"),
            os.environ.get("WANDB_RUN_PATH"),
            os.environ.get("RUN_ITER"),
            os.environ.get("RUN_SEED"),
            os.environ.get("WANDB_RUN_ID"),
        ])
        if isinstance(meta, dict):
            candidates.append(meta.get("wandb_run_id"))
        for candidate in candidates:
            if not candidate:
                continue
            cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in str(candidate))
            cleaned = cleaned.strip("-")
            if cleaned:
                return cleaned
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

    def _resolve_checkpoint_path(self, override: Optional[Path | str]) -> Optional[Path]:
        if override is not None:
            candidate = Path(override).expanduser() if not isinstance(override, Path) else override.expanduser()
            if not candidate.is_absolute():
                candidate = resolve_output_file(str(candidate), self.context.output_root)
            return candidate
        if self.explicit_checkpoint_path is not None:
            return self.explicit_checkpoint_path
        return self.default_checkpoint_path

    def _ensure_rollout_dir(self, path_like: Path | str) -> Path:
        path = Path(path_like).expanduser()
        if not path.is_absolute():
            return resolve_output_dir(str(path), self.context.output_root)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _extract_lap_counts(env, agent_ids: List[str]) -> Dict[str, float]:
        counts: Dict[str, float] = {}
        lap_counts = getattr(env, "lap_counts", None)
        id_to_index = getattr(env, "_agent_id_to_index", {})
        if lap_counts is None or not isinstance(id_to_index, dict):
            return counts
        for agent_id in agent_ids:
            idx = id_to_index.get(agent_id)
            if idx is None:
                continue
            if 0 <= idx < len(lap_counts):
                counts[agent_id] = float(lap_counts[idx])
        return counts

    @staticmethod
    def _trace_transform(step) -> Dict[str, Any]:
        return {
            "step": step.step,
            "obs": {aid: EvalRunner._serialize_obs(obs) for aid, obs in step.observations.items()},
            "actions": {aid: EvalRunner._to_serializable(action) for aid, action in step.actions.items()},
            "rewards": {aid: EvalRunner._to_serializable(reward) for aid, reward in step.rewards.items()},
            "next_obs": {
                aid: EvalRunner._serialize_obs(obs) for aid, obs in step.next_observations.items()
            },
            "done": dict(step.done),
            "collisions": list(step.collisions),
        }

    @staticmethod
    def _serialize_obs(obs: Dict[str, Any]) -> Dict[str, Any]:
        serial: Dict[str, Any] = {}
        for key, value in obs.items():
            if isinstance(value, dict):
                serial[key] = {sub_k: EvalRunner._to_serializable(sub_v) for sub_k, sub_v in value.items()}
            else:
                serial[key] = EvalRunner._to_serializable(value)
        return serial

    @staticmethod
    def _to_serializable(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        return value


__all__ = ["EvalRunner"]
