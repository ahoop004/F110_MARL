"""SB3 evaluation callback with W&B logging."""

from typing import Any, Dict, Optional, List

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from metrics.outcomes import determine_outcome, EpisodeOutcome

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class SB3EvaluationCallback(BaseCallback):
    """Periodic evaluation callback for SB3 with unified W&B metrics."""

    def __init__(
        self,
        eval_env: Any,
        evaluation_config: Any,
        spawn_configs: Dict[str, Any],
        eval_every_n_episodes: int,
        wandb_run: Optional[Any] = None,
        wandb_logging: Optional[Dict[str, Any]] = None,
        curriculum_callback: Optional[Any] = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.config = evaluation_config
        self.spawn_configs = spawn_configs or {}
        self.eval_every_n_episodes = max(1, int(eval_every_n_episodes)) if eval_every_n_episodes else 0
        self.wandb_run = wandb_run
        self.wandb_logging = wandb_logging if isinstance(wandb_logging, dict) else None
        self.curriculum_callback = curriculum_callback
        self.episode_count = 0
        self.total_eval_episodes = 0
        self.eval_run_count = 0

    def _should_log(self, key: str) -> bool:
        if not self.wandb_run:
            return False
        group_config = self._get_group_config()
        if group_config is None:
            return True
        if not group_config.get("sb3_callbacks", False):
            return False
        return bool(group_config.get(key, False))

    def _get_group_config(self) -> Optional[Dict[str, Any]]:
        if not isinstance(self.wandb_logging, dict):
            return None
        if "groups" in self.wandb_logging:
            groups = self.wandb_logging.get("groups")
            return groups if isinstance(groups, dict) else {}
        return self.wandb_logging

    def _get_metrics_config(self) -> Optional[Dict[str, Any]]:
        if not isinstance(self.wandb_logging, dict):
            return None
        metrics = self.wandb_logging.get("metrics")
        if metrics is None:
            return None
        if isinstance(metrics, dict):
            return metrics
        if isinstance(metrics, (list, tuple, set)):
            return {name: True for name in metrics}
        return None

    def _filter_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        metrics_config = self._get_metrics_config()
        if metrics_config is None:
            return metrics
        return {key: value for key, value in metrics.items() if metrics_config.get(key, False)}

    def _should_run_phase_eval(self) -> bool:
        if not self.curriculum_callback:
            return True
        should_run = getattr(self.curriculum_callback, "should_run_eval", None)
        if callable(should_run):
            return bool(should_run())
        return True

    def _build_speed_schedule(self, speed_range: List[float], count: int) -> List[float]:
        if count <= 0:
            return []
        try:
            min_speed = float(speed_range[0])
            max_speed = float(speed_range[1])
        except (TypeError, ValueError, IndexError):
            return []
        if count == 1 or min_speed == max_speed:
            return [min_speed] * max(1, count)
        return list(np.linspace(min_speed, max_speed, count))

    def _resolve_phase_eval_settings(self) -> Dict[str, Any]:
        phase_config = None
        phase_index = None
        if self.curriculum_callback:
            get_phase = getattr(self.curriculum_callback, "get_current_phase_config", None)
            get_index = getattr(self.curriculum_callback, "get_current_phase_index", None)
            if callable(get_phase):
                phase_config = get_phase()
            if callable(get_index):
                phase_index = get_index()

        spawn_points = list(self.config.spawn_points or [])
        spawn_speeds = list(self.config.spawn_speeds or [])
        lock_speed_steps = int(getattr(self.config, "lock_speed_steps", 0))

        if phase_config:
            spawn_cfg = phase_config.get("spawn", {}) if isinstance(phase_config, dict) else {}
            phase_points = spawn_cfg.get("points")
            if phase_points:
                if phase_points == "all" or phase_points == ["all"]:
                    phase_points = list(self.spawn_configs.keys())
                if isinstance(phase_points, (list, tuple)):
                    filtered = [p for p in phase_points if p in self.spawn_configs]
                    if filtered:
                        spawn_points = filtered
            speed_range = spawn_cfg.get("speed_range")
            if isinstance(speed_range, (list, tuple)) and len(speed_range) == 2 and spawn_points:
                spawn_speeds = self._build_speed_schedule(speed_range, len(spawn_points))
            lock_speed_steps = int(phase_config.get("lock_speed_steps", lock_speed_steps))

            # Apply phase FTG settings to eval agents (if any).
            if hasattr(self.eval_env, "other_agents") and self.eval_env.other_agents:
                try:
                    from curriculum.curriculum_env import apply_curriculum_to_agent
                except Exception:
                    apply_curriculum_to_agent = None
                if apply_curriculum_to_agent:
                    for agent_id, agent in self.eval_env.other_agents.items():
                        apply_curriculum_to_agent(agent, agent_id, phase_config)

        return {
            "phase_index": phase_index,
            "spawn_points": spawn_points,
            "spawn_speeds": spawn_speeds,
            "lock_speed_steps": lock_speed_steps,
        }

    def _on_step(self) -> bool:
        if self.eval_every_n_episodes <= 0:
            return True
        done_count = self._count_episode_ends()
        if done_count > 0:
            for _ in range(done_count):
                self.episode_count += 1
                if self.episode_count % self.eval_every_n_episodes == 0:
                    if self._should_run_phase_eval():
                        self._run_eval(training_episode=self.episode_count)
        return True

    def _count_episode_ends(self) -> int:
        dones = self.locals.get("dones")
        if dones is None:
            terminated = self.locals.get("terminateds")
            truncated = self.locals.get("truncateds")
            if terminated is None or truncated is None:
                return 0
            done_flags = np.logical_or(terminated, truncated)
        else:
            done_flags = dones
        if isinstance(done_flags, (list, tuple, np.ndarray)):
            return int(np.sum(done_flags))
        return int(bool(done_flags))

    def _run_eval(self, training_episode: Optional[int] = None) -> None:
        if not self.spawn_configs:
            if self.verbose > 0:
                print("Eval skipped: spawn_configs missing")
            return

        phase_settings = self._resolve_phase_eval_settings()
        phase_index = phase_settings.get("phase_index")

        rewards: List[float] = []
        lengths: List[int] = []
        success_count = 0
        eval_successes: List[bool] = []
        outcome_counts: Dict[str, int] = {}
        spawn_stats: Dict[str, Dict[str, List]] = {}  # Track per-spawn performance

        num_episodes = int(self.config.num_episodes)
        spawn_points = list(phase_settings.get("spawn_points") or [])
        spawn_speeds = list(phase_settings.get("spawn_speeds") or [])
        lock_speed_steps = int(phase_settings.get("lock_speed_steps", 0))

        if not spawn_points:
            if self.verbose > 0:
                print("Eval skipped: no spawn_points configured")
            return

        self.eval_run_count += 1

        for ep_idx in range(num_episodes):
            spawn_point = spawn_points[ep_idx % len(spawn_points)]
            if spawn_speeds:
                spawn_speed = spawn_speeds[ep_idx % len(spawn_speeds)]
            else:
                spawn_speed = 0.44

            obs, _ = self.eval_env.reset(
                options=self._build_reset_options(spawn_point, spawn_speed, lock_speed_steps)
            )
            done = False
            steps = 0
            ep_reward = 0.0
            outcome_value = None
            success = False
            truncated = False

            while not done and steps < self.config.max_steps:
                action, _ = self.model.predict(obs, deterministic=self.config.deterministic)
                obs, reward, terminated, trunc, info = self.eval_env.step(action)
                ep_reward += float(reward)
                steps += 1
                truncated = bool(trunc)
                done = bool(terminated or trunc)

                if done:
                    outcome_value = info.get("outcome")
                    success = bool(info.get("is_success", False))

            if not outcome_value:
                outcome_value = determine_outcome(info, truncated=truncated).value
                success = outcome_value == EpisodeOutcome.TARGET_CRASH.value

            rewards.append(ep_reward)
            lengths.append(steps)
            success_count += int(success)
            eval_successes.append(bool(success))
            outcome_counts[outcome_value] = outcome_counts.get(outcome_value, 0) + 1
            success_rate_so_far = success_count / (ep_idx + 1)

            # Track per-spawn stats for table visualization
            if spawn_point not in spawn_stats:
                spawn_stats[spawn_point] = {"rewards": [], "successes": []}
            spawn_stats[spawn_point]["rewards"].append(ep_reward)
            spawn_stats[spawn_point]["successes"].append(int(success))

            if self._should_log("eval"):
                reward_value = float(ep_reward)
                if not np.isfinite(reward_value):
                    reward_value = 0.0
                self.total_eval_episodes += 1
                # Use training episode as step metric for consistency with run_v2
                training_ep = training_episode if training_episode is not None else self.episode_count
                episode_log = self._filter_metrics({
                    "eval/episode": int(self.total_eval_episodes),
                })
                if episode_log:
                    self.wandb_run.log(
                        episode_log,
                        step=training_ep,
                    )  # FIXED: Use training episode instead of num_timesteps
                details_log = self._filter_metrics({
                    "eval/episode_reward": reward_value,
                    "eval/episode_steps": int(steps),
                    "eval/episode_success": int(success),
                    "eval/episode_success_rate": float(success_rate_so_far),
                    "eval/spawn_point": spawn_point,
                    "eval/training_episode": training_ep,
                })
                if details_log:
                    self.wandb_run.log(
                        details_log,
                        step=training_ep,
                    )  # FIXED: Use training episode instead of num_timesteps

        if not rewards:
            return

        success_rate = success_count / num_episodes if num_episodes > 0 else 0.0
        avg_reward = float(np.mean(rewards))
        std_reward = float(np.std(rewards))
        avg_steps = float(np.mean(lengths))
        std_steps = float(np.std(lengths))

        if self.curriculum_callback:
            record_eval = getattr(self.curriculum_callback, "record_eval_result", None)
            if callable(record_eval):
                record_eval(successes=eval_successes, phase_index=phase_index)

        if self._should_log("eval_agg"):
            # Use training episode as step metric for consistency with run_v2
            training_ep = training_episode if training_episode is not None else self.episode_count
            agg_metrics = {
                "eval/episode": int(self.total_eval_episodes),
                "eval/run": int(self.eval_run_count),
                "eval_agg/success_rate": success_rate,  # Match run_v2 namespace
                "eval_agg/avg_reward": avg_reward,
                "eval_agg/std_reward": std_reward,
                "eval_agg/avg_episode_length": avg_steps,
                "eval_agg/std_episode_length": std_steps,
                "eval_agg/num_episodes": num_episodes,
            }
            for outcome, count in outcome_counts.items():
                pct = (count / num_episodes) if num_episodes > 0 else 0.0
                agg_metrics[f"eval_agg/outcome_{outcome}"] = pct

            agg_metrics = self._filter_metrics(agg_metrics)
            if agg_metrics:
                self.wandb_run.log(
                    agg_metrics,
                    step=training_ep,  # FIXED: Use training episode instead of num_timesteps
                )

            # Add rich visualizations if wandb is available
            if WANDB_AVAILABLE and self.wandb_run:
                viz_metrics = {}

                # 1. Outcome distribution bar chart
                if outcome_counts and num_episodes > 0:
                    outcome_data = [[outcome, count / num_episodes] for outcome, count in outcome_counts.items()]
                    outcome_table = wandb.Table(data=outcome_data, columns=["outcome", "percentage"])
                    viz_metrics["eval_viz/outcome_distribution"] = wandb.plot.bar(
                        outcome_table, "outcome", "percentage", title="Outcome Distribution"
                    )

                # 2. Reward distribution histogram
                if rewards:
                    viz_metrics["eval_viz/reward_distribution"] = wandb.Histogram(rewards)

                # 3. Spawn point performance table
                if spawn_stats:
                    spawn_table_data = []
                    for spawn, stats in spawn_stats.items():
                        avg_reward = float(np.mean(stats["rewards"]))
                        spawn_success_rate = float(np.mean(stats["successes"]))
                        num_eps = len(stats["rewards"])
                        spawn_table_data.append([spawn, spawn_success_rate, avg_reward, num_eps])
                    spawn_table = wandb.Table(
                        data=spawn_table_data,
                        columns=["spawn_point", "success_rate", "avg_reward", "num_episodes"]
                    )
                    viz_metrics["eval_viz/spawn_performance"] = spawn_table

                # Log visualizations
                if viz_metrics:
                    self.wandb_run.log(viz_metrics, step=training_ep)

        if self.verbose > 0:
            print(
                f"[Eval] success_rate={success_rate:.2%}, "
                f"avg_reward={avg_reward:.2f}, avg_steps={avg_steps:.1f}"
            )
            if spawn_stats:
                per_spawn = []
                for spawn in sorted(spawn_stats.keys()):
                    stats = spawn_stats[spawn]
                    if not stats.get("successes"):
                        continue
                    spawn_rate = float(np.mean(stats["successes"]))
                    per_spawn.append(f"{spawn}={spawn_rate:.2%}")
                if per_spawn:
                    print(f"[Eval] per-spawn success_rate: {', '.join(per_spawn)}")

    def _build_reset_options(
        self,
        spawn_point: str,
        spawn_speed: float,
        lock_speed_steps: int
    ) -> Dict[str, Any]:
        if spawn_point not in self.spawn_configs:
            raise ValueError(
                f"Spawn point '{spawn_point}' not found in spawn_configs. "
                f"Available: {list(self.spawn_configs.keys())}"
            )

        spawn_config = self.spawn_configs[spawn_point]
        env = getattr(self.eval_env, "env", None)
        agent_order = getattr(env, "possible_agents", None)
        if not agent_order:
            agent_order = sorted(spawn_config.keys())

        poses = []
        velocities: Dict[str, float] = {}
        for agent_id in agent_order:
            pose = spawn_config.get(agent_id)
            if pose is None:
                continue
            poses.append(pose)
            velocities[agent_id] = float(spawn_speed)

        if not poses:
            raise ValueError(f"No poses found for spawn point '{spawn_point}'")

        return {
            "poses": np.array(poses, dtype=np.float32),
            "velocities": velocities,
            "lock_speed_steps": int(lock_speed_steps),
        }


__all__ = ["SB3EvaluationCallback"]
