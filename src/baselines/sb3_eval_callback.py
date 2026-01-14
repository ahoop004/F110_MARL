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
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.config = evaluation_config
        self.spawn_configs = spawn_configs or {}
        self.eval_every_n_episodes = max(1, int(eval_every_n_episodes)) if eval_every_n_episodes else 0
        self.wandb_run = wandb_run
        self.wandb_logging = wandb_logging if isinstance(wandb_logging, dict) else None
        self.episode_count = 0
        self.total_eval_episodes = 0

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

    def _on_step(self) -> bool:
        if self.eval_every_n_episodes <= 0:
            return True
        done_count = self._count_episode_ends()
        if done_count > 0:
            for _ in range(done_count):
                self.episode_count += 1
                if self.episode_count % self.eval_every_n_episodes == 0:
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

        rewards: List[float] = []
        lengths: List[int] = []
        success_count = 0
        outcome_counts: Dict[str, int] = {}
        spawn_stats: Dict[str, Dict[str, List]] = {}  # Track per-spawn performance

        num_episodes = int(self.config.num_episodes)
        spawn_points = list(self.config.spawn_points or [])
        spawn_speeds = list(self.config.spawn_speeds or [])

        if not spawn_points:
            if self.verbose > 0:
                print("Eval skipped: no spawn_points configured")
            return

        for ep_idx in range(num_episodes):
            spawn_point = spawn_points[ep_idx % len(spawn_points)]
            spawn_speed = (
                spawn_speeds[ep_idx % len(spawn_speeds)]
                if spawn_speeds
                else 0.44
            )

            obs, _ = self.eval_env.reset(
                options=self._build_reset_options(spawn_point, spawn_speed)
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
            outcome_counts[outcome_value] = outcome_counts.get(outcome_value, 0) + 1

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

        if self._should_log("eval_agg"):
            # Use training episode as step metric for consistency with run_v2
            training_ep = training_episode if training_episode is not None else self.episode_count
            agg_metrics = {
                "eval/episode": int(self.total_eval_episodes),
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
                        success_rate = float(np.mean(stats["successes"]))
                        num_eps = len(stats["rewards"])
                        spawn_table_data.append([spawn, success_rate, avg_reward, num_eps])
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

    def _build_reset_options(self, spawn_point: str, spawn_speed: float) -> Dict[str, Any]:
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
            "lock_speed_steps": int(self.config.lock_speed_steps),
        }


__all__ = ["SB3EvaluationCallback"]
