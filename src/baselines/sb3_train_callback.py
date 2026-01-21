"""SB3 training callback for W&B logging."""

from collections import deque
from typing import Any, Dict, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class SB3TrainLoggingCallback(BaseCallback):
    """Log per-episode training metrics to W&B during SB3 runs."""

    def __init__(
        self,
        wandb_run: Optional[Any] = None,
        wandb_logging: Optional[Dict[str, Any]] = None,
        window_size: int = 100,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.wandb_run = wandb_run
        self.wandb_logging = wandb_logging if isinstance(wandb_logging, dict) else None
        self.window_size = max(1, int(window_size))
        self.episode_count = 0
        self.episode_rewards = deque(maxlen=self.window_size)
        self.episode_successes = deque(maxlen=self.window_size)
        self.episode_lengths = deque(maxlen=self.window_size)
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        self.processed_episodes = set()

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
        if "rewards" in self.locals:
            rewards = self.locals["rewards"]
            if isinstance(rewards, (list, np.ndarray)):
                step_reward = float(rewards[0])
            else:
                step_reward = float(rewards)
            self.current_episode_reward += step_reward
            self.current_episode_length += 1

        done = None
        if "dones" in self.locals:
            dones = self.locals["dones"]
            if isinstance(dones, (list, np.ndarray)):
                done = bool(dones[0])
            else:
                done = bool(dones)
        else:
            terminated = self.locals.get("terminateds")
            truncated = self.locals.get("truncateds")
            if terminated is not None or truncated is not None:
                if isinstance(terminated, (list, np.ndarray)):
                    terminated = bool(terminated[0])
                if isinstance(truncated, (list, np.ndarray)):
                    truncated = bool(truncated[0])
                done = bool(terminated or truncated)

        processed_direct = False
        if done:
            info = {}
            if "infos" in self.locals:
                infos = self.locals["infos"]
                if isinstance(infos, list) and infos:
                    info = infos[0]
                elif isinstance(infos, dict):
                    info = infos

            self._process_episode(
                reward=self.current_episode_reward,
                length=self.current_episode_length,
                success=bool(info.get("is_success", False)),
                outcome=info.get("outcome"),
            )
            processed_direct = True

            self.current_episode_reward = 0.0
            self.current_episode_length = 0

        if not processed_direct and len(self.model.ep_info_buffer) > 0:
            for ep_info in self.model.ep_info_buffer:
                if "r" in ep_info and "l" in ep_info:
                    ep_id = (ep_info["r"], ep_info["l"], self.num_timesteps)
                    if ep_id not in self.processed_episodes:
                        self.processed_episodes.add(ep_id)
                        if self.current_episode_reward == 0.0 and self.current_episode_length == 0:
                            self._process_episode(
                                reward=ep_info["r"],
                                length=ep_info["l"],
                                success=bool(ep_info.get("is_success", False)),
                                outcome=ep_info.get("outcome"),
                            )
            self.model.ep_info_buffer.clear()

        if len(self.processed_episodes) > 10000:
            self.processed_episodes.clear()

        return True

    def _process_episode(
        self,
        reward: float,
        length: int,
        success: bool,
        outcome: Optional[str] = None,
    ) -> None:
        self.episode_count += 1
        self.episode_rewards.append(float(reward))
        self.episode_successes.append(bool(success))
        self.episode_lengths.append(int(length))

        if not self._should_log("train"):
            return

        success_rate = float(sum(self.episode_successes) / len(self.episode_successes)) if self.episode_successes else 0.0
        reward_mean = float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0
        steps_mean = float(np.mean(self.episode_lengths)) if self.episode_lengths else 0.0

        log_dict = {
            "train/episode": int(self.episode_count),
            "train/episode_reward": float(reward),
            "train/episode_steps": int(length),
            "train/success": int(success),
            "train/success_rate": success_rate,
            "train/reward_mean": reward_mean,
            "train/steps_mean": steps_mean,
        }
        if outcome is not None:
            log_dict["train/outcome"] = outcome

        log_dict = self._filter_metrics(log_dict)
        if not log_dict:
            return

        self.wandb_run.log(log_dict, step=self.episode_count)


__all__ = ["SB3TrainLoggingCallback"]
