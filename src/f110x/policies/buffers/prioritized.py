"""Prioritized replay buffer implementation shared across agents."""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import numpy as np

from .replay import ReplayBuffer


class PrioritizedReplayBuffer(ReplayBuffer):
    """Proportional prioritized replay buffer with importance sampling weights."""

    def __init__(
        self,
        capacity: int,
        obs_shape: Iterable[int],
        action_shape: Iterable[int],
        *,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment_per_sample: float = 1e-4,
        beta_final: float = 1.0,
        min_priority: float = 1e-3,
        epsilon: float = 1e-6,
        dtype: np.dtype = np.float32,
        store_actions: bool = True,
        store_action_indices: bool = False,
    ) -> None:
        super().__init__(
            capacity,
            obs_shape,
            action_shape,
            dtype=dtype,
            store_actions=store_actions,
            store_action_indices=store_action_indices,
        )
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if not 0.0 <= beta <= 1.0:
            raise ValueError("beta must be in [0, 1]")
        if not 0.0 <= beta_final <= 1.0:
            raise ValueError("beta_final must be in [0, 1]")
        if beta_final < beta:
            raise ValueError("beta_final must be greater than or equal to beta")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.beta_increment_per_sample = float(max(beta_increment_per_sample, 0.0))
        self.beta_target = float(beta_final)
        self.min_priority = float(max(min_priority, 1e-12))
        self.epsilon = float(max(epsilon, 0.0))

        self._priorities = np.zeros((self.capacity,), dtype=np.float32)
        self._max_priority = 1.0

    def add(
        self,
        obs: np.ndarray,
        action: Optional[np.ndarray],
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        info: Optional[Dict[str, Any]] = None,
        *,
        action_index: Optional[int] = None,
    ) -> None:
        idx = self._idx
        super().add(
            obs,
            action,
            reward,
            next_obs,
            done,
            info,
            action_index=action_index,
        )
        priority = max(self._max_priority, self.min_priority)
        self._priorities[idx] = float(priority)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self._size < batch_size:
            raise ValueError("Not enough samples in replay buffer")

        current_size = self._size
        priorities = self._priorities[:current_size]
        if priorities.size == 0:
            raise ValueError("No samples available in prioritized buffer")

        scaled = np.power(np.maximum(priorities, self.min_priority), self.alpha)
        total = float(np.sum(scaled))
        if total <= 0.0:
            scaled = np.ones_like(scaled, dtype=np.float32)
            total = float(np.sum(scaled))
        probs = scaled / total

        indices = np.random.choice(current_size, batch_size, p=probs)
        batch = {
            "obs": self._observations[indices],
            "rewards": self._rewards[indices],
            "next_obs": self._next_observations[indices],
            "dones": self._dones[indices],
        }
        if self.store_actions and self._actions is not None:
            batch["actions"] = self._actions[indices]
        if self.store_action_indices and self._action_indices is not None:
            batch["action_indices"] = self._action_indices[indices]
        infos = [self._infos[i] for i in indices]
        if any(info is not None for info in infos):
            batch["infos"] = infos

        sample_probs = probs[indices]
        weights = np.power(current_size * sample_probs, -self.beta)
        weights /= weights.max() if weights.max() > 0 else 1.0
        batch["weights"] = weights.astype(np.float32).reshape(-1, 1)
        batch["indices"] = indices.astype(np.int64)

        self.beta = min(self.beta_target, self.beta + self.beta_increment_per_sample)
        return batch

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        idxs = np.asarray(indices, dtype=np.int64).reshape(-1)
        if idxs.size == 0:
            return
        abs_errors = np.abs(np.asarray(td_errors, dtype=np.float32)) + self.epsilon
        updated = np.maximum(abs_errors, self.min_priority)
        self._priorities[idxs] = updated
        self._max_priority = max(self._max_priority, float(updated.max()))


__all__ = ["PrioritizedReplayBuffer"]
