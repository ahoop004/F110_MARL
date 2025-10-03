"""Simple replay buffer implementation reusable by TD3/DQN agents."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import numpy as np


@dataclass
class Transition:
    obs: np.ndarray
    action: np.ndarray
    reward: float
    next_obs: np.ndarray
    done: bool
    info: Optional[Dict[str, Any]] = None


class ReplayBuffer:
    """Episode-agnostic replay buffer with numpy-backed storage."""

    def __init__(
        self,
        capacity: int,
        obs_shape: Iterable[int],
        action_shape: Iterable[int],
        dtype: np.dtype = np.float32,
    ) -> None:
        if capacity <= 0:
            raise ValueError("ReplayBuffer capacity must be positive")

        self.capacity = int(capacity)
        self.obs_shape = tuple(obs_shape)
        self.action_shape = tuple(action_shape)
        self.dtype = np.dtype(dtype)

        self._observations = np.zeros((capacity, *self.obs_shape), dtype=self.dtype)
        self._next_observations = np.zeros((capacity, *self.obs_shape), dtype=self.dtype)
        self._actions = np.zeros((capacity, *self.action_shape), dtype=self.dtype)
        self._rewards = np.zeros((capacity, 1), dtype=np.float32)
        self._dones = np.zeros((capacity, 1), dtype=np.float32)

        self._infos: list[Optional[Dict[str, Any]]] = [None] * capacity

        self._idx = 0
        self._size = 0

    def __len__(self) -> int:  # pragma: no cover - trivial access
        return self._size

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        idx = self._idx
        self._observations[idx] = np.asarray(obs, dtype=self.dtype)
        self._actions[idx] = np.asarray(action, dtype=self.dtype)
        self._rewards[idx] = float(reward)
        self._next_observations[idx] = np.asarray(next_obs, dtype=self.dtype)
        self._dones[idx] = float(done)
        self._infos[idx] = info

        self._idx = (self._idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self._size < batch_size:
            raise ValueError("Not enough samples in replay buffer")

        indices = np.random.randint(0, self._size, size=batch_size)
        batch = {
            "obs": self._observations[indices],
            "actions": self._actions[indices],
            "rewards": self._rewards[indices],
            "next_obs": self._next_observations[indices],
            "dones": self._dones[indices],
        }
        infos = [self._infos[i] for i in indices]
        if any(info is not None for info in infos):
            batch["infos"] = infos
        return batch

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
        min_priority: float = 1e-3,
        epsilon: float = 1e-6,
        dtype: np.dtype = np.float32,
    ) -> None:
        super().__init__(capacity, obs_shape, action_shape, dtype=dtype)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if not 0.0 <= beta <= 1.0:
            raise ValueError("beta must be in [0, 1]")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.beta_increment_per_sample = float(max(beta_increment_per_sample, 0.0))
        self.min_priority = float(max(min_priority, 1e-12))
        self.epsilon = float(max(epsilon, 0.0))
        self._priorities = np.zeros((self.capacity,), dtype=np.float32)
        self._max_priority = 1.0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        idx = self._idx
        super().add(obs, action, reward, next_obs, done, info)
        priority = max(self._max_priority, self.min_priority)
        self._priorities[idx] = priority

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
        # resample the batch arrays using prioritized indices so we align data and weights
        batch = {
            "obs": self._observations[indices],
            "actions": self._actions[indices],
            "rewards": self._rewards[indices],
            "next_obs": self._next_observations[indices],
            "dones": self._dones[indices],
        }
        infos = [self._infos[i] for i in indices]
        if any(info is not None for info in infos):
            batch["infos"] = infos

        sample_probs = probs[indices]
        weights = np.power(current_size * sample_probs, -self.beta)
        weights /= weights.max() if weights.max() > 0 else 1.0
        batch["weights"] = weights.astype(np.float32).reshape(-1, 1)
        batch["indices"] = indices.astype(np.int64)

        self.beta = min(1.0, self.beta + self.beta_increment_per_sample)
        return batch

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        idxs = np.asarray(indices, dtype=np.int64)
        if idxs.size == 0:
            return
        abs_errors = np.abs(td_errors).astype(np.float32) + self.epsilon
        updated = np.maximum(abs_errors, self.min_priority)
        self._priorities[idxs] = updated
        self._max_priority = max(self._max_priority, float(updated.max()))
