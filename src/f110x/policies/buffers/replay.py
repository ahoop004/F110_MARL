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

    def last(self) -> Transition:
        if self._size == 0:
            raise IndexError("ReplayBuffer is empty")
        idx = (self._idx - 1) % self._size
        return Transition(
            obs=self._observations[idx].copy(),
            action=self._actions[idx].copy(),
            reward=float(self._rewards[idx]),
            next_obs=self._next_observations[idx].copy(),
            done=bool(self._dones[idx]),
            info=self._infos[idx],
        )
