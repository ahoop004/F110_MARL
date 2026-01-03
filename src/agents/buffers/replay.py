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
        *,
        store_actions: bool = True,
        store_action_indices: bool = False,
    ) -> None:
        if capacity <= 0:
            raise ValueError("ReplayBuffer capacity must be positive")

        self.capacity = int(capacity)
        self.obs_shape = tuple(obs_shape)
        self.action_shape = tuple(action_shape)
        self.dtype = np.dtype(dtype)
        self.store_actions = bool(store_actions)
        self.store_action_indices = bool(store_action_indices)

        self._observations = np.zeros((capacity, *self.obs_shape), dtype=self.dtype)
        self._next_observations = np.zeros((capacity, *self.obs_shape), dtype=self.dtype)
        self._rewards = np.zeros((capacity, 1), dtype=np.float32)
        self._dones = np.zeros((capacity, 1), dtype=np.float32)

        if self.store_actions:
            self._actions = np.zeros((capacity, *self.action_shape), dtype=self.dtype)
        else:
            self._actions = None

        if self.store_action_indices:
            self._action_indices = np.full((capacity,), -1, dtype=np.int64)
        else:
            self._action_indices = None

        self._infos: list[Optional[Dict[str, Any]]] = [None] * capacity

        self._idx = 0
        self._size = 0

    def __len__(self) -> int:  # pragma: no cover - trivial access
        return self._size

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
        self._observations[idx] = np.asarray(obs, dtype=self.dtype)
        self._rewards[idx] = float(reward)
        self._next_observations[idx] = np.asarray(next_obs, dtype=self.dtype)
        self._dones[idx] = float(done)
        self._infos[idx] = info

        if self.store_actions:
            if action is None:
                raise ValueError("ReplayBuffer configured to store actions but received None")
            self._actions[idx] = np.asarray(action, dtype=self.dtype)

        if self.store_action_indices and self._action_indices is not None:
            if action_index is None:
                self._action_indices[idx] = -1
            else:
                self._action_indices[idx] = int(action_index)

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
        return batch

    def sample_sequences(self, batch_size: int, sequence_length: int = 5) -> Dict[str, np.ndarray]:
        """Sample sequences of consecutive transitions.

        Args:
            batch_size: Number of sequences to sample
            sequence_length: Length of each sequence (default: 5)

        Returns:
            Dictionary with arrays of shape (batch_size, sequence_length, ...)

        Raises:
            ValueError: If not enough samples for sequences
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if self._size < sequence_length:
            raise ValueError(f"Not enough samples for sequences ({self._size} < {sequence_length})")

        # Sample start indices ensuring we have enough consecutive samples
        # Avoid wrapping around the buffer by limiting max start index
        max_start_idx = self._size - sequence_length
        if max_start_idx < 0:
            raise ValueError("Not enough consecutive samples in buffer")

        start_indices = np.random.randint(0, max_start_idx + 1, size=batch_size)

        # Build sequences by taking consecutive indices
        # Shape: (batch_size, sequence_length)
        sequence_indices = start_indices[:, None] + np.arange(sequence_length)[None, :]

        # Sample sequences
        batch = {
            "obs": self._observations[sequence_indices],  # (batch, seq_len, obs_dim)
            "rewards": self._rewards[sequence_indices],    # (batch, seq_len, 1)
            "next_obs": self._next_observations[sequence_indices],  # (batch, seq_len, obs_dim)
            "dones": self._dones[sequence_indices],        # (batch, seq_len, 1)
        }

        if self.store_actions and self._actions is not None:
            batch["actions"] = self._actions[sequence_indices]  # (batch, seq_len, action_dim)

        return batch
