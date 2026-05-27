"""Pure PyTorch ring-buffer replay buffer for off-policy algorithms.

Storage strategy
----------------
All transitions are stored on **CPU** regardless of the training device.
This avoids tying up GPU/MPS memory for large buffers (e.g. 1 M transitions
× 121 obs floats × 4 bytes = ~460 MB for a single obs tensor).

At sample time, the sampled batch is moved to the caller's training device
(``self.device``) in one contiguous transfer.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch


class ReplayBuffer:
    """Fixed-capacity ring buffer storing (obs, action, reward, next_obs, done).

    Parameters
    ----------
    capacity:
        Maximum number of transitions stored.
    obs_dim:
        Dimension of one observation vector.
    action_dim:
        Dimension of one action vector.
    device:
        **Training device** — sampled batches are moved here.
        Storage is always on CPU.
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
    ) -> None:
        self.capacity = int(capacity)
        self.device = device  # destination for sampled batches
        self._cpu = torch.device("cpu")

        # All storage lives on CPU — avoids pinning GPU memory for large buffers.
        self._obs      = torch.zeros(self.capacity, obs_dim,    dtype=torch.float32)
        self._actions  = torch.zeros(self.capacity, action_dim, dtype=torch.float32)
        self._rewards  = torch.zeros(self.capacity,             dtype=torch.float32)
        self._next_obs = torch.zeros(self.capacity, obs_dim,    dtype=torch.float32)
        self._dones    = torch.zeros(self.capacity,             dtype=torch.float32)
        self._ptr  = 0
        self._size = 0

    # ------------------------------------------------------------------
    # Write path — always stays on CPU
    # ------------------------------------------------------------------

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        i = self._ptr
        self._obs[i]      = torch.as_tensor(obs,      dtype=torch.float32)
        self._actions[i]  = torch.as_tensor(action,   dtype=torch.float32)
        self._rewards[i]  = float(reward)
        self._next_obs[i] = torch.as_tensor(next_obs, dtype=torch.float32)
        self._dones[i]    = float(done)
        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    # ------------------------------------------------------------------
    # Read path — sample from CPU storage, then ship to training device
    # ------------------------------------------------------------------

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a random batch and move it to ``self.device``."""
        idx = torch.randint(0, self._size, (batch_size,))  # sample index on CPU
        return {
            "obs":      self._obs[idx].to(self.device),
            "actions":  self._actions[idx].to(self.device),
            "rewards":  self._rewards[idx].to(self.device),
            "next_obs": self._next_obs[idx].to(self.device),
            "dones":    self._dones[idx].to(self.device),
        }

    def __len__(self) -> int:
        return self._size
