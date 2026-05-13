"""Pure PyTorch ring-buffer replay buffer for off-policy algorithms."""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch


class ReplayBuffer:
    """Fixed-capacity ring buffer storing (obs, action, reward, next_obs, done).

    All data is stored as float32 tensors on the specified device.
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
    ) -> None:
        self.capacity = int(capacity)
        self.device = device
        self._obs = torch.zeros(self.capacity, obs_dim, dtype=torch.float32, device=device)
        self._actions = torch.zeros(self.capacity, action_dim, dtype=torch.float32, device=device)
        self._rewards = torch.zeros(self.capacity, dtype=torch.float32, device=device)
        self._next_obs = torch.zeros(self.capacity, obs_dim, dtype=torch.float32, device=device)
        self._dones = torch.zeros(self.capacity, dtype=torch.float32, device=device)
        self._ptr = 0
        self._size = 0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        i = self._ptr
        self._obs[i] = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        self._actions[i] = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        self._rewards[i] = float(reward)
        self._next_obs[i] = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        self._dones[i] = float(done)
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        idx = torch.randint(0, self._size, (batch_size,), device=self.device)
        return {
            "obs": self._obs[idx],
            "actions": self._actions[idx],
            "rewards": self._rewards[idx],
            "next_obs": self._next_obs[idx],
            "dones": self._dones[idx],
        }

    def __len__(self) -> int:
        return self._size
