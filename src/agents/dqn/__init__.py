"""Pure PyTorch DQN (Deep Q-Network) agent with discrete action set."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.common.networks import make_mlp
from utils.torch_io import resolve_device


class _DQNQNetwork(nn.Module):
    """Q(obs) → Q-values for all discrete actions."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_dims: List[int], activation: str) -> None:
        super().__init__()
        self.net = make_mlp(obs_dim, hidden_dims, n_actions, activation)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class DQNAgent:
    """DQN with ε-greedy exploration, twin-network (online + target), and soft target updates."""

    def __init__(
        self,
        obs_dim: int,
        action_set: List,
        params: Dict,
    ) -> None:
        self.obs_dim = obs_dim
        self.action_set = np.asarray(action_set, dtype=np.float32)
        self.n_actions = len(self.action_set)
        self.device = resolve_device([params.get("device", "cpu")])

        hidden_dims: List[int] = params.get("hidden_dims", [256, 256])
        activation: str = params.get("activation", "relu")
        lr: float = float(params.get("learning_rate", 1e-4))
        self.gamma: float = float(params.get("gamma", 0.99))
        self.tau: float = float(params.get("tau", 0.005))

        # Epsilon schedule
        self.eps_start: float = float(params.get("exploration_initial_eps", 1.0))
        self.eps_final: float = float(params.get("exploration_final_eps", 0.05))
        self.eps_fraction: float = float(params.get("exploration_fraction", 0.25))
        self._total_steps: int = 0
        self._decay_steps: int = 1  # set by trainer via set_total_steps()

        # Online + target networks
        self.q_net = _DQNQNetwork(obs_dim, self.n_actions, hidden_dims, activation).to(self.device)
        self.q_tgt = _DQNQNetwork(obs_dim, self.n_actions, hidden_dims, activation).to(self.device)
        self.q_tgt.load_state_dict(self.q_net.state_dict())
        self.opt = optim.Adam(self.q_net.parameters(), lr=lr)

    def set_total_steps(self, total_steps: int) -> None:
        self._decay_steps = max(1, int(total_steps * self.eps_fraction))

    def _epsilon(self) -> float:
        frac = min(1.0, self._total_steps / self._decay_steps)
        return self.eps_start + frac * (self.eps_final - self.eps_start)

    def act(self, obs: np.ndarray) -> np.ndarray:
        self._total_steps += 1
        if np.random.random() < self._epsilon():
            return self.explore()
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            idx = int(self.q_net(obs_t).argmax(dim=-1).item())
        return np.array([idx], dtype=np.float32)

    def explore(self) -> np.ndarray:
        return np.array([np.random.randint(self.n_actions)], dtype=np.float32)

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs = batch["obs"]
        action_idx = batch["actions"].long().squeeze(-1)
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        dones = batch["dones"]

        with torch.no_grad():
            next_q = self.q_tgt(next_obs).max(dim=-1).values
            target_q = rewards + self.gamma * (1.0 - dones) * next_q

        q_pred = self.q_net(obs).gather(1, action_idx.unsqueeze(1)).squeeze(1)
        loss = nn.functional.smooth_l1_loss(q_pred, target_q)

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.opt.step()

        # Soft target update
        for p, p_tgt in zip(self.q_net.parameters(), self.q_tgt.parameters()):
            p_tgt.data.mul_(1 - self.tau).add_(p.data, alpha=self.tau)

        return {
            "q_loss": float(loss.item()),
            "epsilon": self._epsilon(),
        }

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"q_net": self.q_net.state_dict()}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.q_tgt.load_state_dict(ckpt["q_net"])
