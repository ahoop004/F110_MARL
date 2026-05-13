"""Pure PyTorch TD3 (Twin Delayed Deep Deterministic Policy Gradient) agent."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.common.networks import make_mlp
from utils.torch_io import resolve_device


class _TD3ActorNet(nn.Module):
    """Deterministic actor: obs → action in [-1, 1]."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: List[int], activation: str) -> None:
        super().__init__()
        self.net = make_mlp(obs_dim, hidden_dims, action_dim, activation, output_activation="tanh")

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class _TD3QNetwork(nn.Module):
    """Q(obs, action) → scalar."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: List[int], activation: str) -> None:
        super().__init__()
        self.net = make_mlp(obs_dim + action_dim, hidden_dims, 1, activation)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1)).squeeze(-1)


class TD3Agent:
    """TD3 with twin critics, delayed actor updates, and target policy smoothing."""

    def __init__(
        self,
        obs_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        params: Dict,
    ) -> None:
        self.obs_dim = obs_dim
        self.action_dim = len(action_low)
        self.device = resolve_device([params.get("device", "cpu")])

        hidden_dims: List[int] = params.get("hidden_dims", [256, 256])
        activation: str = params.get("activation", "relu")
        lr: float = float(params.get("learning_rate", 1e-4))
        self.gamma: float = float(params.get("gamma", 0.99))
        self.tau: float = float(params.get("tau", 0.005))
        self.policy_delay: int = int(params.get("policy_delay", 2))
        self.action_noise_sigma: float = float(params.get("action_noise_sigma", 0.1))
        self.target_policy_noise: float = float(params.get("target_policy_noise", 0.2))
        self.target_noise_clip: float = float(params.get("target_noise_clip", 0.5))
        self._update_step = 0

        # Actor + target
        self.actor = _TD3ActorNet(obs_dim, self.action_dim, hidden_dims, activation).to(self.device)
        self.actor_tgt = _TD3ActorNet(obs_dim, self.action_dim, hidden_dims, activation).to(self.device)
        self.actor_tgt.load_state_dict(self.actor.state_dict())
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)

        # Twin critics + targets
        self.q1 = _TD3QNetwork(obs_dim, self.action_dim, hidden_dims, activation).to(self.device)
        self.q2 = _TD3QNetwork(obs_dim, self.action_dim, hidden_dims, activation).to(self.device)
        self.q1_tgt = _TD3QNetwork(obs_dim, self.action_dim, hidden_dims, activation).to(self.device)
        self.q2_tgt = _TD3QNetwork(obs_dim, self.action_dim, hidden_dims, activation).to(self.device)
        self.q1_tgt.load_state_dict(self.q1.state_dict())
        self.q2_tgt.load_state_dict(self.q2.state_dict())
        self.critic_opt = optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr
        )

    def act(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            action = self.actor(obs_t).squeeze(0).cpu().numpy()
        noise = np.random.normal(0, self.action_noise_sigma, self.action_dim).astype(np.float32)
        return np.clip(action + noise, -1.0, 1.0)

    def explore(self) -> np.ndarray:
        return np.random.uniform(-1.0, 1.0, self.action_dim).astype(np.float32)

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self._update_step += 1
        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        dones = batch["dones"]

        with torch.no_grad():
            noise = (
                torch.randn_like(actions) * self.target_policy_noise
            ).clamp(-self.target_noise_clip, self.target_noise_clip)
            next_action = (self.actor_tgt(next_obs) + noise).clamp(-1.0, 1.0)
            q1_next = self.q1_tgt(next_obs, next_action)
            q2_next = self.q2_tgt(next_obs, next_action)
            target_q = rewards + self.gamma * (1.0 - dones) * torch.min(q1_next, q2_next)

        # Critic update
        critic_loss = (
            nn.functional.mse_loss(self.q1(obs, actions), target_q)
            + nn.functional.mse_loss(self.q2(obs, actions), target_q)
        )
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # Delayed actor + target update
        actor_loss = 0.0
        if self._update_step % self.policy_delay == 0:
            actor_loss_t = -self.q1(obs, self.actor(obs)).mean()
            self.actor_opt.zero_grad()
            actor_loss_t.backward()
            self.actor_opt.step()
            actor_loss = float(actor_loss_t.item())

            for p, p_tgt in zip(self.actor.parameters(), self.actor_tgt.parameters()):
                p_tgt.data.mul_(1 - self.tau).add_(p.data, alpha=self.tau)
            for p, p_tgt in zip(self.q1.parameters(), self.q1_tgt.parameters()):
                p_tgt.data.mul_(1 - self.tau).add_(p.data, alpha=self.tau)
            for p, p_tgt in zip(self.q2.parameters(), self.q2_tgt.parameters()):
                p_tgt.data.mul_(1 - self.tau).add_(p.data, alpha=self.tau)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": actor_loss,
        }

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.q1.load_state_dict(ckpt["q1"])
        self.q2.load_state_dict(ckpt["q2"])
        self.actor_tgt.load_state_dict(ckpt["actor"])
        self.q1_tgt.load_state_dict(ckpt["q1"])
        self.q2_tgt.load_state_dict(ckpt["q2"])
