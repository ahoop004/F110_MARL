"""Pure PyTorch SAC (Soft Actor-Critic) agent."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.common.networks import make_mlp
from utils.torch_io import resolve_device


class _SACActorNet(nn.Module):
    """State-dependent Gaussian actor with tanh squashing."""

    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: List[int], activation: str) -> None:
        super().__init__()
        self.net = make_mlp(obs_dim, hidden_dims, action_dim * 2, activation)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.net(obs)
        mean, log_std = out.chunk(2, dim=-1)
        log_std = log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std.exp()

    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        mean, std = self(obs)
        if deterministic:
            action = torch.tanh(mean)
            log_prob = torch.zeros(obs.shape[0], device=obs.device)
        else:
            dist = torch.distributions.Normal(mean, std)
            raw = dist.rsample()
            action = torch.tanh(raw)
            log_prob = (dist.log_prob(raw) - torch.log(1 - action.pow(2) + 1e-6)).sum(-1)
        return action, log_prob


class _SACQNetwork(nn.Module):
    """Q(obs, action) → scalar."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: List[int], activation: str) -> None:
        super().__init__()
        self.net = make_mlp(obs_dim + action_dim, hidden_dims, 1, activation)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1)).squeeze(-1)


class SACAgent:
    """Soft Actor-Critic with automatic entropy tuning and twin Q-critics."""

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
        lr: float = float(params.get("learning_rate", 3e-4))
        self.gamma: float = float(params.get("gamma", 0.99))
        self.tau: float = float(params.get("tau", 0.005))

        # Actor
        self.actor = _SACActorNet(obs_dim, self.action_dim, hidden_dims, activation).to(self.device)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)

        # Twin critics + targets
        self.q1 = _SACQNetwork(obs_dim, self.action_dim, hidden_dims, activation).to(self.device)
        self.q2 = _SACQNetwork(obs_dim, self.action_dim, hidden_dims, activation).to(self.device)
        self.q1_tgt = _SACQNetwork(obs_dim, self.action_dim, hidden_dims, activation).to(self.device)
        self.q2_tgt = _SACQNetwork(obs_dim, self.action_dim, hidden_dims, activation).to(self.device)
        self.q1_tgt.load_state_dict(self.q1.state_dict())
        self.q2_tgt.load_state_dict(self.q2.state_dict())
        self.critic_opt = optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr
        )

        # Entropy coefficient (auto-tuned)
        target_entropy_cfg = params.get("target_entropy", "auto")
        if target_entropy_cfg == "auto" or target_entropy_cfg is None:
            self.target_entropy = -float(self.action_dim)
        else:
            self.target_entropy = float(target_entropy_cfg)

        ent_coef_cfg = params.get("ent_coef", "auto")
        if ent_coef_cfg == "auto" or ent_coef_cfg is None:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_opt = optim.Adam([self.log_alpha], lr=lr)
            self._auto_alpha = True
        else:
            self.log_alpha = torch.log(torch.tensor(float(ent_coef_cfg), device=self.device))
            self.alpha_opt = None
            self._auto_alpha = False

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def act(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            action, _ = self.actor.get_action(obs_t, deterministic=False)
        return action.squeeze(0).cpu().numpy()

    def explore(self) -> np.ndarray:
        return np.random.uniform(-1.0, 1.0, self.action_dim).astype(np.float32)

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        dones = batch["dones"]

        with torch.no_grad():
            next_action, next_log_prob = self.actor.get_action(next_obs)
            q1_next = self.q1_tgt(next_obs, next_action)
            q2_next = self.q2_tgt(next_obs, next_action)
            q_next = torch.min(q1_next, q2_next) - self.alpha.detach() * next_log_prob
            target_q = rewards + self.gamma * (1.0 - dones) * q_next

        # Critic update
        q1_pred = self.q1(obs, actions)
        q2_pred = self.q2(obs, actions)
        critic_loss = nn.functional.mse_loss(q1_pred, target_q) + nn.functional.mse_loss(q2_pred, target_q)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # Actor update
        new_action, log_prob = self.actor.get_action(obs)
        q1_pi = self.q1(obs, new_action)
        q2_pi = self.q2(obs, new_action)
        actor_loss = (self.alpha.detach() * log_prob - torch.min(q1_pi, q2_pi)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Alpha update
        alpha_loss = 0.0
        if self._auto_alpha:
            alpha_loss_t = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
            self.alpha_opt.zero_grad()
            alpha_loss_t.backward()
            self.alpha_opt.step()
            alpha_loss = float(alpha_loss_t.item())

        # Soft update target networks
        for p, p_tgt in zip(self.q1.parameters(), self.q1_tgt.parameters()):
            p_tgt.data.mul_(1 - self.tau).add_(p.data, alpha=self.tau)
        for p, p_tgt in zip(self.q2.parameters(), self.q2_tgt.parameters()):
            p_tgt.data.mul_(1 - self.tau).add_(p.data, alpha=self.tau)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": alpha_loss,
            "alpha": float(self.alpha.item()),
            "log_prob": float(log_prob.mean().item()),
        }

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "log_alpha": self.log_alpha.detach(),
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.q1.load_state_dict(ckpt["q1"])
        self.q2.load_state_dict(ckpt["q2"])
        self.q1_tgt.load_state_dict(ckpt["q1"])
        self.q2_tgt.load_state_dict(ckpt["q2"])
        if "log_alpha" in ckpt:
            with torch.no_grad():
                self.log_alpha.copy_(ckpt["log_alpha"].to(self.device))
