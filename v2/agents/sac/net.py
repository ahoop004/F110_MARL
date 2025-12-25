"""Neural network blocks for the SAC agent."""

from __future__ import annotations

import torch
from torch import nn

from v2.agents.common.networks import build_mlp, soft_update, hard_update


class GaussianPolicy(nn.Module):
    """Actor network that outputs mean/log_std for a diagonal Gaussian."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dims=(256, 256)) -> None:
        super().__init__()
        self.net = build_mlp(obs_dim, hidden_dims, act_dim * 2)
        self.act_dim = act_dim

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(obs)
        mu, log_std = torch.chunk(out, 2, dim=-1)
        log_std = torch.clamp(log_std, -5.0, 2.0)
        return mu, log_std


class QNetwork(nn.Module):
    """Critic network returning scalar Q-value."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dims=(256, 256)) -> None:
        super().__init__()
        self.net = build_mlp(obs_dim + act_dim, hidden_dims, 1)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, act], dim=-1)
        return self.net(x)
