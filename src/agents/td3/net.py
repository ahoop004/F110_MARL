"""Neural building blocks for TD3 actor/critic networks."""
from __future__ import annotations

import torch
from torch import nn

from agents.common.networks import build_mlp, soft_update, hard_update


class TD3Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims=(256, 256)) -> None:
        super().__init__()
        self.net = build_mlp(obs_dim, hidden_dims, act_dim, output_activation=nn.Tanh())

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class TD3Critic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims=(256, 256)) -> None:
        super().__init__()
        self.net = build_mlp(obs_dim + act_dim, hidden_dims, 1)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, act], dim=-1)
        return self.net(x)
