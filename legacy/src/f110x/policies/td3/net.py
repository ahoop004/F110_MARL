"""Neural building blocks for TD3 actor/critic networks."""
from __future__ import annotations

from typing import Iterable, Optional

import torch
from torch import nn


def _build_mlp(
    input_dim: int,
    hidden_dims: Iterable[int],
    output_dim: int,
    *,
    output_activation: Optional[nn.Module] = None,
) -> nn.Sequential:
    layers = []
    prev = input_dim
    for dim in hidden_dims:
        layers.append(nn.Linear(prev, dim))
        layers.append(nn.ReLU(inplace=True))
        prev = dim
    layers.append(nn.Linear(prev, output_dim))
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)


class TD3Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims=(256, 256)) -> None:
        super().__init__()
        self.net = _build_mlp(obs_dim, hidden_dims, act_dim, output_activation=nn.Tanh())

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class TD3Critic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims=(256, 256)) -> None:
        super().__init__()
        self.net = _build_mlp(obs_dim + act_dim, hidden_dims, 1)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, act], dim=-1)
        return self.net(x)


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


def hard_update(target: nn.Module, source: nn.Module) -> None:
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
