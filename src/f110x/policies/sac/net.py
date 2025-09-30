"""Neural network blocks for the SAC agent."""

from __future__ import annotations

from typing import Iterable, Optional

import torch
from torch import nn


def _build_mlp(
    input_dim: int,
    hidden_dims: Iterable[int],
    output_dim: int,
    *,
    activation: Optional[nn.Module] = None,
) -> nn.Sequential:
    layers = []
    prev = input_dim
    for dim in hidden_dims:
        layers.append(nn.Linear(prev, dim))
        layers.append(nn.ReLU(inplace=True))
        prev = dim
    layers.append(nn.Linear(prev, output_dim))
    if activation is not None:
        layers.append(activation)
    return nn.Sequential(*layers)


class GaussianPolicy(nn.Module):
    """Actor network that outputs mean/log_std for a diagonal Gaussian."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dims=(256, 256)) -> None:
        super().__init__()
        self.net = _build_mlp(obs_dim, hidden_dims, act_dim * 2)
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
