"""Shared MLP building blocks for all PyTorch RL agents."""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


def make_mlp(
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    activation: str = "tanh",
    output_activation: Optional[str] = None,
) -> nn.Sequential:
    """Build a fully-connected MLP."""
    act_map = {"relu": nn.ReLU, "tanh": nn.Tanh, "silu": nn.SiLU, "swish": nn.SiLU}
    Act = act_map.get(activation.lower(), nn.Tanh)

    layers: List[nn.Module] = []
    prev = input_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), Act()]
        prev = h
    layers.append(nn.Linear(prev, output_dim))

    if output_activation:
        OutAct = act_map.get(output_activation.lower())
        if OutAct:
            layers.append(OutAct())

    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Gaussian policy — outputs mean; log_std is a learned parameter.

    Actions are sampled as: action = tanh(mean + std * noise)
    This keeps actions in (-1, 1) matching the normalized action space.
    """

    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int],
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        self.net = make_mlp(obs_dim, hidden_dims, action_dim, activation)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.net(obs)
        log_std = self.log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp()
        return mean, std

    def get_action(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
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


class Critic(nn.Module):
    """Value function — maps obs (or global state for MAPPO) to scalar.

    input_dim=obs_dim  for PPO (local obs)
    input_dim=n_agents*obs_dim  for MAPPO (global state)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        self.net = make_mlp(input_dim, hidden_dims, 1, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
