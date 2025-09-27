"""Neural network building blocks for PPO policies."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import torch
from gymnasium import spaces
from gymnasium.spaces import utils as space_utils
from torch import nn


def _build_mlp(
    input_dim: int,
    hidden_sizes: Iterable[int],
    activation: nn.Module = nn.Tanh,
    output_dim: int | None = None,
) -> nn.Sequential:
    """Create a simple fully connected network."""

    layers = []
    last_dim = int(input_dim)
    for size in hidden_sizes:
        layers.append(nn.Linear(last_dim, int(size)))
        layers.append(activation())
        last_dim = int(size)

    if output_dim is not None:
        layers.append(nn.Linear(last_dim, int(output_dim)))

    return nn.Sequential(*layers)


class PPOActorCritic(nn.Module):
    """Shared-body actor-critic network for continuous control PPO."""

    def __init__(
        self,
        obs_space: spaces.Space,
        action_space: spaces.Box,
        hidden_sizes: Tuple[int, ...] = (256, 256),
        activation: nn.Module = nn.Tanh,
    ) -> None:
        super().__init__()

        if not isinstance(action_space, spaces.Box):
            raise TypeError("PPOActorCritic currently supports Box action spaces only")
        if action_space.low.shape != action_space.high.shape:
            raise ValueError("Action space low/high bounds must have identical shapes")

        flat_obs_space = space_utils.flatten_space(obs_space)
        if not isinstance(flat_obs_space, spaces.Box):
            raise TypeError("Flattened observation space must resolve to a Box")

        self.obs_dim = int(np.prod(flat_obs_space.shape, dtype=np.int64))
        self.act_dim = int(np.prod(action_space.shape, dtype=np.int64))

        hidden_sizes = tuple(int(h) for h in hidden_sizes)
        if not hidden_sizes:
            raise ValueError("At least one hidden layer is required")

        self.body = _build_mlp(self.obs_dim, hidden_sizes, activation)
        last_dim = hidden_sizes[-1]

        self.policy_mean = nn.Linear(last_dim, self.act_dim)
        self.value_head = nn.Linear(last_dim, 1)

        log_std_init = torch.zeros(self.act_dim, dtype=torch.float32)
        self.log_std = nn.Parameter(log_std_init)

        low = torch.as_tensor(action_space.low, dtype=torch.float32)
        high = torch.as_tensor(action_space.high, dtype=torch.float32)
        self.register_buffer("action_low", low)
        self.register_buffer("action_high", high)

    def _forward_body(self, obs: torch.Tensor) -> torch.Tensor:
        obs_flat = obs.view(obs.size(0), -1)
        return self.body(obs_flat)

    def _distribution(self, features: torch.Tensor) -> torch.distributions.Normal:
        mean = self.policy_mean(features)
        log_std = torch.clamp(self.log_std, -20.0, 2.0)
        std = torch.exp(log_std)
        return torch.distributions.Normal(mean, std)

    def act(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action and return (clipped_action, log_prob, value, raw_action)."""

        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        features = self._forward_body(obs)
        dist = self._distribution(features)
        raw_action = dist.rsample()
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        value = self.value_head(features).squeeze(-1)

        clipped = torch.maximum(
            torch.minimum(raw_action, self.action_high.expand_as(raw_action)),
            self.action_low.expand_as(raw_action),
        )

        return clipped.squeeze(0), log_prob.squeeze(0), value.squeeze(0), raw_action.squeeze(0)

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)

        features = self._forward_body(obs)
        dist = self._distribution(features)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        values = self.value_head(features).squeeze(-1)
        return log_prob, entropy, values

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        features = self._forward_body(obs)
        return self.value_head(features).squeeze(-1)
