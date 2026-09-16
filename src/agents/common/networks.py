"""Shared MLP building blocks for all PyTorch RL agents."""
from __future__ import annotations

from functools import partial
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Fixed as part of the "leaky_relu" checkpoint activation contract.
LEAKY_RELU_NEGATIVE_SLOPE = 0.2


def make_mlp(
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    activation: str = "tanh",
    output_activation: Optional[str] = None,
) -> nn.Sequential:
    """Build a fully-connected MLP."""
    act_map = {
        "relu": nn.ReLU, "tanh": nn.Tanh, "silu": nn.SiLU, "swish": nn.SiLU,
        "leaky_relu": partial(nn.LeakyReLU, negative_slope=LEAKY_RELU_NEGATIVE_SLOPE),
    }
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
        # Deterministic quadrature for E[log |d tanh(z)/dz|]. Nonpersistent
        # buffers preserve the parameter layout of existing checkpoints.
        nodes, weights = np.polynomial.hermite.hermgauss(32)
        self.register_buffer("_entropy_nodes", torch.tensor(nodes * np.sqrt(2), dtype=torch.float32), persistent=False)
        self.register_buffer("_entropy_weights", torch.tensor(weights / np.sqrt(np.pi), dtype=torch.float32), persistent=False)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.net(obs)
        log_std = self.log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp()
        return mean, std

    def get_action(
        self, obs: torch.Tensor, deterministic: bool = False, *, return_raw: bool = False
    ):
        mean, std = self(obs)
        if deterministic:
            raw = mean
            action = torch.tanh(mean)
            log_prob = torch.zeros(obs.shape[0], device=obs.device)
        else:
            dist = torch.distributions.Normal(mean, std)
            raw = dist.rsample()
            action = torch.tanh(raw)
            log_prob = self._log_prob(dist, raw)
        return (action, log_prob, raw) if return_raw else (action, log_prob)

    @staticmethod
    def _log_jacobian(raw: torch.Tensor) -> torch.Tensor:
        # log(1 - tanh(z)^2), without cancellation at large |z|.
        return 2.0 * (np.log(2.0) - raw - F.softplus(-2.0 * raw))

    def _log_prob(self, dist, raw):
        return (dist.log_prob(raw) - self._log_jacobian(raw)).sum(-1)

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        raw_actions: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log probabilities for already-sampled squashed actions.

        PPO must compare the current policy probability of each rollout action
        with the probability recorded when that same action was collected.
        Training retains pre-tanh samples because float32 tanh loses their
        identity near the bounds. The inverse fallback supports callers that
        only have nonsaturated actions; it cannot recover saturated samples.
        """
        mean, std = self(obs)
        dist = torch.distributions.Normal(mean, std)
        bounded_actions = actions.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        recovered = torch.atanh(bounded_actions)
        raw_actions = recovered if raw_actions is None else torch.where(
            torch.isfinite(raw_actions), raw_actions, recovered
        )
        log_prob = self._log_prob(dist, raw_actions)
        samples = mean.unsqueeze(-1) + std.unsqueeze(-1) * self._entropy_nodes
        correction = (self._log_jacobian(samples) * self._entropy_weights).sum(-1)
        entropy = (dist.entropy() + correction).sum(-1)
        return log_prob, entropy


class Critic(nn.Module):
    """Value function — maps obs (or global state for MAPPO) to scalar.

    input_dim=obs_dim  for PPO (local obs)
    input_dim=global_state_dim+n_trainable_agents for agent-conditioned MAPPO
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
