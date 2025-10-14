"""Rainbow DQN network components (noisy, dueling, distributional)."""

from __future__ import annotations

import math
from typing import Iterable, List

import torch
from torch import nn
from torch.nn import functional as F


class NoisyLinear(nn.Module):
    """Factorised Gaussian noisy linear layer as used in Rainbow DQN."""

    def __init__(self, in_features: int, out_features: int, sigma0: float = 0.5) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("NoisyLinear requires positive in/out features")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.sigma0 = float(sigma0)

        weight_shape = (self.out_features, self.in_features)
        self.weight_mu = nn.Parameter(torch.empty(weight_shape))
        self.weight_sigma = nn.Parameter(torch.empty(weight_shape))
        self.register_buffer("weight_epsilon", torch.zeros(weight_shape))

        self.bias_mu = nn.Parameter(torch.empty(self.out_features))
        self.bias_sigma = nn.Parameter(torch.empty(self.out_features))
        self.register_buffer("bias_epsilon", torch.zeros(self.out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.bias_mu.data.uniform_(-bound, bound)

        sigma_weight = self.sigma0 / math.sqrt(self.in_features)
        sigma_bias = self.sigma0 / math.sqrt(self.out_features)
        self.weight_sigma.data.fill_(sigma_weight)
        self.bias_sigma.data.fill_(sigma_bias)

    def reset_noise(self) -> None:
        eps_in = self._scale_noise(self.in_features, device=self.weight_mu.device)
        eps_out = self._scale_noise(self.out_features, device=self.weight_mu.device)
        self.weight_epsilon.copy_(eps_out.ger(eps_in))
        self.bias_epsilon.copy_(eps_out)

    @staticmethod
    def _scale_noise(size: int, *, device: torch.device) -> torch.Tensor:
        noise = torch.randn(size, device=device)
        return noise.sign().mul_(noise.abs().sqrt_())

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # pragma: no cover - exercised via Rainbow agent
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(input, weight, bias)


class RainbowQNetwork(nn.Module):
    """Dueling distributional Q-network with optional noisy exploration layers."""

    def __init__(
        self,
        input_dim: int,
        n_actions: int,
        hidden_dims: Iterable[int] = (256, 256),
        *,
        atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        noisy: bool = True,
        sigma0: float = 0.5,
    ) -> None:
        super().__init__()
        if atoms <= 1:
            raise ValueError("RainbowQNetwork requires at least two atoms for the categorical support")
        if n_actions <= 0:
            raise ValueError("RainbowQNetwork requires at least one action")

        self.n_actions = int(n_actions)
        self.atoms = int(atoms)
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.noisy = bool(noisy)
        self.sigma0 = float(sigma0)

        hidden = list(hidden_dims) or [256, 256]
        self.hidden_layers = nn.ModuleList()
        self._noisy_layers: List[NoisyLinear] = []

        prev_dim = int(input_dim)
        for dim in hidden:
            layer = self._make_linear(prev_dim, int(dim))
            self.hidden_layers.append(layer)
            prev_dim = int(dim)

        self.value_head = self._make_linear(prev_dim, self.atoms)
        self.advantage_head = self._make_linear(prev_dim, self.n_actions * self.atoms)

        support = torch.linspace(self.v_min, self.v_max, self.atoms)
        self.register_buffer("support", support)

    def _make_linear(self, in_dim: int, out_dim: int) -> nn.Module:
        if self.noisy:
            layer = NoisyLinear(in_dim, out_dim, sigma0=self.sigma0)
            self._noisy_layers.append(layer)
            return layer
        return nn.Linear(in_dim, out_dim)

    def reset_noise(self) -> None:
        if not self.noisy:
            return
        for layer in self._noisy_layers:
            layer.reset_noise()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs
        for layer in self.hidden_layers:
            x = layer(x)
            x = F.relu(x, inplace=True)

        value = self.value_head(x).view(-1, 1, self.atoms)
        advantage = self.advantage_head(x).view(-1, self.n_actions, self.atoms)
        advantage = advantage - advantage.mean(dim=1, keepdim=True)
        return value + advantage

    def dist(self, obs: torch.Tensor) -> torch.Tensor:
        """Return the categorical action-value distribution (probabilities)."""
        logits = self.forward(obs)
        return torch.softmax(logits, dim=-1)

    def q_values(self, obs: torch.Tensor) -> torch.Tensor:
        """Return expected Q-values computed from the categorical distribution."""
        probs = self.dist(obs)
        return torch.sum(probs * self.support, dim=-1)
