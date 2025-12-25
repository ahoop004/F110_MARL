"""Feed-forward Q-network helper for DQN/Rainbow variants."""
from __future__ import annotations

from typing import Iterable

from torch import nn


class QNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Iterable[int] = (256, 256)) -> None:
        super().__init__()
        layers = []
        prev = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev, dim))
            layers.append(nn.ReLU(inplace=True))
            prev = dim
        layers.append(nn.Linear(prev, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, obs):  # pragma: no cover - simple forward
        return self.model(obs)
