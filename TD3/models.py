# models.py
# MLP-based Actor and Twin Critic networks for TD3.
# - ActorMLP:      obs -> a_norm in [-1, 1]^act_dim  (via tanh)
# - QNetworkMLP:   (obs, a_norm) -> Q(s,a)
# - TwinCriticMLP: wraps two independent Q networks (Q1, Q2)
#
# Targets (actor_targ / critic_targ) are created and soft-updated in agents.py.

from __future__ import annotations
from typing import Sequence, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F



# -------------------------
# Utilities
# -------------------------

def init_linear_(layer: nn.Linear, *, last: bool = False, last_w_scale: float = 1e-3) -> None:
    """
    Initialize a Linear layer.
    - Hidden layers: orthogonal init with gain sqrt(2), bias=0
    - Last layer:    uniform in [-last_w_scale, +last_w_scale], bias=0
    """
    if last:
        nn.init.uniform_(layer.weight, -last_w_scale, last_w_scale)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0.0)
    else:
        nn.init.orthogonal_(layer.weight, gain=2 ** 0.5)  # ~ ReLU-friendly
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0.0)




def _build_mlp(sizes: Sequence[int], activation: nn.Module = nn.ReLU()) -> nn.Sequential:
    """
    Build an MLP (without the final output layer).
    Example: sizes=[obs_dim, 256, 256] -> Linear(obs_dim,256)-Act-Linear(256,256)-Act
    """
    layers = []
    for i in range(len(sizes) - 1):
        lin = nn.Linear(sizes[i], sizes[i + 1])
        init_linear_(lin, last=False)
        layers += [lin, activation]
    return nn.Sequential(*layers)


# -------------------------
# Actor
# -------------------------

class ActorMLP(nn.Module):

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: Sequence[int] = (128, 128),
        activation: nn.Module = nn.ReLU(),
        last_w_scale: float = 1e-3,
    ) -> None:
        super().__init__()
        assert obs_dim > 0 and act_dim > 0

        # Body
        self.body = _build_mlp([obs_dim, *hidden_sizes], activation)

        # Output head
        last_in = hidden_sizes[-1] if len(hidden_sizes) > 0 else obs_dim
        self.mu = nn.Linear(last_in, act_dim)
        init_linear_(self.mu, last=True, last_w_scale=last_w_scale)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: (B, obs_dim) or (obs_dim,)
        returns: (B, act_dim) in [-1, 1]
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        x = self.body(obs)
        a = torch.tanh(self.mu(x))
        return a




# -------------------------
# Q Networks
# -------------------------

class QNetworkMLP(nn.Module):
    """
    Single Q network: (obs, a_norm) -> Q(s,a).

    Args:
        obs_dim:       observation dimensionality
        act_dim:       action dimensionality (normalized action)
        hidden_sizes:  e.g., (256, 256)
        activation:    hidden activation (default ReLU)
        last_w_scale:  initialize final layer weights in [-last_w_scale, +last_w_scale]
    """
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: Sequence[int] = (128, 128),
        activation: nn.Module = nn.ReLU(),
        last_w_scale: float = 1e-3,
    ) -> None:
        super().__init__()
        assert obs_dim > 0 and act_dim > 0

        in_dim = obs_dim + act_dim
        self.body = _build_mlp([in_dim, *hidden_sizes], activation)

        last_in = hidden_sizes[-1] if len(hidden_sizes) > 0 else in_dim
        self.q_out = nn.Linear(last_in, 1)
        init_linear_(self.q_out, last=True, last_w_scale=last_w_scale)

    def forward(self, obs: torch.Tensor, act_norm: torch.Tensor) -> torch.Tensor:
        """
        obs:      (B, obs_dim) or (obs_dim,)
        act_norm: (B, act_dim) or (act_dim,)
        returns:  (B, 1)
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if act_norm.dim() == 1:
            act_norm = act_norm.unsqueeze(0)
        x = torch.cat([obs, act_norm], dim=-1)
        h = self.body(x)
        q = self.q_out(h)
        return q


class TwinCriticMLP(nn.Module):
    """
    TD3 twin critic: wraps two independent Q networks (Q1 and Q2).
    """
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: Sequence[int] = (128, 128),
        activation: nn.Module = nn.ReLU(),
        last_w_scale: float = 1e-3,
    ) -> None:
        super().__init__()
        self.q1 = QNetworkMLP(obs_dim, act_dim, hidden_sizes, activation, last_w_scale)
        self.q2 = QNetworkMLP(obs_dim, act_dim, hidden_sizes, activation, last_w_scale)

    def forward(self, obs: torch.Tensor, act_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns both Q1 and Q2: each shape (B, 1).
        """
        return self.q1(obs, act_norm), self.q2(obs, act_norm)

    def q1_only(self, obs: torch.Tensor, act_norm: torch.Tensor) -> torch.Tensor:
        """Convenience helper: returns only Q1(s,a)."""
        return self.q1(obs, act_norm)
