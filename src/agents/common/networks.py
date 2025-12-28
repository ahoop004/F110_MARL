"""Shared network utilities for RL agents.

This module contains common neural network building blocks used across
multiple agent implementations (PPO, TD3, SAC, etc.).
"""
from __future__ import annotations

from typing import Iterable, Optional

import torch
from torch import nn


def build_mlp(
    input_dim: int,
    hidden_dims: Iterable[int],
    output_dim: int,
    *,
    output_activation: Optional[nn.Module] = None,
) -> nn.Sequential:
    """Build multi-layer perceptron with ReLU activations.

    Creates a feedforward neural network with:
    - Linear layers with specified hidden dimensions
    - ReLU activations between hidden layers
    - Optional activation on output layer

    Args:
        input_dim: Input dimension
        hidden_dims: Sequence of hidden layer sizes
        output_dim: Output dimension
        output_activation: Optional activation module for output layer
            (e.g., nn.Tanh() for bounded actions)

    Returns:
        nn.Sequential: MLP network

    Example:
        >>> # Build 3-layer MLP: 10 -> 64 -> 64 -> 2 with Tanh output
        >>> net = build_mlp(10, [64, 64], 2, output_activation=nn.Tanh())
    """
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


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """Soft update target network parameters.

    Updates target network parameters as a weighted average:
        θ_target = τ * θ_source + (1 - τ) * θ_target

    This is commonly used in off-policy RL algorithms (TD3, SAC, DDPG)
    to slowly track the online network with a target network for stability.

    Args:
        target: Target network to update
        source: Source network to copy from
        tau: Interpolation parameter in [0, 1]
            - tau=0: No update (target unchanged)
            - tau=1: Hard copy (target = source)
            - Typical values: 0.001-0.01

    Example:
        >>> # Slowly update target critic towards current critic
        >>> soft_update(target_critic, critic, tau=0.005)
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


def hard_update(target: nn.Module, source: nn.Module) -> None:
    """Hard copy source network parameters to target.

    Completely replaces target network parameters with source parameters:
        θ_target = θ_source

    This is equivalent to soft_update(target, source, tau=1.0) but more explicit.

    Args:
        target: Target network to update
        source: Source network to copy from

    Example:
        >>> # Initialize target network to match current network
        >>> hard_update(target_q_net, q_net)
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
