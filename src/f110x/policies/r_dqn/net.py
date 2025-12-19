"""Compatibility wrapper exposing the Rainbow network from the new module."""

from __future__ import annotations

from f110x.policies.rainbow.r_dqn_net import NoisyLinear, RainbowQNetwork

__all__ = ["RainbowQNetwork", "NoisyLinear"]
