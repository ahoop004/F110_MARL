"""Compatibility wrapper exposing the Rainbow network from the new module."""

from __future__ import annotations

from v2.agents.rainbow.r_dqn_net import NoisyLinear, RainbowQNetwork

__all__ = ["RainbowQNetwork", "NoisyLinear"]
