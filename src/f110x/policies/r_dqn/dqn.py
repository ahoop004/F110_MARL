"""Compatibility layer that exposes the Rainbow DQN agent from the new module."""

from __future__ import annotations

from f110x.policies.rainbow.r_dqn import DQNAgent, RainbowDQNAgent

__all__ = ["RainbowDQNAgent", "DQNAgent"]
