"""Compatibility layer that exposes the Rainbow DQN agent from the new module."""

from __future__ import annotations

from v2.agents.rainbow.r_dqn import DQNAgent, RainbowDQNAgent

__all__ = ["RainbowDQNAgent", "DQNAgent"]
