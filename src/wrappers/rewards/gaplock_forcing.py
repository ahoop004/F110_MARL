"""Gaplock forcing reward — bonus for pushing target toward track edges."""
from __future__ import annotations

from typing import Dict

from wrappers.rewards.base import RewardComponent


class GaplockForcingComponent(RewardComponent):
    """Reward proportional to forcing_reward supplied by the environment."""

    def __init__(self, config: dict) -> None:
        self.weight = float(config.get("weight", 0.5))

    def compute(self, step_info: dict) -> Dict[str, float]:
        info = step_info.get("info") or {}
        forcing_val = float(info.get("forcing_reward", 0.0))
        return {"gaplock/forcing": self.weight * forcing_val}
