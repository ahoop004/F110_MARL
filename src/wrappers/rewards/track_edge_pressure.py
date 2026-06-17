"""Target track-edge pressure reward components."""
from __future__ import annotations

from typing import Dict

from wrappers.rewards.base import RewardComponent


class TargetEdgePressureComponent(RewardComponent):
    """Reward proportional to an environment-supplied target edge-pressure fact."""

    def __init__(self, config: dict) -> None:
        self.weight = float(config.get("weight", 0.5))

    def compute(self, step_info: dict) -> Dict[str, float]:
        info = step_info.get("info") or {}
        pressure_val = float(info.get("forcing_reward", 0.0))
        return {"target_edge_pressure/bonus": self.weight * pressure_val}
