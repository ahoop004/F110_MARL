"""Target-finish penalty for racing tasks."""
from __future__ import annotations

from typing import Dict

from wrappers.rewards.base import RewardComponent


class TargetFinishComponent(RewardComponent):
    """Penalize ego when its configured target finishes first."""

    def __init__(self, config: dict) -> None:
        self.penalty = float(config.get("penalty", -100.0))

    def compute(self, step_info: dict) -> Dict[str, float]:
        info = step_info.get("info") or {}
        if not bool(info.get("target_finished", False)):
            return {}
        if bool(info.get("collision", False)):
            return {}
        return {"target_finish/penalty": self.penalty}
