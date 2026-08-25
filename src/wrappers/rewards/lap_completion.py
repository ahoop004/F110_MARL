"""Clean lap-completion reward for racing tasks."""
from __future__ import annotations

from typing import Dict

from wrappers.rewards.base import RewardComponent


class LapCompletionComponent(RewardComponent):
    """Sparse final-race bonus tied only to explicit completion facts."""

    def __init__(self, config: dict) -> None:
        self.bonus = float(config.get("bonus", 150.0))
        self.require_clean = bool(config.get("require_clean", True))

    def compute(self, step_info: dict) -> Dict[str, float]:
        info = step_info.get("info") or {}
        if not bool(info.get("lap_crossed", False)):
            return {}
        if not bool(info.get("race_completed", False)):
            return {}

        if self.require_clean and bool(info.get("collision", False)):
            return {}

        return {"lap_completion/bonus": self.bonus}


class PerLapBonusComponent(RewardComponent):
    """Bonus for each accepted crossing, optionally excluding the final lap."""

    def __init__(self, config: dict) -> None:
        self.bonus = float(config.get("bonus", 25.0))
        self.include_final_lap = bool(config.get("include_final_lap", False))

    def compute(self, step_info: dict) -> Dict[str, float]:
        info = step_info.get("info") or {}
        if not bool(info.get("lap_crossed", False)):
            return {}
        if bool(info.get("race_completed", False)) and not self.include_final_lap:
            return {}
        return {"per_lap/bonus": self.bonus}
