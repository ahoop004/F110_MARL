"""Target crash reward components."""
from __future__ import annotations

from typing import Dict

from wrappers.rewards.base import RewardComponent


class TargetCrashBonusComponent(RewardComponent):
    """Sparse bonus when the target crashes and ego does not."""

    def __init__(self, config: dict) -> None:
        self.bonus = float(config.get("bonus", 200.0))
        self._awarded = False

    def reset(self) -> None:
        self._awarded = False

    def compute(self, step_info: dict) -> Dict[str, float]:
        info = step_info.get("info") or {}
        target_crashed = info.get("target_terminal_reason") == "collision"
        ego_crashed = info.get("terminal_reason") == "collision"
        if target_crashed and not ego_crashed and not self._awarded:
            self._awarded = True
            return {"target_crash/bonus": self.bonus}
        return {}
