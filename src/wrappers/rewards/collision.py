"""Collision reward component — per-step penalty for crashes."""
from __future__ import annotations

from typing import Dict

from wrappers.rewards.base import RewardComponent


class CollisionRewardComponent(RewardComponent):
    """One-time penalty when the ego agent collides this step."""

    def __init__(self, config: dict) -> None:
        self.penalty = float(config.get("penalty", -200.0))

    def compute(self, step_info: dict) -> Dict[str, float]:
        info = step_info.get("info") or {}
        collided = bool(info.get("collision", False))
        if collided:
            return {"collision/penalty": self.penalty}
        return {}


class SelfCrashPenaltyComponent(RewardComponent):
    """Sparse penalty when ego crashes but the target does not."""

    def __init__(self, config: dict) -> None:
        self.penalty = float(config.get("penalty", -20.0))

    def compute(self, step_info: dict) -> Dict[str, float]:
        if not (step_info.get("done") or step_info.get("terminated")):
            return {}
        info = step_info.get("info") or {}
        ego_crashed = bool(info.get("collision", False))
        target_crashed = bool(info.get("target_collision", False))
        if ego_crashed and not target_crashed:
            return {"self_crash/penalty": self.penalty}
        return {}
