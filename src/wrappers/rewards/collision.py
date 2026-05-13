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
