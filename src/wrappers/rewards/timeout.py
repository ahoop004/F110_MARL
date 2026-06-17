"""Timeout reward components."""
from __future__ import annotations

from typing import Dict

from wrappers.rewards.base import RewardComponent


class TimeoutPenaltyComponent(RewardComponent):
    """Sparse penalty when an episode ends by time limit without crash outcome."""

    def __init__(self, config: dict) -> None:
        self.penalty = float(config.get("penalty", -100.0))

    def compute(self, step_info: dict) -> Dict[str, float]:
        if not step_info.get("truncated"):
            return {}
        info = step_info.get("info") or {}
        if bool(info.get("collision")) or bool(info.get("target_collision")):
            return {}
        return {"timeout/penalty": self.penalty}
