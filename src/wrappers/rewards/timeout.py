"""Timeout reward components."""
from __future__ import annotations

from typing import Dict

from wrappers.rewards.base import RewardComponent


class TimeoutPenaltyComponent(RewardComponent):
    """Sparse penalty when an episode ends by time limit without crash outcome."""

    def __init__(self, config: dict) -> None:
        self.penalty = float(config.get("penalty", -100.0))

    def compute(self, step_info: dict) -> Dict[str, float]:
        info = step_info.get("info") or {}
        if info.get("terminal_reason") != "time_limit":
            return {}
        return {"timeout/penalty": self.penalty}
