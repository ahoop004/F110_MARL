"""Clean lap-completion reward for racing tasks."""
from __future__ import annotations

from typing import Dict

from wrappers.rewards.base import RewardComponent


class LapCompletionComponent(RewardComponent):
    """Sparse bonus when the ego agent crosses the configured finish line."""

    def __init__(self, config: dict) -> None:
        self.bonus = float(config.get("bonus", 150.0))
        self.require_clean = bool(config.get("require_clean", True))

    def compute(self, step_info: dict) -> Dict[str, float]:
        if not step_info.get("terminated", False):
            return {}

        info = step_info.get("info") or {}
        if not bool(info.get("finish_line", False)):
            return {}

        if self.require_clean and bool(info.get("collision", False)):
            return {}

        return {"lap_completion/bonus": self.bonus}
