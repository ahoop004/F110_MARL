"""Terminal self-crash penalty — fires when ego agent crashes without taking out target."""
from __future__ import annotations

from typing import Dict

from wrappers.rewards.base import RewardComponent


class TerminalSelfCrashComponent(RewardComponent):
    """Sparse penalty when ego crashes but target does not (ego failure)."""

    def __init__(self, config: dict) -> None:
        self.penalty = float(config.get("penalty", -20.0))

    def compute(self, step_info: dict) -> Dict[str, float]:
        if not (step_info.get("done") or step_info.get("terminated")):
            return {}
        info = step_info.get("info") or {}
        ego_crashed = bool(info.get("collision", False))
        target_crashed = bool(info.get("target_collision", False))
        if ego_crashed and not target_crashed:
            return {"terminal/self_crash": self.penalty}
        return {}
