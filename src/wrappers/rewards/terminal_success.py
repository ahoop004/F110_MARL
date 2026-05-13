"""Terminal success reward — fires when target vehicle is eliminated."""
from __future__ import annotations

from typing import Dict

from wrappers.rewards.base import RewardComponent


class TerminalSuccessComponent(RewardComponent):
    """Sparse bonus at episode end when target crashes but ego does not."""

    def __init__(self, config: dict) -> None:
        self.bonus = float(config.get("bonus", 200.0))

    def compute(self, step_info: dict) -> Dict[str, float]:
        if not (step_info.get("done") or step_info.get("terminated")):
            return {}
        info = step_info.get("info") or {}
        target_crashed = bool(info.get("target_collision", False))
        ego_crashed = bool(info.get("collision", False))
        if target_crashed and not ego_crashed:
            return {"terminal/success": self.bonus}
        return {}
