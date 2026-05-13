"""Terminal reward component — sparse signals at episode end."""
from __future__ import annotations

from typing import Dict

from wrappers.rewards.base import RewardComponent


class TerminalRewardComponent(RewardComponent):
    """Sparse rewards on episode termination encoding task success/failure."""

    def __init__(self, config: dict) -> None:
        self.success = float(config.get("success", 200.0))
        self.timeout = float(config.get("timeout", -100.0))
        self.self_crash = float(config.get("self_crash", -20.0))
        self.collision = float(config.get("collision", 0.0))
        self.target_finish = float(config.get("target_finish", -20.0))

    def compute(self, step_info: dict) -> Dict[str, float]:
        done = step_info.get("done", False) or step_info.get("terminated", False)
        if not done:
            return {}

        info = step_info.get("info") or {}
        attacker_crashed = bool(info.get("collision", False))
        target_crashed = bool(info.get("target_collision", False))
        target_finished = bool(info.get("target_finished", False))
        truncated = step_info.get("truncated", False)

        if target_finished:
            return {"terminal/target_finish": self.target_finish}
        if attacker_crashed and target_crashed:
            return {"terminal/collision": self.collision}
        if target_crashed:
            return {"terminal/success": self.success}
        if attacker_crashed:
            return {"terminal/self_crash": self.self_crash}
        if truncated:
            return {"terminal/timeout": self.timeout}
        return {"terminal/timeout": self.timeout}
