"""Speed reward component — incentivize forward velocity."""
from __future__ import annotations

from typing import Dict

import numpy as np

from wrappers.rewards.base import RewardComponent


class SpeedRewardComponent(RewardComponent):
    """Small per-step bonus proportional to forward speed."""

    def __init__(self, config: dict) -> None:
        self.weight = float(config.get("weight", 0.01))
        self.speed_index = int(config.get("speed_index", 1))

    def compute(self, step_info: dict) -> Dict[str, float]:
        action = np.asarray(step_info.get("action", [0.0, 0.0]), dtype=np.float32).ravel()
        speed = float(action[self.speed_index]) if len(action) > self.speed_index else 0.0
        return {"speed/bonus": self.weight * max(speed, 0.0)}
