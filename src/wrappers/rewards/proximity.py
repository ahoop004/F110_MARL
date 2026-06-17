"""Target proximity reward components."""
from __future__ import annotations

from typing import Dict

import numpy as np

from wrappers.rewards.base import RewardComponent


class TargetProximityComponent(RewardComponent):
    """Reward staying near a target at a preferred distance."""

    def __init__(self, config: dict) -> None:
        self.weight = float(config.get("weight", 1.0))
        self.preferred_distance = float(config.get("preferred_distance", 1.5))
        self.distance_tolerance = float(config.get("distance_tolerance", 0.5))

    def compute(self, step_info: dict) -> Dict[str, float]:
        obs = step_info.get("next_obs") or step_info.get("obs") or {}
        target_pose = np.asarray(obs.get("target_pose", [0.0, 0.0, 0.0]), dtype=np.float64)
        ego_pose = np.asarray(obs.get("pose", [0.0, 0.0, 0.0]), dtype=np.float64)
        dist = float(np.linalg.norm(target_pose[:2] - ego_pose[:2]))
        deviation = abs(dist - self.preferred_distance)
        bonus = max(0.0, 1.0 - deviation / max(self.distance_tolerance, 1e-6))
        return {"target_proximity/bonus": self.weight * bonus}
