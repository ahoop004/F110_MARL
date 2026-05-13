"""Gaplock reward component — adversarial gap pressure and forcing."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from wrappers.rewards.base import RewardComponent


class GaplockRewardComponent(RewardComponent):
    """Adversarial reward combining pressure (staying close to target) and forcing.

    pressure: bonus for maintaining a tight gap behind the target vehicle
    forcing:  bonus for pushing the target toward track edges / obstacles
    """

    def __init__(self, config: dict) -> None:
        pressure_cfg = config.get("pressure", {})
        forcing_cfg = config.get("forcing", {})

        self.pressure_enabled = bool(pressure_cfg.get("enabled", True))
        self.pressure_weight = float(pressure_cfg.get("weight", 1.0))
        self.preferred_distance = float(pressure_cfg.get("preferred_distance", 1.5))
        self.distance_tolerance = float(pressure_cfg.get("distance_tolerance", 0.5))

        self.forcing_enabled = bool(forcing_cfg.get("enabled", True))
        self.forcing_weight = float(forcing_cfg.get("weight", 0.5))

    def compute(self, step_info: dict) -> Dict[str, float]:
        result: Dict[str, float] = {}
        info = step_info.get("info") or {}

        if self.pressure_enabled:
            obs = step_info.get("next_obs") or step_info.get("obs") or {}
            target_pose = np.asarray(obs.get("target_pose", [0.0, 0.0, 0.0]), dtype=np.float64)
            ego_pose = np.asarray(obs.get("pose", [0.0, 0.0, 0.0]), dtype=np.float64)
            dist = float(np.linalg.norm(target_pose[:2] - ego_pose[:2]))
            deviation = abs(dist - self.preferred_distance)
            bonus = max(0.0, 1.0 - deviation / max(self.distance_tolerance, 1e-6))
            result["gaplock/pressure"] = self.pressure_weight * bonus

        if self.forcing_enabled:
            forcing_val = float(info.get("forcing_reward", 0.0))
            result["gaplock/forcing"] = self.forcing_weight * forcing_val

        return result
