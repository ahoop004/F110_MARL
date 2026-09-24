"""Target proximity and track-edge pressure rewards."""
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


class TargetEdgePressureComponent(RewardComponent):
    """Reward proportional to an environment-supplied target edge-pressure fact."""

    def __init__(self, config: dict) -> None:
        self.weight = float(config.get("weight", 0.5))

    def compute(self, step_info: dict) -> Dict[str, float]:
        info = step_info.get("info") or {}
        pressure_val = float(info.get("forcing_reward", 0.0))
        return {"target_edge_pressure/bonus": self.weight * pressure_val}


class RacePursuitComponent(RewardComponent):
    """Penalize trailing by unwrapped progress; reward each isolated respawn."""

    def __init__(self, config):
        self.penalty = float(config.get("behind_penalty", -0.01))
        self.bonus = float(config.get("respawn_bonus", 1.0))
        self.target_id = str(config.get("target_id", "car_1"))
        self.reset()

    def reset(self):
        self._gap = None

    def compute(self, step_info):
        info = step_info.get("info") or {}
        other = (step_info.get("all_infos") or {}).get(self.target_id, {})
        ego = info.get("centerline", {})
        target = other.get("centerline", {})
        if not ego or not target:
            raise ValueError("race_pursuit requires both agents' centerline facts")
        if self._gap is None:
            # Initial grid is local; subsequent progress stays unwrapped, so
            # lapping the opponent cannot flip the ordering at the seam.
            self._gap = (target["progress"] - ego["progress"] + .5) % 1.0 - .5
        else:
            self._gap += target["progress_delta"] - ego["progress_delta"]
        return {"race_pursuit/behind": self.penalty if self._gap > 0 else 0.0,
                "race_pursuit/respawn": self.bonus if info.get("target_respawned") else 0.0}
