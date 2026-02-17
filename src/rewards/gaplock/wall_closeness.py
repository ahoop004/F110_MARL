"""Wall closeness reward component.

Rewards the attacker when the target is close to track walls.
Encourages forcing the target into tight spaces near barriers.
"""

from typing import Any, Dict, Optional

import numpy as np

from rewards.base import RewardComponent


class WallClosenessReward(RewardComponent):
    """Reward based on how close the target is to track walls.

    Computes the minimum distance from the target to all wall points
    and returns a reward that increases as the target gets closer.

    Config keys:
        weight: Multiplier for the closeness reward (default: 1.0)
        distance_scale: Distance in meters at which closeness is zero;
                        closer distances produce higher reward (default: 1.0)
    """

    def __init__(self, config: dict) -> None:
        self.weight = float(config.get("weight", 1.0))
        self.distance_scale = float(config.get("distance_scale", 1.0))
        self._wall_points: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._wall_points = None

    def compute(self, step_info: dict) -> Dict[str, float]:
        if self.distance_scale <= 0.0:
            return {}

        walls = step_info.get("walls")
        target_obs = step_info.get("target_obs")
        target_pose = _extract_pose(target_obs)

        if walls is None or target_pose is None:
            return {}

        wall_points = self._get_wall_points(walls)
        if wall_points is None or wall_points.size == 0:
            return {}

        diffs = wall_points - target_pose[:2].astype(np.float32)
        distances = np.linalg.norm(diffs, axis=1)
        min_dist = float(np.min(distances)) if distances.size else self.distance_scale

        closeness = 1.0 - min_dist / self.distance_scale
        value = float(np.clip(closeness, 0.0, 1.0))
        return {"wall_closeness/reward": self.weight * value}

    def _get_wall_points(self, walls: Any) -> Optional[np.ndarray]:
        if self._wall_points is not None:
            return self._wall_points

        if isinstance(walls, dict):
            parts = [
                np.asarray(pts, dtype=np.float32)[:, :2]
                for pts in walls.values()
                if pts is not None
            ]
            if parts:
                self._wall_points = np.vstack(parts)
        elif isinstance(walls, np.ndarray):
            arr = np.asarray(walls, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                self._wall_points = arr[:, :2]

        return self._wall_points


def _extract_pose(obs: Any) -> Optional[np.ndarray]:
    """Extract [x, y, theta] pose from an observation dict."""
    if not isinstance(obs, dict):
        return None
    pose = obs.get("pose")
    if pose is None:
        pose = obs.get("target_pose")
    if pose is None:
        return None
    arr = np.asarray(pose, dtype=np.float32).reshape(-1)
    if arr.size < 3:
        return None
    return arr[:3]


__all__ = ["WallClosenessReward"]
