"""Centerline deviation reward component.

Rewards the attacker when the target deviates from the track centerline.
Larger deviations produce higher reward, encouraging the attacker to
force the target off-line.
"""

from typing import Any, Dict, Optional

import numpy as np

from rewards.base import RewardComponent
from utils.centerline import project_to_centerline


class CenterlineDeviationReward(RewardComponent):
    """Reward based on target's lateral deviation from the centerline.

    Computes the target's lateral error relative to the track centerline
    and returns a normalized reward proportional to that deviation.

    Config keys:
        weight: Multiplier for the deviation reward (default: 1.0)
        scale: Normalization scale in meters; deviation is divided by this
               value and clipped to [0, 1] (default: 1.0)
    """

    def __init__(self, config: dict) -> None:
        self.weight = float(config.get("weight", 1.0))
        self.scale = float(config.get("scale", 1.0))
        self._last_index: Optional[int] = None

    def reset(self) -> None:
        self._last_index = None

    def compute(self, step_info: dict) -> Dict[str, float]:
        if self.scale <= 0.0:
            return {}

        centerline = step_info.get("centerline")
        target_obs = step_info.get("target_obs")
        target_pose = _extract_pose(target_obs)

        if centerline is None or target_pose is None:
            return {}

        projection = project_to_centerline(
            np.asarray(centerline, dtype=np.float32),
            target_pose[:2].astype(np.float32),
            float(target_pose[2]),
            last_index=self._last_index,
        )
        self._last_index = projection.index

        deviation = abs(float(projection.lateral_error))
        value = float(np.clip(deviation / self.scale, 0.0, 1.0))
        return {"centerline_deviation/reward": self.weight * value}


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


__all__ = ["CenterlineDeviationReward"]
