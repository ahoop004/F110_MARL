"""Ego vehicle state observation component."""
from __future__ import annotations

from typing import Dict

import numpy as np

from wrappers.observations.base import ObservationComponent

_ZEROS3 = np.zeros(3, dtype=np.float32)


class EgoStateComponent(ObservationComponent):
    """Ego vehicle state: velocity (always) + optional pose.

    velocity:  [vx, vy, yaw_rate]  — 3 dims
    pose:      [x,  y,  theta]     — 3 dims (optional)
    """

    def __init__(
        self,
        include_velocity: bool = True,
        include_pose: bool = False,
    ) -> None:
        self._vel = include_velocity
        self._pose = include_pose

    @property
    def dim(self) -> int:
        return (3 if self._vel else 0) + (3 if self._pose else 0)

    def compute_into(self, raw_obs: Dict, info: Dict, out: np.ndarray) -> None:
        offset = 0
        if self._vel:
            raw = raw_obs.get("velocity")
            if raw is None:
                out[0:3] = 0.0
            else:
                arr = np.asarray(raw, dtype=np.float32).ravel()
                n = min(arr.shape[0], 3)
                out[0:n] = arr[0:n]
                if n < 3:
                    out[n:3] = 0.0
            offset = 3
        if self._pose:
            raw = raw_obs.get("pose")
            if raw is None:
                out[offset : offset + 3] = 0.0
            else:
                arr = np.asarray(raw, dtype=np.float32).ravel()
                n = min(arr.shape[0], 3)
                out[offset : offset + n] = arr[0:n]
                if n < 3:
                    out[offset + n : offset + 3] = 0.0

    def compute(self, raw_obs: Dict, info: Dict) -> np.ndarray:
        out = np.empty(self.dim, dtype=np.float32)
        self.compute_into(raw_obs, info, out)
        return out
