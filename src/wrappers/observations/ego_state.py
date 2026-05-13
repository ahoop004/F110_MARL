"""Ego vehicle state observation component."""
from __future__ import annotations

from typing import Dict

import numpy as np

from wrappers.observations.base import ObservationComponent


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

    def compute(self, raw_obs: Dict, info: Dict) -> np.ndarray:
        parts = []
        if self._vel:
            vel = np.asarray(raw_obs.get("velocity", [0.0, 0.0, 0.0]), dtype=np.float32).ravel()
            parts.append(vel[:3] if len(vel) >= 3 else np.pad(vel, (0, 3 - len(vel))))
        if self._pose:
            pose = np.asarray(raw_obs.get("pose", [0.0, 0.0, 0.0]), dtype=np.float32).ravel()
            parts.append(pose[:3] if len(pose) >= 3 else np.pad(pose, (0, 3 - len(pose))))
        return np.concatenate(parts).astype(np.float32) if parts else np.zeros(0, dtype=np.float32)
