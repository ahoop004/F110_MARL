"""Target vehicle state observation component."""
from __future__ import annotations

from typing import Dict

import numpy as np

from wrappers.observations.base import ObservationComponent


class TargetStateComponent(ObservationComponent):
    """Opponent vehicle velocity: [vx, vy, yaw_rate] — 3 dims.

    Requires target_id to be set on the agent in the scenario so the env
    populates the central_state / target fields in the obs dict.
    """

    @property
    def dim(self) -> int:
        return 3

    def compute(self, raw_obs: Dict, info: Dict) -> np.ndarray:
        # env places target velocity under central_state or target_velocity
        target_vel = raw_obs.get("target_velocity")
        if target_vel is None:
            central = raw_obs.get("central_state")
            if central is not None:
                arr = np.asarray(central, dtype=np.float32).ravel()
                target_vel = arr[3:6] if len(arr) >= 6 else arr[:3]
        if target_vel is None:
            return np.zeros(3, dtype=np.float32)
        arr = np.asarray(target_vel, dtype=np.float32).ravel()
        return arr[:3].astype(np.float32) if len(arr) >= 3 else np.pad(arr, (0, 3 - len(arr)))
