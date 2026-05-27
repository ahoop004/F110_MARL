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

    def compute_into(self, raw_obs: Dict, info: Dict, out: np.ndarray) -> None:
        target_vel = raw_obs.get("target_velocity")
        if target_vel is None:
            central = raw_obs.get("central_state")
            if central is not None:
                arr = np.asarray(central, dtype=np.float32).ravel()
                if arr.shape[0] >= 6:
                    target_vel = arr[3:6]
                elif arr.shape[0] >= 3:
                    target_vel = arr[:3]

        if target_vel is None:
            out[:] = 0.0
            return

        arr = np.asarray(target_vel, dtype=np.float32).ravel()
        n = min(arr.shape[0], 3)
        out[0:n] = arr[0:n]
        if n < 3:
            out[n:] = 0.0

    def compute(self, raw_obs: Dict, info: Dict) -> np.ndarray:
        out = np.empty(3, dtype=np.float32)
        self.compute_into(raw_obs, info, out)
        return out
