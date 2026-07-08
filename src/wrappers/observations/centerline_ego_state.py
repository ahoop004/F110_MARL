"""Track-relative ego state observation component."""
from __future__ import annotations

import math
from typing import Dict

import numpy as np

from wrappers.observations.base import ObservationComponent


class CenterlineEgoStateComponent(ObservationComponent):
    """Ego dynamics expressed in the track (Frenet) frame — 3 dims.

    Reads from ``info["centerline"]`` which is populated each step when
    ``centerline_features: true`` is set in the environment config.

    Output vector:
        [vs, vd, heading_error]

    vs
        Speed along the track tangent (m/s).  Positive = forward progress.
    vd
        Speed perpendicular to the track tangent (m/s).  Positive = drifting
        left.  Indicates wasted lateral velocity.
    heading_error
        Angle between the car's heading and the track tangent, normalised to
        [-pi, pi].  Zero means the car is perfectly aligned with the track.
    """

    @property
    def dim(self) -> int:
        return 3

    def compute_into(self, raw_obs: Dict, info: Dict, out: np.ndarray) -> None:
        cl = info.get("centerline", {}) if isinstance(info, dict) else {}
        out[0] = float(cl.get("vs", 0.0))
        out[1] = float(cl.get("vd", 0.0))
        out[2] = float(cl.get("heading_error", 0.0))

    def compute(self, raw_obs: Dict, info: Dict) -> np.ndarray:
        out = np.empty(self.dim, dtype=np.float32)
        self.compute_into(raw_obs, info, out)
        return out
