"""Centerline progress observation component."""
from __future__ import annotations

from typing import Dict

import numpy as np

from wrappers.observations.base import ObservationComponent


class ProgressComponent(ObservationComponent):
    """Normalized lap progress [0, 1] and cross-track deviation — 2 dims.

    Requires centerline to be loaded (centerline_autoload: true in env config).
    Progress and deviation are sourced from the step info dict.
    """

    @property
    def dim(self) -> int:
        return 2

    def compute_into(self, raw_obs: Dict, info: Dict, out: np.ndarray) -> None:
        cl_info = info.get("centerline", {}) if isinstance(info, dict) else {}
        out[0] = float(cl_info.get("progress", 0.0))
        out[1] = float(cl_info.get("d", 0.0))

    def compute(self, raw_obs: Dict, info: Dict) -> np.ndarray:
        out = np.empty(2, dtype=np.float32)
        self.compute_into(raw_obs, info, out)
        return out
