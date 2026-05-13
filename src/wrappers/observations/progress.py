"""Centerline progress observation component."""
from __future__ import annotations

from typing import Dict, Optional

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

    def compute(self, raw_obs: Dict, info: Dict) -> np.ndarray:
        cl_info = info.get("centerline", {}) if isinstance(info, dict) else {}
        progress = float(cl_info.get("progress", 0.0))
        deviation = float(cl_info.get("d", 0.0))
        return np.array([progress, deviation], dtype=np.float32)
