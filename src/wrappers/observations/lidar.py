"""Lidar observation component."""
from __future__ import annotations

from typing import Dict

import numpy as np

from wrappers.observations.base import ObservationComponent


class LidarComponent(ObservationComponent):
    """Raw lidar scan — optionally normalized by lidar_range.

    Lidar beam count and range come from the env config; this component
    only decides whether to normalize them.
    """

    def __init__(self, n_beams: int, lidar_range: float, normalize: bool = True) -> None:
        self._n_beams = n_beams
        self._range = float(lidar_range)
        self._normalize = normalize

    @property
    def dim(self) -> int:
        return self._n_beams

    def compute(self, raw_obs: Dict, info: Dict) -> np.ndarray:
        scan = np.asarray(raw_obs.get("lidar", np.zeros(self._n_beams)), dtype=np.float32)
        if scan.shape[0] != self._n_beams:
            scan = np.resize(scan, (self._n_beams,))
        if self._normalize:
            scan = np.clip(scan / self._range, 0.0, 1.0)
        return scan.astype(np.float32)
