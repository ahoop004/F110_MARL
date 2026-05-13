"""Relative pose to target vehicle — derived observation component."""
from __future__ import annotations

from typing import Dict

import numpy as np

from wrappers.observations.base import ObservationComponent


class RelativePoseComponent(ObservationComponent):
    """Relative pose from ego to target: [rel_x, rel_y, sin(Δθ), cos(Δθ), dist] — 5 dims."""

    @property
    def dim(self) -> int:
        return 5

    def compute(self, raw_obs: Dict, info: Dict) -> np.ndarray:
        ego_pose = np.asarray(raw_obs.get("pose", [0.0, 0.0, 0.0]), dtype=np.float64).ravel()
        target_pose = np.asarray(raw_obs.get("target_pose", [0.0, 0.0, 0.0]), dtype=np.float64).ravel()

        if len(ego_pose) < 3:
            ego_pose = np.pad(ego_pose, (0, 3 - len(ego_pose)))
        if len(target_pose) < 3:
            target_pose = np.pad(target_pose, (0, 3 - len(target_pose)))

        rel_x = float(target_pose[0] - ego_pose[0])
        rel_y = float(target_pose[1] - ego_pose[1])
        delta_theta = float(target_pose[2] - ego_pose[2])
        dist = float(np.sqrt(rel_x ** 2 + rel_y ** 2))

        return np.array(
            [rel_x, rel_y, np.sin(delta_theta), np.cos(delta_theta), dist],
            dtype=np.float32,
        )
