"""Centerline reward component — speed along track with deviation penalty."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from wrappers.rewards.base import RewardComponent


class CenterlineRewardComponent(RewardComponent):
    """Reward forward progress along the centerline, penalize deviation and sharp steering."""

    def __init__(self, config: dict) -> None:
        self.vs_weight = float(config.get("vs_weight", 1.0))
        self.vd_weight = float(config.get("vd_weight", 0.01))
        self.d_weight = float(config.get("d_weight", 0.02))
        self.steer_weight = float(config.get("steer_weight", 0.05))
        self.normalize_by_track_length = bool(config.get("normalize_by_track_length", False))
        self.reference_length = float(config.get("reference_length", 400.0))
        self._track_length: Optional[float] = None

    def reset(self) -> None:
        self._track_length = None

    def compute(self, step_info: dict) -> Dict[str, float]:
        info = step_info.get("info") or {}
        cl = info.get("centerline", {}) if isinstance(info, dict) else {}

        vs = float(cl.get("vs", 0.0))
        vd = float(cl.get("vd", 0.0))
        d = float(cl.get("d", 0.0))

        scale = 1.0
        if self.normalize_by_track_length:
            if self._track_length is None:
                self._track_length = step_info.get("track_length") or self.reference_length
            scale = self.reference_length / max(self._track_length, 1.0)

        action = np.asarray(step_info.get("action", [0.0, 0.0]), dtype=np.float32).ravel()
        steer = float(action[0]) if len(action) > 0 else 0.0

        total = (
            self.vs_weight * vs * scale
            - self.vd_weight * abs(vd) * scale
            - self.d_weight * abs(d)
            - self.steer_weight * abs(steer)
        )
        return {"centerline/total": total}
