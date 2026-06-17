"""Centerline reward components."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from wrappers.rewards.base import RewardComponent


def _centerline_info(step_info: dict) -> dict:
    info = step_info.get("info") or {}
    cl = info.get("centerline", {}) if isinstance(info, dict) else {}
    return cl if isinstance(cl, dict) else {}


def _track_length_scale(
    step_info: dict,
    *,
    normalize_by_track_length: bool,
    reference_length: float,
    cached_track_length: Optional[float],
) -> tuple[float, Optional[float]]:
    if not normalize_by_track_length:
        return 1.0, cached_track_length
    track_length = cached_track_length
    if track_length is None:
        track_length = step_info.get("track_length") or reference_length
    return reference_length / max(float(track_length), 1.0), float(track_length)


def _steer_value(step_info: dict, *, steer_index: int = 0) -> float:
    action = np.asarray(step_info.get("action", [0.0, 0.0]), dtype=np.float32).ravel()
    return float(action[steer_index]) if len(action) > steer_index else 0.0


class CenterlineProgressComponent(RewardComponent):
    """Reward forward speed along the centerline."""

    def __init__(self, config: dict) -> None:
        self.weight = float(config.get("weight", config.get("vs_weight", 1.0)))
        self.normalize_by_track_length = bool(config.get("normalize_by_track_length", False))
        self.reference_length = float(config.get("reference_length", 400.0))
        self._track_length: Optional[float] = None

    def reset(self) -> None:
        self._track_length = None

    def compute(self, step_info: dict) -> Dict[str, float]:
        cl = _centerline_info(step_info)
        vs = float(cl.get("vs", 0.0))
        scale, self._track_length = _track_length_scale(
            step_info,
            normalize_by_track_length=self.normalize_by_track_length,
            reference_length=self.reference_length,
            cached_track_length=self._track_length,
        )
        return {"centerline_progress/bonus": self.weight * vs * scale}


class CenterlineLateralVelocityPenaltyComponent(RewardComponent):
    """Penalize speed perpendicular to the centerline tangent."""

    def __init__(self, config: dict) -> None:
        self.weight = float(config.get("weight", config.get("vd_weight", 0.01)))
        self.normalize_by_track_length = bool(config.get("normalize_by_track_length", False))
        self.reference_length = float(config.get("reference_length", 400.0))
        self._track_length: Optional[float] = None

    def reset(self) -> None:
        self._track_length = None

    def compute(self, step_info: dict) -> Dict[str, float]:
        cl = _centerline_info(step_info)
        vd = float(cl.get("vd", 0.0))
        scale, self._track_length = _track_length_scale(
            step_info,
            normalize_by_track_length=self.normalize_by_track_length,
            reference_length=self.reference_length,
            cached_track_length=self._track_length,
        )
        return {"centerline_lateral_velocity/penalty": -self.weight * abs(vd) * scale}


class CenterlineDeviationPenaltyComponent(RewardComponent):
    """Penalize lateral distance from the centerline."""

    def __init__(self, config: dict) -> None:
        self.weight = float(config.get("weight", config.get("d_weight", 0.02)))

    def compute(self, step_info: dict) -> Dict[str, float]:
        cl = _centerline_info(step_info)
        d = float(cl.get("d", 0.0))
        return {"centerline_deviation/penalty": -self.weight * abs(d)}


class SteeringPenaltyComponent(RewardComponent):
    """Penalize steering magnitude."""

    def __init__(self, config: dict) -> None:
        self.weight = float(config.get("weight", config.get("steer_weight", 0.05)))
        self.steer_index = int(config.get("steer_index", 0))

    def compute(self, step_info: dict) -> Dict[str, float]:
        steer = _steer_value(step_info, steer_index=self.steer_index)
        return {"steering/penalty": -self.weight * abs(steer)}


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
        cl = _centerline_info(step_info)

        vs = float(cl.get("vs", 0.0))
        vd = float(cl.get("vd", 0.0))
        d = float(cl.get("d", 0.0))

        scale, self._track_length = _track_length_scale(
            step_info,
            normalize_by_track_length=self.normalize_by_track_length,
            reference_length=self.reference_length,
            cached_track_length=self._track_length,
        )
        steer = _steer_value(step_info)

        total = (
            self.vs_weight * vs * scale
            - self.vd_weight * abs(vd) * scale
            - self.d_weight * abs(d)
            - self.steer_weight * abs(steer)
        )
        return {"centerline/total": total}
