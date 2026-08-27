"""Normalized vehicle/Frenet/track-preview observation."""
from __future__ import annotations

from typing import Dict, Mapping

import numpy as np

from wrappers.observations.base import ObservationComponent


_STATE_KEYS = (
    "vx",
    "vy",
    "u",
    "n",
    "r",
    "delta",
    "delta_ref",
    "omega_ref_dot",
    "omega_ref",
    "omega",
)

_DEFAULT_MAXIMA = {
    "vx": 10.0,
    "vy": 10.0,
    "u": float(np.pi),
    "n": 5.0,
    "r": 10.0,
    "delta": 0.46,
    "delta_ref": 0.46,
    "omega_ref_dot": 20000.0,
    "omega_ref": 200.0,
    "omega": 200.0,
}


class FrenetVehicleTrackComponent(ObservationComponent):
    """Observation ``[vx,vy,u,n,r,δ,δref,ωref_dot,ωref,ω,c[N],w[N]]``.

    Vehicle-state maxima are supplied by configuration. Curvature and width
    maxima are taken from the complete active track geometry and accompany the
    per-step preview in ``info["track_preview"]``.
    """

    def __init__(
        self,
        *,
        points: int,
        wheel_radius: float,
        maxima: Mapping[str, float] | None = None,
        clip: bool = False,
    ) -> None:
        self.points = max(int(points), 1)
        self.wheel_radius = max(float(wheel_radius), 1e-6)
        configured = dict(_DEFAULT_MAXIMA)
        configured.update(dict(maxima or {}))
        self._maxima = np.asarray(
            [max(abs(float(configured[key])), 1e-6) for key in _STATE_KEYS],
            dtype=np.float32,
        )
        self.clip = bool(clip)

    @property
    def dim(self) -> int:
        return 10 + 2 * self.points

    def compute_into(self, raw_obs: Dict, info: Dict, out: np.ndarray) -> None:
        velocity = _vector(raw_obs.get("velocity"), 2)
        centerline = info.get("centerline", {}) if isinstance(info, dict) else {}
        preview = info.get("track_preview", {}) if isinstance(info, dict) else {}
        wheel_radius = self.wheel_radius
        state = np.asarray(
            [
                velocity[0],
                velocity[1],
                _number(centerline.get("heading_error")),
                _number(centerline.get("d")),
                _number(raw_obs.get("angular_velocity")),
                _number(raw_obs.get("steering_angle")),
                _number(raw_obs.get("steering_reference")),
                _number(raw_obs.get("speed_reference_rate")) / wheel_radius,
                _number(raw_obs.get("speed_reference")) / wheel_radius,
                velocity[0] / wheel_radius,
            ],
            dtype=np.float32,
        )
        out[:10] = state / self._maxima

        curvature = _vector(preview.get("curvature"), self.points)
        width = _vector(preview.get("width"), self.points)
        curvature_max = max(abs(_number(preview.get("curvature_max"), 1.0)), 1e-6)
        width_max = max(abs(_number(preview.get("width_max"), 1.0)), 1e-6)
        out[10 : 10 + self.points] = curvature / curvature_max
        out[10 + self.points :] = width / width_max
        np.nan_to_num(out, copy=False)
        if self.clip:
            np.clip(out, -1.0, 1.0, out=out)


def _number(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _vector(value: object, size: int) -> np.ndarray:
    result = np.zeros(size, dtype=np.float32)
    if value is None:
        return result
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    count = min(size, array.size)
    result[:count] = array[:count]
    return result


__all__ = ["FrenetVehicleTrackComponent"]
