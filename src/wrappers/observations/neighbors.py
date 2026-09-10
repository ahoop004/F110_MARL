"""Target-relative state and nearby-agent Frenet observations."""
from __future__ import annotations

import math
from typing import Dict, Mapping

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


class RelativePoseComponent(ObservationComponent):
    """Relative pose from ego to target: [rel_x, rel_y, sin(Δθ), cos(Δθ), dist] — 5 dims.

    Uses ``float32`` throughout (no intermediate float64 conversion) and the
    ``math`` module for scalar sin/cos/sqrt — ~4× faster than the numpy
    equivalents for single values.
    """

    @property
    def dim(self) -> int:
        return 5

    def compute_into(self, raw_obs: Dict, info: Dict, out: np.ndarray) -> None:
        ego_raw = raw_obs.get("pose")
        tgt_raw = raw_obs.get("target_pose")

        if ego_raw is None:
            ea = _ZERO_F32
        else:
            ea = np.asarray(ego_raw, dtype=np.float32).ravel()
            if ea.shape[0] < 3:
                ea = np.pad(ea, (0, 3 - ea.shape[0]))

        if tgt_raw is None:
            ta = _ZERO_F32
        else:
            ta = np.asarray(tgt_raw, dtype=np.float32).ravel()
            if ta.shape[0] < 3:
                ta = np.pad(ta, (0, 3 - ta.shape[0]))

        # Scalar arithmetic — avoids numpy ufunc dispatch overhead.
        rel_x = float(ta[0]) - float(ea[0])
        rel_y = float(ta[1]) - float(ea[1])
        delta_theta = float(ta[2]) - float(ea[2])

        out[0] = rel_x
        out[1] = rel_y
        out[2] = math.sin(delta_theta)
        out[3] = math.cos(delta_theta)
        out[4] = math.sqrt(rel_x * rel_x + rel_y * rel_y)

    def compute(self, raw_obs: Dict, info: Dict) -> np.ndarray:
        out = np.empty(5, dtype=np.float32)
        self.compute_into(raw_obs, info, out)
        return out


_ZERO_F32 = np.zeros(3, dtype=np.float32)


_FIELDS = ("delta_s", "delta_d", "delta_vs", "delta_vd")
_DEFAULT_MAXIMA = {
    "delta_s": 20.0,
    "delta_d": 5.0,
    "delta_vs": 20.0,
    "delta_vd": 10.0,
}


class FrenetNeighborsComponent(ObservationComponent):
    """Nearest-agent slots ``[Δs, Δd, Δvs, Δvd, present]``.

    Neighbors are ordered by absolute wrapped longitudinal distance, with
    agent ID used only as a deterministic tie-breaker. Missing slots are zero.
    """

    def __init__(
        self,
        *,
        max_neighbors: int,
        maxima: Mapping[str, float] | None = None,
        clip: bool = False,
    ) -> None:
        self.max_neighbors = max(int(max_neighbors), 1)
        configured = dict(_DEFAULT_MAXIMA)
        configured.update(dict(maxima or {}))
        self._maxima = np.asarray(
            [max(abs(float(configured[field])), 1e-6) for field in _FIELDS],
            dtype=np.float32,
        )
        self.clip = bool(clip)

    @property
    def dim(self) -> int:
        return 5 * self.max_neighbors

    def compute_into(self, raw_obs: Dict, info: Dict, out: np.ndarray) -> None:
        out.fill(0.0)
        neighbors = info.get("frenet_neighbors", []) if isinstance(info, dict) else []
        if not isinstance(neighbors, (list, tuple)):
            return
        for slot, neighbor in enumerate(neighbors[: self.max_neighbors]):
            if not isinstance(neighbor, Mapping):
                continue
            start = 5 * slot
            values = np.asarray(
                [_finite_number(neighbor.get(field)) for field in _FIELDS],
                dtype=np.float32,
            )
            out[start : start + 4] = values / self._maxima
            out[start + 4] = 1.0
        np.nan_to_num(out, copy=False)
        if self.clip:
            np.clip(out, -1.0, 1.0, out=out)


def _finite_number(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if np.isfinite(result) else 0.0


__all__ = ["TargetStateComponent", "RelativePoseComponent", "FrenetNeighborsComponent"]
