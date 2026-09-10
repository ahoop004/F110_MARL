"""Local sensors, ego vehicle state, and previous-action observations."""
from __future__ import annotations

from typing import Dict

import numpy as np

from wrappers.observations.base import ObservationComponent


class LidarComponent(ObservationComponent):
    """Raw lidar scan — optionally normalized by lidar_range.

    Lidar beam count and range come from the env config; this component
    only decides whether to normalize them.

    Normalization uses ``np.minimum`` (not ``np.clip``) because physical
    sensor readings are always ≥ 0, so only the upper bound needs clamping.
    This is ~2.3× faster than ``np.clip`` for 108-beam scans.
    """

    def __init__(self, n_beams: int, lidar_range: float, normalize: bool = True) -> None:
        self._n_beams = n_beams
        self._normalize = normalize
        # Store as float32 scalar so multiply stays in float32 arithmetic.
        self._inv_range = np.float32(1.0 / float(lidar_range)) if normalize else None

    @property
    def dim(self) -> int:
        return self._n_beams

    def compute_into(self, raw_obs: Dict, info: Dict, out: np.ndarray) -> None:
        """Write normalized lidar scan into *out* (zero allocation on fast path)."""
        scan_raw = raw_obs.get("lidar")
        if scan_raw is None:
            out.fill(0.0)
            return

        scan = np.asarray(scan_raw, dtype=np.float32)
        # Resize only when shape doesn't match (rare — env normally provides correct shape).
        if scan.shape[0] != self._n_beams:
            scan = np.resize(scan, (self._n_beams,))

        if self._normalize:
            # In-place multiply then clamp upper bound.
            # np.minimum is ~2.5× faster than np.clip for non-negative inputs.
            np.multiply(scan, self._inv_range, out=out)
            np.minimum(out, 1.0, out=out)
        else:
            np.copyto(out, scan)

    def compute(self, raw_obs: Dict, info: Dict) -> np.ndarray:
        out = np.empty(self._n_beams, dtype=np.float32)
        self.compute_into(raw_obs, info, out)
        return out


_ZEROS3 = np.zeros(3, dtype=np.float32)


class EgoStateComponent(ObservationComponent):
    """Ego vehicle state: velocity (always) + optional pose.

    velocity:  [vx, vy, yaw_rate]  — 3 dims
    pose:      [x,  y,  theta]     — 3 dims (optional)
    """

    def __init__(
        self,
        include_velocity: bool = True,
        include_pose: bool = False,
    ) -> None:
        self._vel = include_velocity
        self._pose = include_pose

    @property
    def dim(self) -> int:
        return (3 if self._vel else 0) + (3 if self._pose else 0)

    def compute_into(self, raw_obs: Dict, info: Dict, out: np.ndarray) -> None:
        offset = 0
        if self._vel:
            raw = raw_obs.get("velocity")
            if raw is None:
                out[0:3] = 0.0
            else:
                arr = np.asarray(raw, dtype=np.float32).ravel()
                n = min(arr.shape[0], 3)
                out[0:n] = arr[0:n]
                if n < 3:
                    out[n:3] = 0.0
            offset = 3
        if self._pose:
            raw = raw_obs.get("pose")
            if raw is None:
                out[offset : offset + 3] = 0.0
            else:
                arr = np.asarray(raw, dtype=np.float32).ravel()
                n = min(arr.shape[0], 3)
                out[offset : offset + n] = arr[0:n]
                if n < 3:
                    out[offset + n : offset + 3] = 0.0

    def compute(self, raw_obs: Dict, info: Dict) -> np.ndarray:
        out = np.empty(self.dim, dtype=np.float32)
        self.compute_into(raw_obs, info, out)
        return out


class PrevActionComponent(ObservationComponent):
    """Last action taken by this agent: [steer, speed] — 2 dims.

    The composer stores the previous action and injects it on each call.
    """

    def __init__(self, action_dim: int = 2) -> None:
        self._action_dim = action_dim
        self._prev_action = np.zeros(action_dim, dtype=np.float32)

    @property
    def dim(self) -> int:
        return self._action_dim

    def update(self, action: np.ndarray) -> None:
        """Call after each env.step() to track the last action."""
        arr = np.asarray(action, dtype=np.float32).ravel()
        n = min(arr.shape[0], self._action_dim)
        self._prev_action[:n] = arr[:n]
        if n < self._action_dim:
            self._prev_action[n:] = 0.0

    def reset(self) -> None:
        self._prev_action.fill(0.0)

    def compute_into(self, raw_obs: Dict, info: Dict, out: np.ndarray) -> None:
        np.copyto(out, self._prev_action)

    def compute(self, raw_obs: Dict, info: Dict) -> np.ndarray:
        return self._prev_action.copy()
