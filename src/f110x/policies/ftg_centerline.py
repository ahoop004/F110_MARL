from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np

from .gap_follow import FollowTheGapPolicy


def _wrap_angle(angle: float) -> float:
    wrapped = (angle + math.pi) % (2 * math.pi) - math.pi
    return wrapped


class FollowTheGapCenterlinePolicy(FollowTheGapPolicy):
    """Follow-the-gap with an added centerline tracking term."""

    CONFIG_DEFAULTS: Dict[str, Any] = dict(
        FollowTheGapPolicy.CONFIG_DEFAULTS,
        centerline_weight=0.5,
        lookahead_distance=2.0,
        centerline_tracking_gain=1.0,
        centerline_obstacle_min=4.0,
    )

    def __init__(
        self,
        centerline_weight: float = 0.5,
        lookahead_distance: float = 2.0,
        centerline_tracking_gain: float = 1.0,
        centerline_obstacle_min: float = 4.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.centerline_weight = float(np.clip(centerline_weight, 0.0, 1.0))
        self.lookahead_distance = max(float(lookahead_distance), 0.1)
        self.centerline_tracking_gain = float(centerline_tracking_gain)
        self.centerline_obstacle_min = max(float(centerline_obstacle_min), 0.1)
        self._centerline_points: Optional[np.ndarray] = None
        self._centerline_cumulative: Optional[np.ndarray] = None
        self._centerline_total: float = 0.0

    def set_centerline(self, points: Optional[np.ndarray]) -> None:
        if points is None:
            self._centerline_points = None
            self._centerline_cumulative = None
            return
        arr = np.asarray(points, dtype=np.float32)
        if arr.ndim < 2 or arr.shape[1] < 2:
            self._centerline_points = None
            self._centerline_cumulative = None
            return
        pts = arr[:, :2]
        diffs = np.diff(pts, axis=0)
        seg_len = np.linalg.norm(diffs, axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(seg_len)))
        total = float(cumulative[-1])
        if total <= 0.0:
            self._centerline_points = None
            self._centerline_cumulative = None
            return
        self._centerline_points = pts
        self._centerline_cumulative = cumulative
        self._centerline_total = total

    def get_action(self, action_space, obs: Dict[str, Any]):
        action = super().get_action(action_space, obs)
        if self._centerline_points is None:
            return action
        steering_cl = self._centerline_steering(obs)
        if steering_cl is None:
            return action
        weight = self._centerline_weight(obs)
        if weight <= 0.0:
            return action
        blended = (1.0 - weight) * action[0] + weight * steering_cl
        action[0] = float(np.clip(blended, -self.max_steer, self.max_steer))
        return action

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _centerline_weight(self, obs: Dict[str, Any]) -> float:
        scan = np.asarray(obs.get("scans", ()), dtype=np.float32)
        if scan.size == 0:
            return self.centerline_weight
        ranges = scan * self.max_distance if self.normalized else scan
        min_scan = float(np.min(ranges))
        if not np.isfinite(min_scan):
            return self.centerline_weight
        if min_scan <= 0.0:
            return 0.0
        weight = self.centerline_weight
        if min_scan < self.centerline_obstacle_min:
            weight *= float(np.clip(min_scan / self.centerline_obstacle_min, 0.0, 1.0))
        return float(np.clip(weight, 0.0, 1.0))

    def _centerline_steering(self, obs: Dict[str, Any]) -> Optional[float]:
        pose = obs.get("pose")
        if pose is None or self._centerline_points is None:
            return None
        pose_arr = np.asarray(pose, dtype=np.float32).reshape(-1)
        if pose_arr.size < 3:
            return None
        target = self._lookup_centerline_target(pose_arr[:2])
        if target is None:
            return None
        desired_heading = math.atan2(float(target[1] - pose_arr[1]), float(target[0] - pose_arr[0]))
        heading_error = _wrap_angle(desired_heading - float(pose_arr[2]))
        steer = self.centerline_tracking_gain * heading_error
        return float(np.clip(steer, -self.max_steer, self.max_steer))

    def _lookup_centerline_target(self, position: np.ndarray) -> Optional[np.ndarray]:
        pts = self._centerline_points
        cumulative = self._centerline_cumulative
        total = self._centerline_total
        if pts is None or cumulative is None or pts.shape[0] == 0 or total <= 0.0:
            return None
        deltas = pts - position[:2]
        idx = int(np.argmin(np.sum(deltas * deltas, axis=1)))
        target_distance = cumulative[idx] + self.lookahead_distance
        if target_distance > total:
            target_distance -= total * math.floor(target_distance / total)
        target_idx = int(np.searchsorted(cumulative, target_distance, side="left"))
        target_idx = min(target_idx, pts.shape[0] - 1)
        return pts[target_idx]
