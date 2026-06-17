"""Cost helpers for simple racing MPC utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from agents.mpc.track_geometry import (
    CenterlineGeometry,
    heading_error as track_heading_error,
    prepare_centerline_geometry,
    progress_along_centerline,
    project_to_centerline,
)


@dataclass(frozen=True)
class CostWeights:
    path_tracking: float = 1.0
    heading_error: float = 0.25
    target_speed: float = 0.5
    control_effort: float = 0.05
    steering_smoothness: float = 0.1
    progress: float = 0.0


@dataclass(frozen=True)
class MPCCWeights:
    contouring: float = 1.0
    lag: float = 0.1
    heading: float = 0.25
    progress: float = 1.0


def trajectory_cost(
    trajectory: Optional[np.ndarray],
    actions: Optional[np.ndarray],
    *,
    centerline: Optional[np.ndarray] = None,
    target_speed: Optional[float] = None,
    weights: CostWeights = CostWeights(),
) -> float:
    """Compute a weighted trajectory cost from factual geometry terms."""
    traj = _normalize_trajectory(trajectory)
    acts = _normalize_actions(actions)
    if traj.shape[0] == 0:
        return 0.0

    total = 0.0
    total += weights.path_tracking * path_tracking_cost(traj, centerline)
    total += weights.heading_error * heading_error_cost(traj, centerline)
    total += weights.target_speed * target_speed_cost(acts, target_speed)
    total += weights.control_effort * control_effort_cost(acts)
    total += weights.steering_smoothness * steering_smoothness_cost(acts)
    total -= weights.progress * progress_reward(traj, centerline)
    return float(total)


def mpcc_geometry_cost(
    trajectory: Optional[np.ndarray],
    centerline: Optional[np.ndarray | CenterlineGeometry],
    *,
    weights: MPCCWeights = MPCCWeights(),
) -> float:
    """Compute MPCC-style geometry cost over a candidate trajectory.

    This helper is dependency-free and uses centerline projection geometry:
    contouring/lateral error, lag/longitudinal error, heading error, and
    approximate forward arc-length progress.  It returns zero when geometry or
    trajectory inputs are missing/too short.
    """
    traj = _normalize_trajectory(trajectory)
    geometry = (
        centerline
        if isinstance(centerline, CenterlineGeometry)
        else prepare_centerline_geometry(centerline)
    )
    if traj.shape[0] == 0 or not geometry.valid:
        return 0.0

    contour_values = []
    lag_values = []
    heading_values = []
    last_index = None
    for pose in traj:
        projection = project_to_centerline(
            geometry,
            pose[:2],
            heading=float(pose[2]) if pose.shape[0] >= 3 else None,
            last_index=last_index,
        )
        last_index = projection.index
        contour_values.append(projection.contouring_error)
        lag_values.append(projection.lag_error)
        heading_values.append(track_heading_error(geometry, pose, last_index=projection.index))

    contour_arr = np.asarray(contour_values, dtype=np.float32)
    lag_arr = np.asarray(lag_values, dtype=np.float32)
    heading_arr = np.asarray(heading_values, dtype=np.float32)
    progress = progress_along_centerline(geometry, traj)
    total = (
        weights.contouring * float(np.mean(contour_arr * contour_arr))
        + weights.lag * float(np.mean(lag_arr * lag_arr))
        + weights.heading * float(np.mean(heading_arr * heading_arr))
        - weights.progress * float(progress)
    )
    return float(total)


def path_tracking_cost(
    trajectory: Optional[np.ndarray],
    centerline: Optional[np.ndarray],
) -> float:
    """Mean squared lateral distance to the nearest centerline point."""
    traj = _normalize_trajectory(trajectory)
    path = _normalize_centerline(centerline)
    if traj.shape[0] == 0 or path.shape[0] == 0:
        return 0.0

    points = traj[:, :2]
    nearest, _indices = _nearest_path_points(points, path)
    errors = points - nearest
    return float(np.mean(np.sum(errors * errors, axis=1)))


def heading_error_cost(
    trajectory: Optional[np.ndarray],
    centerline: Optional[np.ndarray],
) -> float:
    """Mean squared heading error relative to the nearest path tangent."""
    traj = _normalize_trajectory(trajectory)
    path = _normalize_centerline(centerline)
    if traj.shape[0] == 0 or path.shape[0] < 2:
        return 0.0

    _nearest, indices = _nearest_path_points(traj[:, :2], path)
    headings = np.array([_path_heading(path, int(idx)) for idx in indices], dtype=np.float32)
    errors = _wrap_angles(traj[:, 2] - headings)
    return float(np.mean(errors * errors))


def target_speed_cost(
    actions: Optional[np.ndarray],
    target_speed: Optional[float],
) -> float:
    """Mean squared speed error for ``[steering, speed]`` actions."""
    acts = _normalize_actions(actions)
    if acts.shape[0] == 0 or target_speed is None:
        return 0.0
    target = float(target_speed)
    if not np.isfinite(target):
        return 0.0
    errors = acts[:, 1] - target
    return float(np.mean(errors * errors))


def control_effort_cost(actions: Optional[np.ndarray]) -> float:
    """Mean squared action magnitude."""
    acts = _normalize_actions(actions)
    if acts.shape[0] == 0:
        return 0.0
    return float(np.mean(np.sum(acts * acts, axis=1)))


def steering_smoothness_cost(actions: Optional[np.ndarray]) -> float:
    """Mean squared steering-rate proxy across the candidate sequence."""
    acts = _normalize_actions(actions)
    if acts.shape[0] < 2:
        return 0.0
    deltas = np.diff(acts[:, 0])
    return float(np.mean(deltas * deltas))


def progress_reward(
    trajectory: Optional[np.ndarray],
    centerline: Optional[np.ndarray],
) -> float:
    """Approximate forward progress along a discrete centerline.

    Returns zero when no path is available.  This is a reward term, so
    ``trajectory_cost`` subtracts it when the progress weight is positive.
    """
    traj = _normalize_trajectory(trajectory)
    path = _normalize_centerline(centerline)
    if traj.shape[0] == 0 or path.shape[0] < 2:
        return 0.0

    _nearest, indices = _nearest_path_points(traj[[0, -1], :2], path)
    return float(max(0, int(indices[-1]) - int(indices[0])))


def _normalize_trajectory(trajectory: Optional[np.ndarray]) -> np.ndarray:
    if trajectory is None:
        return np.zeros((0, 3), dtype=np.float32)
    arr = np.asarray(trajectory, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return np.zeros((0, 3), dtype=np.float32)
    result = np.zeros((arr.shape[0], 3), dtype=np.float32)
    cols = min(3, arr.shape[1])
    result[:, :cols] = arr[:, :cols]
    result[~np.isfinite(result)] = 0.0
    return result


def _normalize_actions(actions: Optional[np.ndarray]) -> np.ndarray:
    if actions is None:
        return np.zeros((0, 2), dtype=np.float32)
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    result = np.zeros((arr.shape[0], 2), dtype=np.float32)
    cols = min(2, arr.shape[1])
    result[:, :cols] = arr[:, :cols]
    result[~np.isfinite(result)] = 0.0
    return result


def _normalize_centerline(centerline: Optional[np.ndarray]) -> np.ndarray:
    if centerline is None:
        return np.zeros((0, 2), dtype=np.float32)
    arr = np.asarray(centerline, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 2:
        return np.zeros((0, 2), dtype=np.float32)
    result = arr[:, :2].copy()
    mask = np.isfinite(result).all(axis=1)
    return result[mask]


def _nearest_path_points(points: np.ndarray, path: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    deltas = points[:, None, :] - path[None, :, :]
    dist2 = np.sum(deltas * deltas, axis=2)
    indices = np.argmin(dist2, axis=1)
    return path[indices], indices.astype(np.int32)


def _path_heading(path: np.ndarray, index: int) -> float:
    if path.shape[0] < 2:
        return 0.0
    if index >= path.shape[0] - 1:
        p0 = path[index - 1]
        p1 = path[index]
    else:
        p0 = path[index]
        p1 = path[index + 1]
    delta = p1 - p0
    return float(np.arctan2(delta[1], delta[0]))


def _wrap_angles(angles: np.ndarray) -> np.ndarray:
    return (angles + np.pi) % (2.0 * np.pi) - np.pi


__all__ = [
    "CostWeights",
    "MPCCWeights",
    "control_effort_cost",
    "heading_error_cost",
    "mpcc_geometry_cost",
    "path_tracking_cost",
    "progress_reward",
    "steering_smoothness_cost",
    "target_speed_cost",
    "trajectory_cost",
]
