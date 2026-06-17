"""Track-geometry helpers for MPCC-style fixed-policy controllers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class CenterlineGeometry:
    points: np.ndarray
    segment_lengths: np.ndarray
    arc_lengths: np.ndarray
    total_length: float
    closed: bool = False

    @property
    def valid(self) -> bool:
        return self.points.shape[0] >= 2 and self.total_length > 0.0


@dataclass(frozen=True)
class CenterlineProjection:
    index: int
    segment_index: int
    arc_length: float
    progress: float
    contouring_error: float
    lag_error: float
    heading: float
    distance: float


def prepare_centerline_geometry(
    centerline: Optional[np.ndarray],
    *,
    closed: Optional[bool] = None,
    close_tolerance: float = 1e-3,
) -> CenterlineGeometry:
    """Normalize centerline points and cache cumulative arc lengths."""
    points = _normalize_centerline(centerline)
    if points.shape[0] < 2:
        return CenterlineGeometry(
            points=points,
            segment_lengths=np.zeros(0, dtype=np.float32),
            arc_lengths=np.zeros(points.shape[0], dtype=np.float32),
            total_length=0.0,
            closed=False,
        )

    inferred_closed = bool(
        np.linalg.norm(points[0] - points[-1]) <= float(close_tolerance)
    )
    is_closed = inferred_closed if closed is None else bool(closed)
    work_points = points
    if is_closed and not inferred_closed:
        work_points = np.vstack([points, points[0]])

    diffs = np.diff(work_points, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1).astype(np.float32)
    arc_lengths = np.concatenate(
        [np.zeros(1, dtype=np.float32), np.cumsum(segment_lengths, dtype=np.float32)]
    )
    total_length = float(arc_lengths[-1]) if arc_lengths.size else 0.0
    return CenterlineGeometry(
        points=work_points.astype(np.float32),
        segment_lengths=segment_lengths,
        arc_lengths=arc_lengths.astype(np.float32),
        total_length=total_length,
        closed=is_closed,
    )


def nearest_centerline_index(
    geometry: CenterlineGeometry,
    position: np.ndarray,
    *,
    last_index: Optional[int] = None,
    search_window: int = 80,
) -> int:
    """Return nearest point index, using a local window when possible."""
    if not geometry.valid:
        return 0
    pos = _normalize_position(position)
    n = geometry.points.shape[0]
    if last_index is None or last_index < 0 or last_index >= n:
        lo, hi = 0, n
    else:
        window = max(1, int(search_window))
        if geometry.closed:
            candidates = np.array(
                [wrap_centerline_index(last_index + off, n) for off in range(-window, window + 1)],
                dtype=np.int32,
            )
            dists = np.sum((geometry.points[candidates, :2] - pos) ** 2, axis=1)
            return int(candidates[int(np.argmin(dists))])
        lo = max(0, int(last_index) - window)
        hi = min(n, int(last_index) + window + 1)
    dists = np.sum((geometry.points[lo:hi, :2] - pos) ** 2, axis=1)
    return int(lo + int(np.argmin(dists)))


def project_to_centerline(
    geometry: CenterlineGeometry,
    position: np.ndarray,
    *,
    heading: Optional[float] = None,
    last_index: Optional[int] = None,
    search_window: int = 80,
) -> CenterlineProjection:
    """Project position to the closest centerline segment."""
    if not geometry.valid:
        return CenterlineProjection(
            index=0,
            segment_index=0,
            arc_length=0.0,
            progress=0.0,
            contouring_error=0.0,
            lag_error=0.0,
            heading=0.0,
            distance=0.0,
        )

    pos = _normalize_position(position)
    nearest = nearest_centerline_index(
        geometry,
        pos,
        last_index=last_index,
        search_window=search_window,
    )
    candidate_segments = _candidate_segments(geometry, nearest)
    best = None
    for seg_idx in candidate_segments:
        p0 = geometry.points[seg_idx, :2]
        p1 = geometry.points[seg_idx + 1, :2]
        delta = p1 - p0
        seg_len = float(np.linalg.norm(delta))
        if seg_len <= 1e-9:
            continue
        t = float(np.clip(np.dot(pos - p0, delta) / (seg_len * seg_len), 0.0, 1.0))
        projected = p0 + t * delta
        residual = pos - projected
        dist2 = float(np.dot(residual, residual))
        if best is None or dist2 < best[0]:
            best = (dist2, seg_idx, t, projected, delta, seg_len, residual)

    if best is None:
        return CenterlineProjection(
            index=nearest,
            segment_index=0,
            arc_length=float(geometry.arc_lengths[min(nearest, geometry.arc_lengths.size - 1)]),
            progress=0.0,
            contouring_error=0.0,
            lag_error=0.0,
            heading=local_tangent_heading(geometry, nearest),
            distance=0.0,
        )

    dist2, seg_idx, t, _projected, delta, seg_len, residual = best
    tangent = delta / max(seg_len, 1e-9)
    normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
    signed_contour = float(np.dot(residual, normal))
    lag = float(np.dot(residual, tangent))
    arc = float(geometry.arc_lengths[seg_idx] + t * seg_len)
    progress = arc / max(geometry.total_length, 1e-9)
    tangent_heading = float(np.arctan2(tangent[1], tangent[0]))
    if heading is not None:
        tangent_heading = float(tangent_heading)
    return CenterlineProjection(
        index=nearest,
        segment_index=int(seg_idx),
        arc_length=arc,
        progress=float(np.clip(progress, 0.0, 1.0)),
        contouring_error=signed_contour,
        lag_error=lag,
        heading=tangent_heading,
        distance=float(np.sqrt(dist2)),
    )


def contouring_error(
    geometry: CenterlineGeometry,
    position: np.ndarray,
    *,
    last_index: Optional[int] = None,
) -> float:
    return project_to_centerline(geometry, position, last_index=last_index).contouring_error


def lag_error(
    geometry: CenterlineGeometry,
    position: np.ndarray,
    *,
    last_index: Optional[int] = None,
) -> float:
    return project_to_centerline(geometry, position, last_index=last_index).lag_error


def heading_error(
    geometry: CenterlineGeometry,
    pose: np.ndarray,
    *,
    last_index: Optional[int] = None,
) -> float:
    arr = np.asarray(pose, dtype=np.float32).reshape(-1)
    if arr.size < 3:
        return 0.0
    projection = project_to_centerline(
        geometry,
        arr[:2],
        heading=float(arr[2]),
        last_index=last_index,
    )
    return wrap_angle(float(arr[2]) - projection.heading)


def progress_along_centerline(
    geometry: CenterlineGeometry,
    trajectory: Optional[np.ndarray],
    *,
    last_index: Optional[int] = None,
) -> float:
    """Approximate arc-length progress in metres over a trajectory."""
    if not geometry.valid or trajectory is None:
        return 0.0
    traj = np.asarray(trajectory, dtype=np.float32)
    if traj.ndim != 2 or traj.shape[0] < 2 or traj.shape[1] < 2:
        return 0.0
    start = project_to_centerline(geometry, traj[0, :2], last_index=last_index)
    end = project_to_centerline(geometry, traj[-1, :2], last_index=start.index)
    delta = end.arc_length - start.arc_length
    if geometry.closed and delta < -0.5 * geometry.total_length:
        delta += geometry.total_length
    elif geometry.closed and delta > 0.5 * geometry.total_length:
        delta -= geometry.total_length
    return float(max(0.0, delta))


def local_tangent_heading(geometry: CenterlineGeometry, index: int) -> float:
    if not geometry.valid:
        return 0.0
    idx = max(0, min(int(index), geometry.points.shape[0] - 2))
    delta = geometry.points[idx + 1, :2] - geometry.points[idx, :2]
    if np.allclose(delta, 0.0):
        return 0.0
    return float(np.arctan2(delta[1], delta[0]))


def wrap_centerline_index(index: int, n_points: int) -> int:
    n = max(1, int(n_points))
    return int(index) % n


def wrap_angle(angle: float) -> float:
    return float((float(angle) + np.pi) % (2.0 * np.pi) - np.pi)


def _candidate_segments(geometry: CenterlineGeometry, nearest_index: int) -> list[int]:
    max_seg = max(0, geometry.points.shape[0] - 2)
    candidates = {max(0, min(max_seg, nearest_index))}
    if nearest_index > 0:
        candidates.add(nearest_index - 1)
    if nearest_index < max_seg:
        candidates.add(nearest_index + 1)
    if geometry.closed:
        candidates.add(wrap_centerline_index(nearest_index - 1, max_seg + 1))
        candidates.add(wrap_centerline_index(nearest_index, max_seg + 1))
    return sorted(candidates)


def _normalize_centerline(centerline: Optional[np.ndarray]) -> np.ndarray:
    if centerline is None:
        return np.zeros((0, 2), dtype=np.float32)
    arr = np.asarray(centerline, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 2:
        return np.zeros((0, 2), dtype=np.float32)
    points = arr[:, :2].copy()
    mask = np.isfinite(points).all(axis=1)
    return points[mask].astype(np.float32)


def _normalize_position(position: np.ndarray) -> np.ndarray:
    arr = np.asarray(position, dtype=np.float32).reshape(-1)
    result = np.zeros(2, dtype=np.float32)
    if arr.size:
        result[: min(2, arr.size)] = arr[: min(2, arr.size)]
    result[~np.isfinite(result)] = 0.0
    return result


__all__ = [
    "CenterlineGeometry",
    "CenterlineProjection",
    "contouring_error",
    "heading_error",
    "lag_error",
    "local_tangent_heading",
    "nearest_centerline_index",
    "prepare_centerline_geometry",
    "progress_along_centerline",
    "project_to_centerline",
    "wrap_angle",
    "wrap_centerline_index",
]
