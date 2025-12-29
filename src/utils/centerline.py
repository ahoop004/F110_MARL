"""Helper utilities for working with track centerline waypoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from numba import njit


@dataclass
class CenterlineProjection:
    index: int
    lateral_error: float
    longitudinal_error: float
    heading_error: float
    progress: float


@njit(cache=True)
def _project_to_centerline_core(
    points: np.ndarray,
    position: np.ndarray,
    heading: float,
    tangent_theta: float,
    best_index: int,
) -> Tuple[float, float, float]:
    """JIT-compiled core computation for centerline projection.

    Args:
        points: Centerline waypoints (N, 2)
        position: Agent position (2,)
        heading: Agent heading in radians
        tangent_theta: Centerline heading at nearest point
        best_index: Index of nearest waypoint

    Returns:
        (lateral_error, longitudinal_error, heading_error)
    """
    nearest = points[best_index]

    tangent_cos = np.cos(tangent_theta)
    tangent_sin = np.sin(tangent_theta)
    tangent = np.array([tangent_cos, tangent_sin], dtype=np.float32)
    normal = np.array([-tangent_sin, tangent_cos], dtype=np.float32)

    offset_x = position[0] - nearest[0]
    offset_y = position[1] - nearest[1]

    lateral_error = offset_x * normal[0] + offset_y * normal[1]
    longitudinal_error = offset_x * tangent[0] + offset_y * tangent[1]

    heading_diff = heading - tangent_theta
    heading_error = np.arctan2(np.sin(heading_diff), np.cos(heading_diff))

    return lateral_error, longitudinal_error, heading_error


@njit(cache=True)
def _find_nearest_point(
    points: np.ndarray,
    position: np.ndarray,
    last_index: int,
    search_window: int,
) -> int:
    """JIT-compiled nearest point search.

    Args:
        points: Centerline waypoints (N, 2)
        position: Agent position (2,)
        last_index: Previous closest index (-1 for full search)
        search_window: Window size around last_index

    Returns:
        Index of nearest waypoint
    """
    N = points.shape[0]

    # Full search if no hint
    if last_index < 0 or last_index >= N:
        min_dist_sq = 1e10
        best_idx = 0
        for i in range(N):
            dx = points[i, 0] - position[0]
            dy = points[i, 1] - position[1]
            dist_sq = dx * dx + dy * dy
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                best_idx = i
        return best_idx

    # Local search with window
    start = max(0, last_index - search_window)
    end = min(N, last_index + search_window + 1)

    min_dist_sq = 1e10
    best_idx = last_index
    for i in range(start, end):
        dx = points[i, 0] - position[0]
        dy = points[i, 1] - position[1]
        dist_sq = dx * dx + dy * dy
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            best_idx = i

    return best_idx


def project_to_centerline(
    centerline: np.ndarray,
    position: np.ndarray,
    heading: float,
    *,
    last_index: Optional[int] = None,
    search_window: int = 50,
) -> CenterlineProjection:
    """Project a pose onto the closest point of the track centerline.

    Args:
        centerline: Array of waypoints shaped (N, 2 or 3). Columns are x, y, optional theta.
        position: Cartesian position (x, y).
        heading: Heading angle in radians.
        last_index: Optional hint of the previous closest waypoint index.
        search_window: Number of waypoints to search around ``last_index`` when provided.

    Returns:
        CenterlineProjection containing the closest waypoint index along with lateral/longitudinal
        errors, heading error (ego heading vs. tangent), and normalised progress in [0, 1].
    """

    if centerline.ndim != 2 or centerline.shape[0] == 0:
        raise ValueError("centerline must be a non-empty 2D array")

    points = centerline[:, :2].astype(np.float32, copy=False)
    if position.shape[0] != 2:
        raise ValueError("position must contain (x, y)")

    position_f32 = position.astype(np.float32, copy=False)

    # Use JIT-compiled nearest point search
    last_idx = -1 if last_index is None else int(last_index)
    best_index = _find_nearest_point(points, position_f32, last_idx, int(search_window))

    # Get tangent heading
    if centerline.shape[1] >= 3:
        tangent_theta = float(centerline[best_index, 2])
    else:
        tangent_theta = 0.0

    # Use JIT-compiled projection
    lateral_error, longitudinal_error, heading_error = _project_to_centerline_core(
        points, position_f32, float(heading), tangent_theta, best_index
    )

    progress = best_index / max(len(points) - 1, 1)

    return CenterlineProjection(
        index=best_index,
        lateral_error=float(lateral_error),
        longitudinal_error=float(longitudinal_error),
        heading_error=float(heading_error),
        progress=float(progress),
    )


def centerline_arc_length(centerline: np.ndarray) -> float:
    """Return the total arc length of a polyline centerline."""

    if centerline is None or centerline.ndim != 2 or centerline.shape[0] < 2:
        return 0.0

    points = centerline[:, :2].astype(np.float32, copy=False)
    diffs = np.diff(points, axis=0)
    if diffs.size == 0:
        return 0.0
    segment_lengths = np.linalg.norm(diffs, axis=1)
    return float(segment_lengths.sum())


def progress_from_spacing(
    centerline: np.ndarray,
    spacing: float,
    *,
    start_offset: float = 0.0,
) -> Tuple[float, ...]:
    """Compute progress fractions for evenly spaced distances along the track.

    Args:
        centerline: Waypoint array with at least 2 rows.
        spacing: Desired spacing in metres (must be > 0).
        start_offset: Optional initial offset before the first waypoint (metres).

    Returns:
        Tuple of monotonically increasing fractions in (0, 1) representing the
        requested spacing along the lap. Values at 0 or 1 are omitted.
    """

    try:
        spacing_val = float(spacing)
    except (TypeError, ValueError):
        return ()
    if spacing_val <= 0.0:
        return ()

    total_length = centerline_arc_length(centerline)
    if total_length <= 0.0:
        return ()

    try:
        offset_val = float(start_offset)
    except (TypeError, ValueError):
        offset_val = 0.0
    offset_val = max(float(offset_val), 0.0)

    cumulative = offset_val + spacing_val
    stops: List[float] = []
    while cumulative < total_length:
        progress = cumulative / total_length
        if 0.0 < progress < 1.0:
            stops.append(float(progress))
        cumulative += spacing_val

    if not stops:
        return ()
    # Deduplicate while preserving order.
    seen = set()
    unique: List[float] = []
    for value in stops:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return tuple(unique)
