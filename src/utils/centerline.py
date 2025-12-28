"""Helper utilities for working with track centerline waypoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class CenterlineProjection:
    index: int
    lateral_error: float
    longitudinal_error: float
    heading_error: float
    progress: float


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

    points = centerline[:, :2]
    if position.shape[0] != 2:
        raise ValueError("position must contain (x, y)")

    diffs = points - position.reshape(1, 2)
    dists = np.einsum("ij,ij->i", diffs, diffs)

    if last_index is not None and 0 <= last_index < len(points):
        window = max(int(search_window), 1)
        start = max(0, last_index - window)
        end = min(len(points), last_index + window + 1)
        local = dists[start:end]
        rel_idx = int(np.argmin(local))
        best_index = start + rel_idx
    else:
        best_index = int(np.argmin(dists))

    nearest = points[best_index]
    if centerline.shape[1] >= 3:
        tangent_theta = float(centerline[best_index, 2])
    else:
        tangent_theta = 0.0

    tangent = np.array([np.cos(tangent_theta), np.sin(tangent_theta)], dtype=np.float32)
    normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
    offset = position.astype(np.float32) - nearest.astype(np.float32)

    lateral_error = float(np.dot(offset, normal))
    longitudinal_error = float(np.dot(offset, tangent))
    heading_error = float(np.arctan2(np.sin(heading - tangent_theta), np.cos(heading - tangent_theta)))
    progress = best_index / max(len(points) - 1, 1)

    return CenterlineProjection(
        index=best_index,
        lateral_error=lateral_error,
        longitudinal_error=longitudinal_error,
        heading_error=heading_error,
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
