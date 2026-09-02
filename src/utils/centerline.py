"""Helper utilities for working with track centerline waypoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np


@dataclass
class CenterlineProjection:
    """Continuous projection of a Cartesian pose onto a centerline segment."""

    index: int
    segment_index: int
    arc_length: float
    lateral_error: float
    longitudinal_error: float
    heading_error: float
    tangent_heading: float
    progress: float


@dataclass(frozen=True)
class CenterlineGeometry:
    """Precomputed segment and arc-length data for repeated projections."""

    points: np.ndarray
    segment_starts: np.ndarray
    segment_vectors: np.ndarray
    segment_lengths: np.ndarray
    arc_lengths: np.ndarray
    total_length: float
    closed: bool

    @property
    def valid(self) -> bool:
        return self.segment_lengths.size > 0 and self.total_length > 0.0


def prepare_centerline_geometry(centerline: np.ndarray) -> CenterlineGeometry:
    """Build immutable geometry used by continuous Frenet projection."""
    array = np.asarray(centerline, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] < 2:
        raise ValueError("centerline must contain at least two (x, y) points")
    points = array[:, :2]
    points = points[np.isfinite(points).all(axis=1)]
    if points.shape[0] < 2:
        raise ValueError("centerline must contain at least two finite points")

    # Remove consecutive duplicates before determining topology. Real map
    # centerlines normally stop one sample short of repeating their first point.
    keep = np.concatenate(
        ([True], np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-7)
    )
    points = points[keep]
    open_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    typical_spacing = float(np.median(open_lengths)) if open_lengths.size else 0.0
    closing_length = float(np.linalg.norm(points[-1] - points[0]))
    closed = typical_spacing > 0.0 and closing_length <= 1.5 * typical_spacing

    ends = np.roll(points, -1, axis=0) if closed else points[1:]
    starts = points if closed else points[:-1]
    vectors = ends - starts
    lengths = np.linalg.norm(vectors, axis=1)
    valid = lengths > 1e-7
    starts = starts[valid]
    vectors = vectors[valid]
    lengths = lengths[valid]
    arc_lengths = np.concatenate(
        (np.zeros(1, dtype=np.float32), np.cumsum(lengths, dtype=np.float32))
    )
    total_length = float(arc_lengths[-1])
    for value in (points, starts, vectors, lengths, arc_lengths):
        value.setflags(write=False)
    return CenterlineGeometry(
        points=points,
        segment_starts=starts,
        segment_vectors=vectors,
        segment_lengths=lengths,
        arc_lengths=arc_lengths,
        total_length=total_length,
        closed=closed,
    )


def project_to_centerline(
    centerline: Union[np.ndarray, CenterlineGeometry],
    position: np.ndarray,
    heading: float,
    *,
    last_index: Optional[int] = None,
    search_window: int = 50,
) -> CenterlineProjection:
    """Project a pose onto the closest centerline segment.

    Args:
        centerline: Raw waypoints or geometry from :func:`prepare_centerline_geometry`.
        position: Cartesian position (x, y).
        heading: Heading angle in radians.
        last_index: Optional hint of the previous closest waypoint index.
        search_window: Number of waypoints to search around ``last_index`` when provided.

    Returns:
        A continuous arc-length projection with signed lateral error, heading
        error (ego heading vs. segment tangent), and progress in [0, 1].
    """
    geometry = (
        centerline
        if isinstance(centerline, CenterlineGeometry)
        else prepare_centerline_geometry(centerline)
    )
    if not geometry.valid:
        raise ValueError("centerline must contain a non-zero-length segment")
    position_array = np.asarray(position, dtype=np.float32).reshape(-1)
    if position_array.size != 2:
        raise ValueError("position must contain (x, y)")
    if not np.isfinite(position_array).all():
        raise ValueError("position must contain finite values")

    segment_count = geometry.segment_lengths.size
    if last_index is None or not 0 <= int(last_index) < segment_count:
        candidates = np.arange(segment_count, dtype=np.int64)
    elif geometry.closed:
        window = max(int(search_window), 0)
        offsets = np.arange(-window, window + 1, dtype=np.int64)
        candidates = (int(last_index) + offsets) % segment_count
    else:
        window = max(int(search_window), 0)
        start = max(int(last_index) - window, 0)
        stop = min(int(last_index) + window + 1, segment_count)
        candidates = np.arange(start, stop, dtype=np.int64)

    starts = geometry.segment_starts[candidates]
    vectors = geometry.segment_vectors[candidates]
    lengths = geometry.segment_lengths[candidates]
    relative = position_array - starts
    fractions = np.clip(
        np.einsum("ij,ij->i", relative, vectors) / np.square(lengths),
        0.0,
        1.0,
    )
    projected = starts + fractions[:, None] * vectors
    residuals = position_array - projected
    distances_sq = np.einsum("ij,ij->i", residuals, residuals)
    local_best = int(np.argmin(distances_sq))
    segment_index = int(candidates[local_best])
    fraction = float(fractions[local_best])
    tangent = vectors[local_best] / lengths[local_best]
    normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
    residual = residuals[local_best]
    lateral_error = float(np.dot(residual, normal))
    longitudinal_error = float(np.dot(residual, tangent))
    tangent_heading = float(np.arctan2(tangent[1], tangent[0]))
    heading_difference = float(heading) - tangent_heading
    heading_error = float(
        np.arctan2(np.sin(heading_difference), np.cos(heading_difference))
    )
    arc_length = float(
        geometry.arc_lengths[segment_index]
        + fraction * geometry.segment_lengths[segment_index]
    )
    progress = arc_length / geometry.total_length

    return CenterlineProjection(
        index=segment_index,
        segment_index=segment_index,
        arc_length=arc_length,
        lateral_error=lateral_error,
        longitudinal_error=longitudinal_error,
        heading_error=heading_error,
        tangent_heading=tangent_heading,
        progress=float(np.clip(progress, 0.0, 1.0)),
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


def centerline_heading(centerline: np.ndarray, index: int) -> float:
    """Return heading (theta) at a centerline index."""
    if centerline.ndim == 2 and centerline.shape[1] >= 3:
        theta = float(centerline[index, 2])
        if np.isfinite(theta):
            return theta
    points = centerline[:, :2].astype(np.float32, copy=False)
    prev_idx = max(0, index - 1)
    next_idx = min(points.shape[0] - 1, index + 1)
    delta = points[next_idx] - points[prev_idx]
    if np.allclose(delta, 0.0):
        return 0.0
    return float(np.arctan2(delta[1], delta[0]))


def centerline_pose(centerline: np.ndarray, index: int) -> np.ndarray:
    """Return pose [x, y, theta] at a centerline index."""
    point = np.asarray(centerline[index], dtype=np.float32).reshape(-1)
    theta = centerline_heading(centerline, index)
    return np.array([point[0], point[1], theta], dtype=np.float32)
