"""Uniform arc-length track previews for Frenet observations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np


@dataclass(frozen=True)
class TrackPreviewGeometry:
    """Centerline curvature and track width sampled at uniform arc length."""

    points: np.ndarray
    curvature: np.ndarray
    width: np.ndarray
    spacing: float
    closed: bool
    curvature_max: float
    width_max: float

    @classmethod
    def build(
        cls,
        centerline: Optional[np.ndarray],
        walls: Optional[Mapping[int, np.ndarray]],
        *,
        spacing: float = 0.3,
    ) -> Optional["TrackPreviewGeometry"]:
        path = _valid_path(centerline)
        if path is None:
            return None
        spacing = max(float(spacing), 1e-3)
        points, closed = _resample_uniform(path, spacing)
        curvature = _curvature(points, spacing, closed)
        width = _track_width(points, walls, closed)
        curvature_max = max(float(np.max(np.abs(curvature))), 1e-6)
        width_max = max(float(np.max(width)), 1e-6)
        for array in (points, curvature, width):
            array.setflags(write=False)
        return cls(
            points=points,
            curvature=curvature,
            width=width,
            spacing=spacing,
            closed=closed,
            curvature_max=curvature_max,
            width_max=width_max,
        )

    def nearest_index(
        self,
        position: np.ndarray,
        *,
        last_index: Optional[int] = None,
        search_window: int = 50,
    ) -> int:
        """Find the closest sample, using a seam-aware local search when possible."""
        pos = np.asarray(position, dtype=np.float32).reshape(-1)
        if pos.size < 2 or not np.isfinite(pos[:2]).all():
            return 0
        size = self.points.shape[0]
        if last_index is None or last_index < 0 or last_index >= size:
            indices = np.arange(size, dtype=np.int64)
        elif self.closed:
            offsets = np.arange(-search_window, search_window + 1, dtype=np.int64)
            indices = (int(last_index) + offsets) % size
        else:
            start = max(int(last_index) - search_window, 0)
            stop = min(int(last_index) + search_window + 1, size)
            indices = np.arange(start, stop, dtype=np.int64)
        delta = self.points[indices] - pos[:2]
        return int(indices[int(np.argmin(np.einsum("ij,ij->i", delta, delta)))])

    def preview(
        self,
        position: np.ndarray,
        count: int,
        *,
        start_index: Optional[int] = None,
    ) -> dict[str, np.ndarray | float]:
        """Return *count* samples beginning one interval ahead of the vehicle."""
        count = max(int(count), 1)
        start = self.nearest_index(position) if start_index is None else int(start_index)
        # The first sample is one spacing interval in front of the vehicle.
        indices = start + 1 + np.arange(count, dtype=np.int64)
        if self.closed:
            indices %= self.points.shape[0]
        else:
            indices = np.clip(indices, 0, self.points.shape[0] - 1)
        return {
            "curvature": self.curvature[indices].astype(np.float32, copy=True),
            "width": self.width[indices].astype(np.float32, copy=True),
            "curvature_max": self.curvature_max,
            "width_max": self.width_max,
        }


def _valid_path(centerline: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if centerline is None:
        return None
    path = np.asarray(centerline, dtype=np.float32)
    if path.ndim != 2 or path.shape[0] < 3 or path.shape[1] < 2:
        return None
    path = path[:, :2]
    path = path[np.isfinite(path).all(axis=1)]
    return path if path.shape[0] >= 3 else None


def _resample_uniform(path: np.ndarray, spacing: float) -> tuple[np.ndarray, bool]:
    segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    positive = segment_lengths[segment_lengths > 1e-6]
    typical = float(np.median(positive)) if positive.size else spacing
    closed = float(np.linalg.norm(path[-1] - path[0])) <= max(3.0 * typical, 2.0 * spacing)
    source = np.vstack((path, path[0])) if closed else path
    lengths = np.linalg.norm(np.diff(source, axis=0), axis=1)
    keep = np.concatenate(([True], lengths > 1e-6))
    source = source[keep]
    cumulative = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(source, axis=0), axis=1))))
    total = float(cumulative[-1])
    if total <= 1e-6:
        return path[:1].copy(), False
    distances = np.arange(0.0, total if closed else total + 0.5 * spacing, spacing)
    if distances.size < 3:
        distances = np.linspace(0.0, total, 3, endpoint=not closed)
    x = np.interp(distances, cumulative, source[:, 0])
    y = np.interp(distances, cumulative, source[:, 1])
    return np.column_stack((x, y)).astype(np.float32), closed


def _curvature(points: np.ndarray, spacing: float, closed: bool) -> np.ndarray:
    if closed:
        previous = np.roll(points, 1, axis=0)
        following = np.roll(points, -1, axis=0)
        tangent = following - previous
        headings = np.unwrap(np.arctan2(tangent[:, 1], tangent[:, 0]))
        delta = np.arctan2(
            np.sin(np.roll(headings, -1) - np.roll(headings, 1)),
            np.cos(np.roll(headings, -1) - np.roll(headings, 1)),
        )
        return (delta / (2.0 * spacing)).astype(np.float32)
    tangent = np.gradient(points, spacing, axis=0)
    headings = np.unwrap(np.arctan2(tangent[:, 1], tangent[:, 0]))
    return np.gradient(headings, spacing).astype(np.float32)


def _track_width(
    points: np.ndarray,
    walls: Optional[Mapping[int, np.ndarray]],
    closed: bool,
) -> np.ndarray:
    wall_arrays = [
        np.asarray(value, dtype=np.float32)[:, :2]
        for value in (walls or {}).values()
        if np.asarray(value).ndim == 2 and np.asarray(value).shape[0] >= 2
    ]
    if not wall_arrays:
        return np.ones(points.shape[0], dtype=np.float32)

    if closed:
        tangent = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    else:
        tangent = np.gradient(points, axis=0)
    norm = np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-6)
    normals = np.column_stack((-tangent[:, 1], tangent[:, 0])) / norm

    segment_starts = np.vstack([wall for wall in wall_arrays])
    segment_ends = np.vstack([np.roll(wall, -1, axis=0) for wall in wall_arrays])
    segment_vec = segment_ends - segment_starts
    widths = np.empty(points.shape[0], dtype=np.float32)
    all_wall_points = segment_starts
    for idx, (point, normal) in enumerate(zip(points, normals)):
        # Solve point + t*normal = segment_start + u*segment_vec.
        rel = segment_starts - point
        denominator = _cross_2d(normal, segment_vec)
        valid = np.abs(denominator) > 1e-8
        t = np.full(denominator.shape, np.nan, dtype=np.float32)
        u = np.full(denominator.shape, np.nan, dtype=np.float32)
        t[valid] = _cross_2d(rel[valid], segment_vec[valid]) / denominator[valid]
        u[valid] = _cross_2d(rel[valid], normal) / denominator[valid]
        hits = t[(u >= 0.0) & (u <= 1.0) & np.isfinite(t)]
        positive = hits[hits > 0.0]
        negative = hits[hits < 0.0]
        if positive.size and negative.size:
            widths[idx] = float(np.min(positive) - np.max(negative))
        else:
            # Geometry fallback for incomplete/open wall polylines.
            nearest = float(np.min(np.linalg.norm(all_wall_points - point, axis=1)))
            widths[idx] = 2.0 * nearest
    return np.maximum(widths, 1e-3)


def _cross_2d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


__all__ = ["TrackPreviewGeometry"]
