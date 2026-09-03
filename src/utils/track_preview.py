"""Uniform arc-length track previews for Frenet observations."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Callable, Mapping, Optional

import numpy as np

from .centerline import (
    CenterlineGeometry,
    prepare_centerline_geometry,
    project_to_centerline,
)


TRACK_PREVIEW_PREPROCESSING_VERSION = 1


@dataclass(frozen=True)
class TrackPreviewCacheKey:
    """Identity of every input that can affect preview preprocessing."""

    map_identity: str
    centerline_digest: str
    walls_digest: str
    spacing: float
    preprocessing_version: int = TRACK_PREVIEW_PREPROCESSING_VERSION


def _array_digest(array: Optional[np.ndarray]) -> str:
    digest = hashlib.sha256()
    if array is None:
        digest.update(b"none")
        return digest.hexdigest()
    canonical = np.ascontiguousarray(np.asarray(array, dtype=np.float32))
    digest.update(str(canonical.shape).encode("ascii"))
    digest.update(canonical.tobytes())
    return digest.hexdigest()


def build_track_preview_cache_key(
    *,
    map_identity: str | Path,
    centerline: Optional[np.ndarray],
    walls: Optional[Mapping[int, np.ndarray]],
    spacing: float,
) -> TrackPreviewCacheKey:
    """Return a deterministic content key for preview geometry inputs."""
    wall_digest = hashlib.sha256()
    for wall_id, wall in sorted((walls or {}).items(), key=lambda item: str(item[0])):
        wall_digest.update(str(wall_id).encode("utf-8"))
        wall_digest.update(_array_digest(wall).encode("ascii"))
    return TrackPreviewCacheKey(
        map_identity=str(Path(map_identity).expanduser().resolve()),
        centerline_digest=_array_digest(centerline),
        walls_digest=wall_digest.hexdigest(),
        spacing=max(float(spacing), 1e-3),
    )


class TrackPreviewGeometryCache:
    """Bounded per-environment LRU cache of immutable preview geometry."""

    def __init__(self, max_entries: int) -> None:
        self.max_entries = max(int(max_entries), 1)
        self._entries: OrderedDict[TrackPreviewCacheKey, TrackPreviewGeometry] = (
            OrderedDict()
        )

    def get_or_build(
        self,
        key: TrackPreviewCacheKey,
        centerline: Optional[np.ndarray],
        walls: Optional[Mapping[int, np.ndarray]],
        *,
        builder: Optional[
            Callable[..., Optional["TrackPreviewGeometry"]]
        ] = None,
    ) -> Optional["TrackPreviewGeometry"]:
        geometry = self._entries.get(key)
        if geometry is not None:
            self._entries.move_to_end(key)
            return geometry
        build = builder or TrackPreviewGeometry.build
        geometry = build(centerline, walls, spacing=key.spacing)
        if geometry is None:
            return None
        self._entries[key] = geometry
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
        return geometry

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)


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
    projection_geometry: CenterlineGeometry

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
        projection_geometry = prepare_centerline_geometry(points)
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
            projection_geometry=projection_geometry,
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
        projection = project_to_centerline(
            self.projection_geometry,
            np.asarray(position, dtype=np.float32).reshape(-1)[:2],
            0.0,
            last_index=start_index,
        )
        # The first sample is exactly one configured interval ahead of the
        # continuous vehicle projection, rather than one waypoint ahead.
        distances = projection.arc_length + self.spacing * np.arange(
            1, count + 1, dtype=np.float32
        )
        sample_arc = self.projection_geometry.arc_lengths
        if self.closed:
            sample_arc = sample_arc[:-1]
            distances %= self.projection_geometry.total_length
            interpolation_arc = np.append(
                sample_arc, self.projection_geometry.total_length
            )
            curvature_values = np.append(self.curvature, self.curvature[0])
            width_values = np.append(self.width, self.width[0])
        else:
            distances = np.clip(
                distances, 0.0, self.projection_geometry.total_length
            )
            interpolation_arc = sample_arc
            curvature_values = self.curvature
            width_values = self.width
        return {
            "curvature": np.interp(
                distances, interpolation_arc, curvature_values
            ).astype(np.float32),
            "width": np.interp(
                distances, interpolation_arc, width_values
            ).astype(np.float32),
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


__all__ = [
    "TRACK_PREVIEW_PREPROCESSING_VERSION",
    "TrackPreviewCacheKey",
    "TrackPreviewGeometry",
    "TrackPreviewGeometryCache",
    "build_track_preview_cache_key",
]
