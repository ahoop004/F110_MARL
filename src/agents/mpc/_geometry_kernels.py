"""Compiled MPC geometry; public projection helpers remain the reference.

Keep float32 rounding, local search windows, and first-minimum tie breaking
consistent with track_geometry.py. Do not enable fastmath: small cost changes
can change which discrete control candidate wins.
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def nearest_path_indices(trajectory, path):
    """Nearest discrete points without allocating a poses-by-path distance grid."""
    indices = np.empty(len(trajectory), dtype=np.int32)
    for i in range(len(trajectory)):
        best_distance, nearest = np.inf, 0
        for j in range(len(path)):
            dx = trajectory[i, 0] - path[j, 0]
            dy = trajectory[i, 1] - path[j, 1]
            distance = dx * dx + dy * dy
            if distance < best_distance:
                best_distance, nearest = distance, j
        indices[i] = nearest
    return indices


@njit(cache=True)
def _nearest_index(points, x, y, last_index, closed, window=80):
    n = len(points)
    wrap = closed and 0 <= last_index < n
    if last_index < 0 or last_index >= n:
        lo, hi = 0, n
    elif wrap:
        lo, hi = last_index - window, last_index + window + 1
    else:
        lo, hi = max(0, last_index - window), min(n, last_index + window + 1)
    best_distance, nearest = np.inf, 0
    for offset in range(lo, hi):
        index = offset % n if wrap else offset
        dx, dy = points[index, 0] - x, points[index, 1] - y
        distance = dx * dx + dy * dy
        if distance < best_distance:
            best_distance, nearest = distance, index
    return nearest


@njit(cache=True)
def _project_nearest(points, arcs, x, y, nearest, closed):
    max_segment = len(points) - 2
    # Match sorted(_candidate_segments(...)), including the closed seam.
    candidates = np.array([
        min(max_segment, nearest),
        nearest - 1 if nearest > 0 else -1,
        nearest + 1 if nearest < max_segment else -1,
        (nearest - 1) % (max_segment + 1) if closed else -1,
        nearest % (max_segment + 1) if closed else -1,
    ])
    candidates.sort()
    best_distance, best_segment = np.inf, -1
    arc, contour, lag, heading = 0.0, 0.0, 0.0, 0.0
    previous_segment = -1
    for segment in candidates:
        if segment < 0 or segment == previous_segment:
            continue
        previous_segment = segment
        dx = points[segment + 1, 0] - points[segment, 0]
        dy = points[segment + 1, 1] - points[segment, 1]
        length = np.float64(np.sqrt(dx * dx + dy * dy))
        if length <= 1e-9:
            continue
        ex, ey = x - points[segment, 0], y - points[segment, 1]
        fraction = (ex * dx + ey * dy) / np.float32(length * length)
        fraction = min(max(fraction, np.float32(0.0)), np.float32(1.0))
        rx = x - (points[segment, 0] + fraction * dx)
        ry = y - (points[segment, 1] + fraction * dy)
        distance = rx * rx + ry * ry
        if best_segment < 0 or distance < best_distance:
            best_distance, best_segment = distance, segment
            tx, ty = dx / np.float32(length), dy / np.float32(length)
            contour = np.float64(rx * -ty + ry * tx)
            lag = np.float64(rx * tx + ry * ty)
            arc = np.float64(np.float32(arcs[segment] + np.float32(np.float64(fraction) * length)))
            heading = np.float64(np.arctan2(ty, tx))
    if best_segment < 0:
        arc = np.float64(arcs[nearest])
        index = min(nearest, max_segment)
        dx = points[index + 1, 0] - points[index, 0]
        dy = points[index + 1, 1] - points[index, 1]
        if abs(dx) > 1e-8 or abs(dy) > 1e-8:
            heading = np.float64(np.arctan2(dy, dx))
    return arc, contour, lag, heading


@njit(cache=True)
def mpcc_geometry_terms(trajectory, points, arcs, total_length, closed):
    """Return geometry errors and progress with the reference search semantics.

    Heading normally reuses the projection. On a discontinuous trajectory the
    heading helper's recentered window can find a different nearest point;
    retain that behavior. Progress searches the endpoint from the *first*
    index, independently of the sequential contour projections.
    """
    errors = np.empty((3, len(trajectory)), dtype=np.float32)
    last_index, first_index = -1, -1
    first_arc = 0.0
    for i in range(len(trajectory)):
        x, y = trajectory[i, 0], trajectory[i, 1]
        nearest = _nearest_index(points, x, y, last_index, closed)
        arc, contour, lag, heading = _project_nearest(points, arcs, x, y, nearest, closed)
        if i == 0:
            first_index, first_arc = nearest, arc
        heading_index = _nearest_index(points, x, y, nearest, closed)
        if heading_index != nearest:
            heading = _project_nearest(points, arcs, x, y, heading_index, closed)[3]
        last_index = nearest
        errors[0, i], errors[1, i] = contour, lag
        delta = np.float64(trajectory[i, 2]) - heading
        errors[2, i] = (delta + np.pi) % (2.0 * np.pi) - np.pi
    progress = 0.0
    if len(trajectory) > 1:
        x, y = trajectory[-1, 0], trajectory[-1, 1]
        end_index = _nearest_index(points, x, y, first_index, closed)
        end_arc = _project_nearest(points, arcs, x, y, end_index, closed)[0]
        progress = end_arc - first_arc
        if closed and progress < -0.5 * total_length:
            progress += total_length
        elif closed and progress > 0.5 * total_length:
            progress -= total_length
    return errors, max(0.0, progress)
