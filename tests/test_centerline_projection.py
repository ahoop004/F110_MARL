from __future__ import annotations

import numpy as np
import pytest

from utils.centerline import prepare_centerline_geometry, project_to_centerline
from utils.track_preview import TrackPreviewGeometry


def _numpy_projection_reference(geometry, position, heading, hint, window):
    """Original vectorized projection, independent of the compiled search."""
    size = geometry.segment_lengths.size
    if hint is None or not 0 <= hint < size:
        candidates = np.arange(size)
    elif geometry.closed:
        candidates = (hint + np.arange(-window, window + 1)) % size
    else:
        candidates = np.arange(max(hint - window, 0), min(hint + window + 1, size))
    starts = geometry.segment_starts[candidates]
    vectors = geometry.segment_vectors[candidates]
    lengths = geometry.segment_lengths[candidates]
    relative = position - starts
    fractions = np.clip(np.einsum('ij,ij->i', relative, vectors) / lengths**2, 0., 1.)
    residuals = position - (starts + fractions[:, None] * vectors)
    best = np.argmin(np.einsum('ij,ij->i', residuals, residuals))
    index = int(candidates[best])
    tangent = vectors[best] / lengths[best]
    normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
    tangent_heading = float(np.arctan2(tangent[1], tangent[0]))
    diff = float(heading) - tangent_heading
    arc = float(geometry.arc_lengths[index] + float(fractions[best]) * lengths[best])
    return dict(
        index=index, segment_index=index, arc_length=arc,
        lateral_error=float(np.dot(residuals[best], normal)),
        longitudinal_error=float(np.dot(residuals[best], tangent)),
        heading_error=float(np.arctan2(np.sin(diff), np.cos(diff))),
        tangent_heading=tangent_heading,
        progress=float(np.clip(arc / geometry.total_length, 0., 1.)),
    )


@pytest.mark.parametrize('closed', [False, True])
def test_compiled_search_exactly_matches_numpy_with_hints_ties_and_seams(closed):
    rng = np.random.default_rng(42)
    points = np.cumsum(rng.normal(size=(120, 2)), axis=0).astype(np.float32)
    if closed:
        points[-1] = points[0] + .01
    geometry = prepare_centerline_geometry(points)
    assert geometry.closed == closed
    positions = np.concatenate((points, rng.normal(size=(120, 2)).astype(np.float32) * 20))
    for i, position in enumerate(positions):
        hint = [None, -1, 0, 50, len(points) - 1, len(points)][i % 6]
        window = [0, 1, 50, 200][i % 4]
        heading = float(rng.uniform(-np.pi, np.pi))
        expected = _numpy_projection_reference(geometry, position, heading, hint, window)
        actual = project_to_centerline(geometry, position, heading, last_index=hint, search_window=window)
        assert vars(actual) == expected


def test_projection_uses_continuous_arc_length_on_nonuniform_segments() -> None:
    centerline = np.array(
        [[0.0, 0.0], [1.0, 0.0], [4.0, 0.0]], dtype=np.float32
    )

    projection = project_to_centerline(
        centerline,
        np.array([2.0, 1.0], dtype=np.float32),
        heading=0.0,
    )

    assert projection.segment_index == 1
    assert projection.arc_length == pytest.approx(2.0)
    assert projection.progress == pytest.approx(0.5)
    assert projection.lateral_error == pytest.approx(1.0)
    assert projection.longitudinal_error == pytest.approx(0.0)
    assert projection.heading_error == pytest.approx(0.0, abs=1e-7)


def test_closed_projection_wraps_local_search_across_finish_seam() -> None:
    centerline = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.1]],
        dtype=np.float32,
    )
    geometry = prepare_centerline_geometry(centerline)

    projection = project_to_centerline(
        geometry,
        np.array([-0.2, 0.05], dtype=np.float32),
        heading=-np.pi / 2.0,
        last_index=0,
        search_window=1,
    )

    assert geometry.closed
    assert projection.segment_index == 4
    assert projection.progress == pytest.approx(3.95 / 4.0)
    assert projection.lateral_error == pytest.approx(-0.2)
    assert projection.heading_error == pytest.approx(0.0, abs=1e-7)


def test_track_preview_starts_a_fixed_distance_ahead_of_projection() -> None:
    points = np.array(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
        dtype=np.float32,
    )
    geometry = TrackPreviewGeometry(
        points=points,
        curvature=np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
        width=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        spacing=1.0,
        closed=False,
        curvature_max=3.0,
        width_max=4.0,
        projection_geometry=prepare_centerline_geometry(points),
    )

    preview = geometry.preview(
        np.array([0.25, 0.0], dtype=np.float32),
        count=2,
    )

    assert preview["curvature"] == pytest.approx([1.25, 2.25])
    assert preview["width"] == pytest.approx([2.25, 3.25])
