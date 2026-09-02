from __future__ import annotations

import numpy as np
import pytest

from utils.centerline import prepare_centerline_geometry, project_to_centerline
from utils.track_preview import TrackPreviewGeometry


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
