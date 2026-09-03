from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from utils.centerline import prepare_centerline_geometry, project_to_centerline
from utils.map_loader import MapLoader
from utils.track_preview import TrackPreviewGeometry


CONFIGURED_MAPS = (
    "Budapest_map",
    "circle_map",
    "Melbourne_map",
    "Montreal_map",
    "Shanghai_map",
    "Silverstone_map",
    "Spa_map",
    "Spielberg_map",
)


def _legacy_preview(
    geometry: TrackPreviewGeometry,
    position: np.ndarray,
    count: int,
    *,
    start_index: int | None = None,
) -> dict[str, np.ndarray]:
    projection = project_to_centerline(
        geometry.projection_geometry,
        np.asarray(position, dtype=np.float32).reshape(-1)[:2],
        0.0,
        last_index=start_index,
    )
    distances = projection.arc_length + geometry.spacing * np.arange(
        1, count + 1, dtype=np.float32
    )
    sample_arc = geometry.projection_geometry.arc_lengths
    if geometry.closed:
        sample_arc = sample_arc[:-1]
        distances %= geometry.projection_geometry.total_length
        interpolation_arc = np.append(
            sample_arc, geometry.projection_geometry.total_length
        )
        curvature_values = np.append(geometry.curvature, geometry.curvature[0])
        width_values = np.append(geometry.width, geometry.width[0])
    else:
        distances = np.clip(
            distances, 0.0, geometry.projection_geometry.total_length
        )
        interpolation_arc = sample_arc
        curvature_values = geometry.curvature
        width_values = geometry.width
    return {
        "curvature": np.interp(
            distances, interpolation_arc, curvature_values
        ).astype(np.float32),
        "width": np.interp(distances, interpolation_arc, width_values).astype(
            np.float32
        ),
    }


def test_precomputed_interpolation_matches_legacy_on_every_configured_map() -> None:
    loader = MapLoader(base_dir=Path.cwd())
    for map_name in CONFIGURED_MAPS:
        map_data = loader.load(
            {
                "map_dir": "maps",
                "map_bundle": map_name,
                "centerline_autoload": True,
                "walls_autoload": True,
            }
        )
        geometry = TrackPreviewGeometry.build(
            map_data.centerline, map_data.walls, spacing=0.3
        )
        assert geometry is not None
        positions = (
            geometry.points[0] + np.array([0.0, 0.02], dtype=np.float32),
            geometry.points[-1] + np.array([0.0, -0.02], dtype=np.float32),
            geometry.points[len(geometry.points) // 2],
            np.max(geometry.points, axis=0)
            + np.array([10.0, 10.0], dtype=np.float32),
        )
        last_index = -1
        for position in positions:
            nearest = geometry.nearest_index(position, last_index=last_index)
            expected = _legacy_preview(
                geometry, position, 20, start_index=nearest
            )
            actual = geometry.preview(position, 20, start_index=nearest)
            np.testing.assert_array_equal(actual["curvature"], expected["curvature"])
            np.testing.assert_array_equal(actual["width"], expected["width"])
            last_index = nearest


def test_preview_does_not_allocate_interpolation_tails_per_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    angles = np.linspace(0.0, 2.0 * np.pi, 100, endpoint=False)
    centerline = np.column_stack((np.cos(angles), np.sin(angles))).astype(
        np.float32
    )
    geometry = TrackPreviewGeometry.build(centerline, None, spacing=0.1)
    assert geometry is not None

    def reject_append(*args, **kwargs):
        raise AssertionError("np.append must not run during preview sampling")

    monkeypatch.setattr(np, "append", reject_append)
    geometry.preview(centerline[0], 20)


def test_progress_projection_cannot_be_reused_for_resampled_preview() -> None:
    centerline = np.array(
        [[0, 0], [4, 0], [4, 1], [4, 5], [0, 5], [0, 0.2]],
        dtype=np.float32,
    )
    preview = TrackPreviewGeometry.build(centerline, None, spacing=0.7)
    assert preview is not None
    position = np.array([3.8, 0.4], dtype=np.float32)

    progress_projection = project_to_centerline(
        prepare_centerline_geometry(centerline), position, heading=0.0
    )
    preview_projection = project_to_centerline(
        preview.projection_geometry, position, heading=0.0
    )

    assert progress_projection.segment_index != preview_projection.segment_index
    assert progress_projection.arc_length != pytest.approx(
        preview_projection.arc_length, abs=1e-7
    )


def test_projection_preserves_off_track_reverse_heading_and_invalid_input() -> None:
    centerline = np.array(
        [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [0.0, 0.1]],
        dtype=np.float32,
    )
    geometry = prepare_centerline_geometry(centerline)
    position = np.array([5.0, 1.0], dtype=np.float32)
    forward = project_to_centerline(geometry, position, heading=np.pi / 2.0)
    reverse = project_to_centerline(geometry, position, heading=-np.pi / 2.0)

    assert forward.segment_index == reverse.segment_index
    assert forward.arc_length == reverse.arc_length
    assert forward.lateral_error == reverse.lateral_error
    assert abs(reverse.heading_error) == pytest.approx(np.pi)

    with pytest.raises(ValueError, match="finite"):
        project_to_centerline(
            geometry, np.array([np.nan, 0.0], dtype=np.float32), heading=0.0
        )
