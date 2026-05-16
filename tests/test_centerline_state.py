from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "src")

from src.env.centerline_state import (
    build_render_centerline_points,
    inject_finish_line_info,
    normalize_progress_fractions,
    parse_finish_line,
    reset_finish_line_tracking,
    update_finish_line_progress,
)


def test_finish_line_progress_detects_crossing_and_injects_info():
    finish = parse_finish_line(
        {
            "start": [0.0, -1.0],
            "end": [0.0, 1.0],
            "tolerance": 2.0,
            "padding": 0.0,
        }
    )
    assert finish is not None

    poses_x = np.array([-0.5], dtype=np.float32)
    poses_y = np.array([0.0], dtype=np.float32)
    signed_prev = np.zeros(1, dtype=np.float32)
    crossed = np.zeros(1, dtype=bool)
    reset_finish_line_tracking(finish, signed_prev, crossed, poses_x, poses_y, 1)

    poses_x[0] = 0.5
    completed = update_finish_line_progress(
        finish,
        signed_prev,
        crossed,
        ["car_0"],
        poses_x,
        poses_y,
        np.array([1.0], dtype=np.float32),
        np.array([0.0], dtype=np.float32),
    )

    assert completed == {"car_0": True}
    assert bool(crossed[0]) is True

    infos = {"car_0": {}}
    inject_finish_line_info(finish, crossed, {"car_0": 0}, infos)
    assert infos["car_0"]["finish_line"] is True


def test_normalize_progress_fractions_wraps_and_filters_values():
    result = normalize_progress_fractions([0.25, 1.25, 0.0, "bad", 0.5, 0.25])

    assert result == (0.25, 0.5)


def test_build_render_centerline_points_selects_progress_indices():
    centerline = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
        ],
        dtype=np.float32,
    )

    points = build_render_centerline_points(centerline, (0.25, 0.75), 0.0)

    assert points is not None
    np.testing.assert_allclose(points, np.array([[1.0, 0.0], [3.0, 0.0]], dtype=np.float32))
