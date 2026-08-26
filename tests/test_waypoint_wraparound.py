import numpy as np

from agents.waypoint import _find_nearest, _is_closed_path, _lookahead_point


def test_closed_centerline_nearest_search_wraps_at_finish_seam() -> None:
    points = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.1]],
        dtype=np.float32,
    )

    assert _is_closed_path(points)
    assert _find_nearest(points, np.array([0.0, 0.0]), last_idx=4, window=1) == 0


def test_closed_centerline_lookahead_wraps_instead_of_clamping() -> None:
    points = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.1]],
        dtype=np.float32,
    )

    goal, _ = _lookahead_point(points, start=4, dist=0.6)

    np.testing.assert_allclose(goal, np.array([0.5, 0.0]), atol=1e-6)


def test_open_centerline_lookahead_still_clamps() -> None:
    points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float32)

    assert not _is_closed_path(points)
    goal, index = _lookahead_point(points, start=1, dist=5.0)

    np.testing.assert_array_equal(goal, points[-1])
    assert index == len(points) - 1
