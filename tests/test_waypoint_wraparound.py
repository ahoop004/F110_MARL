import numpy as np
import pytest
import agents.waypoint as waypoint

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


@pytest.mark.parametrize("policy_type", [waypoint.PurePursuitPolicy, waypoint.StanleyPolicy])
def test_controller_geometry_reused_and_invalidated_on_centerline_changes(policy_type, monkeypatch):
    track = [np.array([[0., 0.], [1., 0.], [2., 0.], [3., 0.]], dtype=np.float32)]
    policy = policy_type(lambda: track[0])
    observation = {"pose": np.array([0.1, 0.1, 0.0]), "velocity": np.array([1.0, 0.0])}
    original = waypoint._is_closed_path
    calls = []

    def count_closure(points):
        calls.append(len(points))
        return original(points)

    monkeypatch.setattr(waypoint, "_is_closed_path", count_closure)
    initial = policy.act(observation)
    np.testing.assert_array_equal(policy.act(observation), initial)
    assert len(calls) == 1

    # Both edits to a live map and replacement maps must invalidate geometry.
    track[0][1:, 1] = [0.5, 1.5, 3.0]
    changed = policy.act(observation)
    assert len(calls) == 2
    assert not np.array_equal(changed, initial)
    np.testing.assert_array_equal(changed, policy_type(lambda: track[0]).act(observation))
    track[0] = np.array([[0., 0.], [2., 0.]], dtype=np.float32)
    policy.reset()
    np.testing.assert_array_equal(
        policy.act(observation), policy_type(lambda: track[0]).act(observation)
    )
