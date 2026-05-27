from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "src")

from src.env.centerline_state import (
    CenterlineProgressTracker,
    CenterlineRuntimeState,
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


def test_centerline_runtime_state_owns_render_and_feature_flags():
    cfg = {
        "_centerline_render_user_override": False,
        "_centerline_features_user_override": False,
        "centerline_render": False,
        "centerline_features": False,
        "centerline_render_progress": [0.5],
    }
    state = CenterlineRuntimeState.from_config(cfg)
    centerline = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ],
        dtype=np.float32,
    )

    state.set_centerline(centerline)
    changed = state.register_usage(require_render=True, require_features=True)

    assert changed is True
    assert state.render_enabled is True
    assert state.feature_requested is True
    assert state.features_enabled is True
    assert state.render_points is not None
    np.testing.assert_allclose(state.render_points, np.array([[1.0, 0.0]], dtype=np.float32))
    assert state.points is not None
    assert state.points.flags.writeable is False


# ---------------------------------------------------------------------------
# CenterlineProgressTracker regression tests
# ---------------------------------------------------------------------------

# Simple straight-line centerline running along the x-axis (5 waypoints).
_STRAIGHT_CENTERLINE = np.array(
    [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]],
    dtype=np.float32,
)
_AGENT_INDEX = {"car_0": 0, "car_1": 1}


def _make_tracker(agent_ids=("car_0",)) -> CenterlineProgressTracker:
    return CenterlineProgressTracker(agent_ids=agent_ids)


def test_progress_tracker_returns_facts_for_each_agent():
    tracker = _make_tracker(["car_0", "car_1"])
    poses_x = np.array([1.0, 3.0], dtype=np.float32)
    poses_y = np.array([0.0, 0.0], dtype=np.float32)
    poses_theta = np.array([0.0, 0.0], dtype=np.float32)
    vels_x = np.array([1.0, 1.0], dtype=np.float32)
    vels_y = np.array([0.0, 0.0], dtype=np.float32)

    facts = tracker.update(
        _STRAIGHT_CENTERLINE, poses_x, poses_y, poses_theta, vels_x, vels_y, _AGENT_INDEX
    )

    assert set(facts) == {"car_0", "car_1"}
    for aid in ("car_0", "car_1"):
        for key in ("progress", "progress_delta", "d", "vs", "vd", "heading_error", "wrong_way"):
            assert key in facts[aid], f"missing key {key!r} for {aid}"


def test_progress_tracker_vs_is_positive_when_moving_forward():
    tracker = _make_tracker()
    # Agent at waypoint 1, heading along +x, moving forward.
    facts = tracker.update(
        _STRAIGHT_CENTERLINE,
        np.array([1.0], dtype=np.float32),
        np.array([0.0], dtype=np.float32),
        np.array([0.0], dtype=np.float32),  # heading = 0 rad (along +x)
        np.array([2.0], dtype=np.float32),  # vx = 2 m/s forward
        np.array([0.0], dtype=np.float32),
        {"car_0": 0},
    )
    assert facts["car_0"]["vs"] > 0.0
    assert abs(facts["car_0"]["vd"]) < 1e-5


def test_progress_tracker_lateral_deviation_matches_y_offset():
    tracker = _make_tracker()
    # Agent is 0.5 m to the left of the centerline (y = 0.5) with zero velocity.
    facts = tracker.update(
        _STRAIGHT_CENTERLINE,
        np.array([2.0], dtype=np.float32),
        np.array([0.5], dtype=np.float32),  # lateral offset
        np.array([0.0], dtype=np.float32),
        np.array([0.0], dtype=np.float32),
        np.array([0.0], dtype=np.float32),
        {"car_0": 0},
    )
    # The lateral error should be close to 0.5 m (left of track is positive).
    assert abs(facts["car_0"]["d"]) < 0.6  # within one waypoint spacing


def test_progress_tracker_wrong_way_flag_set_when_facing_backward():
    import math

    tracker = _make_tracker()
    # Agent facing directly backward (-x direction, heading = pi).
    facts = tracker.update(
        _STRAIGHT_CENTERLINE,
        np.array([2.0], dtype=np.float32),
        np.array([0.0], dtype=np.float32),
        np.array([math.pi], dtype=np.float32),  # facing backward
        np.array([-1.0], dtype=np.float32),      # moving backward too
        np.array([0.0], dtype=np.float32),
        {"car_0": 0},
    )
    assert facts["car_0"]["wrong_way"] is True


def test_progress_tracker_wrong_way_false_when_facing_forward():
    tracker = _make_tracker()
    facts = tracker.update(
        _STRAIGHT_CENTERLINE,
        np.array([2.0], dtype=np.float32),
        np.array([0.0], dtype=np.float32),
        np.array([0.0], dtype=np.float32),  # heading = 0 (forward)
        np.array([1.0], dtype=np.float32),
        np.array([0.0], dtype=np.float32),
        {"car_0": 0},
    )
    assert facts["car_0"]["wrong_way"] is False


def test_progress_tracker_delta_is_zero_on_first_step():
    tracker = _make_tracker()
    facts = tracker.update(
        _STRAIGHT_CENTERLINE,
        np.array([1.0], dtype=np.float32),
        np.array([0.0], dtype=np.float32),
        np.array([0.0], dtype=np.float32),
        np.array([1.0], dtype=np.float32),
        np.array([0.0], dtype=np.float32),
        {"car_0": 0},
    )
    assert facts["car_0"]["progress_delta"] == 0.0


def test_progress_tracker_delta_is_positive_after_forward_step():
    tracker = _make_tracker()
    agent_index = {"car_0": 0}

    def _step(x: float) -> dict:
        return tracker.update(
            _STRAIGHT_CENTERLINE,
            np.array([x], dtype=np.float32),
            np.array([0.0], dtype=np.float32),
            np.array([0.0], dtype=np.float32),
            np.array([1.0], dtype=np.float32),
            np.array([0.0], dtype=np.float32),
            agent_index,
        )

    _step(0.0)   # step 0: progress ≈ 0.0 (first step, delta = 0)
    facts = _step(2.0)  # step 1: moved forward, delta should be > 0
    assert facts["car_0"]["progress_delta"] > 0.0


def test_progress_tracker_reset_clears_indices_and_prev_progress():
    tracker = _make_tracker()
    agent_index = {"car_0": 0}

    def _step(x: float):
        return tracker.update(
            _STRAIGHT_CENTERLINE,
            np.array([x], dtype=np.float32),
            np.array([0.0], dtype=np.float32),
            np.array([0.0], dtype=np.float32),
            np.array([1.0], dtype=np.float32),
            np.array([0.0], dtype=np.float32),
            agent_index,
        )

    _step(0.0)
    _step(2.0)
    tracker.reset()
    facts = _step(3.0)
    # After reset the delta on the first step should be 0 again.
    assert facts["car_0"]["progress_delta"] == 0.0


def test_progress_tracker_returns_empty_when_centerline_is_none():
    tracker = _make_tracker()
    facts = tracker.update(
        None,  # type: ignore[arg-type]
        np.array([0.0], dtype=np.float32),
        np.array([0.0], dtype=np.float32),
        np.array([0.0], dtype=np.float32),
        np.array([0.0], dtype=np.float32),
        np.array([0.0], dtype=np.float32),
        {"car_0": 0},
    )
    assert facts == {}
