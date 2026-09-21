from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from env.centerline_state import LapTracker, validate_finish_line
from env.collision_state import RaceLifecycle
from env.types import AgentRaceStatus


RACE_MAPS = (
    "Budapest_map",
    "circle_map",
    "Hockenheim_map",
    "Melbourne_map",
    "Montreal_map",
    "Shanghai_map",
    "Silverstone_map",
    "Spa_map",
    "Spielberg_map",
    "line2",
)


def _line() -> dict:
    return validate_finish_line(
        {
            "start": [0.0, -1.0],
            "end": [0.0, 1.0],
            "direction": [1.0, 0.0],
            "min_speed": 0.1,
            "hysteresis": 0.25,
        }
    )


def _update(tracker: LapTracker, x: float, vx: float, step: int) -> bool:
    return tracker.update(
        np.array([x]), np.array([0.0]), np.array([vx]), np.array([0.0]), step=step
    )["car_0"]


def _return_around_finish(tracker: LapTracker, step: int) -> None:
    """Return behind the line via the track, outside the finite finish segment."""
    line = tracker.finish_line
    mid = (line["start"] + line["end"]) / 2
    direction = line["direction"]
    outside = line["segment"] * (1.0 + line["padding"])
    for offset, point in enumerate((
        mid + direction + outside,
        mid - direction + outside,
        mid - direction,
    )):
        tracker.update(np.array([point[0]]), np.array([point[1]]),
                       np.array([1.0]), np.array([0.0]), step=step + offset)


def test_three_forward_crossings_finish_only_on_lap_three() -> None:
    lifecycle = RaceLifecycle(["car_0"], target_laps=3)
    tracker = LapTracker(["car_0"], _line(), lifecycle)
    tracker.reset(np.array([-1.0]), np.array([0.0]))

    for lap in range(1, 4):
        assert _update(tracker, 0.1, 1.0, lap * 10) is True
        assert lifecycle.records["car_0"].lap_count == lap
        expected = AgentRaceStatus.FINISHED if lap == 3 else AgentRaceStatus.ACTIVE
        assert lifecycle.records["car_0"].status == expected
        if lap < 3:
            _return_around_finish(tracker, lap * 10 + 1)


def test_initial_crossing_can_start_race_without_completing_a_lap() -> None:
    lifecycle = RaceLifecycle(["car_0"], target_laps=1)
    tracker = LapTracker(
        ["car_0"],
        _line(),
        lifecycle,
        count_initial_crossing_as_lap=False,
    )
    tracker.reset(np.array([-1.0]), np.array([0.0]))

    # Crossing from the starting grid begins the lap.
    assert _update(tracker, 0.1, 1.0, 1) is False
    assert lifecycle.records["car_0"].lap_count == 0
    assert lifecycle.records["car_0"].lap_start_step == 1
    assert lifecycle.records["car_0"].status == AgentRaceStatus.ACTIVE

    # The car must travel back around and cross again to complete one circuit.
    _return_around_finish(tracker, 2)
    assert _update(tracker, 0.1, 1.0, 5) is True
    assert lifecycle.records["car_0"].lap_count == 1
    assert lifecycle.records["car_0"].lap_time_steps == 4
    assert lifecycle.records["car_0"].status == AgentRaceStatus.FINISHED


def test_reverse_crossing_and_stationary_jitter_do_not_count() -> None:
    lifecycle = RaceLifecycle(["car_0"], target_laps=1)
    tracker = LapTracker(["car_0"], _line(), lifecycle)
    tracker.reset(np.array([1.0]), np.array([0.0]))

    assert _update(tracker, -1.0, -1.0, 1) is False
    for step, x in enumerate([-0.05, 0.05, -0.03, 0.02], start=2):
        assert _update(tracker, x, 0.0, step) is False
    assert lifecycle.records["car_0"].lap_count == 0


def test_spawn_on_completed_side_cannot_immediately_finish() -> None:
    lifecycle = RaceLifecycle(["car_0"], target_laps=1)
    tracker = LapTracker(["car_0"], _line(), lifecycle)
    tracker.reset(np.array([0.1]), np.array([0.0]))

    assert _update(tracker, 0.2, 1.0, 1) is False
    assert lifecycle.records["car_0"].lap_count == 0


def test_finish_line_validation_rejects_invalid_geometry() -> None:
    with pytest.raises(ValueError, match="distinct"):
        validate_finish_line({"start": [0, 0], "end": [0, 0], "direction": [1, 0]})
    with pytest.raises(ValueError, match="direction"):
        validate_finish_line({"start": [0, -1], "end": [0, 1]})
    with pytest.raises(ValueError, match="run along"):
        lifecycle = RaceLifecycle(["car_0"], 1)
        LapTracker(
            ["car_0"],
            validate_finish_line(
                {"start": [0, -1], "end": [0, 1], "direction": [0, 1]}
            ),
            lifecycle,
        )


@pytest.mark.parametrize("map_name", RACE_MAPS)
def test_race_map_finish_line_is_reproducible_and_ahead_of_grid(map_name: str) -> None:
    root = Path(__file__).resolve().parents[1]
    map_dir = root / "maps" / map_name
    metadata = yaml.safe_load((map_dir / f"{map_name}.yaml").read_text())
    annotation = metadata["annotations"]["finish_line"]
    spawns = np.asarray(
        [item["pose"] for item in metadata["annotations"]["spawn_points"]],
        dtype=np.float32,
    )
    centerline = np.loadtxt(
        map_dir / f"{map_name}_centerline.csv",
        delimiter=",",
        skiprows=1,
        usecols=(0, 1),
    )

    line = validate_finish_line(
        annotation,
        centerline=centerline,
        spawn_poses=spawns,
    )
    midpoint = (line["start"] + line["end"]) * 0.5
    direction = line["direction"]
    segment = line["segment_unit"]

    assert annotation["version"] == 1
    assert abs(float(np.dot(direction, segment))) < 1e-5
    assert np.all((spawns[:, :2] - midpoint) @ direction < -1.0)

    lifecycle = RaceLifecycle(["car_0"], target_laps=3)
    tracker = LapTracker(["car_0"], line, lifecycle)
    behind = midpoint - direction
    ahead = midpoint + 0.1 * direction
    tracker.reset(np.array([behind[0]]), np.array([behind[1]]))
    for lap in range(1, 4):
        crossed = tracker.update(
            np.array([ahead[0]]),
            np.array([ahead[1]]),
            np.array([1.0]),
            np.array([0.0]),
            step=lap * 10,
        )
        assert crossed["car_0"] is True
        assert lifecycle.records["car_0"].lap_count == lap
        if lap < 3:
            _return_around_finish(tracker, lap * 10 + 1)


@pytest.mark.parametrize("count_initial", [False, True])
def test_local_circles_at_actual_finish_cannot_earn_laps(count_initial) -> None:
    root = Path(__file__).resolve().parents[1] / "maps" / "circle_map"
    metadata = yaml.safe_load((root / "circle_map.yaml").read_text())
    line = validate_finish_line(metadata["annotations"]["finish_line"])
    lifecycle = RaceLifecycle(["car_0"], target_laps=3)
    tracker = LapTracker(["car_0"], line, lifecycle,
                         count_initial_crossing_as_lap=count_initial)
    mid = (line["start"] + line["end"]) / 2
    forward = line["direction"]
    left = np.array([-forward[1], forward[0]])
    # A local 0.8 m radius loop fits inside this track and crosses the line in
    # both directions. Longitudinal *body* speed remains positive throughout.
    angles = np.linspace(-np.pi / 2, 8 * np.pi, 2200)
    points = mid + 0.8 * (np.sin(angles)[:, None] * forward + np.cos(angles)[:, None] * left)
    tracker.reset(points[:1, 0], points[:1, 1])
    for step, point in enumerate(points[1:], 1):
        tracker.update(np.array([point[0]]), np.array([point[1]]),
                       np.array([1.0]), np.array([0.0]), step=step)
    assert lifecycle.records["car_0"].lap_count == int(count_initial)
    assert lifecycle.records["car_0"].status == AgentRaceStatus.ACTIVE


def test_reverse_laps_must_be_repaid_and_reset_clears_debt() -> None:
    lifecycle = RaceLifecycle(["car_0"], target_laps=3)
    tracker = LapTracker(["car_0"], _line(), lifecycle)
    tracker.reset(np.array([-1.0]), np.array([0.0]))
    assert _update(tracker, 0.1, 1.0, 1)
    # Repeated reverse crossings, returning ahead outside the segment, make
    # two laps of reverse debt. Merely seeing one forward crossing is not enough.
    for step in (10, 20):
        _update(tracker, -1.0, 1.0, step)
        for offset, (x, y) in enumerate(((-1., 3.), (1., 3.), (1., 0.)), 1):
            tracker.update(np.array([x]), np.array([y]), np.array([1.]), np.array([0.]), step=step + offset)
    for lap in range(3):
        _return_around_finish(tracker, 30 + 10 * lap)
        assert _update(tracker, 0.1, 1.0, 34 + 10 * lap) == (lap == 2)
    assert lifecycle.records["car_0"].lap_count == 2
    _update(tracker, -1.0, 1.0, 60)
    tracker.reset(np.array([-1.0]), np.array([0.0]))
    assert _update(tracker, 0.1, 1.0, 1)
    assert lifecycle.records["car_0"].lap_count == 1


@pytest.mark.parametrize("map_name", [name for name in RACE_MAPS if name != "line2"])
def test_full_centerline_circuits_still_complete_three_lap_race(map_name) -> None:
    root = Path(__file__).resolve().parents[1] / "maps" / map_name
    metadata = yaml.safe_load((root / f"{map_name}.yaml").read_text())
    line = validate_finish_line(metadata["annotations"]["finish_line"])
    points = np.loadtxt(root / f"{map_name}_centerline.csv", delimiter=",", skiprows=1, usecols=(0, 1))
    mid = (line["start"] + line["end"]) / 2
    behind = mid - line["direction"]
    start = np.argmin(np.linalg.norm(points - behind, axis=1))
    points = np.roll(points, -start, axis=0)
    lifecycle = RaceLifecycle(["car_0"], target_laps=3)
    tracker = LapTracker(["car_0"], line, lifecycle, count_initial_crossing_as_lap=False)
    tracker.reset(points[:1, 0], points[:1, 1])
    for step, point in enumerate(np.tile(points, (4, 1)), 1):
        tracker.update(np.array([point[0]]), np.array([point[1]]),
                       np.ones(1), np.zeros(1), step=step)
    assert lifecycle.records["car_0"].lap_count == 3
    assert lifecycle.records["car_0"].status == AgentRaceStatus.FINISHED


def test_reverse_crossing_debt_is_per_agent_and_ignores_body_speed_sign() -> None:
    lifecycle = RaceLifecycle(["car_0", "car_1"], target_laps=2)
    tracker = LapTracker(lifecycle.agent_ids, _line(), lifecycle)
    tracker.reset(np.array([1., -1.]), np.zeros(2))
    tracker.update(np.array([-1., -1.]), np.zeros(2), np.zeros(2), np.zeros(2), step=1)
    crossings = tracker.update(np.ones(2), np.zeros(2), np.ones(2), np.zeros(2), step=2)
    assert crossings == {"car_0": False, "car_1": True}


def test_segment_intersection_uses_crossing_point_not_step_endpoint() -> None:
    lifecycle = RaceLifecycle(["car_0"], target_laps=2)
    tracker = LapTracker(["car_0"], _line(), lifecycle)
    tracker.reset(np.array([1.]), np.array([-5.]))
    # The endpoint is within the line's width, but the intersection is outside.
    tracker.update(np.array([-1.]), np.array([0.]), np.ones(1), np.zeros(1), step=1)
    assert _update(tracker, 1., 1., 2)
    tracker.reset(np.array([1.]), np.array([-5.]))
    # Conversely, both endpoints can be outside while the crossing is inside.
    tracker.update(np.array([-1.]), np.array([5.]), np.ones(1), np.zeros(1), step=1)
    tracker.update(np.array([-1.]), np.array([0.]), np.ones(1), np.zeros(1), step=2)
    assert not _update(tracker, 1., 1., 3)
