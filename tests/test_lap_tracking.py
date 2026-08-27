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
            assert _update(tracker, -1.0, 1.0, lap * 10 + 1) is False


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
    assert lifecycle.records["car_0"].status == AgentRaceStatus.ACTIVE

    # The car must travel back around and cross again to complete one circuit.
    assert _update(tracker, -1.0, 1.0, 2) is False
    assert _update(tracker, 0.1, 1.0, 3) is True
    assert lifecycle.records["car_0"].lap_count == 1
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
            step=lap * 2,
        )
        assert crossed["car_0"] is True
        assert lifecycle.records["car_0"].lap_count == lap
        if lap < 3:
            tracker.update(
                np.array([behind[0]]),
                np.array([behind[1]]),
                np.array([1.0]),
                np.array([0.0]),
                step=lap * 2 + 1,
            )
