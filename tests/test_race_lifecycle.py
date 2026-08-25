from __future__ import annotations

import pytest

from env.collision_state import RaceLifecycle, validate_target_laps
from env.types import AgentRaceStatus, TerminalReason


@pytest.mark.parametrize("value", [0, -1, 1.5, "3", True, None])
def test_target_laps_requires_a_positive_integer(value: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        validate_target_laps(value)


def test_lifecycle_finishes_only_on_target_lap() -> None:
    lifecycle = RaceLifecycle(["car_0", "car_1"], target_laps=3)

    assert lifecycle.record_lap_crossing("car_0", step=10) is False
    assert lifecycle.records["car_0"].lap_count == 1
    lifecycle.begin_step()
    assert lifecycle.records["car_0"].lap_crossed is False

    assert lifecycle.record_lap_crossing("car_0", step=20) is False
    assert lifecycle.record_lap_crossing("car_0", step=30) is True

    result = lifecycle.records["car_0"]
    assert result.status == AgentRaceStatus.FINISHED
    assert result.terminal_reason == TerminalReason.RACE_COMPLETE
    assert result.terminal_step == 30
    assert result.finish_position == 1
    assert lifecycle.active_agents == ("car_1",)
    assert lifecycle.episode_done is False


def test_terminal_transition_is_immutable() -> None:
    lifecycle = RaceLifecycle(["car_0"], target_laps=1)
    assert lifecycle.record_lap_crossing("car_0", step=4) is True

    assert lifecycle.record_collision("car_0", step=5) is False
    result = lifecycle.records["car_0"]
    assert result.status == AgentRaceStatus.FINISHED
    assert result.terminal_reason == TerminalReason.RACE_COMPLETE
    assert result.terminal_step == 4
    assert result.finish_position == 1


def test_time_limit_truncates_only_active_agents() -> None:
    lifecycle = RaceLifecycle(["car_0", "car_1", "car_2"], target_laps=1)
    lifecycle.record_lap_crossing("car_0", step=2)
    lifecycle.record_collision("car_1", step=3)

    assert lifecycle.truncate_active(step=9) == ("car_2",)
    assert lifecycle.records["car_0"].status == AgentRaceStatus.FINISHED
    assert lifecycle.records["car_1"].status == AgentRaceStatus.CRASHED
    assert lifecycle.records["car_2"].status == AgentRaceStatus.TRUNCATED
    assert lifecycle.records["car_2"].terminal_reason == TerminalReason.TIME_LIMIT
    assert lifecycle.episode_done is True


def test_reset_clears_results_and_finish_order() -> None:
    lifecycle = RaceLifecycle(["car_0", "car_1"], target_laps=1)
    lifecycle.record_lap_crossing("car_1", step=1)
    lifecycle.reset()

    assert lifecycle.active_agents == ("car_0", "car_1")
    assert all(record.lap_count == 0 for record in lifecycle.records.values())
    assert all(record.terminal_reason is None for record in lifecycle.records.values())
    lifecycle.record_lap_crossing("car_0", step=2)
    assert lifecycle.records["car_0"].finish_position == 1
