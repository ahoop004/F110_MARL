from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from wrappers.rewards.composer import RewardComposer
from wrappers.rewards.speed import ReverseVelocityPenaltyComponent


def _step_with_speed(speed: float) -> dict:
    state = SimpleNamespace(velocity=np.array([speed, 0.0], dtype=np.float32))
    facts = SimpleNamespace(agent_states={"car_0": state})
    return {"agent_id": "car_0", "last_step_facts": facts}


def test_reverse_velocity_penalty_uses_actual_longitudinal_speed() -> None:
    component = ReverseVelocityPenaltyComponent(
        {"weight": 2.0, "deadband": 0.05, "max_reverse_speed": 1.0}
    )

    assert component.compute(_step_with_speed(2.0)) == {}
    assert component.compute(_step_with_speed(-0.04)) == {}
    assert component.compute(_step_with_speed(-0.5)) == {
        "reverse_velocity/penalty": pytest.approx(-0.9)
    }
    assert component.compute(_step_with_speed(-3.0)) == {
        "reverse_velocity/penalty": pytest.approx(-2.0)
    }


def test_reverse_velocity_penalty_falls_back_to_raw_observation() -> None:
    component = ReverseVelocityPenaltyComponent({"weight": 1.0, "deadband": 0.0})

    assert component.compute(
        {"next_obs": {"velocity": np.array([-0.25, 0.0], dtype=np.float32)}}
    ) == {"reverse_velocity/penalty": pytest.approx(-0.25)}
    assert component.compute({"next_obs": {}}) == {}


def test_complete_four_reward_enables_reverse_velocity_penalty() -> None:
    composer = RewardComposer.from_file(
        "configs/reward/tasks/complete_4_lap_completion.yaml"
    )

    assert any(
        isinstance(component, ReverseVelocityPenaltyComponent)
        for component in composer._components
    )
