"""Legacy physics identity and frozen pre-update rollout regressions."""
from copy import deepcopy

import numpy as np
import pytest

from core.scenario import ScenarioError, load_and_expand_scenario, validate_scenario
from env.f110ParallelEnv import F110ParallelEnv, _default_vehicle_params
from physics.dynamic_models import validate_vehicle_params
from physics.vehicle import RaceCar
from scripts.physics_baseline import DEFAULT_OUTPUT, check


@pytest.mark.parametrize("explicit_model", [False, True])
def test_frozen_ppo_and_mappo_physics_rollouts(explicit_model):
    check(DEFAULT_OUTPUT, explicit_model=explicit_model)


@pytest.mark.parametrize("overrides,field", [
    ({"model": "mf61"}, "model"),
    ({"model": None}, "model"),
    ({"model_version": 2}, "model_version"),
    ({"model_version": True}, "model_version"),
    ({"model_version": 1.0}, "model_version"),
    ({"wheel_radius": 0.05}, "wheel_radius"),
    ({"tire": {}}, "tire"),
    ({"m": 0}, "m"),
    ({"I": -1}, "I"),
    ({"lf": np.nan}, "lf"),
    ({"mu": np.inf}, "mu"),
    ({"mu": -0.1}, "mu"),
    ({"C_Sf": -1}, "C_Sf"),
    ({"h": -1}, "h"),
    ({"a_max": True}, "a_max"),
    ({"a_max": "5.0"}, "a_max"),
    ({"a_max": [5]}, "a_max"),
    ({"v_min": 0}, "v_min"),
    ({"v_max": 0}, "v_max"),
    ({"sv_min": 0}, "sv_min"),
    ({"s_max": np.pi / 2}, "s_max"),
])
def test_invalid_physics_fails_scenario_validation(overrides, field):
    scenario = load_and_expand_scenario("scenarios/legacy/ppo.yaml")
    scenario["environment"].setdefault("vehicle_params", {}).update(overrides)
    with pytest.raises(ScenarioError, match=field):
        validate_scenario(scenario)


def test_partial_configuration_preserves_defaults_and_does_not_mutate_input():
    params = {"model": "legacy_st", "model_version": 1, "a_max": 5}
    original = deepcopy(params)
    result = F110ParallelEnv._configure_vehicle_params(None, {"vehicle_params": params})
    assert result == {**_default_vehicle_params(), **params}
    assert params == original
    assert validate_vehicle_params({}) == {}
    assert validate_vehicle_params({"mu": 0, "h": 0}) == {"mu": 0.0, "h": 0.0}


def test_rejected_runtime_parameter_update_preserves_car_state_and_parameters():
    car = RaceCar(_default_vehicle_params(), seed=42, num_beams=16)
    original = car.params.copy()
    original_dynamics = car._dyn_params
    original_state = car.state.copy()
    with pytest.raises(ValueError, match="v_min"):
        car.update_params({**original, "v_min": 0})
    assert car.params == original
    assert car._dyn_params == original_dynamics
    np.testing.assert_array_equal(car.state, original_state)


@pytest.mark.parametrize("timestep", [0, -0.01, np.inf, np.nan])
def test_car_rejects_invalid_physics_timestep(timestep):
    with pytest.raises(ValueError, match="time_step"):
        RaceCar(_default_vehicle_params(), seed=42, time_step=timestep, num_beams=16)
