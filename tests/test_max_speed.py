"""Shared speed presets must reach both policy actions and fixed MPC commands."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from core.scenario import ScenarioError, load_and_expand_scenario, resolve_max_speed
from core.setup import create_training_setup
from wrappers.actions.composer import ActionComposer


@pytest.mark.parametrize("speed", [5, 10, 15, 20])
def test_shared_speed_reaches_learner_and_mpc(speed):
    scenario = load_and_expand_scenario(
        "scenarios/ppo_1v1_racing_mpc_circle.yaml",
        overrides=[f"environment.max_speed={speed}"],
    )
    env, opponents, _ = create_training_setup(scenario, scenario_dir=Path("scenarios").resolve())
    try:
        radius = env.params["wheel_actuators"]["wheel_radius"]
        bounds = env.action_space("car_0")
        learner = ActionComposer.from_config(
            bounds.low, bounds.high, scenario["agents"]["car_0"]["action_constraints"],
            decision_dt=env.timestep,
        )
        for _ in range(200):
            action = learner.process(np.array([0., 1.]))
        assert action[1] * radius == pytest.approx(speed)
        assert bounds.high[1] * radius == pytest.approx(speed)

        mpc = opponents["car_1"]
        assert mpc.max_speed == pytest.approx(speed)
        # Isolate the real adapter from track-dependent cornering decisions.
        mpc.controller.act = lambda *args, **kwargs: np.array([0., speed + 1.])
        assert mpc.act({})[1] * radius == pytest.approx(speed)
    finally:
        env.close()


@pytest.mark.parametrize("speed", [5, 10, 15, 20])
@pytest.mark.parametrize("path", [
    "scenarios/ppo_lap_completion_transfer_3lap.yaml",
    "scenarios/mappo_2v2_asymmetric.yaml",
    "scenarios/mappo_2v2_race.yaml",
    "scenarios/render/racing_mpc.yaml",
])
def test_presets_resolve_across_workflows(path, speed):
    scenario = load_and_expand_scenario(path, overrides=[f"environment.max_speed={speed}"])
    actuators = scenario["environment"]["vehicle_params"]["wheel_actuators"]
    assert actuators["wheel_speed_max"] * actuators["wheel_radius"] == pytest.approx(speed)
    for agent in scenario["agents"].values():
        if agent["algorithm"] == "racing_mpc":
            assert agent["params"]["max_speed"] == speed


def test_cli_speed_overrides_yaml_and_individual_limits(monkeypatch):
    import run

    monkeypatch.setattr("sys.argv", [
        "run.py", "--scenario", "scenarios/ppo_1v1_racing_mpc_circle.yaml",
        "--set", "environment.max_speed=10",
        "--set", "agents.car_1.params.max_speed=5", "--max-speed", "15",
    ])
    args = run.parse_args()
    scenario = run.apply_cli_overrides(load_and_expand_scenario(args.scenario), args)
    assert scenario["environment"]["max_speed"] == 15
    assert scenario["agents"]["car_1"]["params"]["max_speed"] == 15
    assert scenario["environment"]["vehicle_params"]["wheel_actuators"]["wheel_speed_max"] == 300


def test_cli_set_speed_resolves_without_dedicated_flag(monkeypatch):
    import run

    monkeypatch.setattr("sys.argv", [
        "run.py", "--scenario", "scenarios/ppo_1v1_racing_mpc_circle.yaml",
        "--set", "environment.max_speed=20",
    ])
    args = run.parse_args()
    scenario = run.apply_cli_overrides(load_and_expand_scenario(args.scenario), args)
    assert scenario["agents"]["car_1"]["params"]["max_speed"] == 20
    assert scenario["environment"]["vehicle_params"]["wheel_actuators"]["wheel_speed_max"] == 400


@pytest.mark.parametrize("configured,arguments,expected", [
    (10, [], 10),
    (10, ["--set", "environment.max_speed=!delete"], 5),
    (7, ["--max-speed", "20"], 20),
    (10, ["--set", "environment.max_speed=15", "--max-speed", "5"], 5),
])
def test_main_resolves_yaml_and_cli_before_building_controllers(
    tmp_path, monkeypatch, configured, arguments, expected,
):
    import run

    scenario = load_and_expand_scenario("scenarios/render/racing_mpc.yaml")
    scenario["environment"]["max_speed"] = configured
    path = tmp_path / "speed.yaml"
    path.write_text(yaml.safe_dump(scenario))
    captured = []
    monkeypatch.setattr(run, "_run_heuristic", lambda config, *args: captured.append(config))
    monkeypatch.setattr("sys.argv", ["run.py", "--scenario", str(path), *arguments])
    run.main()
    resolved = captured[0]
    actuators = resolved["environment"]["vehicle_params"]["wheel_actuators"]
    assert actuators["wheel_speed_max"] * actuators["wheel_radius"] == pytest.approx(expected)
    assert resolved["agents"]["car_0"]["params"]["max_speed"] == expected


@pytest.mark.parametrize("value", [0, -5, 7, True, "10", float("nan"), float("inf")])
def test_invalid_speed_is_rejected(value):
    with pytest.raises(ScenarioError, match="environment.max_speed"):
        resolve_max_speed({"environment": {"max_speed": value}})


def test_speed_uses_physical_radius_without_changing_other_contracts():
    scenario = load_and_expand_scenario("scenarios/ppo_1v1_racing_mpc_circle.yaml")
    scenario["environment"]["max_speed"] = 20
    scenario["environment"]["vehicle_params"]["wheel_actuators"]["wheel_radius"] = .1
    original = deepcopy(scenario)
    resolved = resolve_max_speed(scenario)
    assert scenario == original
    assert resolved["environment"]["vehicle_params"]["wheel_actuators"]["wheel_speed_max"] == 200
    resolved["environment"]["vehicle_params"]["wheel_actuators"]["wheel_speed_max"] = 100
    resolved["agents"]["car_1"]["params"]["max_speed"] = 5
    assert resolved == original


@pytest.mark.parametrize("radius", [0, -.1, True, None, float("nan")])
def test_invalid_radius_is_rejected(radius):
    scenario = {"environment": {"max_speed": 10, "vehicle_params": {
        "model": "combined_slip_st", "wheel_actuators": {"wheel_radius": radius},
    }}}
    with pytest.raises(ScenarioError, match="wheel_radius"):
        resolve_max_speed(scenario)


def test_legacy_speed_limit_and_opt_in_behavior():
    scenario = load_and_expand_scenario("scenarios/legacy/ppo_racing.yaml")
    assert resolve_max_speed(scenario) == scenario
    scenario["environment"]["max_speed"] = 15
    assert resolve_max_speed(scenario)["environment"]["vehicle_params"]["v_max"] == 15
