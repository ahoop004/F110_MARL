from types import SimpleNamespace

import pytest

from core.scenario import load_and_expand_scenario
from training.reward_context import build_reward_context
from wrappers.observations.composer import ObservationComposer
from wrappers.rewards.composer import RewardComposer


def test_complete_4_has_consistent_full_circuit_contract_and_held_out_maps() -> None:
    scenario = load_and_expand_scenario("scenarios/complete_4.yaml")
    environment = scenario["environment"]

    assert environment["target_laps"] == 1
    assert environment["lap_counting"]["count_initial_crossing_as_lap"] is False
    assert environment["max_steps"] == 250000
    assert environment["map_bundles_eval"]
    assert set(environment["map_bundles_train"]).isdisjoint(
        environment["map_bundles_eval"]
    )
    assert scenario["wandb"]["enabled"] is True
    assert scenario["wandb"]["group"] == "complete4-full-circuit-v1"
    assert scenario["wandb"]["job_type"] == "train-individual"
    assert "one-full-circuit" in scenario["wandb"]["notes"].lower()


def test_duration_calibration_scenarios_cover_all_maps() -> None:
    one_lap = load_and_expand_scenario(
        "scenarios/calibration/hybrid_pp_ftg_1lap.yaml"
    )
    three_lap = load_and_expand_scenario(
        "scenarios/calibration/pure_pursuit_3lap.yaml"
    )

    assert one_lap["environment"]["target_laps"] == 1
    assert three_lap["environment"]["target_laps"] == 3
    assert one_lap["environment"]["map_bundles_train"] == three_lap["environment"][
        "map_bundles_train"
    ]
    assert one_lap["experiment"]["episodes"] == len(
        one_lap["environment"]["map_bundles_train"]
    )


def test_complete_4_reward_is_lap_normalized_and_penalizes_reverse_progress() -> None:
    composer = RewardComposer.from_file(
        "configs/reward/tasks/complete_4_lap_completion.yaml"
    )
    component_names = {type(component).__name__ for component in composer._components}

    assert "CenterlineProgressComponent" not in component_names
    assert "ProgressDeltaBonusComponent" in component_names
    assert "WrongWayPenaltyComponent" in component_names

    base_info = {
        "collision": False,
        "lap_crossed": False,
        "race_completed": False,
        "terminal_reason": None,
        "time_limit": False,
    }
    forward_total, forward = composer.compute(
        {
            "info": {
                **base_info,
                "centerline": {"progress_delta": 0.001, "wrong_way": False},
            }
        }
    )
    reverse_total, reverse = composer.compute(
        {
            "info": {
                **base_info,
                "centerline": {"progress_delta": -0.001, "wrong_way": True},
            }
        }
    )

    assert forward["progress_delta/bonus"] == pytest.approx(0.1)
    assert reverse["progress_delta/bonus"] == pytest.approx(-0.1)
    assert reverse["wrong_way/penalty"] == pytest.approx(-0.05)
    assert forward_total == pytest.approx(0.095)
    assert reverse_total == pytest.approx(-0.155)


def test_signed_progress_delta_clamps_projection_jumps_symmetrically() -> None:
    composer = RewardComposer.from_file(
        "configs/reward/tasks/complete_4_lap_completion.yaml"
    )
    common = {
        "collision": False,
        "lap_crossed": False,
        "race_completed": False,
        "terminal_reason": None,
        "time_limit": False,
        "wrong_way": False,
    }
    _, positive = composer.compute(
        {"info": {**common, "centerline": {"progress_delta": 0.2}}}
    )
    _, negative = composer.compute(
        {"info": {**common, "centerline": {"progress_delta": -0.2}}}
    )

    assert positive["progress_delta/bonus"] == pytest.approx(2.5)
    assert negative["progress_delta/bonus"] == pytest.approx(-2.5)


def test_reward_context_exposes_active_centerline_track_length() -> None:
    env = SimpleNamespace(
        centerline_track_length=402.5,
        trainable_agents=["car_0"],
        fixed_policy_agents=[],
        last_step_facts=None,
    )
    context = build_reward_context(
        env=env,
        agent_id="car_0",
        info_dict={"car_0": {}},
        obs_dict={"car_0": {}},
        actions={"car_0": []},
    )

    assert context["track_length"] == pytest.approx(402.5)
    context["info"] = {"centerline": {"vs": 5.0}}
    composer = RewardComposer.from_config(
        {
            "reward": {
                "centerline_progress": {
                    "enabled": True,
                    "weight": 0.02,
                    "normalize_by_track_length": True,
                    "reference_length": 400.0,
                }
            }
        }
    )
    _, breakdown = composer.compute(context)
    assert breakdown["centerline_progress/bonus"] == pytest.approx(
        0.02 * 5.0 * 400.0 / 402.5
    )


def test_frenet_observations_are_bounded_for_experiment_configs() -> None:
    for path, expected_dim in (
        ("configs/observations/rl_racer_vehicle_track_frenet.yaml", 158),
        ("configs/observations/rl_racer_vehicle_track_frenet_neighbors.yaml", 173),
    ):
        composer = ObservationComposer.from_file(
            path,
            {"lidar_beams": 108, "lidar_range": 10.0},
        )
        assert composer.obs_dim == expected_dim
        assert all(
            getattr(component, "clip", True) for component in composer.components
        )
