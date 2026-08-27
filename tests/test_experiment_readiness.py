from core.scenario import load_and_expand_scenario


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
