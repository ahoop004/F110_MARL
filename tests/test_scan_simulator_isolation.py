from pathlib import Path

import numpy as np

from core.scenario import load_and_expand_scenario
from core.setup import create_training_setup


def test_training_and_evaluation_environments_keep_independent_scan_maps() -> None:
    scenario_path = Path("scenarios/ppo_lap_completion_pretrain.yaml").resolve()
    scenario = load_and_expand_scenario(str(scenario_path))

    train_env, _, _ = create_training_setup(
        scenario, mode="train", scenario_dir=scenario_path.parent
    )
    try:
        train_scanner = train_env.sim.agents[0].scan_simulator
        train_origin = np.asarray(train_scanner.origin, dtype=np.float64).copy()
        train_obs, _ = train_env.reset(seed=42)
        pose = np.asarray(train_obs["car_0"]["pose"], dtype=np.float64)
        baseline_scan = train_scanner.scan(pose, np.random.default_rng(123)).copy()

        eval_env, _, _ = create_training_setup(
            scenario, mode="eval", scenario_dir=scenario_path.parent
        )
        try:
            eval_scanner = eval_env.sim.agents[0].scan_simulator

            assert train_scanner is not eval_scanner
            assert train_env._map_bundle_active == "Budapest_map"
            assert eval_env._map_bundle_active == "Silverstone_map"
            assert np.array_equal(train_scanner.origin, train_origin)
            assert not np.array_equal(train_scanner.origin, eval_scanner.origin)

            scan = train_scanner.scan(pose, np.random.default_rng(123))
            assert scan.min() < 1.0
            assert np.array_equal(scan, baseline_scan)
        finally:
            eval_env.close()
    finally:
        train_env.close()
