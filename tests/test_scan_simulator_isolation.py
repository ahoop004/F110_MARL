from pathlib import Path

import numpy as np
import pytest

from core.scenario import load_and_expand_scenario
from core.setup import create_training_setup


@pytest.mark.parametrize("pick,shuffle", [("random", False), ("round_robin", False), ("round_robin", True)])
def test_seeded_map_schedule_replays_independently_of_reset_history(pick, shuffle):
    scenario = load_and_expand_scenario("scenarios/ppo.yaml")
    scenario["environment"].update(
        map_bundles_train=["line2", "circle_map"],
        map_bundles_eval=["line2", "circle_map"],
        map_cycle="per_episode", map_pick=pick, epoch_shuffle=shuffle,
    )
    env, _, _ = create_training_setup(scenario, mode="eval", scenario_dir=Path("scenarios").resolve())
    try:
        def snapshot(index):
            obs, _ = env.reset(seed=123, options={"map_episode_index": index})
            return env._map_bundle_active, obs["car_0"]["pose"].copy(), obs["car_0"]["scans"].copy()

        baseline = [snapshot(index) for index in range(4)]
        env.reset(seed=999)
        env.reset()
        for index in reversed(range(4)):
            actual = snapshot(index)
            assert actual[0] == baseline[index][0]
            np.testing.assert_array_equal(actual[1], baseline[index][1])
            np.testing.assert_array_equal(actual[2], baseline[index][2])
        if pick == "round_robin" and not shuffle:
            assert [row[0] for row in baseline] == ["line2", "circle_map"] * 2
            env.reset(seed=123)
            assert env._map_bundle_active == "line2"
            env.reset()
            assert env._map_bundle_active == "circle_map"
            env.reset(seed=123)
            assert env._map_bundle_active == "line2"
        with pytest.raises(ValueError, match="map_episode_index"):
            env.reset(options={"map_episode_index": 1})
        with pytest.raises(ValueError, match="map_episode_index"):
            env.reset(seed=123, options={"map_episode_index": -1})
    finally:
        env.close()


def test_training_and_evaluation_environments_keep_independent_scan_maps() -> None:
    scenario_path = Path("scenarios/ppo_lap_completion_pretrain.yaml").resolve()
    scenario = load_and_expand_scenario(str(scenario_path))
    # The test needs distinct maps regardless of the experiment's current split.
    scenario["environment"]["map_bundles"] = ["Budapest_map", "Silverstone_map"]
    scenario["environment"]["map_bundles_train"] = ["Budapest_map"]
    scenario["environment"]["map_bundles_eval"] = ["Silverstone_map"]

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
