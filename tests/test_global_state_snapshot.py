from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from core.scenario import load_and_expand_scenario
from core.setup import create_training_setup
from env.state_views import build_global_state
from training.reward_context import build_reward_context, transition_lifecycle_fields


def _complete4_env():
    scenario_path = Path("scenarios/complete_4.yaml").resolve()
    scenario = load_and_expand_scenario(str(scenario_path))
    environment = scenario["environment"]
    environment["map_bundles"] = ["circle_map"]
    environment["map_bundles_train"] = ["circle_map"]
    environment["map_bundles_eval"] = ["circle_map"]
    environment["map_bundle_active"] = "circle_map"
    environment["map_pick"] = "first"
    environment["max_steps"] = 3
    env, _, _ = create_training_setup(
        scenario,
        mode="train",
        scenario_dir=scenario_path.parent,
    )
    return env


def test_global_state_snapshot_is_recursively_immutable() -> None:
    state = build_global_state(
        possible_agents=["car_0"],
        active_agents=["car_0"],
        central_vector=np.array([1.0], dtype=np.float32),
        metadata={"spawn_ids": {"car_0": "grid_0"}, "values": [1, 2]},
    )

    with pytest.raises(ValueError):
        state.vector[0] = 2.0
    with pytest.raises(ValueError):
        state.masks["active_mask"][0] = False
    with pytest.raises(TypeError):
        state.masks["other"] = np.ones(1, dtype=bool)
    with pytest.raises(TypeError):
        state.metadata["spawn_ids"]["car_0"] = "changed"
    assert state.metadata["values"] == (1, 2)


def test_global_state_reconstructs_once_per_mutation(monkeypatch) -> None:
    env = _complete4_env()
    module = sys.modules[env.__class__.__module__]
    original_build = module.build_global_state
    build_calls = 0

    def counted_build(*args, **kwargs):
        nonlocal build_calls
        build_calls += 1
        return original_build(*args, **kwargs)

    monkeypatch.setattr(module, "build_global_state", counted_build)
    try:
        env.reset(seed=42)
        initial = env.get_global_state()
        assert build_calls == 1
        assert all(env.get_global_state() is initial for _ in range(8))
        assert build_calls == 1

        expected = build_global_state(
            possible_agents=env.possible_agents,
            active_agents=env.agents,
            central_vector=env._central_state_tensor(env.sim.current_observation()),
            controlled_agents=env.controlled_agents,
            trainable_agents=env.trainable_agents,
            lifecycle_records=env.lifecycle.records,
            metadata={
                "map_bundle": env._map_bundle_active,
                **env._spawn_manager.last_spawn_metadata,
            },
        )
        np.testing.assert_array_equal(initial.vector, expected.vector)
        for name, mask in expected.masks.items():
            np.testing.assert_array_equal(initial.masks[name], mask)

        env.apply_initial_speeds({"car_0": 1.5})
        after_speed = env.get_global_state()
        assert after_speed is not initial
        assert build_calls == 2

        actions = {
            agent_id: np.zeros(2, dtype=np.float32)
            for agent_id in env.decision_agents
        }
        env.step(actions)
        post_step = env.last_step_facts.global_state
        assert env.get_global_state() is post_step
        assert build_calls == 3

        env.reset(seed=42)
        after_reset = env.get_global_state()
        assert after_reset is not post_step
        assert build_calls == 4

        env._apply_map_data(env._map_data, bundle=env._map_bundle_active)
        after_map_update = env.get_global_state()
        assert after_map_update is not after_reset
        assert build_calls == 5
    finally:
        env.close()


def test_supplied_snapshot_avoids_reward_and_lifecycle_state_calls() -> None:
    snapshot = build_global_state(
        possible_agents=["car_0"],
        active_agents=[],
        central_vector=np.array([3.0], dtype=np.float32),
    )

    class RejectStateLookup:
        trainable_agents = ["car_0"]
        fixed_policy_agents = []
        last_step_facts = None
        centerline_track_length = 10.0

        def get_global_state(self):
            raise AssertionError("the supplied snapshot must be reused")

    env = RejectStateLookup()
    context = build_reward_context(
        env=env,
        agent_id="car_0",
        info_dict={"car_0": {}},
        obs_dict={"car_0": {}},
        actions={"car_0": np.zeros(2, dtype=np.float32)},
        global_state=snapshot,
    )
    np.testing.assert_array_equal(context["global_state"], snapshot.vector)

    fields = transition_lifecycle_fields(
        env,
        {"status": "truncated"},
        global_state=snapshot,
    )
    assert fields["lifecycle_masks"]["active_mask"].tolist() == [False]
    assert fields["lifecycle_masks"]["active_mask"].flags.writeable
