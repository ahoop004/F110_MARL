from __future__ import annotations

import os
import sys

import numpy as np

os.environ.setdefault("PYGLET_HEADLESS", "true")
sys.path.insert(0, ".")
sys.path.insert(0, "src")

from src.core.scenario import load_and_expand_scenario
from src.core.setup import create_training_setup


def test_env_reset_step_contract_from_ppo_scenario():
    scenario = load_and_expand_scenario("scenarios/ppo.yaml")
    scenario["environment"]["max_steps"] = 3
    scenario["environment"]["render"] = False

    env, _, reward_strategies = create_training_setup(scenario, mode="train")
    assert reward_strategies == {}
    assert env.possible_agents == ["car_0", "car_1"]
    assert set(env.action_spaces) == set(env.possible_agents)
    assert set(env.observation_spaces) == set(env.possible_agents)

    obs, infos = env.reset(seed=123)
    assert set(obs) == set(env.possible_agents)
    assert set(infos) == set(env.possible_agents)
    assert all("spawn_point" in info for info in infos.values())

    for agent_id, agent_obs in obs.items():
        assert "lidar" in agent_obs
        assert "pose" in agent_obs
        assert "velocity" in agent_obs
        assert "state" in agent_obs
        assert agent_obs["lidar"].dtype == np.float32
        assert agent_obs["state"].shape == (env._central_state_dim,)
        assert env.observation_spaces[agent_id].spaces["state"].shape == (env._central_state_dim,)

    zero_actions = {
        agent_id: np.zeros(env.action_spaces[agent_id].shape, dtype=np.float32)
        for agent_id in env.agents
    }
    next_obs, rewards, terminations, truncations, step_infos = env.step(zero_actions)

    assert set(next_obs) == set(env.possible_agents)
    assert set(rewards) == set(env.possible_agents)
    assert set(terminations) == set(env.possible_agents)
    assert set(truncations) == set(env.possible_agents)
    assert set(step_infos) == set(env.possible_agents)

    for agent_id in env.possible_agents:
        assert isinstance(rewards[agent_id], float)
        assert isinstance(terminations[agent_id], bool)
        assert isinstance(truncations[agent_id], bool)
        assert "collision" in step_infos[agent_id]
        assert "target_collision" in step_infos[agent_id]
        assert "target_finished" in step_infos[agent_id]

    env.close()
