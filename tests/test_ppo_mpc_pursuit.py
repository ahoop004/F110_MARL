from pathlib import Path

import numpy as np
import pytest

from core.scenario import load_and_expand_scenario
from core.setup import create_training_setup
from wrappers.rewards.composer import RewardComposer


@pytest.fixture
def setup():
    scenario = load_and_expand_scenario('scenarios/ppo_1v1_racing_mpc_circle.yaml')
    scenario['agents']['car_0']['params']['device'] = 'cpu'
    env, agents, rewards = create_training_setup(scenario, scenario_dir=Path('scenarios').resolve())
    env.reset(seed=42)
    yield env, agents, rewards
    env.close()


def test_only_opponent_laps_end_race(setup):
    env, _, _ = setup
    for lap in range(6):
        env.lifecycle.record_lap_crossing('car_0', step=lap)
    assert env.lifecycle.records['car_0'].is_active
    for lap in range(4):
        env.lifecycle.record_lap_crossing('car_1', step=lap)
    assert env.lifecycle.records['car_1'].is_active
    env.lifecycle.record_lap_crossing('car_1', step=5)
    assert env.step({})[2]['car_0']


def test_opponent_respawn_preserves_ego_and_laps(setup):
    env, _, _ = setup
    env.lifecycle.record_lap_crossing('car_1', step=0)
    ego_state = env.sim.agents[0].physics_state.copy()
    old_step = env._elapsed_steps
    env._respawn_on_centerline({'car_1'})
    np.testing.assert_array_equal(env.sim.agents[0].physics_state, ego_state)
    assert env.lifecycle.records['car_1'].lap_count == 1
    assert env._elapsed_steps == old_step
    assert env.lifecycle.records['car_1'].is_active


def test_offtrack_opponent_respawns_and_bonus_repeats(setup):
    env, _, _ = setup
    for _ in range(2):
        pose = env.sim.agent_poses[1].copy()
        pose[:2] += 20.0
        env.sim.agents[1].reset(pose)
        _, _, terminated, truncated, infos = env.step({})
        assert not any(terminated.values())
        assert not any(truncated.values())
        assert infos['car_0']['target_respawned']
        assert not infos['car_1']['track_limits']['exceeded']
        assert env.lifecycle.records['car_1'].is_active
    assert not env.step({})[4]['car_0'].get('target_respawned', False)


def test_ego_crash_ends_episode(setup):
    env, _, _ = setup
    pose = env.sim.agent_poses[0].copy()
    pose[:2] += 20.0
    env.sim.agents[0].reset(pose)
    _, _, terminated, _, infos = env.step({})
    assert all(terminated.values())
    assert not infos['car_0'].get('target_respawned', False)


def test_vehicle_collision_ends_episode_without_bonus(setup):
    env, _, _ = setup
    env.sim.agents[1].reset(env.sim.agent_poses[0].copy())
    _, _, terminated, _, infos = env.step({})
    assert all(terminated.values())
    assert not infos['car_0'].get('target_respawned', False)


def test_reward_preserves_progress_and_wraps_order():
    reward = RewardComposer.from_file('configs/reward/tasks/race_1v1_pursuit.yaml')
    info = {'centerline': {'progress': .98, 'progress_delta': .01},
            'track_limits': {'exceeded': False}}
    target = {'centerline': {'progress': .01, 'progress_delta': .01}}
    step = {'info': info, 'all_infos': {'car_1': target}, 'track_length': 100.}
    assert reward.compute(step)[0] == pytest.approx(.99)
    info['target_respawned'] = True
    assert reward.compute(step)[0] == pytest.approx(1.99)
    assert reward.compute(step)[0] == pytest.approx(1.99)
    info['target_respawned'] = False
    info['centerline']['progress_delta'] = .06
    assert reward.compute(step)[0] == pytest.approx(6.)


def test_evaluation_uses_twenty_opponent_laps():
    scenario = load_and_expand_scenario('scenarios/ppo_1v1_racing_mpc_circle.yaml')
    scenario['agents']['car_0']['params']['device'] = 'cpu'
    env, _, _ = create_training_setup(scenario, mode='eval', scenario_dir=Path('scenarios').resolve())
    try:
        env.reset(seed=42)
        assert env.target_laps == 20
        for lap in range(19):
            env.lifecycle.record_lap_crossing('car_1', step=lap)
        assert env.lifecycle.records['car_1'].is_active
        env.lifecycle.record_lap_crossing('car_1', step=20)
        assert not env.lifecycle.records['car_1'].is_active
        assert env.lifecycle.records['car_0'].is_active
    finally:
        env.close()
