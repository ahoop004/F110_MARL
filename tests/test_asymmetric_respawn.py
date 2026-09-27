"""Recovery must preserve racing, rewards and policy state across teleports."""
from pathlib import Path

import numpy as np
import pytest

from core.scenario import load_and_expand_scenario, validate_scenario, ScenarioError
from core.setup import create_training_setup, build_reward_composers
from env.respawn import reset_respawned
from metrics.racing_eval import create_episode_facts, update_agent_step_facts, episode_race_record
from utils.centerline import project_to_centerline
from wrappers.rewards.events import OpponentCrashBonusComponent

IDS = ['car_0', 'car_1', 'car_2', 'car_3']


@pytest.fixture
def recovery():
    scenario = load_and_expand_scenario('scenarios/mappo_2v2_asymmetric.yaml')
    env, opponents, _ = create_training_setup(scenario, scenario_dir=Path('scenarios'))
    env.reset(seed=42)
    # Keep all cars well separated, independent of scenario spawn randomization.
    geometry = env._centerline_progress_tracker._geometry
    points = geometry.segment_starts
    poses = env.sim.agent_poses.copy()
    for i in range(4):
        k = int([.0625, .3125, .5625, .92][i] * len(points))
        direction = geometry.segment_vectors[k]
        poses[i] = [*points[k], np.arctan2(direction[1], direction[0])]
    env.sim.reset(poses)
    env._lap_tracker.reset(poses[:, 0], poses[:, 1])
    env.step({})
    yield env, scenario, opponents
    env.close()


@pytest.mark.parametrize('aid', IDS)
def test_every_car_recovers_boundary_once_without_terminal_or_teleport_reward(recovery, aid):
    env, scenario, _ = recovery
    env.lifecycle.record_lap_crossing(aid, step=0)
    laps = env.lifecycle.records[aid].lap_count
    idx = IDS.index(aid)
    pose = env.sim.agent_poses[idx].copy()
    pose[:2] += 1000.
    env.sim.agents[idx].reset(pose)
    obs, _, terms, truncs, infos = env.step({})
    assert not terms[aid] and not truncs[aid]
    assert aid in env.agents
    assert infos[aid]['respawn_reason'] == 'track_boundary'
    assert infos[aid]['boundary_event']
    assert obs[aid]['wheel_speed_reference'] == 0.
    assert obs[aid]['wheel_speed_reference_rate'] == 0.
    assert not infos[aid]['track_limits']['exceeded']
    assert infos[aid]['centerline']['progress_delta'] == 0.
    assert env.lifecycle.records[aid].lap_count == laps
    if aid in IDS[:2]:
        rewards = build_reward_composers(scenario['agents'], IDS[:2], Path('scenarios'))
        value, parts = rewards[aid].compute({'agent_id': aid, 'info': infos[aid],
            'all_infos': infos, 'track_length': env._centerline_progress_tracker._geometry.total_length,
            'opponent_agent_ids': IDS[2:]})
        assert value == -1.
        assert parts['progress_delta/boundary'] == -1.
    next_infos = env.step({})[4]
    assert not next_infos[aid].get('respawned')
    assert abs(next_infos[aid]['centerline']['progress_delta']) < 1e-5


def test_mpc_collision_recovers_half_lap_ahead_even_when_learner_crashes(recovery):
    env, _, _ = recovery
    # car_1 leads by a completed lap, irrespective of track coordinate.
    env.lifecycle.record_lap_crossing('car_1', step=0)
    geometry = env._centerline_progress_tracker._geometry
    leader_s = project_to_centerline(geometry, env.sim.agent_poses[1, :2], 0.).arc_length
    expected = (leader_s + geometry.total_length / 2) % geometry.total_length
    env.sim.agents[2].reset(env.sim.agent_poses[0].copy())
    _, _, terms, _, infos = env.step({})
    assert terms['car_0'] and infos['car_0']['terminal_reason'] == 'collision'
    assert not terms['car_2'] and infos['car_2']['respawn_reason'] == 'collision'
    assert env.lifecycle.records['car_2'].is_active
    assert env.lifecycle.records['car_2'].lap_count == 0
    assert infos['car_2']['centerline']['progress_delta'] == 0.
    actual = project_to_centerline(geometry, env.sim.agent_poses[2, :2], 0.).arc_length
    error = abs((actual - expected + geometry.total_length / 2) % geometry.total_length - geometry.total_length / 2)
    assert error <= max(geometry.segment_lengths) * 1.1
    reward = OpponentCrashBonusComponent({'bonus': 1.})
    context = {'all_infos': infos, 'opponent_agent_ids': IDS[2:]}
    assert reward.compute(context) == {'opponent_crash/bonus': 1.}
    assert reward.compute(context) == {}  # Original once-per-opponent cap survives recovery.
    facts = create_episode_facts(episode=0, agent_ids=IDS, trainable_ids=IDS[:2], opponent_ids=IDS[2:])
    update_agent_step_facts(facts, step_idx=1, infos=infos, terminations=terms)
    record = episode_race_record(facts, timestep=env.timestep)
    assert record['opponent_collision_respawns'] == 1
    assert record['opponent_collision_dnf_count'] == 0
    assert record['own_collision_dnf_count'] == 1


def test_simultaneous_mpc_crashes_get_separate_clear_respawns(recovery):
    env, _, _ = recovery
    env.sim.agents[3].reset(env.sim.agent_poses[2].copy())
    infos = env.step({})[4]
    assert all(infos[aid]['respawn_reason'] == 'collision' for aid in IDS[2:])
    assert np.linalg.norm(env.sim.agent_poses[2, :2] - env.sim.agent_poses[3, :2]) > 2 * env.sim.params['length']
    assert not any(env.step({})[4][aid].get('respawned') for aid in IDS)


def test_recovery_restarts_speed_reference_and_mpc_plan(recovery):
    from wrappers.actions.composer import ActionComposer
    env, scenario, controllers = recovery
    actions = {aid: ActionComposer.from_config(
        env.action_spaces[aid].low, env.action_spaces[aid].high,
        scenario['agents'][aid]['action_constraints'], decision_dt=env.timestep)
        for aid in IDS[:2]}
    for action in actions.values():
        action.process(np.array([0., 1.]))
        action.process(np.array([0., 1.]))
    for controller in controllers.values():
        controller.controller._warm = np.ones((2, 2))
    reset_respawned({'car_0': {'respawned': True}, 'car_2': {'respawned': True}},
                   controllers=controllers, actions=actions, observations={})
    assert actions['car_0'].process(np.array([0., 1.]))[1] == pytest.approx(5.)
    assert actions['car_1'].process(np.array([0., 1.]))[1] == pytest.approx(15.)
    assert controllers['car_2'].controller._warm is None
    assert controllers['car_3'].controller._warm is not None


@pytest.mark.parametrize('config', [
    {'boundary_agents': ['unknown']}, {'boundary_agents': 'car_0'},
    {'collision_agents': ['car_0']}, {'collision_placement': 'random'},
])
def test_invalid_recovery_configuration_is_rejected(config):
    scenario = load_and_expand_scenario('scenarios/mappo_2v2_asymmetric.yaml')
    scenario['environment']['respawn'] = config
    with pytest.raises(ScenarioError, match='respawn'):
        validate_scenario(scenario)
