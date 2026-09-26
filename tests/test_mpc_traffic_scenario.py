from pathlib import Path

import numpy as np
import pytest

from core.scenario import load_and_expand_scenario
from core.setup import create_training_setup
from wrappers.rewards.composer import RewardComposer


@pytest.fixture
def traffic():
    scenario = load_and_expand_scenario('scenarios/ppo_1v1_racing_mpc_circle.yaml', overrides=['environment.spawn.policy="centerline_random"',
         'environment.spawn.centerline.min_distance=2.0',
         'environment.respawn_agents=["car_1","car_2","car_3","car_4","car_5","car_6"]',
         'environment.respawn_on_vehicle_collision=true',
         'environment.rendering={"vehicle_colors":{"car_0":"#3288ff","car_1":"#ff2020","car_2":"#a0a0a0","car_3":"#a0a0a0","car_4":"#a0a0a0","car_5":"#a0a0a0","car_6":"#a0a0a0"}}',
         'wandb.group="ppo-1v1-mpc-traffic-circle"',
         'wandb.tags=["ppo","1v1","circle","randomized-traffic","fixed-mpc","pretrained"]',
         'wandb.notes="Same pursuit reward, no lap limit, against red car_1; five mixed fixed MPC traffic '
         'cars, independently randomized track spawns with 2 m clearance. Only target respawns earn a '
         'bonus."',
         'experiment.name="ppo_1v1_mpc_traffic_circle"',
         'agents.car_2={"algorithm":"kinematic_mpc","trainable":false,"role":"traffic","target_id":"car_0","action_adapter":"rolling_speed_to_wheel_v1","params":{"dt":0.05,"target_speed":1.5,"min_speed":0.5,"max_speed":2.0}}',
         'agents.car_3={"algorithm":"obstacle_aware_mpc","trainable":false,"role":"traffic","target_id":"car_0","action_adapter":"rolling_speed_to_wheel_v1","params":{"dt":0.05,"target_speed":2.0,"min_speed":0.5,"max_speed":2.5}}',
         'agents.car_4={"algorithm":"defensive_mpc","trainable":false,"role":"traffic","target_id":"car_0","action_adapter":"rolling_speed_to_wheel_v1","params":{"dt":0.05,"target_speed":2.0,"min_speed":0.5,"max_speed":2.5}}',
         'agents.car_5={"algorithm":"cbf_mpc","trainable":false,"role":"traffic","target_id":"car_0","action_adapter":"rolling_speed_to_wheel_v1","params":{"dt":0.05,"target_speed":2.5,"min_speed":0.5,"max_speed":3.0}}',
         'agents.car_6={"algorithm":"mpcc","trainable":false,"role":"traffic","target_id":"car_0","action_adapter":"rolling_speed_to_wheel_v1","params":{"dt":0.05,"target_speed":2.5,"min_speed":0.5,"max_speed":3.0}}'])
    env, agents, _ = create_training_setup(scenario, scenario_dir=Path('scenarios').resolve())
    env.reset(seed=42)
    yield env, agents, scenario
    env.close()


def test_random_spawns_are_separated_reproducible_and_change_target(traffic):
    env, _, scenario = traffic
    first = env.sim.agent_poses.copy()
    env.reset(seed=42)
    np.testing.assert_array_equal(first, env.sim.agent_poses)
    env.reset(seed=43)
    assert not np.allclose(first[1], env.sim.agent_poses[1])
    assert not np.allclose(first[2:], env.sim.agent_poses[2:])
    poses = env.sim.agent_poses[:, :2]
    distances = np.linalg.norm(poses[:, None] - poses[None], axis=2)
    np.fill_diagonal(distances, np.inf)
    assert distances.min() >= 2.0
    assert env.get_target_id('car_0') == 'car_1'
    assert scenario['environment']['rendering']['vehicle_colors']['car_1'] == '#ff2020'
    assert env._vehicle_colors['car_1'][0] == 1.0
    assert env._vehicle_colors['car_1'][1] < .15
    assert not env.lifecycle.finish_on_laps
    assert scenario['evaluation']['lap_completion'] is False


def test_all_fixed_mpcs_produce_finite_actions(traffic):
    env, agents, _ = traffic
    observations, _ = env.reset(seed=42)
    actions = {}
    for aid, controller in agents.items():
        controller.set_env(env)
        actions[aid] = controller.act(observations[aid])
        assert actions[aid].shape == (2,)
        assert np.isfinite(actions[aid]).all()
    _, _, terminated, _, _ = env.step(actions)
    assert not any(terminated.values())


def test_background_crash_has_no_target_bonus_or_episode_reset(traffic):
    env, _, _ = traffic
    env.lifecycle.record_lap_crossing('car_2', step=0)
    pose = env.sim.agent_poses[2].copy()
    pose[:2] += 30.0
    env.sim.agents[2].reset(pose)
    _, _, terminated, truncated, infos = env.step({})
    assert not any(terminated.values()) and not any(truncated.values())
    assert not infos['car_0'].get('target_respawned', False)
    assert not infos['car_2']['track_limits']['exceeded']
    assert env.lifecycle.records['car_2'].lap_count == 1
    reward = RewardComposer.from_file('configs/reward/tasks/race_1v1_pursuit.yaml')
    _, breakdown = reward.compute({'info': infos['car_0'], 'all_infos': infos,
                                  'track_length': env.centerline_track_length})
    assert breakdown['race_pursuit/respawn'] == 0.0


@pytest.mark.parametrize('first,second,ends,bonus', [
    (2, 3, False, False), (1, 2, False, True), (0, 2, True, False),
])
def test_vehicle_collision_rules(traffic, first, second, ends, bonus):
    env, _, _ = traffic
    env.sim.agents[second].reset(env.sim.agent_poses[first].copy())
    _, _, terminated, _, infos = env.step({})
    assert any(terminated.values()) == ends
    assert bool(infos['car_0'].get('target_respawned')) == bonus
    if not ends:
        assert len(env.agents) == 7
        assert np.linalg.norm(env.sim.agent_poses[first, :2] - env.sim.agent_poses[second, :2]) > .5


def test_traffic_laps_do_not_end_race(traffic):
    env, _, _ = traffic
    for lap in range(6):
        env.lifecycle.record_lap_crossing('car_2', step=lap)
    assert not any(env.step({})[2].values())
    for lap in range(5):
        env.lifecycle.record_lap_crossing('car_1', step=lap)
    assert not any(env.step({})[2].values())


def test_random_spawn_rejects_impossible_clearance():
    from env.spawn import sample_random_centerline_spawn
    with pytest.raises(ValueError, match='Cannot place'):
        sample_random_centerline_spawn(centerline=np.array([[0, 0], [1, 0], [0, 1]]),
            agent_ids=['a', 'b'], rng=np.random.default_rng(42), config={'min_distance': 10.0})
