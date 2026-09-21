"""Prediction, physical limits, geometry, traffic, and real controller wiring."""
from types import SimpleNamespace
import numpy as np
import pytest

from agents.mpc.racing import RacingMPCAgent, _distance, _shoot, predict_step


P = np.array([.3302, .17145, .5, 3.2, .15, 500., 5., 1.0489, .58, .31])


def test_prediction_preserves_actuator_lag_and_acceleration_bound():
    initial = np.zeros(6)
    future = predict_step(initial, np.array([.4189, 3.5]), P, .05)
    assert 0 < future[4] < .4189
    assert 0 < future[5] < 3.5
    assert 0 < future[3] <= 5*.05
    assert future[0] > 0
    assert abs(future[4]) <= 3.2*.05


def test_map_distance_uses_origin_rotation_and_rejects_outside():
    field = np.tile(np.arange(10, dtype=float), (10, 1))
    assert _distance(field, 3., 4., np.zeros(3), 1.) == pytest.approx(3.)
    assert _distance(field, 6., 23., np.array([10., 20., np.pi/2]), 1.) == pytest.approx(3.)
    assert _distance(field, -1., 2., np.zeros(3), 1.) < 0


def shooting_args(traffic=None, footprint=None):
    # Straight lane: y=0, boundaries at +/-0.5m, nonzero world origin.
    xs = np.arange(-.5, 6., .1)
    path = np.column_stack([xs, np.zeros_like(xs), xs+.5])
    field = np.tile((.5-np.abs(np.arange(80)*.1-4.))[:, None], (1, 100))
    return (np.array([0., 0., 0., 1., 0., 1., 1.]), path, field,
            np.array([-2., -4., 0.]), .1,
            np.array([[0., 0.]]) if footprint is None else footprint,
            np.empty((0, 5)) if traffic is None else traffic,
            P, .05, 5, 2., .1, 5.)


def test_wall_cost_checks_footprint_even_when_center_is_inside():
    controls = np.tile([0., 1.], (4, 1))
    center_cost, center_clearance, _ = _shoot(controls, *shooting_args())
    body_cost, body_clearance, _ = _shoot(controls, *shooting_args(footprint=np.array([[0., .6]])))
    assert center_clearance > 0 and body_clearance < 0
    assert body_cost > center_cost


def test_moving_vehicle_prediction_changes_collision_cost():
    controls = np.tile([0., 1.], (4, 1))
    stopped = np.array([[1.4, 0., 0., 0., 0.]])
    departing = np.array([[1.4, 0., 0., 3., 0.]])
    static_cost, static_clearance, _ = _shoot(controls, *shooting_args(stopped))
    moving_cost, moving_clearance, _ = _shoot(controls, *shooting_args(departing))
    assert static_cost > moving_cost
    assert static_clearance < 0 < moving_clearance


def test_traffic_rotates_body_velocity_and_keeps_stationary_cars():
    controller = RacingMPCAgent({'agent_id': 'ego'})
    states = {
        'moving': SimpleNamespace(pose=np.array([2., 0., np.pi/2]), velocity=np.array([2., 1.])),
        'wreck': SimpleNamespace(pose=np.array([3., 0., 0.]), velocity=np.zeros(2)),
        'far': SimpleNamespace(pose=np.array([20., 0., 0.]), velocity=np.zeros(2)),
    }
    controller.env = SimpleNamespace(possible_agents=['ego', *states], get_agent_state=states.__getitem__)
    rows = controller._traffic(np.zeros(3), 'ego')
    assert rows.shape == (2, 5)
    np.testing.assert_allclose(rows[0, 3:], [-1., 2.], atol=1e-12)
    np.testing.assert_array_equal(rows[1, 3:], [0., 0.])
    with pytest.raises(ValueError, match='agent id'):
        controller._traffic(np.zeros(3), None)


@pytest.mark.parametrize('config', [{'horizon': 31}, {'max_speed': float('nan')},
                                  {'margin': -1}, {'knots': 0}])
def test_invalid_configuration_rejected(config):
    with pytest.raises(ValueError):
        RacingMPCAgent(config)


def test_active_team_scenarios_use_one_fixed_mpc_opponent_profile():
    from pathlib import Path
    from core.scenario import load_and_expand_scenario, load_yaml_config
    profile = load_yaml_config(Path('configs/controllers/racing_mpc.yaml'))
    paths = sorted(Path('scenarios').glob('mappo_2v2_*.yaml'))
    assert len(paths) == 8
    for path in paths:
        scenario = load_and_expand_scenario(str(path))
        for aid, target in [('car_2', 'car_0'), ('car_3', 'car_1')]:
            assert scenario['agents'][aid] == {**profile, 'role': 'opponent', 'target_id': target}
        assert [aid for aid, cfg in scenario['agents'].items() if cfg['trainable']] == ['car_0', 'car_1']


def test_hybrid_benchmark_stays_hybrid_after_matrix_opponent_change():
    from scripts.benchmark_racing_opponents import scenario_for
    for focal in ('hybrid_pp_ftg', 'racing_mpc'):
        scenario = scenario_for(focal, 'circle_map', 'traffic', 10042, 10, 3)
        assert scenario['agents']['car_0']['algorithm'] == focal
        assert all(scenario['agents'][aid]['algorithm'] == 'hybrid_pp_ftg'
                   for aid in ['car_1', 'car_2', 'car_3'])


def test_mpc_opponents_act_in_actual_training_setup_on_both_maps():
    from pathlib import Path
    from core.scenario import load_and_expand_scenario
    from core.setup import create_training_setup
    path = Path('scenarios/mappo_2v2_penalties_scratch.yaml').resolve()
    scenario = load_and_expand_scenario(str(path))
    env, opponents, _ = create_training_setup(scenario, mode='train', scenario_dir=path.parent)
    try:
        for agent in opponents.values():
            agent.set_env(env)
        for index in range(2):
            obs, _ = env.reset(seed=42, options={'map_episode_index': index})
            for agent in opponents.values():
                agent.reset()
            moved = False
            for _ in range(10):
                actions = {aid: np.zeros(2) for aid in ('car_0', 'car_1') if aid in env.agents}
                for aid, agent in opponents.items():
                    if aid not in env.agents:
                        continue
                    actions[aid] = agent.act(obs[aid])  # same implicit id as training/evaluation
                    assert np.isfinite(actions[aid]).all()
                    assert abs(actions[aid][0]) <= .418901
                    assert 0 <= actions[aid][1] <= 70.00001
                    assert agent.last_plan['traffic_count'] <= 3
                    moved |= actions[aid][1] > 0
                obs, _, _, _, _ = env.step(actions)
            assert moved
    finally:
        env.close()


def test_real_setup_identity_action_units_limits_reset_and_map_switch():
    from scripts.benchmark_racing_opponents import ROOT, scenario_for
    from core.setup import create_training_setup
    scenario = scenario_for('racing_mpc', 'circle_map', 'solo', 10042, 20, 1)
    for key in ('map_bundles', 'map_bundles_train', 'map_bundles_eval'):
        scenario['environment'][key] = ['circle_map', 'Budapest_map']
    env, controllers, _ = create_training_setup(scenario, mode='eval', scenario_dir=ROOT/'scenarios')
    try:
        obs, _ = env.reset(seed=10042)
        controller = controllers['car_0']
        controller.set_env(env)
        initial = controller.act(obs['car_0'])  # runners do not supply aid
        assert abs(initial[0]) <= .4189
        assert 0 <= initial[1] <= 5.00001  # 0.25 m/s reference = 5 rad/s
        controller.reset()
        np.testing.assert_array_equal(initial, controller.act(obs['car_0']))
        for _ in range(4):
            action = controller.act(obs['car_0'])
            obs, _, _, _, _ = env.step({'car_0': action})
            assert np.isfinite(obs['car_0']['pose']).all()
        old_path = controller.controller.path.copy()
        obs, _ = env.reset(seed=10042, options={'map_episode_index': 1})
        controller.reset()
        assert np.isfinite(controller.act(obs['car_0'])).all()
        assert not np.array_equal(old_path, controller.controller.path)
        controller.controller.field[:] = -1.  # no feasible predicted route
        action = controller.act(obs['car_0'])
        assert controller.last_plan['brake_fallback']
        assert action[1] == 0.  # standing reset can request a complete stop
    finally:
        env.close()
