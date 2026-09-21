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
