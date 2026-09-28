"""Prediction, physical limits, geometry, traffic, and real controller wiring."""
from types import SimpleNamespace
import numpy as np
import pytest

from agents.mpc.racing import RacingMPCAgent, _distance, _shoot, predict_step, _limit_command, _speed_profile, _chassis_rhs
from physics.dynamic_models import first_order_actuator_step


from pathlib import Path
import yaml
from physics.tire_models import MF61_KEYS

VEHICLE = yaml.safe_load(Path('scenarios/render/racing_mpc.yaml').read_text())['environment']['vehicle_params']
P = np.array([.3302, .17145, .5, 3.2, .15, 500., 5., 1.0489, .58, .31,
              VEHICLE['m'], VEHICLE['I'], VEHICLE['slip_speed_floor'],
              *[VEHICLE['front_tire'][key] for key in MF61_KEYS],
              *[VEHICLE['rear_tire'][key] for key in MF61_KEYS]])


def test_prediction_preserves_actuator_lag_and_grip_bound():
    initial = np.zeros(8)
    future = predict_step(initial, np.array([.4189, 3.5]), P, .05)
    assert 0 < future[4] < .4189
    assert 0 < future[5] < 3.5
    assert 0 < future[3] <= P[7]*9.81*.05
    assert future[0] > 0
    assert abs(future[4]) <= 3.2*.05


@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.int64])
def test_prediction_does_not_mutate_or_quantize_input(dtype):
    state = np.array([0, 0, 0, 1, 0, 1, 0, 0], dtype=dtype)
    original = state.copy()
    control = np.array([.1, 2.5])
    expected = predict_step(state.astype(np.float64), control, P, .05)
    result = predict_step(state, control, P, .05)
    assert result.dtype == np.float64
    np.testing.assert_array_equal(state, original)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize('mu', [0., .4, 1.0489, 1.2])
def test_prediction_matches_allocating_midpoint_reference(mu):
    # Keep the previous array-expression integrator as a numerical reference
    # for the allocation-free RHS and reusable midpoint buffer.
    rng = np.random.default_rng(319)
    params = P.copy()
    params[7] = mu
    dt = .05
    for _ in range(12):
        initial = np.array([0., 0., 0., rng.uniform(0, 5), rng.uniform(-.2, .2),
                            rng.uniform(0, 5), rng.uniform(-.5, .5), rng.uniform(-1, 1)])
        command = np.array([rng.uniform(-.4, .4), rng.uniform(0, 5)])
        state = initial.copy()
        front_speed = state[3]*np.cos(state[4])+(state[6]+(params[0]-params[1])*state[7])*np.sin(state[4])
        contact_speed = min(abs(state[3]), abs(front_speed))
        contact_speed -= dt*(params[7]*9.81+abs(state[7]*state[6]))
        steps = max(1, int(np.ceil(dt/min(.02, .004*max(.5, contact_speed)/.5))))
        h = dt/steps
        for _ in range(steps):
            delta, wheel = state[4], state[5]
            mid = state + .5*h*np.asarray(_chassis_rhs(state, delta, wheel, params))
            mid_delta = first_order_actuator_step(delta, command[0], h/2, params[2], -params[3], params[3])
            mid_wheel = first_order_actuator_step(wheel, command[1], h/2, params[4], -params[5], params[5])
            state += h*np.asarray(_chassis_rhs(mid, mid_delta, mid_wheel, params))
            state[4] = first_order_actuator_step(delta, command[0], h, params[2], -params[3], params[3])
            state[5] = first_order_actuator_step(wheel, command[1], h, params[4], -params[5], params[5])
        np.testing.assert_array_equal(predict_step(initial, command, params, dt), state)


def test_map_distance_uses_origin_rotation_and_rejects_outside():
    field = np.tile(np.arange(10, dtype=float), (10, 1))
    assert _distance(field, 3., 4., np.zeros(3), 1.) == pytest.approx(3.)
    assert _distance(field, 6., 23., np.array([10., 20., np.pi/2]), 1.) == pytest.approx(3.)
    assert _distance(field, -1., 2., np.zeros(3), 1.) < 0


def shooting_args(traffic=None, footprint=None):
    # Straight lane: y=0, boundaries at +/-0.5m, nonzero world origin.
    xs = np.arange(-.5, 6., .1)
    path = np.column_stack([xs, np.zeros_like(xs), xs+.5, np.full_like(xs, 2.)])
    field = np.tile((.5-np.abs(np.arange(80)*.1-4.))[:, None], (1, 100))
    return (np.array([0., 0., 0., 1., 0., 1., 0., 0., 1., 0.]), path, field,
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


@pytest.mark.parametrize('moving_traffic', [False, True])
def test_shooting_without_trajectory_keeps_cost_clearance_and_inputs(moving_traffic):
    controls = np.array([[.1, 2.5], [-.1, 1.5], [0., 0.], [.2, 1.]])
    traffic = np.array([[1.4, 0., .2, .5, .1]]) if moving_traffic else None
    args = shooting_args(traffic=traffic, footprint=np.array([[0., 0.], [.2, .15]]))
    controls_before = controls.copy()
    arrays_before = [arg.copy() for arg in args if isinstance(arg, np.ndarray)]
    recorded = _shoot(controls, *args)
    unrecorded = _shoot(controls, *args, record_trajectory=False)
    assert recorded[:2] == unrecorded[:2]
    assert recorded[2].shape == (20, 8)
    assert unrecorded[2].shape == (0, 8)
    np.testing.assert_array_equal(controls, controls_before)
    for before, after in zip(arrays_before, [arg for arg in args if isinstance(arg, np.ndarray)]):
        np.testing.assert_array_equal(before, after)


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
                                  {'margin': -1}, {'knots': 0},
                                  {'grip_utilization': 0}, {'grip_utilization': 1.1},
                                  {'max_steering_reference_rate': 0}])
def test_invalid_configuration_rejected(config):
    with pytest.raises(ValueError):
        RacingMPCAgent(config)


def test_active_team_scenarios_use_one_fixed_mpc_opponent_profile():
    from pathlib import Path
    from core.scenario import load_and_expand_scenario, load_yaml_config
    profile = load_yaml_config(Path('configs/controllers/racing_mpc.yaml'))
    paths = sorted(Path('scenarios').glob('mappo_2v2_*.yaml'))
    assert {'mappo_2v2_continuous', 'mappo_2v2_race', 'mappo_2v2_asymmetric'} <= {path.stem for path in paths}
    for path in paths:
        scenario = load_and_expand_scenario(str(path))
        if scenario.get('two_team', {}).get('enabled', False):
            continue
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
    path = Path('scenarios/mappo_2v2_race.yaml').resolve()
    scenario = load_and_expand_scenario(str(path), overrides=['training_defaults.update_version="parallel-mappo-grouped256-batch2048-v1"',
         'training_defaults.batch_size=2048',
         'training_defaults.rollout_steps_per_env=256',
         'training_defaults.checkpoint_every_steps=1024000',
         'wandb.group="mappo-2v2-penalties-current-physics"',
         'wandb.tags=["mappo","2v2","terminal-incidents-v1","current-pretrain-physics","racing-mpc-opponents"]',
         'wandb.notes="Matched physics, observations, racing MPC opponents and rewards. Both-finished '
         'rate, then rank plus recorded penalties select checkpoints; clean finish time breaks successful '
         'ties."',
         'experiment.name="mappo_2v2_penalties_scratch"',
         'experiment.num_envs=400',
         'experiment.num_workers=100',
         'experiment.worker_startup_batch_size=8',
         'experiment.worker_startup_timeout_s=600',
         'experiment.worker_response_timeout_s=120',
         'experiment.terminal_recent_episodes=100',
         'experiment.terminal_every_updates=10',
         'experiment.terminal_diagnostic_every_updates=100',
         'experiment.terminal_episode_detail=false',
         'evaluation.selection_strategy="team_combined_penalties"',
         'evaluation.every_steps=1024000',
         'agents.car_0.reward.task.name="race_team_2v2_penalties"',
         'agents.car_0.reward.task.description="Shared completion/placement reward with recorded terminal '
         'race penalties."',
         'agents.car_0.reward.reward.collision.enabled=false',
         'agents.car_0.reward.reward.team_race_penalties={"enabled":true,"policy":"terminal_incidents_v1"}',
         'agents.car_1.reward.task.name="race_team_2v2_penalties"',
         'agents.car_1.reward.task.description="Shared completion/placement reward with recorded terminal '
         'race penalties."',
         'agents.car_1.reward.reward.collision.enabled=false',
         'agents.car_1.reward.reward.team_race_penalties={"enabled":true,"policy":"terminal_incidents_v1"}'])
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
                    assert 0 <= actions[aid][1] <= 100.00001
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
        controller.controller.field = np.full_like(controller.controller.field, controller.controller.margin/2)
        controller.act(obs['car_0'])
        assert controller.last_plan['brake_fallback']  # no overlap, but insufficient margin
        assert controller.last_plan['predicted_safety_slack_m'] < 0
        controller.controller.field = np.full_like(controller.controller.field, -1.)  # no feasible predicted route
        action = controller.act(obs['car_0'])
        assert controller.last_plan['brake_fallback']
        assert action[1] == 0.  # standing reset can request a complete stop
    finally:
        env.close()


def test_prediction_retains_lateral_and_yaw_momentum_at_zero_grip():
    p = P.copy()
    p[7] = 0.
    initial = np.array([0., 0., 0., 2., 0., 2., .5, 1.])
    future = predict_step(initial, np.array([0., 0.]), p, .05)
    assert future[1] == pytest.approx(.025, abs=1e-5)
    assert future[2] == pytest.approx(.05)
    assert future[7] == pytest.approx(1.)
    assert np.hypot(future[3], future[6]) == pytest.approx(np.hypot(2., .5), rel=1e-5)


@pytest.mark.parametrize('mu,speed', [(0., 2.), (.4, .3), (.8, 1.), (1.0489, 2.), (1.2, 3.5), (1.0489, 5.)])
def test_prediction_agrees_with_plant_for_short_sliding_rollout(mu, speed):
    from physics.vehicle import CombinedSlipVehicle
    config = {k: v for k, v in VEHICLE.items() if k not in {'length', 'width', 'wheel_actuators'}}
    plant = CombinedSlipVehicle({**config, 'mu': mu}, VEHICLE['wheel_actuators'])
    plant.reset(velocity=(speed, .25), yaw_rate=.8, steering_angle=.12, wheel_speed=speed/.05)
    plant.command(.15, speed/.05)
    params = P.copy()
    params[7] = mu
    predicted = predict_step(np.array([0., 0., 0., speed, .12, speed, .25, .8]),
                             np.array([.15, speed]), params, .05)
    actual = plant.advance(.05)
    np.testing.assert_allclose(predicted[:3], actual[:3], atol=.003)
    np.testing.assert_allclose(predicted[[3, 6, 7]], actual[[3, 4, 5]], atol=.08)


def test_wall_margin_is_required_even_without_body_overlap():
    controls = np.tile([0., 1.], (4, 1))
    _, slack, _ = _shoot(controls, *shooting_args(footprint=np.array([[0., .45]])))
    assert slack == pytest.approx(-.05)


def test_first_executed_steering_command_is_rate_limited_and_costed():
    previous = np.array([-.2, 1.])
    command = _limit_command(np.array([.4, 3.5]), previous, .05, 5., 1.5)
    np.testing.assert_allclose(command, [-.125, 1.25])
    args = list(shooting_args())
    args[0] = args[0].copy()
    args[0][3:9] = 0.  # no movement; isolate first steering transition cost
    args[9] = 1
    unchanged, _, _ = _shoot(np.array([[0., 0.]]), *args)
    changed, _, traj = _shoot(np.array([[.4, 0.]]), *args)
    assert changed-unchanged == pytest.approx(.075**2/.05)
    assert 0 < traj[0, 4] < .075


def test_grip_and_curvature_reduce_speed_and_anticipate_braking():
    straight = np.column_stack((np.arange(0., 2., .1), np.zeros(20)))
    angle = np.linspace(-np.pi/2, 0., 20)
    bend = np.column_stack((2.+.5*np.cos(angle), .5+.5*np.sin(angle)))
    points = np.vstack((straight, bend))
    high = _speed_profile(points, 3.5, 1., .8, 5.)
    low = _speed_profile(points, 3.5, .4, .8, 5.)
    assert np.all(low <= high+1e-10)
    assert low[-10] < high[-10] < 3.5
    assert low[15] < low[0]  # slows on the straight before reaching the bend
    np.testing.assert_array_equal(_speed_profile(points, 3.5, 0., .8, 5.), 0.)


def test_episode_grip_and_executed_command_match_prediction_after_reset():
    from scripts.benchmark_racing_opponents import ROOT, scenario_for
    from core.setup import create_training_setup
    scenario = scenario_for('racing_mpc', 'circle_map', 'solo', 10042, 20, 1)
    scenario['environment']['friction']['eval'] = {'mode': 'grid', 'values': [.4, 1.2]}
    env, agents, _ = create_training_setup(scenario, mode='eval', scenario_dir=ROOT/'scenarios')
    try:
        adapter = agents['car_0']
        adapter.set_env(env)
        seen = []
        for _ in range(2):
            obs, info = env.reset()
            adapter.reset()
            action = adapter.act(obs['car_0'])
            mpc = adapter.controller
            seen.append(mpc.last_plan['friction_mu'])
            assert seen[-1] == info['car_0']['physics']['mu']
            initial = np.array([*obs['car_0']['pose'], *[float(obs['car_0']['velocity'][0]),
                               float(obs['car_0']['steering_angle']),
                               float(obs['car_0']['wheel_speed'])*mpc.radius],
                               float(obs['car_0']['velocity'][1]),
                               float(env.get_agent_state('car_0').angular_velocity)])
            expected = predict_step(initial, np.array([action[0], action[1]*mpc.radius]), mpc.p, mpc.dt)
            np.testing.assert_allclose(mpc.last_plan['trajectory'][0], expected, atol=1e-7)
            assert abs(action[0]-obs['car_0']['steering_reference']) <= 1.5*env.timestep+1e-7
        assert sorted(seen) == [.4, 1.2]
    finally:
        env.close()


@pytest.mark.parametrize('mu', [-1., np.nan, np.inf])
def test_benchmark_rejects_invalid_grip(mu):
    from scripts.benchmark_racing_opponents import scenario_for
    with pytest.raises(ValueError, match='friction_mu'):
        scenario_for('racing_mpc', 'circle_map', 'solo', 10042, 20, 1, friction_mu=mu)


def test_benchmark_grip_override_applies_to_evaluation():
    from scripts.benchmark_racing_opponents import scenario_for
    scenario = scenario_for('racing_mpc', 'circle_map', 'pair', 10042, 20, 1, friction_mu=.8)
    assert scenario['environment']['friction']['eval'] == {'mode': 'fixed', 'mu': .8}


def test_active_workflows_share_five_mps_cap_with_learners_and_mpc():
    from env.spaces_builder import build_action_spaces
    from wrappers.actions.composer import ActionComposer, WheelReferenceAdapter
    paths = sorted(Path('scenarios').glob('*.yaml')) + sorted(Path('scenarios/render').glob('*.yaml'))
    checked = 0
    for path in paths:
        scenario = yaml.safe_load(path.read_text())
        vehicle = scenario.get('environment', {}).get('vehicle_params', {})
        if vehicle.get('model') != 'combined_slip_st':
            continue
        checked += 1
        actuators = vehicle['wheel_actuators']
        assert actuators['wheel_radius'] * actuators['wheel_speed_max'] == pytest.approx(5.), path
        space, _ = build_action_spaces(list(scenario['agents']), vehicle)
        dt = scenario['environment']['timestep'] * scenario['environment'].get('action_repeat', 1)
        for agent in scenario['agents'].values():
            if agent['algorithm'] in {'ppo', 'mappo'}:
                composer = ActionComposer.from_config(space.low, space.high, agent['action_constraints'], decision_dt=dt)
                for _ in range(100):
                    command = composer.process([0., 1.])
                assert command[1]*actuators['wheel_radius'] == pytest.approx(5.)
                assert composer.process([0., -1.])[1] < command[1]  # no integrator windup
            elif agent['algorithm'] == 'racing_mpc':
                assert agent['params']['max_speed'] == 5., path
                # The physical adapter uses the same shared ceiling, even if
                # a fixed controller accidentally requests too much speed.
                adapter = WheelReferenceAdapter(SimpleNamespace(act=lambda _: [0., 20.]), actuators)
                assert adapter.act({})[1]*actuators['wheel_radius'] == pytest.approx(5.)
    assert checked >= 10
