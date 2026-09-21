"""Nonlinear physics must agree across reset, sensors, controls, and checkpoints."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from core.provenance import physics_contract, build_run_provenance
from core.scenario import load_and_expand_scenario, load_yaml_config, validate_scenario, ScenarioError
from core.setup import create_training_setup
from env.state_buffer import TerminalVehicleController, TerminalAgentConfig
from env.types import AgentRaceStatus
from physics.vehicle import RaceCar
from wrappers.actions.composer import ActionComposer
from wrappers.observations.composer import ObservationComposer
from wrappers.observations.track import FrenetVehicleTrackComponent


@pytest.fixture
def scenario():
    config = load_and_expand_scenario('scenarios/ppo_lap_completion_pretrain.yaml')
    config['experiment'].update(num_envs=1, total_steps=None)
    config['environment'].pop('track_limits', None)
    config['environment']['terminate_on_collision'] = True
    config['environment']['episode_termination']['lap_completion'] = True
    config['environment'].pop('spawn')
    for key in ('map_bundles', 'map_bundles_train', 'map_bundles_eval'):
        config['environment'][key] = ['circle_map']
    config['environment'].pop('friction')
    config['environment'].update(timestep=.01, action_repeat=2, max_steps=32)
    config['agents']['car_0']['algorithm'] = 'mappo'
    config['agents']['car_1'] = deepcopy(config['agents']['car_0'])
    return config


@pytest.fixture
def env(scenario):
    environment, _, _ = create_training_setup(scenario, scenario_dir=Path('scenarios'))
    yield environment
    environment.close()


def test_runtime_profile_retains_component_calibration(scenario):
    profile = load_yaml_config('configs/vehicle/combined_slip_development.yaml')
    params = scenario['environment']['vehicle_params']
    assert params['wheel_actuators'] == profile['wheel_actuators']
    assert {k: v for k, v in params.items() if k not in {'wheel_actuators', 'length', 'width'}} == profile['combined_slip_vehicle']


def test_rolling_reset_actual_wheel_observation_and_public_velocity(env, scenario):
    observations, infos = env.reset(seed=42, options={'velocities': {'car_0': 2.0, 'car_1': -1.0}})
    car = env.sim.agents[0]
    assert car.physics_state.shape == (8,)
    assert car.state.shape == (7,)  # Geometry compatibility view, not integration state.
    assert observations['car_0']['wheel_speed'] == pytest.approx(40)
    assert observations['car_0']['wheel_speed_reference'] == pytest.approx(40)
    assert observations['car_1']['wheel_speed'] == pytest.approx(-20)
    np.testing.assert_allclose(env.get_agent_state('car_0').velocity, [2, 0])
    assert env.get_agent_state('car_0').metadata['wheel_speed'] == pytest.approx(40)
    composer = ObservationComposer.from_file('configs/observations/rl_racer_simulated_wheel.yaml', scenario['environment'])
    wrapped = composer.wrap(observations['car_0'], infos['car_0'])
    assert wrapped.shape == (158,)
    assert wrapped[108 + 9] == pytest.approx(40 / 400)
    global_size = env.get_global_state().vector.size
    for _ in range(8):
        obs, *_ = env.step({'car_0': np.array([.15, 100.0]), 'car_1': np.array([0, -20.0])})
    state = car.physics_state
    np.testing.assert_allclose(obs['car_0']['velocity'], state[3:5], atol=1e-6)
    assert abs(state[7] - state[3] / .05) > .01
    assert env.get_global_state().vector.size == global_size
    assert np.isfinite(composer.wrap(obs['car_0'])).all()
    snapshot = car.physics_state
    snapshot[:] = 0
    assert car.physics_state[7] != 0
    env.reset(seed=42)
    assert car.physics_state[7] == 0
    np.testing.assert_array_equal(car.control_reference, [0, 0])


def test_command_clipping_and_reference_derivative_use_wheel_units(env):
    env.reset(seed=42)
    obs, *_ = env.step({'car_0': np.array([2, 1000])})
    raw = obs['car_0']
    assert raw['wheel_speed_reference'] == 400
    assert raw['wheel_speed_reference_rate'] == pytest.approx(400 / .02)
    assert 'speed_reference' not in raw
    assert raw['steering_reference'] == pytest.approx(.4189)
    assert raw['wheel_speed'] == pytest.approx(400 * (1 - np.exp(-.01/.15)))
    with pytest.raises(ValueError, match='finite'):
        env.step({'car_0': np.array([0, np.nan])})


def test_wheel_reference_integrates_once_per_decision_and_clamps():
    constraints = {'speed_control': 'wheel_acceleration', 'max_wheel_acceleration': 100,
                   'max_wheel_deceleration': 50, 'prevent_reverse': False}
    composer = ActionComposer.from_config(np.array([-.4, -4]), np.array([.4, 4]), constraints, decision_dt=.02)
    first = composer.process(np.array([.5, 1]))
    np.testing.assert_allclose(first, [.2, 2])
    for _ in range(10):
        assert composer.process(np.array([0, 1]))[1] == 4
    assert composer.process(np.array([0, -1]))[1] == 3  # no windup
    composer.reset()
    assert composer.process(np.array([0, -1]))[1] == -1
    assert ActionComposer.contract_from_config(constraints, .02)['rate_units'] == 'rad/s^2'


def test_low_speed_lateral_motion_is_not_discarded(scenario):
    # A pure lateral low-speed condition cannot be recovered via legacy's <0.5 switch.
    params = scenario['environment']['vehicle_params']
    car = RaceCar(params, seed=42, num_beams=16, integrator='RK4')
    car._physics.reset(velocity=(.01, .1), wheel_speed=.2)
    car._sync_physics_view()
    np.testing.assert_allclose(car.body_velocity, [.01, .1])
    car.set_longitudinal_speed(-.01)
    np.testing.assert_allclose(car.body_velocity, [-.01, .1])
    assert car.physics_state[7] == pytest.approx(-.2)


def test_wall_collision_rolls_back_full_state_and_stops_wheels(env, monkeypatch):
    env.reset(seed=42, options={'velocities': {'car_0': 2}})
    car = env.sim.agents[0]
    old_pose = car.physics_state[:3].copy()
    original_scan = car.compute_scan
    def collision_scan():
        result = original_scan()
        car.in_collision = True
        return result
    monkeypatch.setattr(car, 'compute_scan', collision_scan)
    env.step({'car_0': np.array([.2, 100]), 'car_1': np.array([0, 20])})
    np.testing.assert_array_equal(car.physics_state[:3], old_pose)
    np.testing.assert_array_equal(car.physics_state[[3, 4, 5, 7]], np.zeros(4))
    assert 'car_0' not in env.agents
    for _ in range(3):
        env.step({'car_1': np.array([0, 20])})
    np.testing.assert_array_equal(car.physics_state[:3], old_pose)
    assert car.physics_state[7] == 0
    monkeypatch.setattr(car, 'compute_scan', original_scan)
    env.reset(seed=42)
    env.step({'car_0': np.array([0, 40])})
    assert car.physics_state[7] > 0  # reset releases the terminal freeze


@pytest.mark.parametrize('status', [AgentRaceStatus.CRASHED, AgentRaceStatus.TRUNCATED, AgentRaceStatus.FINISHED])
def test_terminal_controller_clears_wheel_state_and_stays_stopped(env, status):
    env.reset(seed=42, options={'velocities': {'car_0': 2}})
    car = env.sim.agents[0]
    controller = TerminalVehicleController(env.possible_agents, TerminalAgentConfig(finish_clearance_steps=2))
    controller.capture('car_0', status=status, terminal_step=0, action=np.array([.2, 80]), vehicle_state=car.physics_state)
    commands = np.zeros((2, 2))
    controller.apply(commands, agent_index={'car_0': 0, 'car_1': 1}, simulator=env.sim, step=1)
    if status == AgentRaceStatus.FINISHED:
        np.testing.assert_allclose(commands[0], [.1, 40])
    controller.apply(commands, agent_index={'car_0': 0, 'car_1': 1}, simulator=env.sim, step=2)
    pose = car.physics_state[:3].copy()
    car.update_pose(.4, 400)
    np.testing.assert_array_equal(car.physics_state[:3], pose)
    np.testing.assert_array_equal(car.physics_state[[3, 4, 5, 7]], np.zeros(4))


def test_reset_reproducibility_and_environment_isolation(env, scenario):
    other, _, _ = create_training_setup(scenario, scenario_dir=Path('scenarios'))
    try:
        env.reset(seed=42)
        other.reset(seed=42)
        commands = {'car_0': np.array([.1, 50]), 'car_1': np.array([-.1, 60])}
        for _ in range(5):
            env.step(commands)
            other.step(commands)
        for left, right in zip(env.sim.agents, other.sim.agents):
            np.testing.assert_array_equal(left.physics_state, right.physics_state)
        before = env.sim.agents[0].physics_state
        other.reset(seed=43)
        np.testing.assert_array_equal(env.sim.agents[0].physics_state, before)
        assert env.sim.scan_simulator is not other.sim.scan_simulator
    finally:
        other.close()


def test_speed_locking_and_runtime_model_change_are_explicit_errors(env):
    with pytest.raises(ValueError, match='locking'):
        env.reset(seed=42, options={'velocities': {'car_0': 2}, 'lock_speed_steps': 2})
    env.reset(seed=42)
    with pytest.raises(ValueError, match='Recreate'):
        env.update_params(env.params)


@pytest.mark.parametrize('bad', ['legacy_action', 'fixed_controller', 'wheel_index', 'missing_actuators', 'legacy_coefficient'])
def test_reject_incompatible_scenario_contracts(scenario, bad):
    config = deepcopy(scenario)
    agent = config['agents']['car_0']
    if bad == 'legacy_action':
        agent['action_constraints']['speed_control'] = 'acceleration'
    elif bad == 'fixed_controller':
        agent['algorithm'] = 'ftg'
    elif bad == 'wheel_index':
        agent['action_constraints']['speed_index'] = 0
    elif bad == 'missing_actuators':
        del config['environment']['vehicle_params']['wheel_actuators']
    else:
        config['environment']['vehicle_params']['C_Sf'] = 5
    with pytest.raises(ScenarioError):
        validate_scenario(config)


def test_reject_observation_radius_and_wheel_source_mismatches(scenario):
    config = load_yaml_config('configs/observations/rl_racer_simulated_wheel.yaml')
    config['observation']['frenet_vehicle_track']['wheel_radius'] = .06
    with pytest.raises(ValueError, match='radius'):
        ObservationComposer.from_config(config, scenario['environment'])
    with pytest.raises(ValueError, match='simulated_v1'):
        ObservationComposer.from_file('configs/observations/rl_racer_vehicle_track_frenet.yaml', scenario['environment'])
    wheel = FrenetVehicleTrackComponent(points=1, wheel_radius=.05, wheel_speed_source='simulated_v1')
    with pytest.raises(KeyError, match='wheel_speed'):
        wheel.compute_into({'velocity': [2, 0]}, {}, np.empty(12))


@pytest.mark.parametrize('algorithm', ['ppo', 'mappo'])
def test_checkpoint_rejects_same_shape_physics_and_observation_changes(tmp_path, scenario, algorithm):
    from agents.ppo import PPOAgent
    from agents.mappo import MAPPOAgent
    physics = physics_contract(scenario['environment'])
    params = {'hidden_dims': [4], 'n_steps': 2, 'device': 'cpu',
              '_physics_contract': physics, '_observation_contract': {'wheel': 'simulated_v1'}}
    def make(config):
        if algorithm == 'ppo':
            return PPOAgent(3, -np.ones(2), np.ones(2), config)
        return MAPPOAgent(3, 6, -np.ones(2), np.ones(2), ['car_0', 'car_1'], config)
    source = make(params)
    path = tmp_path / 'checkpoint.pt'
    source.save(str(path))
    make(params).load(str(path))
    for key, changed in [('_physics_contract', None), ('_observation_contract', None),
                         ('_observation_contract', {'wheel': 'rolling_estimate'})]:
        with pytest.raises(ValueError, match='contract'):
            make({**params, key: changed}).load(str(path))
    changed = deepcopy(physics)
    changed['vehicle_params']['mu'] = .8
    with pytest.raises(ValueError, match='physics_contract'):
        make({**params, '_physics_contract': changed}).load(str(path))
    # Missing metadata in an old checkpoint is not permission to use new semantics.
    make({**params, '_physics_contract': None, '_observation_contract': None}).save(str(path))
    with pytest.raises(ValueError, match='physics_contract'):
        make(params).load(str(path))


def test_ppo_to_mappo_actor_transfer_checks_physics_before_loading(tmp_path, scenario):
    from agents.ppo import PPOAgent
    from agents.mappo import MAPPOAgent
    params = {'hidden_dims': [4], 'n_steps': 2, 'device': 'cpu',
              '_physics_contract': physics_contract(scenario['environment'])}
    source = PPOAgent(3, -np.ones(2), np.ones(2), params)
    path = tmp_path / 'actor.pt'
    source.save(str(path))
    target = MAPPOAgent(3, 6, -np.ones(2), np.ones(2), ['car_0', 'car_1'], params)
    target.load_pretrained_actor(str(path))
    legacy = MAPPOAgent(3, 6, -np.ones(2), np.ones(2), ['car_0', 'car_1'], {**params, '_physics_contract': None})
    with pytest.raises(ValueError, match='physics_contract'):
        legacy.load_pretrained_actor(str(path))


def test_provenance_records_resolved_parameters_and_units(scenario):
    result = build_run_provenance(scenario, scenario_path='scenarios/ppo_lap_completion_pretrain.yaml',
                                 run_id='physics-test', algorithm='mappo', trainable_agents=['car_0', 'car_1'])
    contract = result['physics_contract']
    assert contract['vehicle_params']['calibration']['status'] == 'uncalibrated'
    assert contract['action_units'] == ['rad', 'rad/s']
    assert contract['state_layout'][-1] == 'omega'


def test_retired_reduced_model_version_cannot_load_as_mf61(scenario):
    config = deepcopy(scenario)
    config['environment']['vehicle_params']['model_version'] = 1
    with pytest.raises(ScenarioError, match='model_version'):
        validate_scenario(config)
    current = physics_contract(scenario['environment'])
    assert current['vehicle_params']['tire_model'] == 'mf61_planar'
    assert current['vehicle_params']['model_version'] == 2
