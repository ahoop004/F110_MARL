"""Shared grip, explicit fixed-controller units, and episode provenance."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest

from core.agent_builder import build_fixed_policy_agents, get_trainable_agent_ids
from core.config import register_builtin_agents
from core.provenance import physics_contract
from core.scenario import load_and_expand_scenario, validate_scenario, ScenarioError
from core.setup import create_training_setup
from env.friction import EpisodeFriction, validate_friction_protocol
from training.hooks import PhysicsEpisodeHook
from wrappers.actions.composer import WheelReferenceAdapter
from wrappers.observations.composer import ObservationComposer


@pytest.fixture
def scenario():
    return load_and_expand_scenario('scenarios/ppo_combined_slip_vs_ftg_development.yaml')


@pytest.mark.parametrize('algorithm', ['ftg', 'pure_pursuit', 'stanley', 'hybrid_pp_ftg', 'kinematic_mpc'])
def test_controller_adapter_preserves_output_and_episode_reset(scenario, algorithm):
    register_builtin_agents()
    config = {'car_1': {'algorithm': algorithm, 'params': {}, 'action_adapter': 'rolling_speed_to_wheel_v1'}}
    wrapper = build_fixed_policy_agents(config, vehicle_params=scenario['environment']['vehicle_params'])['car_1']
    assert get_trainable_agent_ids(config) == []
    env, _, _ = create_training_setup(scenario, scenario_dir=Path('scenarios'))
    try:
        obs, _ = env.reset(seed=42)
        if hasattr(wrapper, 'set_env'):
            wrapper.set_env(env)
        wrapper.reset()
        expected = np.array(wrapper.controller.act(obs['car_1']), copy=True)
        expected[1] /= .05
        wrapper.reset()
        actual = wrapper.act(obs['car_1'])
        np.testing.assert_allclose(actual, np.clip(expected, wrapper.low, wrapper.high), atol=1e-6)
        assert actual.shape == (2,) and np.isfinite(actual).all()
        wrapper.reset()
        np.testing.assert_allclose(wrapper.act(obs['car_1']), actual)
    finally:
        env.close()


def test_adapter_conversion_negative_speed_clipping_and_no_input_mutation(scenario):
    values = np.array([.2, -2.])
    controller = SimpleNamespace(act=lambda obs: values, set_action_space=lambda space: setattr(controller, 'space', space))
    adapter = WheelReferenceAdapter(controller, scenario['environment']['vehicle_params']['wheel_actuators'])
    np.testing.assert_allclose(adapter.act({}), [.2, -40])
    np.testing.assert_array_equal(values, [.2, -2])
    values[:] = [2, 30]
    np.testing.assert_allclose(adapter.act({}), [.4189, 400])
    adapter.set_action_space(SimpleNamespace(low=np.array([-.4, -400]), high=np.array([.4, 400])))
    np.testing.assert_allclose(controller.space.high, [.4, 20])
    values[1] = np.nan
    with pytest.raises(ValueError, match='finite'):
        adapter.act({})


def test_randomization_is_shared_constant_and_does_not_leak_to_actor(scenario):
    env, _, _ = create_training_setup(scenario, scenario_dir=Path('scenarios'))
    composer = ObservationComposer.from_file('configs/observations/rl_racer_simulated_wheel.yaml', scenario['environment'])
    try:
        obs, infos = env.reset(seed=42)
        metadata = infos['car_0']['physics']
        assert .8 <= metadata['mu'] < 1.1
        assert infos['car_1']['physics'] == metadata
        assert metadata['mu'] != env.params['mu']  # nominal config is not mutated
        for car in env.sim.agents:
            assert car._physics.params['mu'] == metadata['mu']
        assert env.get_global_state().metadata['physics']['mu'] == metadata['mu']
        assert 'physics' not in obs['car_0'] and 'mu' not in obs['car_0']
        wrapped = composer.wrap(obs['car_0'], infos['car_0'])
        changed = deepcopy(infos['car_0']); changed['physics']['mu'] = 100
        np.testing.assert_array_equal(composer.wrap(obs['car_0'], changed), wrapped)
        infos['car_0']['physics']['mu'] = 100
        assert env.get_global_state().metadata['physics']['mu'] != 100
        for _ in range(5):
            _, _, _, _, infos = env.step({'car_0': [0, 20], 'car_1': [0, 20]})
        assert infos['car_0']['physics']['mu'] == env.sim.agents[0]._physics.params['mu']
        assert infos['car_0']['physics']['draw'] == 0
    finally:
        env.close()


def test_rng_reproducibility_and_independence(scenario):
    cfg = scenario['environment']['friction']
    a = EpisodeFriction(cfg, nominal_mu=1, seed=42, phase='train')
    b = EpisodeFriction(cfg, nominal_mu=1, seed=42, phase='train')
    sequence = [a.sample()['mu'] for _ in range(20)]
    np.random.seed(999)
    np.random.random(1000)
    assert [b.sample()['mu'] for _ in range(20)] == sequence
    a.reseed(42)
    assert a.sample()['mu'] == sequence[0]
    a.reseed(43)
    assert a.sample()['mu'] != sequence[0]
    assert len(set(sequence)) == len(sequence)


def test_reset_and_eval_grid_reproduce_without_changing_spawn(scenario):
    train, _, _ = create_training_setup(scenario, scenario_dir=Path('scenarios'))
    evaluation, _, _ = create_training_setup(scenario, mode='eval', scenario_dir=Path('scenarios'))
    try:
        observed = []
        for seed in range(42, 45):
            left, ti = train.reset(seed=seed)
            right, ei = evaluation.reset(seed=seed)
            np.testing.assert_array_equal(left['car_0']['pose'], right['car_0']['pose'])
            np.testing.assert_array_equal(left['car_0']['lidar'], right['car_0']['lidar'])
            observed.append(ei['car_0']['physics']['mu'])
            assert ei['car_0']['physics']['phase'] == 'eval'
            assert ti['car_0']['physics']['protocol']['mode'] == 'uniform'
        assert observed == [.8, .95, 1.1]
        _, first = train.reset(seed=42)
        train.reset()
        _, again = train.reset(seed=42)
        assert first['car_0']['physics'] == again['car_0']['physics']
    finally:
        train.close(); evaluation.close()


def test_zero_grip_spin_uses_sampled_force_parameter(scenario):
    config = deepcopy(scenario)
    config['environment']['friction']['train'] = {'mode': 'fixed', 'mu': 0}
    env, _, _ = create_training_setup(config, scenario_dir=Path('scenarios'))
    try:
        env.reset(seed=42)
        before = env.sim.agents[0].physics_state
        for _ in range(5):
            env.step({'car_0': [0, 100]})
        after = env.sim.agents[0].physics_state
        np.testing.assert_array_equal(before[:6], after[:6])
        assert after[7] > 0
        np.testing.assert_array_equal(env.sim.agents[0].physics_diagnostics()['tire_forces'][:,:2], np.zeros((2,2)))
    finally:
        env.close()


@pytest.mark.parametrize('mutation', ['eval_random', 'missing_eval', 'scope', 'negative', 'bounds', 'empty_grid', 'legacy'])
def test_invalid_protocol_fails_before_setup(scenario, mutation):
    config = deepcopy(scenario)
    protocol = config['environment']['friction']
    if mutation == 'eval_random': protocol['eval'] = protocol['train']
    elif mutation == 'missing_eval': del protocol['eval']
    elif mutation == 'scope': protocol['scope'] = 'independent'
    elif mutation == 'negative': protocol['train']['low'] = -1
    elif mutation == 'bounds': protocol['train']['low'] = 2
    elif mutation == 'empty_grid': protocol['eval']['values'] = []
    else: config['environment']['vehicle_params'] = {'model': 'legacy_st'}
    with pytest.raises(ScenarioError):
        validate_scenario(config)


def test_episode_log_deduplicates_agents_and_rejects_changing_grip(tmp_path, scenario):
    hook = PhysicsEpisodeHook(tmp_path)
    sample = EpisodeFriction(scenario['environment']['friction'], nominal_mu=1, seed=42, phase='train').sample()
    record = SimpleNamespace(episode_id='worker0_ep0', map_id='circle', info={'physics': sample})
    hook.on_step(record); hook.on_step(record)
    record.episode_id = 'worker1_ep0'
    hook.on_step(record)
    rows = [json.loads(line) for line in hook.path.read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]['physics']['mu'] == sample['mu']
    record.info = {'physics': {**sample, 'mu': .5}}
    with pytest.raises(ValueError, match='within an episode'):
        hook.on_step(record)


def test_protocol_is_part_of_checkpoint_identity(scenario):
    first = physics_contract(scenario['environment'])
    changed = deepcopy(scenario['environment'])
    changed['friction']['eval']['values'] = [.1]
    assert first != physics_contract(changed)
    assert 'friction_protocol' in first


def test_legacy_adapter_and_missing_opt_in_rejected(scenario):
    config = deepcopy(scenario)
    del config['agents']['car_1']['action_adapter']
    with pytest.raises(ScenarioError, match='adapter'):
        validate_scenario(config)
    legacy = load_and_expand_scenario('scenarios/ppo.yaml')
    legacy['agents']['car_1']['action_adapter'] = 'rolling_speed_to_wheel_v1'
    with pytest.raises(ScenarioError, match='adapter'):
        validate_scenario(legacy)
