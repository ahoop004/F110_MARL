"""Real spawned races preserve the serial evaluation protocol and learner state."""
import os
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from agents.mappo import MAPPOAgent
from core.scenario import load_and_expand_scenario, validate_scenario, ScenarioError
from core.setup import create_training_setup, build_obs_composers, resolve_training_params
from training.mappo_evaluator import DeterministicMAPPOEvaluator
from training.parallel_mappo_evaluator import ParallelMAPPOEvaluator, _infer_actions, evaluation_workers
from wrappers.actions.composer import ActionComposer


def make_evaluator(workers, agent=None):
    directory = Path('scenarios').resolve()
    scenario = load_and_expand_scenario(str(directory / 'mappo_1v1_attack.yaml'))
    scenario['experiment']['seed'] = scenario['evaluation']['seed']
    for key in ('map_bundles', 'map_bundles_train', 'map_bundles_eval'):
        scenario['environment'][key] = ['circle_map', 'Budapest_map']
    scenario['environment']['max_steps'] = scenario['evaluation']['max_steps'] = 6
    scenario['agents']['car_1']['params'].update(horizon=2, knots=2, iterations=1, max_evaluations=5)
    scenario['agents']['car_0']['params'].update(device='cpu', pi_hidden_dims=[8], vf_hidden_dims=[8])
    env, opponents, _ = create_training_setup(scenario, mode='eval', scenario_dir=directory)
    obs = build_obs_composers(scenario['agents'], ['car_0'], scenario['environment'], directory)
    for controller in opponents.values():
        controller.set_env(env)
    space = env.action_spaces['car_0']
    if agent is None:
        params = resolve_training_params(scenario['agents']['car_0'], scenario)
        agent = MAPPOAgent(obs['car_0'].obs_dim, len(env.get_global_state().vector),
                           space.low, space.high, ['car_0'], params)
    actions = ActionComposer.from_config(space.low, space.high,
        scenario['agents']['car_0']['action_constraints'], decision_dt=env.timestep)
    kwargs = dict(env=env, trainable_ids=['car_0'], other_agents=opponents, obs_composers=obs,
                  action_composer=actions, episodes=3, base_seed=scenario['evaluation']['seed'])
    evaluator = (ParallelMAPPOEvaluator(scenario=scenario, scenario_dir=directory,
        num_workers=workers, **kwargs) if workers > 1 else DeterministicMAPPOEvaluator(**kwargs))
    return evaluator.bind_agent(agent)


def assert_nested_close(a, b, path='summary'):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_nested_close(a[key], b[key], f'{path}.{key}')
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for index, (x, y) in enumerate(zip(a, b)):
            assert_nested_close(x, y, f'{path}[{index}]')
    elif isinstance(a, float):
        assert a == pytest.approx(b, rel=1e-5, abs=1e-6), path
    else:
        assert a == b


def test_parallel_matches_serial_reuses_workers_and_observes_updated_policy():
    before_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    serial = make_evaluator(1)
    parallel = make_evaluator(2, serial.agent)
    processes = []
    try:
        progress = []
        parallel.set_progress_callback(progress.append)
        serial.agent.last_raw_actions = {'car_0': np.array([.3, -.4])}
        expected = serial.evaluate()
        numpy_state, python_state = np.random.get_state(), random.getstate()
        torch_state = torch.get_rng_state().clone()
        weights = {key: value.clone() for key, value in serial.agent.actor.state_dict().items()}
        threads = {key: os.environ.get(key) for key in ('OMP_NUM_THREADS', 'NUMBA_NUM_THREADS')}
        actual = parallel.evaluate()
        processes = list(parallel._processes)
        assert_nested_close(expected, actual)
        assert [r['seed'] for r in actual['episode_results']] == [10042, 10043, 10044]
        assert [r['map_id'] for r in actual['episode_results']] == ['circle_map', 'Budapest_map', 'circle_map']
        assert progress[-1] is None
        assert {r['episode'] for r in progress if r and r['status'] == 'complete'} == {1, 2, 3}
        assert all(r['workers'] == 2 for r in progress if r)
        assert serial.agent.actor.training
        np.testing.assert_array_equal(serial.agent.last_raw_actions['car_0'], [.3, -.4])
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
        assert np.random.get_state()[2:] == numpy_state[2:]
        assert random.getstate() == python_state
        assert torch.equal(torch.get_rng_state(), torch_state)
        assert threads == {key: os.environ.get(key) for key in threads}
        assert all(torch.equal(value, serial.agent.actor.state_dict()[key]) for key, value in weights.items())
        assert_nested_close(actual, parallel.evaluate())
        assert_nested_close(expected, serial.evaluate())
        with torch.no_grad():
            for parameter in serial.agent.actor.parameters():
                parameter.zero_()
            serial.agent.actor.net[-1].bias[1] = 1.
        updated = parallel.evaluate()
        assert parallel._processes == processes and all(p.is_alive() for p in processes)
        assert_nested_close(serial.evaluate(), updated)
        assert updated['episode_results'] != actual['episode_results']
    finally:
        parallel.close()
        serial.close()
        torch.set_num_threads(before_threads)
    assert all(not process.is_alive() for process in processes)


def test_worker_startup_failure_restores_parent_and_reaps_processes():
    evaluator = make_evaluator(2)
    evaluator.scenario['agents']['car_1']['algorithm'] = 'missing_controller'
    progress = []
    evaluator.set_progress_callback(progress.append)
    rng = torch.get_rng_state().clone()
    try:
        with pytest.raises(RuntimeError, match='Evaluation worker .* failed'):
            evaluator.evaluate()
        assert evaluator.agent.actor.training
        assert torch.equal(torch.get_rng_state(), rng)
        assert progress == [None]
        assert not evaluator._processes and not evaluator._connections
    finally:
        evaluator.close()


def test_inference_failure_closes_existing_workers(monkeypatch):
    evaluator = make_evaluator(2)
    processes = []
    try:
        evaluator.evaluate()
        processes = list(evaluator._processes)
        def fail(*args, **kwargs):
            raise RuntimeError('inference failed')
        monkeypatch.setattr(evaluator.agent, 'actor_actions', fail)
        with pytest.raises(RuntimeError, match='inference failed'):
            evaluator.evaluate()
        assert evaluator.agent.actor.training
        assert not evaluator._connections
        assert all(not p.is_alive() for p in processes)
    finally:
        evaluator.close()


def test_worker_timeout_is_not_hidden_by_other_workers():
    evaluator = ParallelMAPPOEvaluator.__new__(ParallelMAPPOEvaluator)
    with pytest.raises(RuntimeError, match='worker 0 timed out'):
        evaluator._ready({0, 1}, {0: 0., 1: float('inf')}, 1)


def test_recording_uses_serial_evaluator_without_spawning(monkeypatch):
    evaluator = make_evaluator(2)
    try:
        evaluator.recording = SimpleNamespace(close=lambda: None)
        monkeypatch.setattr(DeterministicMAPPOEvaluator, '_collect_episodes', lambda *args: ['serial'])
        with pytest.warns(RuntimeWarning, match='recording or rendering'):
            assert evaluator._collect_episodes({}) == ['serial']
        assert not evaluator._processes
    finally:
        evaluator.close()


@pytest.mark.parametrize('mode', ['shared', 'independent', 'lora'])
def test_ready_inference_routes_repeated_ids(mode):
    params = dict(device='cpu', hidden_dims=[8], actor_mode='independent' if mode == 'independent' else 'shared')
    if mode == 'lora':
        params['lora'] = dict(mode='per_agent', rank=2, alpha=2.)
    agent = MAPPOAgent(3, 4, -np.ones(2), np.ones(2), ['a', 'b'], params)
    agent._lora_ready = True
    requests = {0: (['a', 'b'], [np.zeros(3), np.ones(3)]), 1: (['b'], [np.full(3, 2.)])}
    result = _infer_actions(agent, requests)
    for worker, (ids, rows) in requests.items():
        expected, _ = agent.act_batch(ids, rows, deterministic=True)
        for aid in ids:
            np.testing.assert_array_equal(result[worker][aid], expected[aid])


def test_worker_count_auto_and_explicit_caps():
    scenario = dict(experiment=dict(num_envs=400, num_workers=100), evaluation=dict(num_workers='auto'))
    assert evaluation_workers(scenario, 8) == 8
    assert evaluation_workers(scenario, 3) == 3
    scenario['experiment']['num_envs'] = 1
    assert evaluation_workers(scenario, 8) == 1
    scenario['evaluation']['num_workers'] = 4
    assert evaluation_workers(scenario, 8) == 4


@pytest.mark.parametrize('value', [0, -1, True, 1.5, 'invalid'])
def test_invalid_worker_configuration(value):
    scenario = load_and_expand_scenario('scenarios/mappo_1v1_attack.yaml')
    scenario['evaluation']['num_workers'] = value
    with pytest.raises(ScenarioError, match='evaluation.num_workers'):
        validate_scenario(scenario)


def test_episode_spawn_index_uses_configured_range_and_wraps():
    from env.spawn import sample_centerline_relative_spawn
    angles = np.linspace(0, 2 * np.pi, 21)
    centerline = np.column_stack([10 * np.cos(angles), 10 * np.sin(angles)])
    kwargs = dict(spawn_policy='centerline_relative', centerline=centerline, map_split_mode='eval',
        spawn_centerline_cfg=dict(mode='random', min_progress=.2, max_progress=.4, avoid_finish=False),
        spawn_offsets_cfg={}, spawn_target_cfg={}, spawn_ego_cfg={}, agent_ids=['car_0'])
    for history in (0, 4, 19):
        result = sample_centerline_relative_spawn(**kwargs, rng=np.random.default_rng(42),
                                                  current_index=history, episode_index=7)
        assert result.metadata['spawn_s'] == 6 / 20
        assert result.next_index == 7


def test_invalid_episode_spawn_indices_are_rejected():
    evaluator = make_evaluator(1)
    try:
        for value in (None, True, -1, 1.5):
            with pytest.raises(ValueError, match='spawn_episode_index'):
                evaluator.env.reset(seed=42, options=dict(spawn_episode_index=value))
        with pytest.raises(ValueError, match='spawn_episode_index'):
            evaluator.env.reset(options=dict(spawn_episode_index=0))
    finally:
        evaluator.close()


def test_training_entrypoint_evaluates_with_live_collectors(tmp_path, monkeypatch):
    import run
    scenario = load_and_expand_scenario('scenarios/mappo_1v1_attack.yaml')
    scenario['experiment'].update(total_steps=8, num_envs=2, num_workers=2, torch_threads=1)
    scenario['training_defaults'].update(rollout_steps_per_env=2,
        pretrained_actor_checkpoint=None, require_pretrained_actor=False)
    scenario['agents']['car_0']['params'].update(device='cpu', pi_hidden_dims=[8],
        vf_hidden_dims=[8], batch_size=4, n_epochs=1, checkpoint_every_steps=4)
    scenario['agents']['car_1']['params'].update(horizon=2, knots=2, iterations=1, max_evaluations=5)
    for key in ('map_bundles', 'map_bundles_train', 'map_bundles_eval'):
        scenario['environment'][key] = ['circle_map']
    scenario['environment']['max_steps'] = 3
    scenario['evaluation'].update(num_workers='auto', every_steps=4, episodes=3, max_steps=3)
    scenario['wandb']['enabled'] = False
    monkeypatch.setattr(run, 'load_and_expand_scenario', lambda *args, **kwargs: scenario)
    monkeypatch.setattr('sys.argv', ['run.py', '--scenario', 'scenarios/mappo_1v1_attack.yaml',
        '--quiet', '--no-wandb', '--output-dir', str(tmp_path)])
    processes = []
    collect = ParallelMAPPOEvaluator._collect_episodes
    def capture(evaluator, protocol):
        result = collect(evaluator, protocol)
        assert evaluator.num_workers == 2
        if processes:
            assert evaluator._processes == processes
        else:
            processes.extend(evaluator._processes)
        return result
    monkeypatch.setattr(ParallelMAPPOEvaluator, '_collect_episodes', capture)
    before_threads = torch.get_num_threads()
    try:
        run.main()
    finally:
        torch.set_num_threads(before_threads)
    history = [json.loads(line) for line in (tmp_path / 'evaluation_history.jsonl').read_text().splitlines()]
    assert [row['environment_steps'] for row in history] == [4, 8]
    assert all(len(row['episode_results']) == 3 for row in history)
    assert (tmp_path / 'best_model.pt').is_file() and (tmp_path / 'final_model.pt').is_file()
    assert processes and all(not process.is_alive() for process in processes)
