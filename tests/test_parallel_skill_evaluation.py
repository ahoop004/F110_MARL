"""Real spawned skill suites must preserve serial trials and selection facts."""
from copy import deepcopy
import json
import random

import numpy as np
import pytest
import torch

from agents.mappo import MAPPOAgent
from core.setup import build_obs_composers, create_training_setup, resolve_training_params
from training.skill_curriculum import SkillCurriculum
from training.skill_evaluator import SkillEvaluator
from test_skill_adapters import DIRECTORY, tiny_scenario


def setup(tmp_path, skill):
    cfg, _ = tiny_scenario(tmp_path, skill, 4)
    cfg['experiment']['num_workers'] = 2
    cfg['evaluation']['num_workers'] = 2
    for key in ('map_bundles', 'map_bundles_train', 'map_bundles_eval'):
        cfg['environment'][key] = ['circle_map', 'Budapest_map']
    # Different stage maps, uneven worker assignments, and multiple seeds/map.
    for stage, maps, count in zip(cfg['skill_curriculum']['stages'],
            [['circle_map', 'Budapest_map'], ['Budapest_map'], ['Budapest_map', 'circle_map']], [1, 3, 1]):
        stage.update(maps=maps, evaluation_episodes_per_map=count)
    cfg['evaluation']['episodes'] = 7
    cfg['evaluation']['final_test']['episodes'] = 14
    env, _, _ = create_training_setup(cfg, scenario_dir=DIRECTORY)
    try:
        obs = build_obs_composers(cfg['agents'], ['car_0'], cfg['environment'], DIRECTORY)
        snapshot = env.get_global_state()
        params = resolve_training_params(cfg['agents']['car_0'], cfg)
        params.update(_observation_contract=obs['car_0'].contract,
            _global_state_contract_version=snapshot.metadata['vector_contract_version'])
        space = env.action_spaces['car_0']
        agent = MAPPOAgent(obs['car_0'].obs_dim, len(snapshot.vector), space.low, space.high, ['car_0'], params)
        agent.load_pretrained_actor(cfg['training_defaults']['pretrained_actor_checkpoint'])
    finally:
        env.close()
    return cfg, agent


@pytest.mark.parametrize('skill', ['pass', 'defend', 'pressure', 'recovery'])
def test_skill_suites_match_serial_with_one_shared_pool_and_updated_adapter(tmp_path, skill):
    cfg, agent = setup(tmp_path, skill)
    serial_cfg = deepcopy(cfg)
    serial_cfg['evaluation']['num_workers'] = 1
    serial = SkillEvaluator(scenario=serial_cfg, scenario_dir=DIRECTORY, agent=agent, output_dir=tmp_path / 'serial')
    parallel = SkillEvaluator(scenario=cfg, scenario_dir=DIRECTORY, agent=agent, output_dir=tmp_path / 'parallel')
    processes = []
    try:
        expected = serial.evaluate()
        progress, target_inputs = [], []
        parallel.set_progress_callback(progress.append)
        act = agent.act_batch
        def capture(ids, rows, deterministic=False):
            if len(progress) and progress[-1] and progress[-1]['suite'] == 'solo_retention':
                target_inputs.extend(np.asarray(row)[-5:].copy() for row in rows)
            return act(ids, rows, deterministic=deterministic)
        agent.act_batch = capture
        agent.last_raw_actions = {'car_0': np.array([.3, -.4])}
        numpy_state, python_state, torch_state = np.random.get_state(), random.getstate(), torch.get_rng_state().clone()
        weights = {k: v.clone() for k, v in agent.actor.state_dict().items()}
        actual = parallel.evaluate()
        assert actual == expected
        assert len(actual['episode_results']) == 7 and len(actual['solo']['episode_results']) == 2
        assert actual['retention_passed'] is False  # Short execution probes cannot qualify a lap.
        assert [r['map_id'] for r in actual['episode_results']] == [
            'circle_map', 'Budapest_map', 'Budapest_map', 'Budapest_map', 'Budapest_map', 'Budapest_map', 'circle_map']
        assert [r['agents']['car_0']['skill']['stage_index'] for r in actual['episode_results']] == [0, 0, 1, 1, 1, 2, 2]
        if skill != 'recovery':
            assert target_inputs and all(np.array_equal(row, np.zeros(5)) for row in target_inputs)
        else:
            assert agent.obs_dim == 158  # Solo recovery has no target extension.
        assert {r['policy'] for r in progress if r} == {'base', 'adapter'}
        assert {r['workers'] for r in progress if r} == {2}
        assert progress[-1] is None
        assert agent.actor.training
        np.testing.assert_array_equal(agent.last_raw_actions['car_0'], [.3, -.4])
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
        assert np.random.get_state()[2:] == numpy_state[2:]
        assert random.getstate() == python_state and torch.equal(torch.get_rng_state(), torch_state)
        assert all(torch.equal(v, agent.actor.state_dict()[k]) for k, v in weights.items())
        processes = list(parallel._worker_pool._processes)
        assert len(processes) == 2 and all(p.is_alive() for p in processes)
        assert len(parallel._evaluators) == 4
        baseline_path = tmp_path / 'parallel' / 'skill_base_selection.json'
        baseline = baseline_path.read_bytes()
        progress.clear()
        assert parallel.evaluate() == expected
        assert {r['policy'] for r in progress if r} == {'adapter'}
        assert baseline_path.read_bytes() == baseline
        # Existing workers must request the newly bound adapter, not a stale copy
        # of either the baseline or the previous policy.
        with torch.no_grad():
            for bank in agent.actor.adapters:
                for residual in bank.values():
                    residual.B.fill_(.2)
        updated = parallel.evaluate()
        assert updated == serial.evaluate()
        assert updated['episode_results'] != expected['episode_results']
        assert parallel._worker_pool._processes == processes
        assert baseline_path.read_bytes() == baseline
        # Only these identical summaries reach the curriculum state machine.
        left, right = SkillCurriculum(cfg['skill_curriculum']), SkillCurriculum(cfg['skill_curriculum'])
        assert left.observe(expected) == right.observe(actual)
    finally:
        parallel.close()
        serial.close()
    assert processes and all(not p.is_alive() for p in processes)


def test_parallel_final_cross_skill_suite_preserves_counts_seeds_and_worker_budget(tmp_path, monkeypatch):
    import training.skill_evaluator as module
    cfg, agent = setup(tmp_path, 'pass')
    other, _ = tiny_scenario(tmp_path, 'defend', 1)
    monkeypatch.setattr(module, 'load_and_expand_scenario', lambda path: deepcopy(other))
    serial_cfg = deepcopy(cfg)
    serial_cfg['evaluation']['num_workers'] = 1
    serial = SkillEvaluator(scenario=serial_cfg, scenario_dir=DIRECTORY, agent=agent,
                           output_dir=tmp_path / 'serial', protocol='final')
    parallel = SkillEvaluator(scenario=cfg, scenario_dir=DIRECTORY, agent=agent,
                             output_dir=tmp_path / 'parallel', protocol='final')
    processes = []
    try:
        expected = serial.evaluate_final()
        actual = parallel.evaluate_final()
        assert actual == expected
        assert len(actual['episode_results']) == 14 and len(actual['solo']['episode_results']) == 4
        assert len(actual['cross_skill']['episode_results']) == 6
        assert 'curriculum' not in actual
        assert set(actual['base']) == {'pass', 'defend', 'solo'}
        assert actual['episode_results'][0]['seed'] == cfg['evaluation']['final_test']['seed']
        assert actual['cross_skill']['episode_results'][0]['seed'] == cfg['evaluation']['final_test']['seed'] + 100000
        assert actual['solo']['episode_results'][0]['seed'] == cfg['evaluation']['final_test']['seed'] + 200000
        assert len(parallel._evaluators) == 7
        assert all(e.num_workers == 2 for e in parallel._evaluators.values())
        processes = list(parallel._worker_pool._processes)
        assert len(processes) == 2
        assert not (tmp_path / 'parallel' / 'best_model.pt').exists()
    finally:
        parallel.close()
        serial.close()
    assert all(not p.is_alive() for p in processes)


def test_shared_pool_failure_reaps_every_stage_worker(tmp_path, monkeypatch):
    cfg, agent = setup(tmp_path, 'pass')
    evaluator = SkillEvaluator(scenario=cfg, scenario_dir=DIRECTORY, agent=agent, output_dir=tmp_path)
    processes = []
    try:
        evaluator.evaluate()
        processes = list(evaluator._worker_pool._processes)
        def fail(*args, **kwargs):
            raise RuntimeError('policy failed')
        monkeypatch.setattr(agent, 'act_batch', fail)
        with pytest.raises(RuntimeError, match='policy failed'):
            evaluator.evaluate()
        assert not evaluator._worker_pool._connections
        assert agent.actor.training and all(not p.is_alive() for p in processes)
    finally:
        evaluator.close()


def test_skill_training_evaluates_in_parallel_with_live_collectors(tmp_path, monkeypatch):
    import run
    cfg, _ = tiny_scenario(tmp_path, 'pass', 2)
    cfg['experiment'].update(total_steps=8, num_workers=2)
    for stage in cfg['skill_curriculum']['stages']:
        stage['evaluation_episodes_per_map'] = 2
    cfg['skill_curriculum']['retention']['episodes_per_map'] = 2
    cfg['evaluation'].update(num_workers='auto', episodes=6)
    cfg['evaluation']['final_test']['episodes'] = 12
    monkeypatch.setattr(run, 'load_and_expand_scenario', lambda *args, **kwargs: deepcopy(cfg))
    monkeypatch.setattr('sys.argv', ['run.py', '--scenario', str(DIRECTORY / 'mappo_1v1_pass_lora.yaml'),
        '--quiet', '--no-wandb', '--output-dir', str(tmp_path / 'train')])
    processes = []
    evaluate = SkillEvaluator.evaluate
    def capture(evaluator):
        result = evaluate(evaluator)
        if processes:
            assert evaluator._worker_pool._processes == processes
        else:
            processes.extend(evaluator._worker_pool._processes)
        return result
    monkeypatch.setattr(SkillEvaluator, 'evaluate', capture)
    run.main()
    history = [json.loads(row) for row in (tmp_path / 'train' / 'evaluation_history.jsonl').read_text().splitlines()]
    assert [row['environment_steps'] for row in history] == [4, 8]
    assert all(len(row['episode_results']) == 6 and not row['retention_passed'] for row in history)
    assert (tmp_path / 'train' / 'final_model.pt').is_file()
    assert not (tmp_path / 'train' / 'best_model.pt').exists()
    assert len(processes) == 2 and all(not p.is_alive() for p in processes)
    processes.clear()
    (tmp_path / 'source.pt').unlink()
    monkeypatch.setattr('sys.argv', ['run.py', '--scenario', str(DIRECTORY / 'mappo_1v1_pass_lora.yaml'),
        '--quiet', '--no-wandb', '--eval', '--eval-protocol', 'selection',
        '--checkpoint', str(tmp_path / 'train' / 'final_model.pt'), '--output-dir', str(tmp_path / 'eval')])
    run.main()
    report = json.loads((tmp_path / 'eval' / 'evaluation_report.json').read_text())
    assert len(report['summary']['episode_results']) == 6
    assert len(report['summary']['solo']['episode_results']) == 2
    assert len(processes) == 2 and all(not p.is_alive() for p in processes)
