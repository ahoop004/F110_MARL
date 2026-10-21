"""Frozen PPO and MAPPO evaluations must produce matched, usable W&B records."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from agents.common.networks import Actor
from agents.ppo import PPOAgent
from core.scenario import load_and_expand_scenario
from core.setup import build_obs_composer, resolve_training_params
from env.spaces_builder import build_action_spaces
from loggers.evaluation import EvaluationWandbLogger
from loggers.metric_policy import MetricPolicy


@pytest.fixture
def fake_wandb(monkeypatch):
    import wandb
    state = SimpleNamespace(inits=[], logs=[], definitions=[], finishes=0,
                            run=SimpleNamespace(id='test', name='test', url='https://example.test/run', summary={}))

    def init(**kwargs):
        state.inits.append(kwargs)
        return state.run

    def finish():
        state.finishes += 1

    monkeypatch.setattr(wandb, 'init', init)
    monkeypatch.setattr(wandb, 'log', lambda row, **kw: state.logs.append(dict(row)))
    monkeypatch.setattr(wandb, 'finish', finish)
    monkeypatch.setattr(wandb, 'config', SimpleNamespace(update=lambda *a, **kw: None))
    monkeypatch.setattr(wandb, 'define_metric', lambda *a, **kw: state.definitions.append((a, kw)))
    return state


def test_comparison_scenarios_share_the_evaluation_contract():
    ppo = load_and_expand_scenario('scenarios/ppo_circle_lap_completion_eval.yaml')
    mappo = load_and_expand_scenario('scenarios/mappo_solo_lap_completion_transfer.yaml')
    assert ppo['environment'] == mappo['environment']
    assert ppo['evaluation'] == mappo['evaluation']
    assert ppo['agents']['car_0']['action_constraints'] == mappo['agents']['car_0']['action_constraints']
    assert ppo['agents']['car_0']['reward'] == mappo['agents']['car_0']['reward']
    assert ppo['environment']['map_bundles_eval'] == ['circle_map']
    assert ppo['environment']['max_speed'] == 8.
    assert ppo['wandb']['group'] == mappo['wandb']['group']
    assert ppo['wandb']['project'] == mappo['wandb']['project']
    assert ppo['wandb']['name'] != mappo['wandb']['name']


def test_logger_keeps_lap_times_and_uses_paired_episode_axis(fake_wandb, tmp_path):
    scenario = load_and_expand_scenario('scenarios/ppo_circle_lap_completion_eval.yaml')
    logger = EvaluationWandbLogger(scenario, 'evaluation', Path('best_model.pt'), 'sha',
                                    123456, None, {'seeds': [20042]}, tmp_path)
    summary = {'completion_rate': 1., 'mean_valid_lap_time_s': 43.4,
               'mean_clean_finish_time_s': 137.2, 'race_count': 1, 'timeout_rate': 0.,
               'mean_net_progress': 3., 'unavailable': None, 'nonfinite': float('nan')}
    logger.log_episode(summary, 1, 20042)
    logger.finish(summary)
    assert fake_wandb.logs[0]['eval_episode/mean_valid_lap_time_s'] == 43.4
    assert fake_wandb.logs[0]['eval_episode/mean_clean_finish_time_s'] == 137.2
    assert fake_wandb.logs[0]['eval_episode/number'] == 1
    assert fake_wandb.logs[0]['eval_episode/seed'] == 20042
    assert (('eval_episode/*',), {'step_metric': 'eval_episode/number'}) in fake_wandb.definitions
    assert fake_wandb.run.summary['eval/mean_valid_lap_time_s'] == 43.4
    assert fake_wandb.finishes == 1
    assert fake_wandb.inits[0]['config']['standalone_evaluation/checkpoint_environment_steps'] == 123456
    assert fake_wandb.inits[0]['job_type'] == 'solo-circle-evaluation'
    assert not MetricPolicy({'groups': {'eval': False}}).accepts('eval_episode/completion_rate')


@pytest.mark.parametrize('algorithm', ['ppo', 'mappo'])
def test_cli_logs_frozen_policy_episodes_and_summary(tmp_path, monkeypatch, fake_wandb, algorithm):
    import run
    source = ('scenarios/ppo_circle_lap_completion_eval.yaml' if algorithm == 'ppo'
              else 'scenarios/mappo_solo_lap_completion_transfer.yaml')
    scenario = load_and_expand_scenario(source)
    scenario['evaluation'].update(max_steps=4)
    scenario['evaluation']['final_test']['episodes'] = 2
    cfg = scenario['agents']['car_0']
    cfg['params'].update(pi_hidden_dims=[8], vf_hidden_dims=[8])
    composer = build_obs_composer(cfg, scenario['environment'], Path('scenarios'))
    params = resolve_training_params(cfg, scenario)
    space, _ = build_action_spaces(['car_0'], scenario['environment']['vehicle_params'])
    checkpoint = tmp_path / 'best_model.pt'
    if algorithm == 'ppo':
        params['_observation_contract'] = composer.contract
        PPOAgent(composer.obs_dim, space.low, space.high, params).save(str(checkpoint))
    else:
        actor = Actor(composer.obs_dim, 2, [8], 'leaky_relu')
        torch.save(dict(algorithm='mappo', actor_mode='independent', lora_contract=None,
            agent_ids=['car_0'], actors={'car_0': actor.state_dict()}, obs_dims={'car_0': 192},
            action_dim=2, action_low=space.low, action_high=space.high,
            action_contract=params['_action_contract'], physics_contract=params['_physics_contract'],
            observation_contracts={'car_0': composer.contract}, actor_hidden_dims=[8],
            activation='leaky_relu'), checkpoint)
    original_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    monkeypatch.setattr(run, 'load_and_expand_scenario', lambda *a, **kw: deepcopy(scenario))
    monkeypatch.setattr(sys, 'argv', ['run.py', '--scenario', source, '--eval',
        *(['--eval-actor', 'car_0'] if algorithm == 'mappo' else []),
        '--checkpoint', str(checkpoint), '--eval-protocol', 'final', '--wandb', '--no-render',
        '--quiet', '--output-dir', str(tmp_path / 'eval')])
    run.main()
    report = json.loads((tmp_path / 'eval' / 'evaluation_report.json').read_text())
    assert len(fake_wandb.inits) == 1 and fake_wandb.finishes == 1
    rows = [row for row in fake_wandb.logs if 'eval_episode/number' in row]
    assert [r['eval_episode/number'] for r in rows] == [1, 2]
    assert [r['eval_episode/seed'] for r in rows] == report['seeds'] == [20042, 20043]
    assert report['wandb_url'] == fake_wandb.run.url
    assert fake_wandb.run.summary['eval/completion_rate'] == report['summary']['completion_rate']
    assert fake_wandb.run.summary['eval/valid_laps'] == report['summary']['valid_laps']
    assert fake_wandb.run.summary['evaluation/completed_episodes'] == 2
    assert [row['map_bundle'] for row in report['episode_results']] == ['circle_map', 'circle_map']
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == original_hash


def test_wandb_disabled_never_initializes_a_run(tmp_path, fake_wandb):
    scenario = load_and_expand_scenario('scenarios/ppo_circle_lap_completion_eval.yaml')
    scenario['wandb']['mode'] = 'disabled'
    logger = EvaluationWandbLogger(scenario, 'eval', Path('checkpoint.pt'), 'sha', None,
                                    None, {}, tmp_path)
    logger.log_episode({'completion_rate': 1.}, 1, 20042)
    logger.finish({'completion_rate': 1.})
    assert fake_wandb.inits == fake_wandb.logs == []
