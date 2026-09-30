"""Tactical task events, safe starts, retention and executable LoRA workflows."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from core.scenario import load_and_expand_scenario, validate_scenario
from core.setup import create_training_setup
from env.skills import SkillTracker, sample_skill_spawn, reset_skill_opponent
from metrics.outcomes import determine_outcome
from training.skill_curriculum import SkillCurriculum
from training.skill_evaluator import SkillEvaluator, retention_result, summarize_skill
from training.hooks import EvaluationCheckpointHook
from wrappers.rewards.skills import SkillRewardComponent


DIRECTORY = Path('scenarios').resolve()


def scenario(skill='pass'):
    return load_and_expand_scenario(str(DIRECTORY / f'mappo_1v1_{skill}_lora.yaml'))


def tracker(skill='pass', lead=-2., **kwargs):
    task = {**scenario(skill)['environment']['skill_task'], **kwargs}
    result = SkillTracker(task)
    result.reset(lead)
    return result


def advance(track, time, ego=0., target=0., speed=2., failed=False, opponent_failed=False, timed_out=False):
    infos = {'car_0': dict(centerline=dict(progress_delta=ego / 100., vs=speed),
                          track_limits=dict(exceeded=failed), target_frenet=dict(delta_s=-49.)),
             'car_1': dict(centerline=dict(progress_delta=target / 100., vs=2.),
                          track_limits=dict(exceeded=opponent_failed))}
    return track.update(time=time, infos=infos, collisions={'car_0': False, 'car_1': False},
                        track_length=100., timed_out=timed_out)


def test_pass_requires_sustained_forward_lead_and_awards_once():
    task = tracker()
    assert not advance(task, .1, ego=4., target=1.)['success']
    assert not advance(task, 1.)['success']
    facts = advance(task, 1.1, ego=2., target=2.)
    assert facts['success'] and facts['event']
    assert facts['ego_lead'] == pytest.approx(1.)
    reward = SkillRewardComponent({})
    assert reward.compute({'info': {'skill': facts}})['skill/success'] == 10.
    assert sum(reward.compute({'info': {'skill': advance(task, 2.)}}).values()) == 0.
    assert determine_outcome({'skill': facts}).value == 'skill_success'


@pytest.mark.parametrize('speed,delta', [(0., 0.), (2., -2.)])
def test_pass_confirmation_resets_when_stopped_or_lead_lost(speed, delta):
    task = tracker(lead=2.)
    advance(task, .1)
    assert not advance(task, .9, ego=delta, speed=speed)['success']
    assert not advance(task, 1.2, ego=-delta)['success']
    assert advance(task, 2.2)['success']


@pytest.mark.parametrize('failed,opponent_failed,outcome', [
    (True, False, 'ego_failure'), (False, True, 'opponent_failure'), (True, True, 'ego_failure')])
def test_failure_has_priority_over_simultaneous_pass(failed, opponent_failed, outcome):
    task = tracker(lead=2.)
    advance(task, .1)
    facts = advance(task, 1.1, ego=2., failed=failed, opponent_failed=opponent_failed)
    assert facts['outcome'] == outcome and not facts['success']
    reward = SkillRewardComponent({}).compute({'info': {'skill': facts}})
    assert 'skill/success' not in reward
    if failed:
        assert sum(reward.values()) == -20.


def test_order_uses_signed_earned_progress_not_wrapped_target_distance():
    task = tracker(lead=-2.)
    assert advance(task, .1, ego=1., target=.5)['ego_lead'] == -1.5
    assert advance(task, .2, ego=-1., target=-.5)['ego_lead'] == -2.
    # Half-track wrapping of target_frenet (fixed at -49 above) has no influence.
    assert advance(task, .3, ego=51., target=51.)['ego_lead'] == -2.


@pytest.mark.parametrize('progress,expected', [(39., 'defense_incomplete'), (40., 'success'), (70., 'success')])
def test_defense_requires_pace_and_accepts_pulling_away(progress, expected):
    task = tracker('defend', lead=3.)
    facts = advance(task, 20., ego=progress, target=35., timed_out=True)
    assert facts['outcome'] == expected


def test_defense_lost_lead_requires_confirmation_and_parking_earns_no_lead_bonus():
    task = tracker('defend', lead=3.)
    assert advance(task, .1, speed=0.)['lead_reward_s'] == 0.
    assert not advance(task, .2, target=5.)['done']
    facts = advance(task, 1.2)
    assert facts['outcome'] == 'lead_lost'
    assert SkillRewardComponent({}).compute({'info': {'skill': facts}})['skill/lead_lost'] == -10.


def test_pass_timeout_and_early_time_limit_are_unsuccessful():
    assert advance(tracker(), 30.)['outcome'] == 'timeout'
    assert advance(tracker(), .1, timed_out=True)['outcome'] == 'timeout'


def test_scenarios_preserve_frozen_base_contract_and_independent_defaults():
    base = load_and_expand_scenario(str(DIRECTORY / 'mappo_1v1_attack_lora.yaml'))
    for skill in ('pass', 'defend'):
        cfg = scenario(skill)
        assert cfg['training_defaults'] == base['training_defaults']
        assert cfg['agents']['car_0']['params'] == base['agents']['car_0']['params']
        for key in ('observation', 'action_constraints'):
            assert cfg['agents']['car_0'][key] == base['agents']['car_0'][key]
        assert cfg['environment']['vehicle_params'] == base['environment']['vehicle_params']
        assert cfg['environment']['map_bundles_train'] == ['circle_map']
        assert cfg['evaluation']['episodes'] == 80 and cfg['evaluation']['final_test']['episodes'] == 160


def test_frozen_zero_target_columns_still_learn_opponent_conditioning_through_lora():
    from agents.common.networks import Actor
    from agents.common.lora import LoRAActor
    torch.manual_seed(71)
    base = Actor(163, 2, [8, 8], activation='leaky_relu')
    with torch.no_grad():
        base.net[0].weight[:, 158:].zero_()
    frozen = deepcopy(base.net.state_dict())
    actor = LoRAActor(base, dict(mode='shared', rank=4, alpha=4., train_log_std=True), 1)
    obs = torch.ones(2, 163)
    obs[:, 158] = torch.tensor([-1., 1.])
    torch.testing.assert_close(actor(obs), base(obs), rtol=0, atol=0)
    torch.testing.assert_close(actor(obs)[0][0], actor(obs)[0][1], rtol=0, atol=0)
    optimizer = torch.optim.Adam([p for p in actor.parameters() if p.requires_grad], lr=.03)
    for _ in range(30):
        optimizer.zero_grad()
        loss = (actor(obs)[0][:, 0] - obs[:, 158]).square().mean()
        loss.backward()
        optimizer.step()
    assert (actor(obs)[0][1, 0] - actor(obs)[0][0, 0]).item() > .2
    for key, value in actor.net.state_dict().items():
        assert torch.equal(value, frozen[key])


@pytest.mark.parametrize('mutation', [
    lambda s: s['environment'].update(respawn_agents=['car_1']),
    lambda s: s['environment']['skill_task'].update(lead_margin=-1),
    lambda s: s['skill_curriculum']['stages'][0].update(gap=[4., 2.]),
    lambda s: s['skill_curriculum']['stages'][0].update(opponent_speed=[0., 0.]),
    lambda s: s['evaluation']['final_test'].update(seed=10042),
    lambda s: s['evaluation'].update(episodes=1),
])
def test_invalid_skill_configuration_fails_early(mutation):
    cfg = scenario()
    mutation(cfg)
    with pytest.raises(ValueError):
        # ScenarioError historically extends Exception, rather than ValueError.
        from training.skill_curriculum import validate_skill_curriculum
        validate_skill_curriculum(cfg)


@pytest.fixture
def skill_env():
    cfg = scenario()
    env, controllers, _ = create_training_setup(cfg, scenario_dir=DIRECTORY)
    try:
        yield env, controllers
    finally:
        env.close()


def test_safe_spawns_are_seeded_stage_changes_wait_for_reset_and_mpc_refreshes(skill_env):
    env, controllers = skill_env
    env.reset(seed=8)
    first = env.sim.agent_poses.copy()
    metadata = deepcopy(env.skill_spawn)
    env.reset(seed=8)
    np.testing.assert_array_equal(first, env.sim.agent_poses)
    assert metadata == env.skill_spawn
    assert not env.sim.current_observation()['collisions'].any()
    assert env._skill_tracker.lead == pytest.approx(metadata['ego_lead'], abs=.01)
    env.set_skill_stage(2)
    assert env._skill_stage == 0 and env._map_bundle_active == 'circle_map'
    _, infos = env.reset(seed=8, options={'map_episode_index': 1})
    reset_skill_opponent(env, controllers)
    assert env._skill_stage == 2
    assert env._map_bundle_active == 'circle_map'
    assert controllers['car_1'].controller.max_speed == env.skill_spawn['opponent_speed']
    assert 4. <= controllers['car_1'].max_speed <= 4.5
    assert not any(v['track_limits']['exceeded'] for v in infos.values())


def test_spawn_rejects_impossible_footprint_clearance(skill_env):
    env, _ = skill_env
    env.reset(seed=9)
    with pytest.raises(RuntimeError, match='256 attempts'):
        sample_skill_spawn(geometry=env._centerline_progress_tracker._geometry, walls=env.walls,
            rng=np.random.default_rng(1), stage=env._skill_stages[0], task=env._skill_tracker.config,
            agent_ids=env.possible_agents, length=10000., width=10000.)


def test_generalization_spawns_are_safe_on_every_configured_map(skill_env):
    env, _ = skill_env
    env.set_skill_stage(2)
    maps = env._skill_stages[2]['maps']
    for index, name in enumerate(maps):
        for seed in (5, 17):
            _, infos = env.reset(seed=seed, options={'map_episode_index': index})
            assert env._map_bundle_active == name
            assert not env.sim.current_observation()['collisions'].any()
            assert not any(v['track_limits']['exceeded'] for v in infos.values())
            assert -8.01 <= infos['car_0']['skill']['ego_lead'] <= -1.99


@pytest.mark.parametrize('outcome,terminated,truncated', [
    ('success', True, False), ('timeout', False, True), ('duration_timeout', False, True)])
def test_task_termination_does_not_finish_laps_and_timeout_bootstraps(skill_env, outcome, terminated, truncated):
    env, _ = skill_env
    env.reset(seed=42)
    if outcome == 'success':
        env._skill_tracker.lead = 2.
        env._skill_tracker.confirm_since = -2.
    elif outcome == 'timeout':
        env.max_steps = 1
    else:
        env.max_steps = 10
        env._skill_tracker.config['duration_s'] = .05
        outcome = 'timeout'
    _, _, terms, truncs, infos = env.step({'car_0': np.array([0., 40.]), 'car_1': np.array([0., 40.])})
    assert terms['car_0'] is terminated and truncs['car_0'] is truncated
    assert not infos['car_0']['race_completed']
    assert infos['car_0']['finish_position'] is None
    assert infos['car_0']['skill']['outcome'] == outcome
    assert not env.agents
    assert not env.get_global_state().masks['active_mask'].any()


def test_curriculum_requires_success_safety_retention_and_prior_stage_retention():
    curriculum = SkillCurriculum(scenario()['skill_curriculum'])
    good = dict(retention_passed=True, stages=[dict(success_rate=.8, ego_failure_rate=.1)] * 3)
    assert curriculum.observe(good)['stage'] == 0
    bad = {**good, 'retention_passed': False}
    assert curriculum.observe(bad)['streak'] == 0
    curriculum.observe(good)
    assert curriculum.observe(good)['stage'] == 1
    bad = deepcopy(good)
    bad['stages'][0] = dict(success_rate=.7, ego_failure_rate=0.)
    assert curriculum.observe(bad)['streak'] == 0
    bad = deepcopy(good)
    bad['stages'][1] = dict(success_rate=.9, ego_failure_rate=.2)
    assert curriculum.observe(bad)['streak'] == 0
    curriculum.observe(good)
    assert curriculum.observe(good)['stage'] == 2


def solo_rows(count=20, time=10., failures=0):
    return [dict(map_id='circle_map', seed=i, agents={'car_0': dict(finished=i >= failures,
        collision_dnf=i < failures, boundary_dnf=False, boundary_respawns=0, collision_respawns=0,
        mean_valid_lap_time_s=time if i >= failures else None)}) for i in range(count)]


def test_retention_pairs_clean_laps_and_rejects_missing_timing_or_bad_rates():
    base = solo_rows()
    assert retention_result(base, solo_rows(time=11., failures=1), 'car_0', {})['retention_passed']
    assert not retention_result(base, solo_rows(time=11.01), 'car_0', {})['retention_passed']
    assert not retention_result(base, solo_rows(failures=2), 'car_0', {})['retention_passed']
    assert not retention_result(solo_rows(failures=20), solo_rows(failures=20), 'car_0', {})['retention_passed']
    with pytest.raises(ValueError, match='identical'):
        retention_result(base, solo_rows(count=19), 'car_0', {})


def test_checkpoint_selection_never_promotes_failed_retention(tmp_path):
    agent = SimpleNamespace(save=lambda path: torch.save({}, path))
    evaluator = SimpleNamespace(evaluate=lambda: dict(skill_success_rate=1., skill_ego_failure_rate=0.,
        skill_tiebreak=1., retention_passed=False))
    hook = EvaluationCheckpointHook(agent, str(tmp_path), evaluator, evaluate_every=1, selection_strategy='skill')
    hook._evaluate_checkpoint(1)
    assert not (tmp_path / 'best_model.pt').exists()
    evaluator.evaluate = lambda: dict(skill_success_rate=.8, skill_ego_failure_rate=.1,
        skill_tiebreak=1., retention_passed=True)
    hook._evaluate_checkpoint(2)
    assert (tmp_path / 'best_model.pt').exists()
    before = (tmp_path / 'best_model.pt').read_bytes()
    evaluator.evaluate = lambda: dict(skill_success_rate=1., skill_ego_failure_rate=0.,
        skill_tiebreak=2., retention_passed=False)
    hook._evaluate_checkpoint(3)
    assert (tmp_path / 'best_model.pt').read_bytes() == before


def tiny_scenario(tmp_path, skill, num_envs):
    """Real compatible actor transfer, with small networks and bounded trials."""
    import run
    from agents.ppo import PPOAgent
    from env.spaces_builder import build_action_spaces
    torch.set_num_threads(1)
    cfg = scenario(skill)
    cfg['experiment'].update(total_steps=16, num_envs=num_envs, num_workers=1)
    for key in ('map_bundles', 'map_bundles_train', 'map_bundles_eval'):
        cfg['environment'][key] = ['circle_map']
    cfg['environment']['max_steps'] = 3
    for stage in cfg['skill_curriculum']['stages']:
        stage['maps'] = ['circle_map']
        stage['evaluation_episodes_per_map'] = 1
    cfg['skill_curriculum'].update(success_threshold=0., max_ego_failure_rate=1., required_evaluations=1)
    cfg['skill_curriculum']['retention'].update(episodes_per_map=1, max_steps=3)
    cfg['evaluation'].update(episodes=3, every_steps=4, max_steps=3)
    cfg['evaluation']['final_test']['episodes'] = 6
    cfg['training_defaults'].update(rollout_steps_per_env=2, checkpoint_every_steps=4)
    cfg['agents']['car_0']['params'].update(pi_hidden_dims=[8, 8], vf_hidden_dims=[8],
        device='cpu', n_steps=4, batch_size=2, n_epochs=1, checkpoint_every_steps=4)
    cfg['agents']['car_1']['params'].update(horizon=2, knots=2, iterations=1, max_evaluations=5)
    source_cfg = deepcopy(cfg['agents']['car_0'])
    source_cfg['observation']['observation'].pop('target_frenet')
    composer = run.build_obs_composer(source_cfg, cfg['environment'], DIRECTORY)
    params = run.resolve_training_params(source_cfg, cfg)
    params['_observation_contract'] = composer.contract
    space, _ = build_action_spaces(['car_0'], cfg['environment']['vehicle_params'])
    source = PPOAgent(158, space.low, space.high, params)
    checkpoint = tmp_path / 'source.pt'
    source.save(str(checkpoint))
    cfg['training_defaults']['pretrained_actor_checkpoint'] = str(checkpoint)
    validate_scenario(cfg)
    return cfg, source


@pytest.mark.parametrize('skill,num_envs', [('pass', 1), ('pass', 2), ('defend', 1), ('defend', 2)])
def test_real_training_curriculum_checkpoint_and_standalone_suite(tmp_path, monkeypatch, skill, num_envs):
    import sys
    import run
    import training.skill_evaluator as module
    from utils.torch_io import safe_load
    cfg, source = tiny_scenario(tmp_path, skill, num_envs)
    original_setup = module.create_training_setup
    solo_inputs = []

    def setup(*args, **kwargs):
        env, *rest = original_setup(*args, **kwargs)
        if env.n_agents == 1:
            original_step = env.step

            def step(actions):
                # Inject a valid measured lap to exercise the real retention
                # pipeline without running hundreds of thousands of physics steps.
                if env._elapsed_steps == 1:
                    env.lifecycle.records['car_0'].lap_start_step = 0
                    env.lifecycle.record_lap_crossing('car_0', step=env._elapsed_steps)
                result = original_step(actions)
                if result[4]['car_0']['race_completed']:
                    result[4]['car_0']['lap_crossed'] = True
                solo_inputs.append(result[4]['car_0'].get('target_frenet'))
                return result
            env.step = step
        return env, *rest

    monkeypatch.setattr(module, 'create_training_setup', setup)
    monkeypatch.setattr(run, 'load_and_expand_scenario', lambda *_a, **_kw: deepcopy(cfg))
    output = tmp_path / 'train'
    path = str(DIRECTORY / f'mappo_1v1_{skill}_lora.yaml')
    monkeypatch.setattr(sys, 'argv', ['run.py', '--scenario', path, '--no-wandb', '--quiet', '--output-dir', str(output)])
    run.main()
    assert solo_inputs and all(value is None for value in solo_inputs)
    final = safe_load(str(output / 'final_model.pt'), map_location='cpu')
    assert final['environment_steps'] == 16
    assert final['skill_curriculum']['stage'] == 2
    assert final['skill_curriculum']['complete']
    assert final['obs_dim'] == 163 and final['lora_contract']['target_layers'] == [0, 2]
    for key, value in source.actor.net.state_dict().items():
        actual = final['actor']['net.' + key]
        if key == '0.weight':
            assert torch.count_nonzero(actual[:, 158:]) == 0
            actual = actual[:, :158]
        assert torch.equal(actual, value)
    history = [json.loads(line) for line in (output / 'evaluation_history.jsonl').read_text().splitlines()]
    assert all(row['retention_passed'] for row in history)
    assert len(history) >= 2
    races = [json.loads(line) for line in (output / 'race_metrics.jsonl').read_text().splitlines()]
    stages_seen = {row['agents']['car_0']['skill']['stage_index'] for row in races}
    assert 0 in stages_seen and max(stages_seen) > 0
    assert (output / 'best_model.pt').exists()
    # Base results are reused unchanged for subsequent evaluations.
    baseline = json.loads((output / 'skill_base_selection.json').read_text())
    assert len(baseline['results']['solo']['episode_results']) == 1
    assert len(baseline['results'][skill]['episode_results']) == 3
    # Evaluation needs only the self-contained checkpoint, not its PPO source.
    (tmp_path / 'source.pt').unlink()
    evaluation = tmp_path / 'eval'
    monkeypatch.setattr(sys, 'argv', ['run.py', '--scenario', path, '--no-wandb', '--quiet',
        '--eval', '--eval-protocol', 'selection', '--checkpoint', str(output / 'best_model.pt'),
        '--output-dir', str(evaluation)])
    run.main()
    report = json.loads((evaluation / 'evaluation_report.json').read_text())
    assert report['summary']['retention_passed']
    assert report['summary']['episodes'] == 3


def test_final_protocol_doubles_counts_tests_both_skills_and_never_advances(tmp_path, monkeypatch):
    from training.skill_evaluator import base_policy
    import run
    from agents.mappo import MAPPOAgent
    cfg, _ = tiny_scenario(tmp_path, 'pass', 1)
    env, _, _ = create_training_setup(cfg, scenario_dir=DIRECTORY)
    try:
        composer = run.build_obs_composer(cfg['agents']['car_0'], cfg['environment'], DIRECTORY)
        params = run.resolve_training_params(cfg['agents']['car_0'], cfg)
        params['_observation_contract'] = composer.contract
        snapshot = env.get_global_state()
        params['_global_state_contract_version'] = snapshot.metadata['vector_contract_version']
        space = env.action_spaces['car_0']
        agent = MAPPOAgent(163, len(snapshot.vector), space.low, space.high, ['car_0'], params)
        agent.load_pretrained_actor(cfg['training_defaults']['pretrained_actor_checkpoint'])
        evaluator = SkillEvaluator(scenario=cfg, scenario_dir=DIRECTORY, agent=agent,
                                   output_dir=tmp_path, protocol='final')
        try:
            assert evaluator._get_evaluator(cfg, 0).episodes == 2
            assert evaluator._get_evaluator(cfg).episodes == 2
            assert evaluator._get_evaluator(cfg, 0).base_seed == cfg['evaluation']['final_test']['seed']
            seen = []

            def run_trial(config, owner, *, solo=False):
                seen.append(('base' if owner is evaluator.base else 'adapter',
                             'solo' if solo else config['environment']['skill_task']['skill']))
                if solo:
                    return dict(episode_results=solo_rows())
                return dict(skill_success_rate=.8, skill_ego_failure_rate=0., stages=[])
            monkeypatch.setattr(evaluator, '_run', run_trial)
            result = evaluator.evaluate_final()
            assert result['retention_passed']
            assert set(seen) == {(p, s) for p in ('base', 'adapter') for s in ('pass', 'defend', 'solo')}
            assert 'curriculum' not in result
            assert not (tmp_path / 'best_model.pt').exists()
        finally:
            evaluator.close()
    finally:
        env.close()
