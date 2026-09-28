"""Transfer, heterogeneous collection, target switching and terminal removal."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from agents.mappo import MAPPOAgent
from agents.ppo import PPOAgent
from core.scenario import load_and_expand_scenario, resolve_mappo_config, validate_scenario, ScenarioError
from core.setup import build_obs_composers, resolve_training_params, create_training_setup
from env.spaces_builder import build_action_spaces
from training.parallel_mappo import CollectorAgent, infer_requests

DIRECTORY = Path('scenarios').resolve()


def config(name='mappo_2v2_attack_transfer'):
    s = load_and_expand_scenario(str(DIRECTORY / (name + '.yaml')))
    for key in ('map_bundles', 'map_bundles_train', 'map_bundles_eval'):
        s['environment'][key] = ['circle_map']
    for a in s['agents'].values():
        if a.get('trainable'):
            a['params'].update(device='cpu', pi_hidden_dims=[8, 8], vf_hidden_dims=[8],
                               n_steps=4, n_epochs=1, batch_size=3)
        else:
            a['params'].update(horizon=2, knots=2, iterations=1, max_evaluations=5)
    s['training_defaults'].update(device='cpu', rollout_steps_per_env=2)
    return s


def agent_for(s, global_dim=32):
    ids = [a for a, v in s['agents'].items() if v.get('trainable')]
    composers = build_obs_composers(s['agents'], ids, s['environment'], DIRECTORY)
    p = resolve_training_params(s['agents'][ids[0]], s)
    p.update(resolve_mappo_config(s))
    p.update(_observation_contract=composers[ids[0]].contract,
             _observation_contracts={a: c.contract for a, c in composers.items()},
             _observation_dims={a: c.obs_dim for a, c in composers.items()})
    space, _ = build_action_spaces(ids, s['environment']['vehicle_params'])
    return MAPPOAgent(composers[ids[0]].obs_dim, global_dim, space.low, space.high, ids, p), composers


def source_checkpoint(tmp_path):
    source, _ = agent_for(config('mappo_1v1_attack_lora'))
    driving_contract = deepcopy(source.observation_contract)
    driving_contract['observation'].pop('target_frenet')
    p = dict(pi_hidden_dims=[8, 8], vf_hidden_dims=[8], device='cpu', activation=source.activation,
             _physics_contract=source.physics_contract, _action_contract=source.action_contract,
             _observation_contract=driving_contract)
    base = PPOAgent(158, source.action_low, source.action_high, p)
    baseline = tmp_path / 'baseline.pt'
    base.save(str(baseline))
    source.load_pretrained_actor(str(baseline))
    with torch.no_grad():
        for residual in source.actor.adapters[0].values():
            residual.B.fill_(.12)
        source.actor.log_std.fill_(-.7)
    path = tmp_path / 'attack.pt'
    source.save(str(path))
    return source, base, path


def recipient(tmp_path):
    source, base, path = source_checkpoint(tmp_path)
    target, _ = agent_for(config())
    target.load_pretrained_adapter(str(path), source_agent='car_0', target_agent='car_1')
    return target, source, base, path


def test_transfer_preserves_attacker_and_base_racer_with_different_inputs(tmp_path):
    torch.set_num_threads(1)
    target, source, base, _ = recipient(tmp_path)
    assert target.obs_dims == {'car_0': 192, 'car_1': 201}
    assert target.actor.adapters[0]['0'].A.shape == (4, 192)
    assert target.actor.adapters[1]['0'].A.shape == (4, 201)
    assert not target.optimizer.state
    assert all(not p.requires_grad for p in target.actor.net.parameters())
    observations = torch.randn(10, 201)
    old = torch.cat((observations[:, :158], observations[:, 192:197]), dim=1)
    with torch.no_grad():
        expected, expected_std = source.actor(old)
        actual, actual_std = target.actor(observations, adapter_indices=torch.ones(10, dtype=torch.long))
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_std, expected_std.expand_as(actual_std))
        racer = observations.clone()
        racer[:, 192:] = 0
        actual, _ = target.actor(racer, adapter_indices=torch.zeros(10, dtype=torch.long))
        expected, _ = base.actor(racer[:, :158])
        torch.testing.assert_close(actual, expected)
    assert target.actor.adapters[1]['0'].A[:, 158:192].count_nonzero() == 0
    assert target.actor.adapters[1]['0'].A[:, 197:].count_nonzero() == 0


def collect(agent, collector=None, only=None):
    for step in range(3):
        ids = only or (['car_1', 'car_0'] if step < 2 else ['car_1'])
        observations = {a: np.full(agent.obs_dims[a], .1 + step, np.float32) for a in ids}
        state = np.zeros(agent.global_state_dim, np.float32)
        actions, logs = agent.act_batch(ids, observations)
        values = agent.evaluate_states(state, ids)
        for owner in [agent] + ([collector] if collector else []):
            owner.store_batch(ids, observations=observations, global_state=state, actions=actions,
                rewards={a: float(step + 1) for a in ids}, log_probs=logs, values=values,
                terminated={a: step == 2 or a == 'car_0' and step == 1 for a in ids},
                truncated={a: False for a in ids}, raw_actions=agent.last_raw_actions)


def test_serial_parallel_updates_and_adapter_isolation(tmp_path):
    agent, _, _, _ = recipient(tmp_path)
    parallel = deepcopy(agent)
    names = ('agent_ids', 'obs_dim', 'obs_dims', 'global_state_dim', 'action_dim', 'gamma',
             'gae_lambda', 'critic_mode', 'reward_mode', 'team_return_mode')
    collector = CollectorAgent({k: getattr(agent, k) for k in names}, 4)
    collect(agent, collector)
    collector.finish_fragment({'car_0': 0., 'car_1': 0.})
    rng = torch.get_rng_state()
    expected = agent.update(np.zeros(agent.global_state_dim))
    torch.set_rng_state(rng)
    actual = parallel.update_rollouts(collector.take_fragments())
    assert actual == pytest.approx(expected, abs=1e-6)
    torch.testing.assert_close(agent.actor.state_dict(), parallel.actor.state_dict())
    before = deepcopy(agent.actor.state_dict())
    agent.clear_buffers()
    collect(agent, only=['car_0'])
    agent.update(np.zeros(agent.global_state_dim))
    for k, value in before.items():
        if k.startswith(('net.', 'adapters.1.')) or k == 'log_stds.1':
            assert torch.equal(value, agent.actor.state_dict()[k]), k
    assert any(not torch.equal(value, agent.actor.state_dict()[k])
               for k, value in before.items() if k.startswith('adapters.0.'))


def test_checkpoint_and_grouped_inference_after_source_is_deleted(tmp_path):
    agent, _, _, path = recipient(tmp_path)
    collect(agent)
    agent.update(np.zeros(agent.global_state_dim))
    saved = tmp_path / 'team.pt'
    agent.save(str(saved))
    path.unlink()
    restored, _ = agent_for(config())
    restored.load(str(saved))
    torch.testing.assert_close(restored.actor.state_dict(), agent.actor.state_dict())
    torch.testing.assert_close(restored.optimizer.state_dict(), agent.optimizer.state_dict())
    rows = agent.pack_observations(['car_1', 'car_0'], [np.ones(201), np.ones(192)])
    requests = {0: ('act', (['car_1', 'car_0'], rows, np.zeros(agent.global_state_dim))),
                1: ('act', (['car_0'], rows[1:], np.zeros(agent.global_state_dim)))}
    torch.manual_seed(9)
    expected = infer_requests(agent, requests)
    torch.manual_seed(9)
    actual = infer_requests(restored, requests)
    for key in expected:
        for aid in expected[key][0]:
            np.testing.assert_array_equal(expected[key][0][aid], actual[key][0][aid])
    bad, _ = agent_for(config())
    bad.observation_contracts['car_1']['observation']['target_frenet']['maxima']['delta_s'] = 3.
    with pytest.raises(ValueError, match='observation contract'):
        bad.load(str(saved))


@pytest.mark.parametrize('mode', ['train', 'eval'])
@pytest.mark.parametrize('survivor_outcome', ['finish', 'failure'])
def test_five_laps_removal_and_opponent_respawn_preserves_removed_mask(mode, survivor_outcome, monkeypatch):
    s = config()
    env, _, _ = create_training_setup(s, mode=mode, scenario_dir=DIRECTORY)
    try:
        env.reset(seed=42)
        env._elapsed_steps = 12001
        for _ in range(5):
            env.lifecycle.record_lap_crossing('car_0', step=env._elapsed_steps)
        _, _, terms, truncs, infos = env.step({})
        assert terms['car_0'] and not any(truncs.values())
        assert env.agents == ['car_1', 'car_2', 'car_3']
        assert not env.sim.collidable_mask[0]
        assert all(n['agent_id'] != 'car_0' for n in infos['car_1']['frenet_neighbors'])
        original = env._inject_track_previews

        def boundary(infos):
            original(infos)
            infos['car_2']['track_limits']['exceeded'] = True

        monkeypatch.setattr(env, '_inject_track_previews', boundary)
        _, _, terms, truncs, infos = env.step({})
        assert infos['car_2']['respawned'] and not terms['car_2']
        assert not env.sim.collidable_mask[0]
        assert all(n['agent_id'] != 'car_0' for n in infos['car_1']['frenet_neighbors'])
        assert infos['car_2']['centerline']['progress_delta'] == 0.
        monkeypatch.setattr(env, '_inject_track_previews', original)
        if survivor_outcome == 'finish':
            for _ in range(5):
                env.lifecycle.record_lap_crossing('car_1', step=env._elapsed_steps)
        else:
            def learner_boundary(infos):
                original(infos)
                infos['car_1']['track_limits']['exceeded'] = True
            monkeypatch.setattr(env, '_inject_track_previews', learner_boundary)
        _, _, terms, truncs, _ = env.step({})
        assert terms['car_1'] and not any(truncs.values()) and not env.agents
        assert not env.sim.collidable_mask[:2].any()
        env.reset(seed=42)
        assert env.sim.collidable_mask.all()
    finally:
        env.close()


@pytest.mark.parametrize('num_envs', [1, 2])
def test_training_checkpoint_selection_and_standalone_evaluation(tmp_path, monkeypatch, num_envs):
    import run
    from utils.torch_io import safe_load
    _, _, source = source_checkpoint(tmp_path)
    s = config()
    s['training_defaults']['adapter_transfer']['checkpoint'] = str(source)
    s['experiment'].update(total_steps=8, num_envs=num_envs, num_workers=1)
    s['evaluation'].update(every_steps=4, episodes=1)
    original = run.create_training_setup

    def setup(*args, **kwargs):
        env, *rest = original(*args, **kwargs)
        if kwargs.get('mode') == 'eval':
            original_step = env.step

            def step(actions):
                if env._elapsed_steps == 2:
                    for aid in ('car_0', 'car_1'):
                        for _ in range(5):
                            env.lifecycle.record_lap_crossing(aid, step=env._elapsed_steps)
                return original_step(actions)

            env.step = step
        return env, *rest

    monkeypatch.setattr(run, 'create_training_setup', setup)
    monkeypatch.setattr(run, 'load_and_expand_scenario', lambda *_a, **_kw: deepcopy(s))
    path = str(DIRECTORY / 'mappo_2v2_attack_transfer.yaml')
    output = tmp_path / 'train'
    monkeypatch.setattr(sys, 'argv', ['run.py', '--scenario', path, '--quiet', '--no-wandb', '--output-dir', str(output)])
    run.main()
    saved = output / 'best_model.pt'
    state = safe_load(str(saved), map_location='cpu')
    assert state['obs_dims'] == {'car_0': 192, 'car_1': 201}
    assert state['checkpoint_selection']['selection_strategy'] == 'racer_attack'
    assert state['checkpoint_selection']['team_both_finished_rate'] == 1.
    assert state['checkpoint_selection']['attack_score_budget'] == 5
    source.unlink()
    monkeypatch.setattr(sys, 'argv', ['run.py', '--scenario', path, '--eval', '--checkpoint', str(saved),
        '--eval-episodes', '1', '--quiet', '--no-wandb', '--output-dir', str(tmp_path / 'eval')])
    run.main()
    report = json.loads((tmp_path / 'eval' / 'evaluation_report.json').read_text())
    assert report['summary']['team_both_finished_rate'] == 1.
    assert report['summary']['attack_score_basis'] == 'scheduled_laps'


def test_nearest_ahead_switches_across_seam_and_excludes_teammate():
    s = config()
    env, _, _ = create_training_setup(s, scenario_dir=DIRECTORY)
    try:
        env.reset(seed=42)
        length = env._centerline_progress_tracker.track_length
        env._last_centerline_facts = {a: dict(s=v, d=0., vs=1., vd=0.) for a, v in
            [('car_0', length - .5), ('car_1', length - 1.), ('car_2', 1.), ('car_3', 2.)]}
        infos = {}
        env._inject_frenet_neighbors(infos)
        assert env.get_target_id('car_1') == 'car_2'
        assert infos['car_1']['target_frenet']['delta_s'] == pytest.approx(2.)
        env._last_centerline_facts['car_3']['s'] = .5
        env._inject_frenet_neighbors(infos)
        assert env.get_target_id('car_1') == 'car_3'
        env._last_centerline_facts['car_2']['s'] = length - 2.
        env._last_centerline_facts['car_3']['s'] = length - 3.
        env._inject_frenet_neighbors(infos)
        assert env.get_target_id('car_1') is None
        assert infos['car_1']['target_frenet'] is None
    finally:
        env.close()


def test_target_switching_and_respawns_do_not_create_attack_credit():
    from env.attack import MultiTargetAttackTracker
    tracker = MultiTargetAttackTracker(config()['environment']['attack_task'])

    def step(time, target, crash=None, ego_failed=False):
        infos = {a: dict(track_limits=dict(exceeded=False, lateral_error=0., half_width=1.))
                 for a in ('car_0', 'car_1', 'car_2', 'car_3')}
        infos['car_1'].update(centerline={'vs': 1.}, frenet_neighbors=[
            dict(agent_id=a, delta_s=d, delta_d=0., delta_vs=0., delta_vd=0.)
            for a, d in [('car_2', 1.), ('car_3', 2.)]])
        collisions = {a: a == crash or a == 'car_1' and ego_failed for a in infos}
        tracker.update(time=time, infos=infos, collisions=collisions, active_target=target)
        return infos['car_1']['attack']

    assert step(0., 'car_2')['success'] == 0
    # Unselected car_3 has no interaction history, despite being nearby.
    assert step(.05, 'car_2', crash='car_3')['eligible_crash'] == 0
    assert step(.1, 'car_2', crash='car_2')['eligible_crash'] == 1
    switched = step(.15, 'car_3')
    assert switched['approach_delta'] == switched['edge_delta'] == 0.
    assert step(.6, None)['success'] == 1  # Credit stays with the old target.
    assert step(.65, None)['success'] == 0
    step(.7, 'car_3', crash='car_3')
    assert step(.75, None, ego_failed=True)['ego_failed']
    assert step(1.3, None)['success'] == 0  # Failure cancelled pending credit.


@pytest.mark.parametrize('failure', ['boundary', 'collision'])
def test_removed_learner_has_one_failure_transition_and_survivor_continues(tmp_path, monkeypatch, failure):
    from core.setup import build_reward_composers
    from training.marl_trainer import MARLTrainer
    from training.hooks import TrainingHook
    from wrappers.actions.composer import ActionComposer
    s = config()
    env, opponents, _ = create_training_setup(s, scenario_dir=DIRECTORY)
    _, _, source = source_checkpoint(tmp_path)
    agent, obs = agent_for(s, len(env.get_global_state().vector))
    agent.load_pretrained_adapter(str(source), source_agent='car_0', target_agent='car_1')
    records = []

    class Hook(TrainingHook):
        def on_step(self, record):
            records.append(record)

    if failure == 'boundary':
        original = env._inject_track_previews

        def boundary(infos):
            original(infos)
            if env._elapsed_steps == 0:
                infos['car_0']['track_limits']['exceeded'] = True
            if env._elapsed_steps == 3:
                infos['car_1']['track_limits']['exceeded'] = True

        monkeypatch.setattr(env, '_inject_track_previews', boundary)
    else:
        original = env.sim.step

        def collision(actions):
            result = original(actions)
            if env._elapsed_steps in (0, 3):
                index = 0 if env._elapsed_steps == 0 else 1
                result['collisions'][index] = 1.
                env.sim.collisions[index] = 1.
            return result

        monkeypatch.setattr(env.sim, 'step', collision)
    for opponent in opponents.values():
        opponent.set_env(env)
    actions = ActionComposer.from_config(agent.action_low, agent.action_high,
        s['agents']['car_0']['action_constraints'], decision_dt=env.timestep)
    trainer = MARLTrainer(env, agent, ['car_0', 'car_1'], opponents, obs,
        build_reward_composers(s['agents'], ['car_0', 'car_1'], DIRECTORY), actions,
        hooks=[Hook()], reward_mode='individual')
    try:
        trainer.train(n_episodes=1, total_steps=4)
        racer = [r for r in records if r.agent_id == 'car_0']
        attacker = [r for r in records if r.agent_id == 'car_1']
        assert len(racer) == 1 and len(attacker) == 4
        assert racer[0].reward == -20.
        assert attacker[-1].reward == -20.
        assert sum(r.reward_components['attack/ego_crash'] for r in attacker) == -20.
        assert not env.sim.collidable_mask[:2].any()
        assert env.episode_done and not env.agents
    finally:
        env.close()


@pytest.mark.parametrize('field,value', [('rank', 2), ('alpha', 3.)])
def test_transfer_rejects_incompatible_adapter_without_mutating_actor(tmp_path, field, value):
    _, _, path = source_checkpoint(tmp_path)
    s = config()
    s['training_defaults']['lora'][field] = value
    target, _ = agent_for(s)
    before = deepcopy(target.actor.state_dict())
    with pytest.raises(ValueError, match=f'Incompatible adapter {field}'):
        target.load_pretrained_adapter(str(path), source_agent='car_0', target_agent='car_1')
    torch.testing.assert_close(target.actor.state_dict(), before)


@pytest.mark.parametrize('kind', ['learner_respawn', 'no_removal', 'wrong_finishers', 'wrong_transfer_target'])
def test_rejects_invalid_team_lifecycle_and_transfer(kind):
    s = config()
    if kind == 'learner_respawn':
        s['environment']['respawn']['boundary_agents'].append('car_0')
    elif kind == 'no_removal':
        s['environment']['terminal_agents']['remove_after_clearance'] = False
    elif kind == 'wrong_finishers':
        s['environment']['episode_termination']['lap_finish_agents'] = ['car_0']
    else:
        s['training_defaults']['adapter_transfer']['target_agent'] = 'car_2'
    with pytest.raises(ScenarioError):
        validate_scenario(s)
