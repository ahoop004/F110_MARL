"""Matched attack transfer, repeated successes, relocation and executable workflows."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from core.scenario import load_and_expand_scenario, validate_scenario
from core.setup import create_training_setup
from env.attack import AttackTracker
from env.respawn import reset_respawned
from metrics.racing_eval import create_episode_facts, update_agent_step_facts, aggregate_eval_episodes
from training.hooks import EvaluationCheckpointHook
from wrappers.rewards.attack import AttackRewardComponent


DIRECTORY = Path('scenarios').resolve()


def scenario(lora=False):
    name = 'mappo_1v1_attack' + ('_lora' if lora else '')
    return load_and_expand_scenario(str(DIRECTORY / (name + '.yaml')))


def test_arms_differ_only_in_actor_training_and_labels():
    full, adapted = scenario(), scenario(True)
    assert adapted['training_defaults'].pop('lora') == dict(mode='shared', rank=4, alpha=4., train_log_std=True)
    adapted['experiment']['name'] = full['experiment']['name']
    adapted['wandb'] = full['wandb']
    assert adapted == full
    source = load_and_expand_scenario(str(DIRECTORY / 'ppo_lap_completion_pretrain.yaml'))
    assert full['environment']['vehicle_params'] == source['environment']['vehicle_params']
    original = source['agents']['car_0']
    actor = full['agents']['car_0']
    assert actor['action_constraints'] == original['action_constraints']
    obs = deepcopy(actor['observation']['observation'])
    obs.pop('target_frenet')
    assert obs == original['observation']['observation']


@pytest.mark.parametrize('override', [
    'agents.car_1.params.max_speed=7.5',
    'agents.car_1.params.max_acceleration=4.0',
    'agents.car_1.params.max_steering_reference_rate=1.5',
    'agents.car_0.action_constraints.max_wheel_acceleration=80.0',
    'agents.car_0.action_constraints.max_wheel_deceleration=60.0',
    'agents.car_0.action_constraints.prevent_reverse=false',
    'agents.car_0.action_constraints.speed_control=wheel_speed',
    'environment.vehicle_params.wheel_actuators.wheel_speed_max=150.0',
    'environment.vehicle_params.wheel_actuators.steering_max=0.5',
    'environment.respawn.random_ahead.speed=7.5',
    'environment.timestep=0.025',
])
@pytest.mark.parametrize('lora', [False, True])
def test_attack_parameters_can_be_overridden_independently(monkeypatch, override, lora):
    import run

    path = str(DIRECTORY / ('mappo_1v1_attack' + ('_lora' if lora else '') + '.yaml'))
    monkeypatch.setattr('sys.argv', ['run.py', '--scenario', path, '--set', override])
    args = run.parse_args()
    config = run.apply_cli_overrides(load_and_expand_scenario(path), args)
    validate_scenario(config)
    key, raw = override.split('=', 1)
    actual = config
    for part in key.split('.'):
        actual = actual[part]
    assert actual == yaml.safe_load(raw)


def attack_step(tracker, time, *, crashed=False, failed=False, gap=2.):
    infos = {'car_0': {'target_frenet': {'delta_s': gap, 'delta_d': 0.}, 'centerline': {'vs': 2.},
                       'track_limits': {'exceeded': failed}},
             'car_1': {'track_limits': {'exceeded': crashed, 'lateral_error': .2, 'half_width': 1.}}}
    tracker.update(time=time, infos=infos, collisions={'car_0': False, 'car_1': False})
    return infos['car_0']['attack']


def test_repeated_crashes_survival_gate_and_no_mutual_or_distant_credit():
    tracker = AttackTracker(scenario()['environment']['attack_task'])
    reward = AttackRewardComponent({})
    for start in (0., 2.):
        event = attack_step(tracker, start, crashed=True)
        assert event['eligible_crash'] and event['success'] == 0
        assert attack_step(tracker, start + .49)['success'] == 0
        event = attack_step(tracker, start + .5)
        assert event['success'] == 1
        assert reward.compute({'info': {'attack': event}})['attack/success'] == 10.
        assert attack_step(tracker, start + .6)['success'] == 0
    tracker.reset()
    assert not attack_step(tracker, 0., crashed=True, gap=20.)['eligible_crash']
    assert not attack_step(tracker, 1., crashed=True, failed=True)['eligible_crash']
    assert attack_step(tracker, 1.5)['success'] == 0
    attack_step(tracker, 2., crashed=True)
    failure = attack_step(tracker, 2.1, failed=True)
    assert reward.compute({'info': {'attack': failure}})['attack/ego_crash'] == -20.
    assert attack_step(tracker, 2.5)['success'] == 0


def test_recent_interaction_and_shaping_do_not_cross_respawn():
    tracker = AttackTracker(scenario()['environment']['attack_task'])
    attack_step(tracker, 0., gap=2.)
    event = attack_step(tracker, .5, gap=4., crashed=True)
    assert event['eligible_crash']
    assert event['approach_delta'] == event['edge_delta'] == 0.
    assert attack_step(tracker, .55, gap=8.)['approach_delta'] == 0.
    assert not attack_step(tracker, .6, gap=8., crashed=True)['eligible_crash']
    tracker.reset()
    attack_step(tracker, 0.)
    assert not attack_step(tracker, 1.01, gap=4., crashed=True)['eligible_crash']


@pytest.fixture
def attack_env():
    config = scenario()
    for key in ('map_bundles', 'map_bundles_train', 'map_bundles_eval'):
        config['environment'][key] = ['circle_map']
    env, controllers, _ = create_training_setup(config, scenario_dir=DIRECTORY)
    for controller in controllers.values():
        controller.set_env(env)
    env.reset(seed=42)
    yield env, config, controllers
    env.close()


def test_respawn_is_random_ahead_reproducible_and_preserves_ego(attack_env):
    env, _, controllers = attack_env
    gaps = []
    for _ in range(4):
        controllers['car_1'].controller._warm = np.ones((2, 2))
        env.sim.agents[1].reset(env.sim.agent_poses[1] + [1000., 1000., 0.])
        ego = env.sim.agent_poses[0].copy()
        raw, _, terms, truncs, infos = env.step({})
        assert not terms['car_0'] and not terms['car_1'] and not truncs['car_0']
        assert infos['car_1']['respawn_reason'] == 'track_boundary'
        assert infos['car_1']['centerline']['progress_delta'] == 0.
        assert not infos['car_1']['track_limits']['exceeded']
        assert infos['car_0']['attack']['target_crash']
        assert not infos['car_0']['attack']['eligible_crash']  # Ego is stationary.
        assert 3. <= infos['car_0']['target_frenet']['delta_s'] <= 10.
        assert infos['car_0']['target_frenet']['delta_d'] == pytest.approx(0., abs=1e-4)
        assert raw['car_1']['velocity'][0] == pytest.approx(2.)
        assert raw['car_1']['wheel_speed_reference'] == pytest.approx(40.)
        np.testing.assert_allclose(env.sim.agent_poses[0], ego, atol=1e-6)
        reset_respawned(infos, controllers=controllers, actions={}, observations={})
        assert controllers['car_1'].controller._warm is None
        gaps.append(infos['car_0']['target_frenet']['delta_s'])
    assert len(set(gaps)) == 4
    env.reset(seed=42)
    env.sim.agents[1].reset(env.sim.agent_poses[1] + [1000., 1000., 0.])
    assert env.step({})[4]['car_0']['target_frenet']['delta_s'] == gaps[0]


def test_mutual_collision_ends_episode_without_success(attack_env):
    env, _, _ = attack_env
    env.sim.agents[1].reset(env.sim.agent_poses[0].copy())
    _, _, terms, _, infos = env.step({})
    assert terms['car_0'] and not env.agents
    assert infos['car_0']['attack']['ego_failed']
    assert infos['car_0']['attack']['target_crash']
    assert not infos['car_0']['attack']['eligible_crash']
    assert infos['car_0']['attack']['success'] == 0


def test_ahead_respawn_wraps_finish_seam(attack_env):
    env, _, _ = attack_env
    geometry = env._centerline_progress_tracker._geometry
    k = len(geometry.segment_starts) - 2
    vector = geometry.segment_vectors[k]
    pose = [*geometry.segment_starts[k], np.arctan2(vector[1], vector[0])]
    env.reset(seed=42, options={'poses': np.array([pose, np.array(pose) + [1000., 1000., 0.]])})
    infos = env.step({})[4]
    assert infos['car_0']['centerline']['progress'] > .95
    assert infos['car_1']['centerline']['progress'] < .5
    assert 3. <= infos['car_0']['target_frenet']['delta_s'] <= 10.


def test_effective_speed_braking_and_steering_limits_match(attack_env):
    from wrappers.actions.composer import ActionComposer
    from agents.mpc.racing import _limit_command
    env, config, controllers = attack_env
    space = env.action_spaces['car_0']
    ego = ActionComposer.from_config(space.low, space.high,
        config['agents']['car_0']['action_constraints'], decision_dt=env.timestep)
    mpc = controllers['car_1'].controller
    wheel = config['environment']['vehicle_params']['wheel_actuators']
    assert mpc.steer_min == wheel['steering_min']
    assert mpc.steer_max == wheel['steering_max']
    assert mpc.p[3] == wheel['steering_rate_max'] == -wheel['steering_rate_min']
    previous = np.zeros(2)
    for command in [1.] * 25 + [-1.] * 25:
        steering = wheel['steering_max'] if command > 0 else wheel['steering_min']
        target = np.array([steering, mpc.max_speed if command > 0 else 0.])
        actual = _limit_command(target, previous, mpc.dt, mpc.acceleration, mpc.steering_reference_rate)
        learner = ego.process(np.array([command, command]))
        np.testing.assert_allclose(learner * [1., mpc.radius], actual, atol=1e-6)
        previous = actual


@pytest.mark.parametrize('horizon,laps,basis,failed_score', [
    (1200, 0, 'scheduled_minutes', -1.),
    (0, 20, 'scheduled_laps', -.05),
])
def test_selection_penalizes_failure_and_does_not_reward_early_termination(horizon, laps, basis, failed_score):
    def result(steps, failed):
        facts = create_episode_facts(episode=0, agent_ids=['car_0', 'car_1'],
                                    trainable_ids=['car_0'], opponent_ids=['car_1'])
        for step in range(1, steps + 1):
            attack = dict(success=int(step == 1), target_crash=False, eligible_crash=False,
                          ego_failed=failed and step == steps, horizon_steps=horizon, target_laps=laps)
            update_agent_step_facts(facts, step_idx=step, infos={'car_0': {'attack': attack}})
        return aggregate_eval_episodes([facts], timestep=.05)
    safe, failed, later_failure = result(100, False), result(10, True), result(100, True)
    assert safe['attack_successes'] == 1
    assert failed['attack_ego_crash_rate'] == 1.
    assert safe['attack_score_basis'] == basis
    assert failed['attack_score'] == later_failure['attack_score'] == failed_score
    assert EvaluationCheckpointHook.selection_score(safe, 'attack') > EvaluationCheckpointHook.selection_score(failed, 'attack')


@pytest.mark.parametrize('lora', [False, True])
def test_actor_transfer_keeps_prefix_and_lora_frozen(tmp_path, lora):
    import run
    from agents.ppo import PPOAgent
    from agents.mappo import MAPPOAgent
    from env.spaces_builder import build_action_spaces
    config = scenario(lora)
    cfg = config['agents']['car_0']
    composer = run.build_obs_composer(cfg, config['environment'], DIRECTORY)
    params = run.resolve_training_params(cfg, config)
    params.update(device='cpu', pi_hidden_dims=[8, 8], vf_hidden_dims=[8], _observation_contract=composer.contract)
    original_params = deepcopy(params)
    original_params['_observation_contract']['observation'].pop('target_frenet')
    space, _ = build_action_spaces(['car_0'], config['environment']['vehicle_params'])
    source = PPOAgent(158, space.low, space.high, original_params)
    path = tmp_path / 'source.pt'
    source.save(str(path))
    torch.manual_seed(1)
    agent = MAPPOAgent(163, 20, space.low, space.high, ['car_0'], params)
    agent.load_pretrained_actor(str(path))
    obs = torch.randn(12, 163)
    with torch.no_grad():
        expected, _ = source.actor(obs[:, :158])
        actual, _ = agent.actor(obs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if lora:
        before = deepcopy(agent.actor.net.state_dict())
        objective = agent.actor(obs)[0].square().mean()
        agent.optimizer.zero_grad()
        objective.backward()
        agent.optimizer.step()
        torch.testing.assert_close(before, agent.actor.net.state_dict(), rtol=0, atol=0)
        assert any(v.count_nonzero() for k, v in agent.actor.state_dict().items() if k.endswith('.B'))
        assert not torch.equal(agent.actor(obs)[0], expected)


@pytest.mark.parametrize('lora,num_envs', [(False, 1), (True, 1), (False, 2), (True, 2)])
def test_training_and_standalone_checkpoint_evaluation(tmp_path, monkeypatch, lora, num_envs):
    import sys
    import run
    from agents.ppo import PPOAgent
    from env.spaces_builder import build_action_spaces
    from utils.torch_io import safe_load
    from training.collector_progress import CollectorProgress
    evaluation_progress = []
    original_progress = CollectorProgress.evaluation_progress

    def capture_progress(progress, row):
        evaluation_progress.append(None if row is None else dict(row))
        original_progress(progress, row)

    monkeypatch.setattr(CollectorProgress, 'evaluation_progress', capture_progress)
    torch.set_num_threads(1)
    config = scenario(lora)
    config['experiment'].update(total_steps=8, num_envs=num_envs, num_workers=1)
    # Rollout updates and the aggregate training budget must work within a long
    # episode, including in the grouped collector.
    assert config['environment']['max_steps'] == 40000
    for key in ('map_bundles', 'map_bundles_train', 'map_bundles_eval'):
        config['environment'][key] = ['circle_map']
    config['evaluation'].update(every_steps=4, episodes=1)
    assert config['evaluation']['max_steps'] == 40000
    config['training_defaults'].update(rollout_steps_per_env=2, device='cpu')
    config['agents']['car_0']['params'].update(pi_hidden_dims=[8, 8], vf_hidden_dims=[8],
        device='cpu', n_steps=4, batch_size=2, n_epochs=1)
    config['agents']['car_1']['params'].update(horizon=2, knots=2, iterations=1, max_evaluations=5)
    source_cfg = deepcopy(config['agents']['car_0'])
    source_cfg['observation']['observation'].pop('target_frenet')
    composer = run.build_obs_composer(source_cfg, config['environment'], DIRECTORY)
    params = run.resolve_training_params(source_cfg, config)
    params['_observation_contract'] = composer.contract
    space, _ = build_action_spaces(['car_0'], config['environment']['vehicle_params'])
    source = PPOAgent(158, space.low, space.high, params)
    source_path = tmp_path / 'source.pt'
    source.save(str(source_path))
    config['training_defaults']['pretrained_actor_checkpoint'] = str(source_path)
    original_setup = run.create_training_setup

    def setup(*args, **kwargs):
        env, *rest = original_setup(*args, **kwargs)
        if kwargs.get('mode') == 'eval':
            original_step = env.step

            def step(actions):
                # Supply accepted lap crossings to keep this integration test
                # short while exercising real lap termination before the timeout.
                if env._elapsed_steps == 2:
                    for _ in range(env.target_laps):
                        env.lifecycle.record_lap_crossing('car_0', step=env._elapsed_steps)
                return original_step(actions)

            env.step = step
        return env, *rest

    monkeypatch.setattr(run, 'create_training_setup', setup)
    monkeypatch.setattr(run, 'load_and_expand_scenario', lambda *_a, **_kw: deepcopy(config))
    path = str(DIRECTORY / ('mappo_1v1_attack' + ('_lora' if lora else '') + '.yaml'))
    output = tmp_path / 'train'
    monkeypatch.setattr(sys, 'argv', ['run.py', '--scenario', path, '--no-wandb', '--quiet',
                                    '--output-dir', str(output)])
    run.main()
    checkpoint = output / 'best_model.pt'
    payload = safe_load(str(checkpoint), map_location='cpu')
    assert payload['obs_dim'] == 163
    assert payload['checkpoint_selection']['selection_strategy'] == 'attack'
    assert payload['checkpoint_selection']['environment_steps'] > 0
    assert payload['checkpoint_selection']['attack_score_basis'] == 'scheduled_minutes'
    assert payload['checkpoint_selection']['attack_score_budget'] == pytest.approx(2000 / 60)
    assert payload['checkpoint_selection']['focal_completion_rate'] == 1.
    if num_envs > 1:
        assert evaluation_progress[-1] is None
        finished = [row for row in evaluation_progress if row and row['status'] == 'complete']
        assert finished
        assert all(row['laps'] == 'car_0:5/5' for row in finished)
        assert all(row['outcome'] == 'car_0:race_complete' for row in finished)
        assert all(row['map'] == 'circle_map' and row['max_steps'] == 40000 for row in finished)
        assert all('attack_successes' in row and 'target_crashes' in row for row in finished)
    if lora:
        assert payload['lora_contract']['rank'] == 4
        for key, value in source.actor.net.state_dict().items():
            actual = payload['actor']['net.' + key]
            if key == '0.weight':
                assert torch.equal(actual[:, :158], value)
                assert actual[:, 158:].count_nonzero() == 0
            else:
                assert torch.equal(actual, value)
        assert any(v.count_nonzero() for k, v in payload['actor'].items() if k.endswith('.B'))
    source_path.unlink()
    monkeypatch.setattr(sys, 'argv', ['run.py', '--scenario', path, '--eval', '--checkpoint',
        str(checkpoint), '--eval-episodes', '1', '--no-wandb', '--quiet', '--output-dir', str(tmp_path / 'eval')])
    run.main()
    report = json.loads((tmp_path / 'eval' / 'evaluation_report.json').read_text())
    assert report['summary']['attack_score_basis'] == 'scheduled_minutes'
    assert report['summary']['focal_completion_rate'] == 1.
