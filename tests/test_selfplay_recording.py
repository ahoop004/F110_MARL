import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from core.scenario import load_and_expand_scenario
from src.replay.race_reader import load_clips, iter_race_frames


def config(num_envs):
    scenario = load_and_expand_scenario('scenarios/mappo_2v2_selfplay.yaml')
    scenario['experiment'].update(num_envs=num_envs, num_workers=min(num_envs, 2), total_steps=11,
        seed=42, worker_startup_timeout_s=60, worker_response_timeout_s=60)
    scenario['training_defaults'].update(device='cpu', pi_hidden_dims=[4], vf_hidden_dims=[4],
        n_steps=2, rollout_steps_per_env=2, n_epochs=1, batch_size=16)
    scenario['environment'].update(max_steps=3)
    scenario['evaluation']['enabled'] = False
    scenario['wandb']['enabled'] = False
    scenario['recording'] = dict(enabled=False, sample_probability=1., chunk_frames=3)
    return scenario


@pytest.mark.parametrize('num_envs', [1, 3])
def test_recorded_selfplay_matches_training_and_replays_both_teams(tmp_path, monkeypatch, num_envs):
    import run
    from src.analysis.run_review import load_run, filter_clips
    from src.analysis.clip_review import load_window
    from src.analysis.annotations import AnnotationStore, segment

    before_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        for enabled in (False, True):
            random.seed(42); np.random.seed(42); torch.manual_seed(42)
            output = tmp_path/('enabled' if enabled else 'disabled')
            argv = ['run.py', '--scenario', 'scenarios/mappo_2v2_selfplay.yaml',
                    '--output-dir', str(output), '--no-wandb', '--run-id', 'recording_test']
            if enabled:
                argv.append('--record-races')
            monkeypatch.setattr('sys.argv', argv)
            run._run_two_team(config(num_envs), run.parse_args(), run.ConsoleLogger(verbose=False), Path('scenarios').resolve())
    finally:
        torch.set_num_threads(before_threads)
    for team in ('team_0.pt', 'team_1.pt'):
        off = torch.load(tmp_path/'disabled'/'final_pair'/team, map_location='cpu', weights_only=False)
        on = torch.load(tmp_path/'enabled'/'final_pair'/team, map_location='cpu', weights_only=False)
        for network in ('actor', 'critic'):
            assert all(torch.equal(v, on[network][k]) for k, v in off[network].items())
    assert not (tmp_path/'disabled'/'behavior').exists()
    dataset = tmp_path/'enabled'/'behavior'
    frames = list(iter_race_frames(dataset))
    assert len(frames) == 11
    assert len({(f['episode_id'], f['physics_index']) for f in frames}) == 11
    assert {f['environment_id'] for f in frames} == set(range(num_envs))
    assert all(set(f['team_policy_versions']) == {'team_a', 'team_b'} for f in frames)
    assert all(len(f['commands']) == len(f['learners']) == 4 for f in frames)
    assert all(f['team_rewards'][f['agent_teams'][a]] == r['reward']
               for f in frames for a, r in f['learners'].items())
    assert all(f['team_rewards'][team] == pytest.approx(sum(v for k, v in f['team_reward_components'].items() if k.startswith(team+'/')))
               for f in frames for team in ('team_a', 'team_b'))
    last = {}
    for f in frames:
        prev = last.get(f['episode_id'])
        if prev:
            assert prev['post_state'] == f['pre_state']
        last[f['episode_id']] = f
    clips = load_clips(dataset)
    samples = [c for c in clips if c['kind'] == 'representative_race']
    assert sum(c['complete'] for c in samples) == 3
    assert any(c['end_reason']=='budget_cut' for c in samples)
    assert all(c['agent_teams']['car_2']=='team_b' for c in clips)
    assert all(c['team_policy_versions_end']['team_a'] == c['policy_version_end'] for c in clips)
    report = load_run(tmp_path/'enabled')
    assert set(report.team_races.team) == {'team_a', 'team_b'}
    assert 'train/team_b/policy_loss' in report.updates
    assert not filter_clips(report.clips, team='team_b').empty
    assert report.metadata['training_races'] == 3
    window = load_window(dataset, samples[0])
    annotation = AnnotationStore(tmp_path/'review'/'annotations.json').save(segment(window,
        start=window.boundaries[0], end=window.boundaries[-1], scope='individual',
        participants=['car_2'], label='uncertain'))
    assert annotation['source']['agent_teams']['car_2'] == 'team_b'
    assert annotation['source']['team_policy_versions_start'] == samples[0]['team_policy_versions_start']


def test_recording_keeps_inactive_team_rewards_and_clearance_removal(tmp_path, monkeypatch):
    import run
    from src.analysis.clip_review import load_window, draw_review
    from env.types import AgentRaceStatus

    original_setup = run.create_training_setup

    def setup(*args, **kwargs):
        env, *rest = original_setup(*args, **kwargs)
        original_step = env.step

        def step(actions):
            obs, rewards, terms, truncs, infos = original_step(actions)
            # End team A first, then one opponent: A must retain its later bonus
            # even though neither of its actors contributes another sample.
            victims = {1: ['car_0', 'car_1'], 2: ['car_2']}.get(env._elapsed_steps, [])
            for aid in victims:
                index = env.possible_agents.index(aid)
                env.lifecycle.record_collision(aid, step=env._elapsed_steps-1)
                env._terminal_controller.capture(aid, status=AgentRaceStatus.CRASHED,
                    terminal_step=env._elapsed_steps-1, action=actions[aid],
                    vehicle_state=env.sim.agents[index].physics_state)
                infos[aid].update(status='crashed', terminal_reason='collision',
                                 terminal_step=env._elapsed_steps-1)
                terms[aid] = True
            env.agents = [aid for aid in env.agents if aid not in victims]
            return obs, rewards, terms, truncs, infos

        env.step = step
        return env, *rest

    monkeypatch.setattr(run, 'create_training_setup', setup)
    scenario = config(1)
    scenario['experiment']['total_steps'] = 3
    scenario['environment']['terminal_agents']['crash_clearance_steps'] = 1
    monkeypatch.setattr('sys.argv', ['run.py', '--scenario', 'scenarios/mappo_2v2_selfplay.yaml',
        '--output-dir', str(tmp_path), '--record-races', '--no-wandb'])
    before_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        run._run_two_team(scenario, run.parse_args(), run.ConsoleLogger(verbose=False), Path('scenarios').resolve())
    finally:
        torch.set_num_threads(before_threads)
    dataset = tmp_path/'behavior'
    frames = list(iter_race_frames(dataset))
    assert len(frames) == 3
    assert set(frames[1]['learners']) == {'car_2', 'car_3'}
    assert frames[1]['team_reward_components']['team_a/opponent_crash'] == 1
    assert frames[1]['team_rewards']['team_a'] == 1
    assert frames[1]['commands']['car_0']['requested'] is None
    assert frames[1]['commands']['car_0']['applied'] == [0., 0.]
    assert frames[0]['post_state']['car_0']['present']
    assert not frames[1]['post_state']['car_0']['present']
    sample = next(c for c in load_clips(dataset) if c['kind'] == 'representative_race')
    window = load_window(dataset, sample)
    fig, seek = draw_review(window)
    seek(2)
    cars = {patch.get_label(): patch for patch in fig.axes[0].patches}
    assert not cars['car_0 (team_a)'].get_visible()
    assert cars['car_3 (team_b)'].get_visible()


def test_parallel_recording_enforces_one_shared_storage_cap(tmp_path, monkeypatch):
    import run

    scenario = config(3)
    scenario['recording']['max_frames'] = 2
    monkeypatch.setattr('sys.argv', ['run.py', '--scenario', 'scenarios/mappo_2v2_selfplay.yaml',
        '--output-dir', str(tmp_path), '--record-races', '--no-wandb'])
    before_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        run._run_two_team(scenario, run.parse_args(), run.ConsoleLogger(verbose=False), Path('scenarios').resolve())
    finally:
        torch.set_num_threads(before_threads)
    frames = list(iter_race_frames(tmp_path/'behavior'))
    assert len(frames) == 2
    summary = json.loads((tmp_path/'run_summary.json').read_text())
    assert summary['environment_steps'] == 11  # Only recording stops.
    ends = {f['episode_id']: f for f in frames}
    for clip in load_clips(tmp_path/'behavior'):
        if clip['end_reason'] != 'storage_limit':
            continue
        assert not clip['complete']
        retained = ends.get(clip['episode_id'])
        assert clip['team_policy_versions_end'] == (retained['team_policy_versions'] if retained else None)
