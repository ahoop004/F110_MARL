"""Evaluation capture must preserve policy behavior and checkpoint selection."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from core.scenario import load_and_expand_scenario
from src.analysis.run_review import load_run, filter_clips
from src.analysis.clip_review import load_window, draw_review
from src.analysis.annotations import source_reference
from src.replay.race_reader import iter_race_frames, load_clips


def small_scenario(algorithm):
    source = ('scenarios/ppo_lap_completion_pretrain.yaml' if algorithm == 'ppo'
              else 'scenarios/mappo_2v2_combined.yaml')
    scenario = load_and_expand_scenario(source)
    scenario['experiment'].update(num_envs=1, total_steps=4, seed=42, torch_threads=1)
    scenario['environment'].update(max_steps=5, action_repeat=2 if algorithm == 'ppo' else 1,
        map_bundles=['circle_map'], map_bundles_train=['circle_map'], map_bundles_eval=['circle_map'])
    scenario['evaluation'].update(enabled=True, every_steps=2, every_episodes=1, episodes=1,
        max_steps=5, final_test=dict(episodes=1, seed=20042))
    params = dict(device='cpu', n_steps=2, n_epochs=1, batch_size=4,
        pi_hidden_dims=[4], vf_hidden_dims=[4], checkpoint_every_steps=2)
    scenario['training_defaults'].update(params, pretrained_actor_checkpoint=None)
    for cfg in scenario['agents'].values():
        if cfg.get('trainable'):
            cfg.setdefault('params', {}).update(params)
    scenario['recording'] = dict(enabled=False, sample_probability=1., chunk_frames=2, max_frames=100)
    scenario['wandb']['enabled'] = False
    return source, scenario


@pytest.mark.parametrize('algorithm', ['ppo', 'mappo'])
def test_selection_and_standalone_recordings_preserve_results(tmp_path, monkeypatch, algorithm):
    import run
    import matplotlib.pyplot as plt
    source, base = small_scenario(algorithm)

    def launch(name, enabled, checkpoint=None):
        scenario = deepcopy(base)
        if enabled and checkpoint is None:
            scenario['evaluation']['recording'] = dict(enabled=True)
        argv = ['run.py', '--scenario', source, '--no-wandb', '--quiet',
                '--output-dir', str(tmp_path/name), '--run-id', 'evaluation_test']
        if checkpoint:
            # Exercise CLI opt-in independently of the YAML switch.
            argv += ['--eval', '--eval-protocol', 'final', '--checkpoint', str(checkpoint)]
            if enabled:
                argv.append('--record-races')
        monkeypatch.setattr(run, 'load_and_expand_scenario', lambda *_a, **_kw: scenario)
        monkeypatch.setattr('sys.argv', argv)
        run.main()

    before_threads = torch.get_num_threads()
    try:
        launch('off', False)
        launch('on', True)
        off, on = [torch.load(tmp_path/p/'final_model.pt', map_location='cpu', weights_only=False)
                   for p in ('off', 'on')]
        for network in ('actor', 'critic'):
            assert all(torch.equal(v, on[network][k]) for k, v in off[network].items())
        histories = [[json.loads(line) for line in (tmp_path/p/'evaluation_history.jsonl').read_text().splitlines()]
                     for p in ('off', 'on')]
        assert len(histories[0]) == len(histories[1]) >= 2
        for a, b in zip(*histories):
            assert a['selection_score'] == b['selection_score']
            assert a['is_best'] == b['is_best']
            for key, value in a.items():
                if isinstance(value, (float, int)):
                    assert value == b[key]
        assert not (tmp_path/'off'/'evaluation_behavior').exists()
        dataset = tmp_path/'on'/'evaluation_behavior'
        frames = list(iter_race_frames(dataset))
        assert len(frames) == len(histories[1])*5
        assert all(f['reward_availability'] == 'not_computed_by_selection_evaluator' for f in frames)
        assert all(r['reward'] is None for f in frames for r in f['learners'].values())
        clips = [c for c in load_clips(dataset) if c['kind'] == 'representative_race']
        assert len(clips) == len(histories[1]) and all(c['complete'] for c in clips)
        for clip in clips:
            assert hashlib.sha256(Path(clip['checkpoint']).read_bytes()).hexdigest() == clip['checkpoint_sha256']
            assert clip['phase'] == 'evaluation' and clip['protocol'] == 'selection'
            window = load_window(dataset, clip)
            repeat = base['environment']['action_repeat']
            assert [f['substep_index'] for f in window.frames] == [i % repeat for i in range(5)]
            assert [f['decision_index'] for f in window.frames] == [i // repeat for i in range(5)]
            for left, right in zip(window.frames, window.frames[1:]):
                assert left['post_state'] == right['pre_state']
            assert source_reference(window)['checkpoint_sha256'] == clip['checkpoint_sha256']
        review = load_run(tmp_path/'on')
        assert review.clips.race_mode.notna().all()
        assert not filter_clips(review.clips, checkpoint=clips[0]['checkpoint']).empty
        assert set(review.evaluations.checkpoint) == {c['checkpoint'] for c in clips}
        fig, draw = draw_review(load_window(dataset, clips[0]))
        draw(2)
        plt.close(fig)

        checkpoint = tmp_path/'off'/'best_model.pt'
        selected = torch.load(checkpoint, map_location='cpu', weights_only=False)['checkpoint_selection']
        original_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        launch('final_off', False, checkpoint)
        launch('final_on', True, checkpoint)
        reports = [json.loads((tmp_path/p/'evaluation_report.json').read_text()) for p in ('final_off', 'final_on')]
        assert reports[0]['summary'] == reports[1]['summary']
        assert reports[0]['episode_results'] == reports[1]['episode_results']
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == original_hash
        final = load_run(tmp_path/'final_on')
        assert set(final.clips.protocol) == {'final'}
        assert final.clips.race_mode.notna().all()
        assert final.races.environment_steps.eq(selected['environment_steps']).all()
        assert final.clips.policy_version_start.eq(selected['policy_version']).all()
        dataset = tmp_path/'final_on'/'behavior'
        frames = list(iter_race_frames(dataset))
        assert len(frames) == 5
        assert all(f['reward_availability'] == 'computed' for f in frames)
        assert all(isinstance(r['reward'], float) and r['reward_components']
                   for f in frames for r in f['learners'].values())
        assert all(len(f['commands']) == (1 if algorithm == 'ppo' else 4) for f in frames)
    finally:
        torch.set_num_threads(before_threads)


@pytest.mark.parametrize('algorithm', ['ppo', 'mappo'])
def test_selection_recording_preserves_learner_terminal_boundary(tmp_path, monkeypatch, algorithm):
    import run
    from env.types import AgentRaceStatus

    source, scenario = small_scenario(algorithm)
    scenario['experiment']['total_steps'] = 2
    scenario['evaluation']['recording'] = dict(enabled=True)
    if algorithm == 'ppo':
        _, team = small_scenario('mappo')
        scenario['agents']['car_1'] = deepcopy(team['agents']['car_2'])
        scenario['agents']['car_0']['reward'] = '../configs/reward/tasks/race_1v1_completion.yaml'
        scenario['environment'].pop('spawn', None)  # Use separated grid spawns for two cars.
        scenario['environment'].update(agent_teams={'car_0': 'learner', 'car_1': 'opponent'},
            episode_termination=dict(mode='all_agents', lap_completion=True),
            track_limits=dict(enabled=False), terminate_on_collision=True)
    original_setup = run.create_training_setup

    def setup(*args, **kwargs):
        env, *rest = original_setup(*args, **kwargs)
        if kwargs.get('mode') != 'eval':
            return env, *rest
        original_step = env.step

        def step(actions):
            obs, rewards, terms, truncs, infos = original_step(actions)
            if env._elapsed_steps == 1:
                victims = [aid for aid, cfg in scenario['agents'].items() if cfg.get('trainable')]
                for aid in victims:
                    index = env.possible_agents.index(aid)
                    env.lifecycle.record_collision(aid, step=0)
                    env._terminal_controller.capture(aid, status=AgentRaceStatus.CRASHED,
                        terminal_step=0, action=actions[aid], vehicle_state=env.sim.agents[index].physics_state)
                    infos[aid].update(status='crashed', terminal_reason='collision', terminal_step=0)
                    terms[aid] = True
                env.agents = [aid for aid in env.agents if aid not in victims]
            return obs, rewards, terms, truncs, infos

        env.step = step
        return env, *rest

    monkeypatch.setattr(run, 'create_training_setup', setup)
    monkeypatch.setattr(run, 'load_and_expand_scenario', lambda *_a, **_kw: scenario)
    monkeypatch.setattr('sys.argv', ['run.py', '--scenario', source, '--no-wandb', '--quiet',
                                   '--output-dir', str(tmp_path)])
    before_threads = torch.get_num_threads()
    try:
        run.main()
    finally:
        torch.set_num_threads(before_threads)
    frames = list(iter_race_frames(tmp_path/'evaluation_behavior'))
    sample = next(c for c in load_clips(tmp_path/'evaluation_behavior') if c['kind'] == 'representative_race')
    if algorithm == 'ppo':
        assert len(frames) == 1
        assert sample['end_reason'] == 'evaluator_boundary' and not sample['complete']
        assert frames[0]['post_state']['car_1']['active']
    else:
        assert len(frames) == 5 and sample['complete']
        assert all(not f['learners'] for f in frames[1:])
        assert all(f['commands']['car_0']['source'] == 'terminal_controller' for f in frames[1:])
        assert all(f['commands']['car_0']['applied'] == [0., 0.] for f in frames[1:])
