from copy import deepcopy
import json
import random

import numpy as np
import pytest

from src.replay.race_recorder import RaceRecorder, EventDetector, recording_config
from src.replay.dataset_writer import RaceDatasetWriter, detect_dataset_schema
from src.replay.race_reader import load_clips, clip_frames, iter_race_frames, render_state


def state(s=10., delta=0., *, active=True, terminal=None, outside=False):
    return dict(pose=[s, 0., 0.], velocity_body=[1., 0.], angular_velocity=0.,
        active=active, terminal_reason=terminal, collision=terminal == 'collision',
        lap_crossed=False, centerline=dict(s=s, d=0., progress_delta=delta),
        track_limits=dict(exceeded=outside))


def frame(i, *, boundary=False, terminal=False):
    before = {a: state(10. + n*2) for n, a in enumerate(('a', 'b', 'c', 'd'))}
    after = deepcopy(before)
    if boundary:
        after['b']['track_limits']['exceeded'] = True
    if terminal:
        after['c'].update(active=False, terminal_reason='collision')
    return dict(physics_index=i, physics_index_end=i+1, decision_index=i//2, substep_index=i%2,
                simulation_time_s=i*.05, simulation_time_end_s=(i+1)*.05, timestep_s=.05,
                action_repeat=2, policy_version=i//4, pre_state=before, post_state=after)


def start(recorder, episode=0):
    recorder.start(episode_id=f'run_env{recorder.env_id}_ep{episode}', episode=episode,
        map_id='circle_map', seed=42, timestep=.05, action_repeat=2,
        trainable_ids=['a', 'b'], agent_ids=['a', 'b', 'c', 'd'], track_length=100.,
        policy_version=0, spawn={}, physics={}, termination={'mode': 'all_agents'})


def test_sample_selection_is_reproducible_and_does_not_touch_training_rng():
    random.seed(4)
    np.random.seed(4)
    before = random.getstate(), np.random.get_state()
    selections = []
    for run_id in ('one', 'another'):
        recorder = RaceRecorder(dict(sample_probability=.4, events=False), lambda _: None,
                                run_id=run_id, environment_id=2, num_envs=3)
        chosen = []
        for episode in range(30):
            start(recorder, episode)
            chosen.append(recorder.sample is not None)
            recorder.end(False)
        selections.append(chosen)
    assert selections[0] == selections[1] and any(selections[0]) and not all(selections[0])
    assert random.getstate() == before[0]
    np.testing.assert_array_equal(np.random.get_state()[1], before[1][1])


def test_shared_clips_merge_across_cars_and_survive_incremental_roundtrip(tmp_path):
    cfg = dict(sample_probability=1., pre_steps=2, post_steps=2, max_clip_steps=20,
               encounter_distance_m=.5, chunk_frames=2)
    writer = RaceDatasetWriter(tmp_path/'data', config=cfg)
    recorder = RaceRecorder(cfg, writer.add_event, run_id='run')
    start(recorder)
    for i in range(10):
        recorder.step(frame(i, boundary=i == 3, terminal=i == 6))
        assert len(recorder.buffer) <= 3
    recorder.end(True)
    writer.close()
    assert detect_dataset_schema(tmp_path/'data') == '3.0'
    clips = load_clips(tmp_path/'data')
    sample, event = sorted(clips, key=lambda c: c['kind'], reverse=True)
    assert sample['kind'] == 'representative_race' and sample['complete']
    assert event['kind'] == 'event_clip'
    assert event['start_physics_index'] == 1 and event['end_physics_index'] == 8
    assert event['retention_reasons'] == ['boundary_departure', 'terminal']
    assert {tuple(e['participants']) for e in event['events']} == {('b',), ('c',)}
    frames = list(iter_race_frames(tmp_path/'data'))
    assert [f['physics_index'] for f in frames] == list(range(10))  # Shared, not duplicated for each clip.
    selected = list(clip_frames(tmp_path/'data', event))
    assert [f['physics_index'] for f in selected] == list(range(1, 9))
    assert len(render_state(selected[-1]['post_state'])) == 4
    assert all(f['simulation_time_end_s'] > f['simulation_time_s'] for f in selected)


def test_caps_and_budget_cuts_remain_explicit(tmp_path):
    cfg = dict(sample_probability=1., events=False, max_frames=3, chunk_frames=2)
    writer = RaceDatasetWriter(tmp_path/'data', config=cfg)
    recorder = RaceRecorder(cfg, writer.add_event, run_id='run')
    start(recorder)
    for i in range(8):
        recorder.step(frame(i))
    recorder.end(False)
    writer.close()
    clip = load_clips(tmp_path/'data')[0]
    assert clip['end_reason'] == 'frame_limit' and not clip['complete']
    assert clip['end_physics_index'] == 2
    assert len(list(iter_race_frames(tmp_path/'data'))) == 3
    events = []
    recorder = RaceRecorder(dict(sample_probability=1., events=False), events.append, run_id='run')
    start(recorder)
    recorder.step(frame(0))
    recorder.end(False)
    assert events[-1][1]['end_reason'] == 'budget_cut' and not events[-1][1]['complete']
    tiny = RaceDatasetWriter(tmp_path/'tiny', config=dict(max_bytes=1))
    start_event = events[0]
    tiny.add_event(start_event)
    tiny.add_event(events[1])
    tiny.add_event(events[2])
    tiny.close()
    assert tiny.storage_full
    assert not load_clips(tmp_path/'tiny')[0]['complete']
    assert json.loads((tmp_path/'tiny'/'metadata.json').read_text())['serialized_frame_bytes'] == 0


def test_event_only_selection_discards_unselected_frames_before_transfer():
    events = []
    recorder = RaceRecorder(dict(sample_probability=0., pre_steps=2, post_steps=2,
                                 max_clip_steps=20, encounter_distance_m=.5), events.append, run_id='run')
    start(recorder)
    for i in range(13):
        recorder.step(frame(i, boundary=i == 3, terminal=i == 6))
    recorder.end(True)
    assert [f['physics_index'] for kind, f in events if kind == 'frame'] == list(range(1, 9))
    closed = [c for kind, c in events if kind == 'clip' and c['status'] == 'closed']
    assert len(closed) == 1 and closed[0]['post_context_complete']


def test_interrupted_writer_preserves_frames_without_claiming_completion(tmp_path):
    with pytest.raises(RuntimeError, match='interrupted'):
        with RaceDatasetWriter(tmp_path/'data', config={}) as writer:
            recorder = RaceRecorder(dict(sample_probability=1.), writer.add_event, run_id='run')
            start(recorder)
            recorder.step(frame(0))
            raise RuntimeError('interrupted')
    assert not json.loads((tmp_path/'data'/'metadata.json').read_text())['complete']
    assert len(list(iter_race_frames(tmp_path/'data'))) == 1
    assert all(c['status'] == 'open' for c in load_clips(tmp_path/'data'))


def test_early_episode_boundary_does_not_invent_opponent_termination():
    from types import SimpleNamespace
    from src.env.types import AgentLifecycleRecord
    from src.replay.race_recorder import capture_state
    record = AgentLifecycleRecord(agent_id='opponent', target_laps=3)
    env = SimpleNamespace(agents=[], lifecycle=SimpleNamespace(records={'opponent': record}),
        get_agent_state=lambda aid: SimpleNamespace(pose=np.zeros(3), velocity=np.zeros(2),
                                                   angular_velocity=0., collision=False))
    row = capture_state(env, {}, {}, ['opponent'])['opponent']
    assert row['active'] and not row['decision_active']
    assert row['terminal_reason'] is None


def test_reset_provenance_does_not_require_transition_traffic(tmp_path):
    from training.hooks import PhysicsEpisodeHook, transition_record_hooks
    from src.replay.dataset_writer import RaceDatasetHook
    physics = PhysicsEpisodeHook(tmp_path, transition_records=False)
    writer = RaceDatasetWriter(tmp_path/'races', config={})
    hook = RaceDatasetHook(writer)
    assert transition_record_hooks([physics, hook]) == []
    physics.on_episode_start(dict(episode_id='partial', map_id='circle_map', physics={'mu': 1.}))
    assert json.loads((tmp_path/'physics_episodes.jsonl').read_text())['episode_id'] == 'partial'
    writer.close(complete=False)


def test_pass_candidates_use_unwrapped_progress_sustain_and_active_filter():
    cfg = recording_config(dict(pass_sustain_steps=2, pass_hysteresis_m=.1))
    detector = EventDetector(cfg, 100.)
    # B is just ahead of A across the centerline seam. A crosses the seam;
    # the order remains the same, then A moves clearly ahead for two samples.
    positions = [(99., 1.), (99.5, 1.5), (0., 2.), (3., 2.), (3.5, 2.2)]
    events = []
    for i, (before, after) in enumerate(zip(positions, positions[1:])):
        pre = {aid: state(s) for aid, s in zip(('a', 'b'), before)}
        post = {aid: state(s, ((s-old+50)%100-50)/100) for aid, s, old in zip(('a', 'b'), after, before)}
        current = detector.detect(pre, post, i)
        if i < 3:
            assert not any(e['kind'] == 'candidate_pass' for e in current)
        events.extend(current)
    passes = [e for e in events if e['kind'] == 'candidate_pass']
    assert len(passes) == 1 and passes[0]['passing_agent'] == 'a' and passes[0]['source'] == 'heuristic'
    post['b'].update(active=False, terminal_reason='collision')
    assert not any(e['kind'] == 'candidate_pass' for e in detector.detect(pre, post, 5))


def window_config(**overrides):
    return dict(sample_probability=1., events=False, max_frames=5, max_bytes=100000,
        windows=[dict(start_step=0, end_step=3, max_frames=2, max_bytes=50000),
                 dict(start_step=5, end_step=10, max_frames=3, max_bytes=50000)], **overrides)


def test_window_budget_resumes_mid_race_without_stitching_gaps(tmp_path):
    cfg = window_config()
    writer = RaceDatasetWriter(tmp_path, config=cfg)
    recorder = RaceRecorder(cfg, writer.add_event, run_id='run')
    start(recorder)
    for i in range(9):
        recorder.prepare(i, i, i//4, exhausted_windows=writer.exhausted_windows)
        recorder.step(frame(i))
    recorder.end(True)
    writer.close()
    frames = list(iter_race_frames(tmp_path))
    assert [f['physics_index'] for f in frames] == [0, 1, 5, 6, 7]
    assert [f['recording_window_index'] for f in frames] == [0, 0, 1, 1, 1]
    clips = load_clips(tmp_path)
    assert len(clips) == 2 and len({c['clip_id'] for c in clips}) == 2
    assert not any(c['complete'] for c in clips)
    assert clips[1]['kind'] == 'representative_segment'
    assert clips[1]['start_physics_index'] == 5 and clips[1]['policy_version_start'] == 1
    assert [f['physics_index'] for f in clip_frames(tmp_path, clips[1])] == [5, 6, 7]
    metadata = json.loads((tmp_path/'metadata.json').read_text())
    assert [w['frames'] for w in metadata['recording_windows']] == [2, 3]
    assert all(w['exhausted'] for w in metadata['recording_windows'])


def test_shared_window_byte_cap_preserves_later_allocation_and_event_context(tmp_path):
    cfg = window_config()
    cfg['events'] = True
    cfg['sample_probability'] = 0.
    cfg['windows'][0]['max_bytes'] = 1  # No frame can fit, but the later budget survives.
    writer = RaceDatasetWriter(tmp_path, config=cfg)
    recorders = [RaceRecorder(cfg, writer.add_event, run_id='run', environment_id=i) for i in range(2)]
    for recorder in recorders:
        start(recorder)
        for step in range(7):
            recorder.prepare(step, step, step//4, exhausted_windows=writer.exhausted_windows,
                             clock='last_completed_collection_barrier')
            recorder.step(frame(step, boundary=step in (0, 6)))
        recorder.end(False)
    writer.close()
    frames = list(iter_race_frames(tmp_path))
    assert len(frames) == 3  # Shared cap across collectors, not three per collector.
    assert all(f['recording_window_index'] == 1 and f['physics_index'] >= 5 for f in frames)
    assert not writer.storage_full and writer.exhausted_windows == {0, 1}
    assert all(not c['complete'] for c in load_clips(tmp_path))
    assert all(c['start_physics_index'] >= 5 for c in load_clips(tmp_path) if c['recording_window_index'] == 1)


@pytest.mark.parametrize('change', [
    dict(start_step=2, end_step=2), dict(start_step=-1), dict(max_frames=True), dict(max_bytes=100001),
])
def test_recording_windows_validate_ranges_and_reservations(change):
    cfg = window_config()
    cfg['windows'][0].update(change)
    with pytest.raises(ValueError):
        recording_config(cfg)
