from copy import deepcopy
import asyncio
import json
from pathlib import Path

import numpy as np
import pytest

from src.analysis.annotations import AnnotationStore, segment
from src.analysis.clip_review import ClipWindow, load_window, telemetry, draw_review
from src.analysis.run_review import filter_clips
from src.replay.dataset_writer import RaceDatasetWriter
import pandas as pd


@pytest.fixture
def window(tmp_path):
    states = {}
    for i in range(4):
        states[f'car_{i}'] = dict(pose=[i, 0., 0.], velocity_body=[1., 0.], active=True,
            terminal_reason=None, lap_count=0, centerline=dict(s=98.+i, d=i*.1, progress_delta=0.),
            controls=dict(steering_angle=0., wheel_speed_reference_rate=2.))
    frames = []
    t = 0.
    for index, dt in enumerate([.05, .1, .025, .05]):
        pre = deepcopy(states)
        for i, state in enumerate(states.values()):
            delta = .02 if i == 0 else .01
            state['centerline']['s'] = (state['centerline']['s']+delta*100)%100
            state['centerline']['progress_delta'] = delta
            state['pose'][0] += delta*100
        frames.append(dict(run_id='run', environment_id=0, episode_id='ep', physics_index=index,
            physics_index_end=index+1, simulation_time_s=t, simulation_time_end_s=t+dt,
            timestep_s=dt, policy_version=0, pre_state=pre, post_state=deepcopy(states),
            commands={a: dict(applied=[0., 2.]) for a in states}, learners={}, events=[]))
        t += dt
    clip = dict(run_id='run', environment_id=0, episode_id='ep', clip_id='sample', map_id='map',
        start_physics_index=0, end_physics_index=3, policy_version_start=0, policy_version_end=0,
        agent_ids=list(states), trainable_ids=['car_0', 'car_1'], track_length_m=100.,
        kind='representative_race', complete=True, detector_version='test')
    metadata = dict(action_contract=dict(units=['rad', 'rad/s'], speed_control='wheel_acceleration', rate_units='rad/s^2'))
    dataset = tmp_path/'raw'
    with RaceDatasetWriter(dataset, config={}, metadata=metadata) as writer:
        for frame in frames:
            writer.add_event(('frame', frame))
    return ClipWindow(dataset, clip, metadata, frames)


def individual(w, start=0, end=2, actor='car_0', **kwargs):
    return segment(w, start=start, end=end, scope='individual', participants=[actor], label='pass attempt', **kwargs)


def test_bounded_loading_timing_alignment_and_seam_gap(window):
    loaded = load_window(window.dataset, window.clip, start=1, end=2, max_frames=2)
    assert loaded.boundaries == [1, 2, 3]
    assert loaded.delay(0, 2) == pytest.approx(.05)
    assert len(load_window(window.dataset, {**window.clip, 'end_physics_index': float('nan')}).frames) == 4
    with pytest.raises(ValueError, match='at most'):
        load_window(window.dataset, window.clip, max_frames=2)
    with pytest.raises(ValueError, match='identity'):
        load_window(window.dataset, {**window.clip, 'environment_id': 1})
    series, units, _ = telemetry(window, 'car_1')
    # No seam jump: car_0 gains 1 m on car_1 each interval.
    np.testing.assert_allclose(series['car_0']['gap'], [-1., 0., 1., 2., 3.])
    assert units == ['rad', 'rad/s']
    assert window.state(0) == window.frames[0]['pre_state']
    assert window.state(1) == window.frames[0]['post_state']
    fig, draw = draw_review(window)
    draw(2)
    assert fig.axes[0].get_title().endswith('boundary 2')


def test_roundtrip_combined_links_edits_evidence_and_cross_race_rejection(window, tmp_path):
    store = AnnotationStore(tmp_path/'review'/'annotations.json')
    a = store.save(individual(window, end=2))
    b = store.save(individual(window, start=1, end=4, actor='car_1'))
    team = segment(window, start=0, end=4, scope='combined', participants=['car_0', 'car_1'],
        label='sequential teammate passes', constituent_ids=[a['annotation_id'], b['annotation_id']],
        roles={'car_0': 'first passer', 'car_1': 'second passer'},
        role_changes=[dict(physics_index=2, roles={'car_0': 'yielding'})])
    c = store.save(team)
    assert c['temporal_relations'][0]['relation'] == 'overlap'
    reopened = AnnotationStore(store.path)
    assert len(reopened.records) == 3
    a['notes'] = 'custom review note'
    assert reopened.save(a)['annotation_id'] == a['annotation_id']
    team['team_outcome'] = 'beneficial'
    with pytest.raises(ValueError, match='evidence'):
        reopened.save(team)
    team['team_outcome'] = 'unknown'
    team['source']['episode_id'] = 'another_race'
    team['annotation_id'] = 'new'
    with pytest.raises(ValueError, match='same recorded race'):
        reopened.save(team)
    # Editing a linked individual must preserve all existing parent invariants.
    b['participants'] = ['car_2']
    with pytest.raises(ValueError, match='constituent actors'):
        reopened.save(b)
    assert len(AnnotationStore(store.path).records) == 3


def test_concurrent_edit_rejected_and_raw_directory_protected(window, tmp_path):
    first = AnnotationStore(tmp_path/'annotations.json')
    second = AnnotationStore(first.path)
    first.save(individual(window))
    with pytest.raises(ValueError, match='Reload'):
        second.save(individual(window, actor='car_1'))
    second.reload()
    second.save(individual(window, actor='car_1'))
    assert len(AnnotationStore(first.path).records) == 2
    with pytest.raises(ValueError, match='outside'):
        AnnotationStore(window.dataset/'annotations.json').save(individual(window))


def test_missing_frame_and_unrelated_chunks(window):
    # An unrelated, unreadable chunk must not be opened for this clip.
    manifest = window.dataset/'frame_chunks.jsonl'
    with manifest.open('a') as f:
        f.write(json.dumps(dict(path='missing.gz', episodes={'other_episode': [0, 9]}, frames=10))+'\n')
    assert len(load_window(window.dataset, window.clip).frames) == 4
    with pytest.raises(ValueError, match='fully retained'):
        load_window(window.dataset, {**window.clip, 'end_physics_index': 4})


def test_checkpoint_filter_does_not_invent_final_checkpoint_identity():
    clips = pd.DataFrame([dict(clip_id='old', policy_version_start=0, policy_version_end=2),
                          dict(clip_id='new', policy_version_start=3, policy_version_end=4, checkpoint_sha256='hash')])
    assert filter_clips(clips, policy_version=2).clip_id.tolist() == ['old']
    assert filter_clips(clips, checkpoint='hash').clip_id.tolist() == ['new']
    assert filter_clips(clips, checkpoint='unknown').empty


def test_widget_save_reload_controls_and_playback(window, tmp_path):
    from src.analysis.review_widget import ClipReviewer
    reviewer = ClipReviewer(window, annotation_path=tmp_path/'annotations.json')
    reviewer.end.value = 2
    a = reviewer.save_annotation()
    reviewer.saved.value = None
    reviewer.participants.value = ('car_1',)
    reviewer.start.value, reviewer.end.value = 2, 4
    b = reviewer.save_annotation()
    reviewer.saved.value = None
    reviewer.scope.value = 'combined'
    reviewer.participants.value = ('car_0', 'car_1')
    reviewer.constituents.value = (a['annotation_id'], b['annotation_id'])
    reviewer.label.value = 'custom combined label'
    c = reviewer.save_annotation()
    assert c['temporal_relations'][0]['relation'] == 'before'
    reviewer.notes.value = 'edited'
    assert reviewer.save_annotation()['annotation_id'] == c['annotation_id']
    reviewer.close()
    reopened = ClipReviewer(window, annotation_path=tmp_path/'annotations.json')
    assert tuple(reopened.canvas._size) == (1100, 800)
    reopened.saved.value = c['annotation_id']
    assert reopened.notes.value == 'edited'
    assert reopened.constituents.value == (a['annotation_id'], b['annotation_id'])
    async def exercise():
        reopened.speed.value = 4.
        reopened.play.value = True
        await asyncio.sleep(.02)
        reopened.play.value = False
        await asyncio.sleep(0)
        assert reopened.cursor.value > 0
        reopened.seek(0)
        before = np.asarray(reopened.canvas.buffer_rgba()).copy()
        reopened.seek(len(window.frames))
        after = np.asarray(reopened.canvas.buffer_rgba()).copy()
        assert not np.array_equal(before, after)  # Canvas pixels must advance with the clock.
        reopened.seek(0)
        reopened.play.value = True
        await asyncio.wait_for(reopened._task, timeout=10)
        assert reopened.cursor.value == len(window.frames)
        assert not reopened.play.value
    asyncio.run(exercise())
    reopened.close()
