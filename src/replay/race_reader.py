"""Read shared frames using chunk spans; preserve sampling and clip metadata."""
import gzip
import json
from pathlib import Path


def load_clips(directory):
    path = Path(directory) / 'clips.jsonl'
    clips = {}
    if path.exists():
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                clips[row['clip_id']] = row
    return list(clips.values())


def iter_race_frames(directory, *, episode_id=None, start=0, end=None):
    directory = Path(directory)
    path = directory / 'frame_chunks.jsonl'
    if not path.exists():
        return
    with path.open() as manifest:
        for line in manifest:
            chunk = json.loads(line)
            if episode_id is not None:
                span = chunk['episodes'].get(episode_id)
                if span is None or span[1] < start or (end is not None and span[0] > end):
                    continue
            with gzip.open(directory / chunk['path'], 'rt', encoding='utf-8') as stream:
                for line in stream:
                    frame = json.loads(line)
                    if (episode_id is None or frame['episode_id'] == episode_id) and frame['physics_index'] >= start:
                        if end is None or frame['physics_index'] <= end:
                            yield frame


def clip_frames(directory, clip):
    yield from iter_race_frames(directory, episode_id=clip['episode_id'],
                                start=clip['start_physics_index'], end=clip['end_physics_index'])


def render_state(state):
    return {aid: dict(poses_x=s['pose'][0], poses_y=s['pose'][1], poses_theta=s['pose'][2],
                      linear_vels_x=s['velocity_body'][0], linear_vels_y=s['velocity_body'][1],
                      ang_vels_z=s['angular_velocity'], collision=s['collision'])
            for aid, s in state.items()}
