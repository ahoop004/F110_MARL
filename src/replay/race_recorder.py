"""Collector-local sampling and bounded, shared-scene event clips.

Selection never consumes training RNG. Frames describe one physics interval;
learner observations/actions refer to its enclosing policy decision.
"""
from collections import deque
from itertools import combinations
import hashlib
import math

import numpy as np


DEFAULTS = dict(enabled=True, sample_probability=0.01, seed=42,
                events=True, pre_steps=60, post_steps=100, max_clip_steps=600,
                max_clips_per_episode=16, max_events_per_clip=128,
                max_frames=100000, max_bytes=2_000_000_000, chunk_frames=128,
                encounter_distance_m=8.0, lateral_distance_m=3.0,
                pass_hysteresis_m=0.3, pass_sustain_steps=3, windows=[])


def recording_config(config=None):
    config = dict(config or {})
    unknown = set(config) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown recording settings: {sorted(unknown)}")
    cfg = {**DEFAULTS, **config}
    if not isinstance(cfg['enabled'], bool) or not isinstance(cfg['events'], bool):
        raise ValueError("recording enabled/events must be booleans")
    probability = cfg['sample_probability']
    if isinstance(probability, bool) or not isinstance(probability, (int, float)) or not 0 <= probability <= 1:
        raise ValueError("recording.sample_probability must be between zero and one")
    for name in ('seed', 'pre_steps', 'post_steps', 'max_clip_steps', 'max_clips_per_episode',
                 'max_events_per_clip', 'max_frames', 'max_bytes', 'chunk_frames', 'pass_sustain_steps'):
        value = cfg[name]
        minimum = 0 if name in ('seed', 'pre_steps', 'post_steps') else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"recording.{name} must be an integer >= {minimum}")
    if cfg['max_clip_steps'] < cfg['pre_steps'] + cfg['post_steps'] + 1:
        raise ValueError("max_clip_steps must cover the initial pre/event/post interval")
    for name in ('encounter_distance_m', 'lateral_distance_m', 'pass_hysteresis_m'):
        if not math.isfinite(float(cfg[name])) or cfg[name] <= 0:
            raise ValueError(f"recording.{name} must be finite and positive")
    windows = cfg['windows']
    if not isinstance(windows, list):
        raise ValueError('recording.windows must be a list')
    previous_end = 0
    for window in windows:
        if not isinstance(window, dict) or set(window) != {'start_step', 'end_step', 'max_frames', 'max_bytes'}:
            raise ValueError('Each recording window needs start_step, end_step, max_frames, max_bytes')
        for key, value in window.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < (0 if key == 'start_step' else 1):
                raise ValueError(f'recording window {key} must be a valid nonnegative/positive integer')
        if window['start_step'] < previous_end or window['end_step'] <= window['start_step']:
            raise ValueError('Recording windows must be ordered, non-overlapping [start_step, end_step) ranges')
        previous_end = window['end_step']
    for limit in ('max_frames', 'max_bytes'):
        if sum(w[limit] for w in windows) > cfg[limit]:
            raise ValueError(f'Recording window {limit} allocations exceed the dataset-wide cap')
    cfg['windows'] = [dict(w) for w in windows]
    return cfg


def plain(value):
    """Only structured numeric/string values; never stringify unknown objects."""
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported recording value: {type(value).__name__}")


def capture_state(env, infos, observations, agent_ids):
    """Copy selected simulator facts. Geometry fallback is read-only."""
    result = {}
    lifecycle = getattr(getattr(env, 'lifecycle', None), 'records', {})
    geometry = getattr(env, '_track_preview_geometry', None)
    for aid in agent_ids:
        info = infos.get(aid, {})
        state = env.get_agent_state(aid)
        record = lifecycle.get(aid)
        status = info.get('status', getattr(getattr(record, 'status', None), 'value', None))
        reason = info.get('terminal_reason', getattr(getattr(record, 'terminal_reason', None), 'value', None))
        centerline = info.get('centerline') or getattr(env, '_last_centerline_facts', {}).get(aid, {})
        preview = info.get('track_preview')
        if preview is None and geometry is not None:
            preview = geometry.preview(np.asarray(state.pose)[:2], 1)
        preview = preview or {}
        limits = info.get('track_limits')
        obs = observations.get(aid, {})
        mask = getattr(getattr(env, 'sim', None), 'collidable_mask', None)
        collidable = bool(mask[list(env.possible_agents).index(aid)]) if mask is not None else None
        result[aid] = plain({
            'pose': state.pose, 'velocity_body': state.velocity,
            'angular_velocity': state.angular_velocity,
            # Early episode termination can clear env.agents while opponents
            # remain alive. Preserve lifecycle truth separately from scheduling.
            'active': record.is_active if record is not None else (status == 'active' if status else aid in env.agents),
            'decision_active': aid in env.agents, 'status': status,
            'collidable': collidable,
            'present': collidable if collidable is not None else True,
            'terminal_reason': reason, 'terminal_step': info.get('terminal_step', getattr(record, 'terminal_step', None)),
            'finish_position': info.get('finish_position', getattr(record, 'finish_position', None)),
            'lap_count': info.get('lap_count', getattr(record, 'lap_count', None)),
            'lap_crossed': info.get('lap_crossed', False),
            'lap_time_steps': info.get('lap_time_steps'),
            'collision': info.get('collision', bool(state.collision)),
            'collision_partner': None, 'collision_fault': None,
            'centerline': {k: centerline.get(k) for k in ('s', 'd', 'vs', 'vd', 'progress', 'progress_delta', 'heading_error')},
            'track': {k: preview.get(k) for k in ('current_width', 'current_lateral_error', 'curvature', 'width')},
            'track_limits': ({k: limits.get(k) for k in ('half_width', 'lateral_error', 'offtrack_distance', 'exceeded')}
                             if limits is not None else None),
            'controls': {k: obs.get(k) for k in ('steering_angle', 'steering_reference',
                'speed_reference', 'speed_reference_rate', 'wheel_speed',
                'wheel_speed_reference', 'wheel_speed_reference_rate')},
        })
    return result


class EventDetector:
    """Candidate encounters/passes are hypotheses; terminal/boundary facts are measured."""
    version = 'traffic_candidates_v1'

    def __init__(self, cfg, track_length):
        self.cfg, self.length = cfg, track_length
        self.pairs = {}
        self.distance = {}

    def detect(self, pre, post, step):
        events = []
        def emit(kind, participants, source='simulator', **extra):
            events.append(dict(kind=kind, participants=list(participants), source=source,
                               physics_index=step, detector_version=self.version, **extra))
        for aid, state in post.items():
            old = pre[aid]
            if state['terminal_reason'] and not old['terminal_reason']:
                emit('terminal', [aid], terminal_reason=state['terminal_reason'], collision_partner=None)
            if (state.get('track_limits') or {}).get('exceeded') and not (old.get('track_limits') or {}).get('exceeded'):
                emit('boundary_departure', [aid])
            if state.get('lap_crossed'):
                emit('lap_crossing', [aid])
            delta = state['centerline'].get('progress_delta')
            if delta is not None and old['active']:
                self.distance[aid] = self.distance.get(aid, 0.) + delta * self.length
        for a, b in combinations(post, 2):
            key = (a, b)
            if not all(pre[x]['active'] and post[x]['active'] for x in key):
                self.pairs.pop(key, None)
                continue
            sa, sb = pre[a]['centerline'].get('s'), pre[b]['centerline'].get('s')
            da, db = post[a]['centerline'].get('d'), post[b]['centerline'].get('d')
            if None in (sa, sb, da, db) or self.length <= 0:
                continue
            if key not in self.pairs:
                gap = (sb - sa + self.length / 2) % self.length - self.length / 2
                # Distance already includes this frame, so remove its delta for initialization.
                delta = sum(sign * (post[x]['centerline'].get('progress_delta') or 0.) * self.length
                            for x, sign in ((b, 1), (a, -1)))
                offset = gap - (self.distance.get(b, 0.) - self.distance.get(a, 0.) - delta)
                self.pairs[key] = dict(offset=offset, order=0, pending=0, count=0, near=False)
            pair = self.pairs[key]
            gap = pair['offset'] + self.distance.get(b, 0.) - self.distance.get(a, 0.)
            near = abs(gap) <= self.cfg['encounter_distance_m'] and abs(db-da) <= self.cfg['lateral_distance_m']
            if near and not pair['near']:
                emit('encounter', key, 'heuristic', longitudinal_gap_m=gap, lateral_gap_m=db-da)
            pair['near'] = near
            sign = 1 if gap > self.cfg['pass_hysteresis_m'] else -1 if gap < -self.cfg['pass_hysteresis_m'] else 0
            if not near:
                pair.update(order=sign, pending=0, count=0)
            elif sign:
                pair['count'] = pair['count'] + 1 if pair['pending'] == sign else 1
                pair['pending'] = sign
                if pair['count'] >= self.cfg['pass_sustain_steps']:
                    if pair['order'] and pair['order'] != sign:
                        emit('candidate_pass', key, 'heuristic', passing_agent=a if sign == -1 else b,
                             passed_agent=b if sign == -1 else a)
                    pair['order'] = sign
        return events


class RaceRecorder:
    """One recorder per environment. Emits retained frames and clip index updates."""
    def __init__(self, cfg, emit, *, run_id, environment_id=0, num_envs=1):
        self.cfg = recording_config(cfg)
        self.emit, self.run_id, self.env_id = emit, run_id, environment_id
        # The writer enforces the shared cap and tells collectors to stop at
        # the next barrier. Do not divide it into tiny per-environment budgets:
        # at 400 environments that would prevent long sampled races completing.
        self.quota = self.cfg['max_frames']
        self.frames_sent = 0
        self.stopped = False
        self.buffer = deque(maxlen=self.cfg['pre_steps'] + 1)
        self.sample = self.clip = None
        self.clip_count = 0
        self.window_index = None
        self.window_frames = 0
        self.window_stopped = False
        self.progress = None

    @property
    def capturing(self):
        return (not self.stopped and not self.window_stopped and
            (not self.cfg['windows'] or self.window_index is not None) and
            (self.sample is not None or (self.cfg['events'] and
            (self.clip is not None or self.clip_count < self.cfg['max_clips_per_episode']))))

    def prepare(self, progress, physics_index, policy_version, *, exhausted_windows=(),
                clock='joint_environment_decisions', team_policy_versions=None):
        """Select a budget using the authoritative serial count or parallel barrier.

        Called before copying a physics frame. Switching windows closes existing
        clips and drops pre-event context; no clip bridges a recording gap.
        """
        if not self.cfg['windows'] or self.stopped:
            return
        self.progress = progress
        index = next((i for i, w in enumerate(self.cfg['windows'])
                      if progress is not None and w['start_step'] <= progress < w['end_step']), None)
        changed = index != self.window_index
        if changed:
            self._pause('window_end')
            self.window_index, self.window_frames, self.window_stopped = index, 0, False
            self.clip_count = 0
        if index is None:
            return
        if index in exhausted_windows:
            if not self.window_stopped:
                self._pause('window_storage_limit')
            return
        if self.window_stopped:
            return
        if changed or self._window_needs_start:
            self._window_needs_start = False
            self.last_frame = self.last_retained = None
            self.last_sent = physics_index-1
            self.detector = EventDetector(self.cfg, self.context['track_length_m'])
            self.context.update(recording_window_index=index, recording_window=self.cfg['windows'][index],
                recording_progress_start=progress, recording_progress_clock=clock,
                capture_policy_version_start=policy_version,
                capture_team_policy_versions_start=team_policy_versions,
                coverage_scope='environment_episode' if physics_index == 0 else 'partial_environment_episode')
            if self._sample_selected:
                self.sample = self._open('representative_race' if physics_index == 0 else 'representative_segment', physics_index)

    def _pause(self, reason):
        self._close(self.sample, self.last_sent, reason)
        self._close(self.clip, self.last_sent, reason)
        self.sample = self.clip = None
        self.window_stopped = True
        self.buffer.clear()

    def start(self, *, episode_id, episode, map_id, seed, timestep, action_repeat,
              trainable_ids, agent_ids, track_length, policy_version, spawn, physics, termination,
              agent_teams=None, team_policy_versions=None, evaluation=None):
        self.buffer.clear()
        self.last_sent = -1
        self.last_frame = None
        self.last_retained = None
        self.clip_count = 0
        self.detector = EventDetector(self.cfg, track_length)
        self.context = plain(dict(run_id=self.run_id, environment_id=self.env_id,
            episode_id=episode_id, environment_episode=episode, map_id=map_id, environment_seed=seed,
            timestep_s=timestep, action_repeat=action_repeat, trainable_ids=trainable_ids,
            agent_ids=agent_ids, track_length_m=track_length, episode_policy_version_start=policy_version,
            spawn=spawn, physics=physics, episode_termination=termination,
            sampling=dict(rule='sha256(seed,environment_id,episode_index)', seed=self.cfg['seed'],
                          probability=self.cfg['sample_probability']), detector_version=EventDetector.version))
        self.context['shared_frame_budget'] = self.quota
        self.context['coverage_scope'] = 'environment_episode'
        self.context['phase'] = 'evaluation' if evaluation is not None else 'training'
        if evaluation is not None:
            self.context.update(plain(evaluation))
        if agent_teams is not None:
            self.context.update(agent_teams=plain(agent_teams),
                                episode_team_policy_versions_start=plain(team_policy_versions))
        draw = int.from_bytes(hashlib.sha256(f"{self.cfg['seed']}:{self.env_id}:{episode}".encode()).digest()[:8], 'big') / 2**64
        self.sample = self.clip = None
        self._window_needs_start = True
        self._sample_selected = draw < self.cfg['sample_probability']
        if self.frames_sent >= self.quota:
            self.stopped = True
        if not self.stopped and self._sample_selected and not self.cfg['windows']:
            self.sample = self._open('representative_race', 0)

    def _open(self, kind, start):
        suffix = 'sample' if kind.startswith('representative_') else f'event{self.clip_count:04d}'
        if self.cfg['windows']:
            suffix = f'window{self.window_index:04d}_{suffix}'
        clip = {**self.context, 'clip_id': f"{self.context['episode_id']}_{suffix}",
                'kind': kind, 'start_physics_index': start, 'end_physics_index': None,
                'policy_version_start': (self.buffer[0]['policy_version'] if kind == 'event_clip' and self.buffer
                                         else self.context.get('capture_policy_version_start', self.context['episode_policy_version_start'])),
                'status': 'open', 'events': [], 'event_count': 0, 'retention_reasons': [],
                'complete': False, 'episode_complete': False, 'all_cars_terminal': False}
        if 'agent_teams' in self.context:
            clip['team_policy_versions_start'] = (self.buffer[0].get('team_policy_versions')
                if kind == 'event_clip' and self.buffer else self.context.get(
                    'capture_team_policy_versions_start', self.context['episode_team_policy_versions_start']))
        self.emit(('clip', dict(clip)))
        return clip

    def _close(self, clip, end, reason, episode_complete=False):
        if clip is None:
            return
        frame = self.last_retained
        clip = dict(clip)
        end_context = clip.pop('_end_context', {})
        clip.update(status='closed', end_physics_index=min(end, self.last_sent), end_reason=reason,
                    complete=reason in ('episode_end', 'post_event_complete') and clip['kind'] != 'representative_segment',
                    episode_complete=episode_complete,
                    all_cars_terminal=end_context.get('all_cars_terminal', bool(frame and all(not s['active'] for s in frame['post_state'].values()))),
                    policy_version_end=end_context.get('policy_version', frame['policy_version'] if frame else None),
                    post_context_complete=(reason == 'post_event_complete' or (reason == 'episode_end' and
                        (clip['kind'].startswith('representative_') or
                         (clip['end_physics_index'] is not None and end >= clip['end_physics_index'])))))
        if 'agent_teams' in self.context:
            clip['team_policy_versions_end'] = end_context.get('team_policy_versions',
                frame.get('team_policy_versions') if frame else None)
        self.emit(('clip', clip))

    def _send(self, frame):
        step = frame['physics_index']
        if step <= self.last_sent:
            return
        if self.frames_sent >= self.quota:
            self.stop('frame_limit')
            return
        if self.cfg['windows'] and self.window_frames >= self.cfg['windows'][self.window_index]['max_frames']:
            self._pause('window_frame_limit')
            return
        self.emit(('frame', frame))
        self.frames_sent += 1
        self.window_frames += 1
        self.last_sent = step
        self.last_retained = frame

    def step(self, frame):
        if not self.capturing:
            return
        self.last_frame = frame
        step = frame['physics_index']
        frame.update(run_id=self.run_id, environment_id=self.env_id,
                     episode_id=self.context['episode_id'], map_id=self.context['map_id'])
        frame.update({key: self.context[key] for key in
            ('phase', 'evaluation_id', 'protocol', 'checkpoint', 'checkpoint_sha256') if key in self.context})
        if self.cfg['windows']:
            frame.update(recording_window_index=self.window_index, recording_progress=self.progress,
                         recording_progress_clock=self.context['recording_progress_clock'])
        events = self.detector.detect(frame['pre_state'], frame['post_state'], step)
        frame['events'] = events
        self.buffer.append(frame)
        triggers = [e for e in events if e['kind'] != 'lap_crossing'] if self.cfg['events'] else []
        if self.clip is not None and step > self.clip['end_physics_index'] + self.cfg['pre_steps']:
            self._close(self.clip, self.clip['end_physics_index'], 'post_event_complete')
            self.clip = None
        if triggers:
            if self.clip is None and self.clip_count < self.cfg['max_clips_per_episode']:
                self.clip_count += 1
                self.clip = self._open('event_clip', self.buffer[0]['physics_index'])
                self.clip['end_physics_index'] = step
            if self.clip is not None:
                clip = self.clip
                cap = clip['start_physics_index'] + self.cfg['max_clip_steps'] - 1
                clip['end_physics_index'] = min(cap, step + self.cfg['post_steps'])
                clip['event_count'] += len(triggers)
                clip['events'] = (clip['events'] + triggers)[:self.cfg['max_events_per_clip']]
                clip['retention_reasons'] = sorted(set(clip['retention_reasons']) | {e['kind'] for e in triggers})
                self.emit(('clip', {k: v for k, v in clip.items() if not k.startswith('_')}))
                for previous in tuple(self.buffer):
                    if previous['physics_index'] <= clip['end_physics_index'] and self.capturing:
                        self._send(previous)
        if self.capturing and (self.sample is not None or (self.clip and step <= self.clip['end_physics_index'])):
            self._send(frame)
        if self.clip and step <= self.clip['end_physics_index']:
            self.clip['_end_context'] = dict(policy_version=frame['policy_version'],
                team_policy_versions=frame.get('team_policy_versions'),
                all_cars_terminal=all(not s['active'] for s in frame['post_state'].values()))
        if self.clip and step >= self.clip['start_physics_index'] + self.cfg['max_clip_steps'] - 1:
            self._close(self.clip, self.clip['end_physics_index'], 'clip_length_limit')
            self.clip = None

    def end(self, episode_complete, *, reason=None):
        step = self.last_frame['physics_index'] if self.last_frame else -1
        reason = reason or ('episode_end' if episode_complete else 'budget_cut')
        self._close(self.sample, step, reason, episode_complete)
        self._close(self.clip, min(step, self.clip['end_physics_index']) if self.clip else step,
                    reason if self.clip and step < self.clip['end_physics_index'] else 'post_event_complete', episode_complete)
        self.sample = self.clip = None
        self.buffer.clear()

    def stop(self, reason='storage_limit'):
        self._close(self.sample, self.last_sent, reason)
        self._close(self.clip, self.last_sent, reason)
        self.sample = self.clip = None
        self.stopped = True
        self.buffer.clear()
