"""Skill facts and reproducible, collision-free training starts."""
from collections.abc import Mapping
import math

import numpy as np

from utils.track_preview import _distance_to_wall

SKILL_SEED_OFFSETS = {'pass': 0, 'defend': 100000, 'solo': 200000, 'recovery': 300000, 'pressure': 400000}


def validate_skill_task(config, agent_ids):
    if config is None:
        return None
    defaults = dict(lead_margin=1., confirmation_s=1., moving_speed=.5,
                    duration_s=30., min_progress=0.)
    required = {"skill", "ego_id"}
    if not isinstance(config, Mapping) or not required <= config.keys():
        raise ValueError("skill_task requires skill and ego_id")
    if config['skill'] == 'recovery':
        defaults.update(lateral_tolerance=.2, heading_tolerance=.15)
    else:
        required.add('target_id')
    if config['skill'] == 'pressure':
        defaults.update(pressure_distance=6., min_pressure_fraction=.8,
                        max_opponent_speed_fraction=.8)
    if (not isinstance(config, Mapping) or not required <= config.keys()
            or set(config) - required - defaults.keys()):
        raise ValueError("skill_task requires skill, ego_id, target_id and optional task thresholds")
    cfg = {**defaults, **config}
    if cfg['skill'] not in {'pass', 'defend', 'recovery', 'pressure'}:
        raise ValueError("skill_task.skill must be pass, defend, recovery or pressure")
    if (cfg['ego_id'] not in agent_ids or (cfg['skill'] != 'recovery' and
            (cfg['ego_id'] == cfg['target_id'] or cfg['target_id'] not in agent_ids))):
        raise ValueError("skill_task requires distinct known participants")
    positive = {'lead_margin', 'confirmation_s', 'duration_s', 'lateral_tolerance',
                'heading_tolerance', 'pressure_distance', 'max_opponent_speed_fraction'}
    for key in defaults:
        value = cfg[key]
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(value) or value < 0 or
                (key in positive and value == 0)):
            raise ValueError(f"skill_task.{key} must be finite and {'positive' if key in positive else 'nonnegative'}")
        if key.endswith('_fraction') and value > 1:
            raise ValueError(f"skill_task.{key} must be at most 1")
    return cfg


class SkillTracker:
    """Ordering is initial separation plus earned distance, never wrapped gap."""

    def __init__(self, config):
        self.config = validate_skill_task(config, [config['ego_id'], config.get('target_id')])
        self.reset(0.)

    def reset(self, ego_lead, *, opponent_speed=None):
        self.lead = float(ego_lead)
        self.progress = self.ahead_time = self.time = 0.
        self.opponent_progress = self.pressure_time = 0.
        self.opponent_speed = opponent_speed
        self.lateral_error = self.heading_error = 0.
        self.confirm_since = None
        self.done = False
        self.outcome = 'active'

    def update(self, *, time, infos, collisions, track_length, timed_out=False):
        cfg = self.config
        ego = infos[cfg['ego_id']]
        target = infos.get(cfg.get('target_id'))
        dt = max(0., time - self.time)
        if self.done:
            return self.facts(dt=0.)
        self.time = time
        delta = float(ego['centerline']['progress_delta']) * track_length
        target_delta = float(target['centerline']['progress_delta']) * track_length if target else 0.
        relative = delta - target_delta
        self.lead += relative
        self.progress += delta
        self.opponent_progress += target_delta
        moving = ego['centerline']['vs'] > cfg['moving_speed']
        ego_failed = bool(collisions[cfg['ego_id']] or ego['track_limits']['exceeded'])
        target_failed = bool(target and (collisions[cfg['target_id']] or target['track_limits']['exceeded']))
        ahead = self.lead > 0 and moving and not (ego_failed or target_failed)
        if ahead:
            self.ahead_time += dt
        if cfg['skill'] == 'recovery':
            self.lateral_error = abs(float(ego['centerline']['d']))
            self.heading_error = abs(float(ego['centerline']['heading_error']))
            confirming = (moving and self.lateral_error <= cfg['lateral_tolerance'] and
                          self.heading_error <= cfg['heading_tolerance'])
        else:
            confirming = ((self.lead >= cfg['lead_margin'] and moving) if cfg['skill'] == 'pass'
                          else self.lead <= -cfg['lead_margin'])
        if cfg['skill'] == 'pressure':
            if self.opponent_speed is None or self.opponent_speed <= 0:
                raise ValueError('Pressure requires the sampled opponent speed at reset')
            if ahead and self.lead <= cfg['pressure_distance']:
                self.pressure_time += dt
        if confirming:
            if self.confirm_since is None:
                self.confirm_since = time
        else:
            self.confirm_since = None
        confirmed = self.confirm_since is not None and time - self.confirm_since >= cfg['confirmation_s'] - 1e-9
        if ego_failed:
            self.outcome = 'ego_failure'
        elif target_failed:
            self.outcome = 'opponent_failure'
        elif confirmed and cfg['skill'] == 'recovery' and self.progress >= cfg['min_progress']:
            self.outcome = 'success'
        elif confirmed and cfg['skill'] != 'recovery':
            self.outcome = 'success' if cfg['skill'] == 'pass' else 'lead_lost'
        elif time >= cfg['duration_s'] - 1e-9:
            if cfg['skill'] == 'defend':
                self.outcome = ('success' if self.lead >= cfg['lead_margin'] and
                                self.progress >= cfg['min_progress'] else 'defense_incomplete')
            elif cfg['skill'] == 'pressure':
                # This is an explicit pace target relative to the sampled MPC
                # cap, not a claim of causal delay versus a counterfactual race.
                self.outcome = ('success' if
                    self.progress >= cfg['min_progress'] and moving and
                    0 < self.lead <= cfg['pressure_distance'] and
                    self.pressure_time / time >= cfg['min_pressure_fraction'] and
                    self.opponent_progress / (self.opponent_speed * time) <= cfg['max_opponent_speed_fraction']
                    else 'pressure_incomplete')
            else:
                self.outcome = 'timeout'
        elif timed_out:
            self.outcome = 'timeout'
        self.done = self.outcome != 'active'
        return self.facts(dt=dt, progress_delta=delta, relative_delta=relative,
                          lead_reward_s=dt if ahead else 0., idle_s=dt if not moving else 0., event=self.done)

    def facts(self, *, dt=0., progress_delta=0., relative_delta=0., lead_reward_s=0., idle_s=0., event=False):
        return dict(skill=self.config['skill'], outcome=self.outcome, done=self.done,
                    success=self.outcome == 'success', ego_failed=self.outcome == 'ego_failure',
                    event=event, elapsed_s=self.time, dt=dt, ego_lead=self.lead,
                    ego_progress=self.progress, progress_delta=progress_delta,
                    relative_progress_delta=relative_delta, lead_reward_s=lead_reward_s,
                    opponent_progress=self.opponent_progress,
                    opponent_progress_delta=progress_delta - relative_delta,
                    opponent_pace_ratio=(self.opponent_progress / (self.opponent_speed * self.time)
                                         if self.opponent_speed and self.time else None),
                    pressure_fraction=self.pressure_time / self.time if self.time else 0.,
                    lateral_error=self.lateral_error, heading_error=self.heading_error, idle_s=idle_s,
                    lead_retention=self.ahead_time / self.time if self.time else 0.,
                    recovery_time_s=self.time if self.outcome == 'success' and self.config['skill'] == 'recovery' else None,
                    pass_time_s=self.time if self.outcome == 'success' and self.config['skill'] == 'pass' else None)


def sample_skill_spawn(*, geometry, walls, rng, stage, task, agent_ids, length, width):
    """Sample metric arc positions inside a conservative footprint-clear tube."""
    if geometry is None or not geometry.closed or not walls:
        raise ValueError("Skill spawning requires a closed centerline and walls")
    recovery = task['skill'] == 'recovery'
    if not recovery and stage['gap'][1] >= geometry.total_length / 2:
        raise ValueError("Skill spawn gaps must be below half the track length")
    radius = math.hypot(length, width) / 2 + .05
    wall_lines = [np.asarray(points)[:, :2] for points in walls.values() if len(points) > 1]
    if not wall_lines:
        raise ValueError("Skill spawning requires wall segments")

    def pose(s, d):
        s %= geometry.total_length
        index = min(np.searchsorted(geometry.arc_lengths, s, side='right') - 1,
                    len(geometry.segment_lengths) - 1)
        vec = geometry.segment_vectors[index]
        norm = geometry.segment_lengths[index]
        center = geometry.segment_starts[index] + vec * ((s - geometry.arc_lengths[index]) / norm)
        clearance = min(float(_distance_to_wall(center[None], w)[0]) for w in wall_lines)
        if clearance < abs(d) + radius:
            return None
        xy = center + np.array([-vec[1], vec[0]]) / norm * d
        return np.array([*xy, math.atan2(vec[1], vec[0])])

    for _ in range(256):
        origin = rng.uniform(0., geometry.total_length)
        if recovery:
            offset = float(rng.uniform(*stage['lateral_offset']))
            ego = pose(origin, offset)
            if ego is None:
                continue
            heading = float(rng.uniform(*stage['heading_error']))
            ego[2] += heading
            velocities = {task['ego_id']: float(rng.uniform(*stage['initial_speed']))}
            return np.array([ego]), velocities, dict(stage=stage['name'], ego_lead=0.,
                lateral_offsets=[offset], heading_error=heading, initial_s=float(origin), velocities=velocities)
        gap = rng.uniform(*stage['gap'])
        lead = -gap if task['skill'] == 'pass' else gap
        offsets = rng.uniform(*stage['lateral_offset'], size=2)
        target = pose(origin, offsets[1])
        ego = pose(origin + lead, offsets[0])
        if ego is None or target is None or np.linalg.norm(ego[:2] - target[:2]) < 2 * radius:
            continue
        cap = float(rng.uniform(*stage['opponent_speed']))
        velocities = {task['ego_id']: float(rng.uniform(*stage['initial_speed'])),
                      task['target_id']: min(cap, float(rng.uniform(*stage['initial_speed'])))}
        poses = {task['ego_id']: ego, task['target_id']: target}
        return np.array([poses[aid] for aid in agent_ids]), velocities, dict(
            stage=stage['name'], ego_lead=lead, lateral_offsets=offsets.tolist(),
            opponent_speed=cap, initial_s=float(origin), velocities=velocities)
    raise RuntimeError("No safe skill spawn found after 256 attempts; check gap, offsets and map clearance")


def reset_skill_opponent(env, controllers):
    """Rebuild MPC physical limits at resets, after the episode speed draw."""
    spawn = getattr(env, 'skill_spawn', None)
    if spawn is not None and env._skill_tracker.config.get('target_id') is not None:
        target = env._skill_tracker.config['target_id']
        wrapped = controllers[target]
        controller = getattr(wrapped, 'controller', wrapped)
        controller.max_speed = spawn['opponent_speed']
        controller.set_env(env)
