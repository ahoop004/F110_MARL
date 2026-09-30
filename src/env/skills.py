"""Passing/defending facts and reproducible, collision-free tactical starts."""
from collections.abc import Mapping
import math

import numpy as np

from utils.track_preview import _distance_to_wall


def validate_skill_task(config, agent_ids):
    if config is None:
        return None
    defaults = dict(lead_margin=1., confirmation_s=1., moving_speed=.5,
                    duration_s=30., min_progress=0.)
    required = {"skill", "ego_id", "target_id"}
    if (not isinstance(config, Mapping) or not required <= config.keys()
            or set(config) - required - defaults.keys()):
        raise ValueError("skill_task requires skill, ego_id, target_id and optional task thresholds")
    cfg = {**defaults, **config}
    if cfg['skill'] not in {'pass', 'defend'}:
        raise ValueError("skill_task.skill must be pass or defend")
    if (cfg['ego_id'] == cfg['target_id'] or
            any(cfg[k] not in agent_ids for k in ('ego_id', 'target_id'))):
        raise ValueError("skill_task requires distinct known participants")
    for key in defaults:
        value = cfg[key]
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(value) or value < 0 or
                (key in {'lead_margin', 'confirmation_s', 'duration_s'} and value == 0)):
            raise ValueError(f"skill_task.{key} must be finite and {'positive' if key in {'lead_margin', 'confirmation_s', 'duration_s'} else 'nonnegative'}")
    return cfg


class SkillTracker:
    """Ordering is initial separation plus earned distance, never wrapped gap."""

    def __init__(self, config):
        self.config = validate_skill_task(config, [config['ego_id'], config['target_id']])
        self.reset(0.)

    def reset(self, ego_lead):
        self.lead = float(ego_lead)
        self.progress = self.ahead_time = self.time = 0.
        self.confirm_since = None
        self.done = False
        self.outcome = 'active'

    def update(self, *, time, infos, collisions, track_length, timed_out=False):
        cfg = self.config
        ego, target = infos[cfg['ego_id']], infos[cfg['target_id']]
        dt = max(0., time - self.time)
        self.time = time
        if self.done:
            return self.facts(dt=0.)
        delta = float(ego['centerline']['progress_delta']) * track_length
        relative = delta - float(target['centerline']['progress_delta']) * track_length
        self.lead += relative
        self.progress += delta
        moving = ego['centerline']['vs'] > cfg['moving_speed']
        ego_failed = bool(collisions[cfg['ego_id']] or ego['track_limits']['exceeded'])
        target_failed = bool(collisions[cfg['target_id']] or target['track_limits']['exceeded'])
        ahead = self.lead > 0 and moving and not (ego_failed or target_failed)
        if ahead:
            self.ahead_time += dt
        confirming = ((self.lead >= cfg['lead_margin'] and moving) if cfg['skill'] == 'pass'
                      else self.lead <= -cfg['lead_margin'])
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
        elif confirmed:
            self.outcome = 'success' if cfg['skill'] == 'pass' else 'lead_lost'
        elif time >= cfg['duration_s'] - 1e-9:
            if cfg['skill'] == 'defend':
                self.outcome = ('success' if self.lead >= cfg['lead_margin'] and
                                self.progress >= cfg['min_progress'] else 'defense_incomplete')
            else:
                self.outcome = 'timeout'
        elif timed_out:
            self.outcome = 'timeout'
        self.done = self.outcome != 'active'
        return self.facts(dt=dt, progress_delta=delta, relative_delta=relative,
                          lead_reward_s=dt if ahead else 0., event=self.done)

    def facts(self, *, dt=0., progress_delta=0., relative_delta=0., lead_reward_s=0., event=False):
        return dict(skill=self.config['skill'], outcome=self.outcome, done=self.done,
                    success=self.outcome == 'success', ego_failed=self.outcome == 'ego_failure',
                    event=event, elapsed_s=self.time, dt=dt, ego_lead=self.lead,
                    ego_progress=self.progress, progress_delta=progress_delta,
                    relative_progress_delta=relative_delta, lead_reward_s=lead_reward_s,
                    lead_retention=self.ahead_time / self.time if self.time else 0.,
                    pass_time_s=self.time if self.outcome == 'success' and self.config['skill'] == 'pass' else None)


def sample_skill_spawn(*, geometry, walls, rng, stage, task, agent_ids, length, width):
    """Sample metric arc positions inside a conservative footprint-clear tube."""
    if geometry is None or not geometry.closed or not walls:
        raise ValueError("Skill spawning requires a closed centerline and walls")
    if stage['gap'][1] >= geometry.total_length / 2:
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
    if spawn is not None:
        target = env._skill_tracker.config['target_id']
        wrapped = controllers[target]
        controller = getattr(wrapped, 'controller', wrapped)
        controller.max_speed = spawn['opponent_speed']
        controller.set_env(env)
