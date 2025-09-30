"""Utilities for decomposing reward shaping logic."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from f110x.utils.geometry import compute_position2, heading_alignment, relative_bearing


@dataclass(frozen=True)
class P2Params:
    enabled: bool
    d_back: float
    lat_offset: float
    dist_thresh: float
    angle_align_thresh: float
    blind_angle: float
    hold_decay: float
    distance_scale: float
    angle_scale: float
    blind_bonus: float
    lateral_scale: float
    camp_speed_thresh: float
    camping_penalty: float
    reward_clip: float


@dataclass(frozen=True)
class ScalingParams:
    horizon: float | None
    clip: float | None


P2Contribution = Dict[str, float]
P2Metrics = Dict[str, float | Tuple[float, float] | bool]


def evaluate_position2(
    ego_pose: Tuple[float, float, float],
    target_pose: Tuple[float, float, float],
    speed: float,
    state_hold: float,
    params: P2Params,
    dt: float,
) -> Tuple[float, P2Contribution, P2Metrics, float]:
    """Compute Position-2 shaping contributions and updated hold time."""

    if not params.enabled:
        return 0.0, {}, {}, state_hold

    x, y, _ = ego_pose
    tx, ty, _ = target_pose

    xp2, yp2 = compute_position2(target_pose, params.d_back, params.lat_offset)
    dist = float(math.hypot(x - xp2, y - yp2))
    phi = float(relative_bearing(target_pose, (x, y)))
    angle_diff = float(heading_alignment(ego_pose, (tx, ty)))

    blind_ok = abs(phi) > params.blind_angle
    align_ok = params.angle_align_thresh <= 0.0 or angle_diff <= params.angle_align_thresh
    range_ok = dist <= params.dist_thresh

    if blind_ok and align_ok and range_ok:
        hold_time = state_hold + dt
        in_position = True
    else:
        hold_time = 0.0
        in_position = False

    contributions: P2Contribution = {}
    total = 0.0

    if params.distance_scale:
        distance_term = -params.distance_scale * dist
        contributions["p2_distance"] = distance_term
        total += distance_term

    if params.angle_scale:
        angle_term = -params.angle_scale * angle_diff
        contributions["p2_angle"] = angle_term
        total += angle_term

    if params.blind_bonus and blind_ok:
        decay = math.exp(-hold_time / params.hold_decay) if params.hold_decay > 0 else 1.0
        blind_term = params.blind_bonus * decay
        contributions["p2_blind_bonus"] = blind_term
        total += blind_term

    lateral_term = 0.0
    if params.lateral_scale and abs(params.lat_offset) > 1e-6:
        heading_vec = np.array([math.cos(target_pose[2]), math.sin(target_pose[2])], dtype=np.float32)
        vec_to_attacker = np.array([x - tx, y - ty], dtype=np.float32)
        heading_norm = float(np.linalg.norm(heading_vec))
        vec_norm = float(np.linalg.norm(vec_to_attacker))
        if heading_norm > 1e-6 and vec_norm > 1e-6:
            cross = heading_vec[0] * vec_to_attacker[1] - heading_vec[1] * vec_to_attacker[0]
            sin_angle = cross / (heading_norm * vec_norm)
            desired_sign = 1.0 if params.lat_offset >= 0.0 else -1.0
            lateral_term = params.lateral_scale * desired_sign * sin_angle
            contributions["p2_lateral"] = lateral_term
            total += lateral_term

    if (
        params.camping_penalty
        and hold_time > params.hold_decay
        and speed < params.camp_speed_thresh
    ):
        excess = hold_time - params.hold_decay
        penalty = -params.camping_penalty * min(1.0, excess / params.hold_decay)
        contributions["p2_camping_penalty"] = penalty
        total += penalty

    if params.reward_clip > 0.0:
        clipped = float(np.clip(total, -params.reward_clip, params.reward_clip))
        if clipped != total:
            contributions["p2_clip_delta"] = clipped - total
            total = clipped

    if contributions:
        contributions["p2_total"] = total

    metrics: P2Metrics = {
        "point": (xp2, yp2),
        "distance": dist,
        "phi": phi,
        "angle_diff": angle_diff,
        "hold_time": hold_time,
        "in_blind": bool(blind_ok),
        "in_position": bool(in_position),
    }

    return total, contributions, metrics, hold_time


def apply_reward_scaling(
    total: float,
    components: Dict[str, float],
    scaling: ScalingParams,
) -> Tuple[float, Dict[str, float]]:
    """Scale and clip rewards while updating component bookkeeping."""

    adjusted = dict(components)
    if scaling.horizon:
        scale = 1.0 / scaling.horizon
        total *= scale
        for key in list(adjusted.keys()):
            adjusted[key] *= scale
        adjusted["scale_factor"] = scale

    if scaling.clip:
        clipped = float(np.clip(total, -scaling.clip, scaling.clip))
        if clipped != total:
            adjusted["clip_delta"] = clipped - total
            total = clipped

    adjusted["total_return"] = total
    return total, adjusted
