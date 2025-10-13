"""Reusable reward component helpers shared across strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import math
import numpy as np


@dataclass
class RewardAccumulator:
    """Utility that tracks total shaped reward and component breakdown."""

    total: float = 0.0
    components: Dict[str, float] = field(default_factory=dict)

    def add(self, key: str, value: float) -> None:
        if value:
            self.total += value
            self.components[key] = self.components.get(key, 0.0) + float(value)

    def extend(self, mapping: Dict[str, float]) -> None:
        for name, value in mapping.items():
            self.add(name, float(value))


def apply_progress(acc: RewardAccumulator, delta: float, weight: float) -> float:
    contribution = float(weight) * float(delta)
    acc.add("progress", contribution)
    return contribution


def apply_speed_bonus(acc: RewardAccumulator, speed: float, timestep: float, weight: float) -> float:
    bonus = float(weight) * float(speed) * float(timestep)
    acc.add("speed", bonus)
    return bonus


def apply_lateral_penalty(acc: RewardAccumulator, lateral_error: float, penalty: float) -> float:
    value = -abs(float(lateral_error)) * float(penalty)
    acc.add("lateral_penalty", value)
    return value


def apply_heading_penalty(acc: RewardAccumulator, heading_error: float, penalty: float) -> float:
    value = -abs(float(heading_error)) * float(penalty)
    acc.add("heading_penalty", value)
    return value


def apply_reverse_penalty(acc: RewardAccumulator, delta: float, penalty: float) -> float:
    if delta < 0.0:
        value = -abs(float(delta)) * float(penalty)
        acc.add("reverse_penalty", value)
        return value
    return 0.0


def apply_idle_penalty(
    acc: RewardAccumulator,
    *,
    idle_counter: int,
    threshold: int,
    penalty: float,
) -> float:
    if threshold <= 0:
        threshold = 1
    if idle_counter >= threshold and penalty:
        value = -abs(float(penalty))
        acc.add("idle_penalty", value)
        return value
    return 0.0


def apply_collision_penalty(
    acc: RewardAccumulator,
    *,
    collided: bool,
    already_applied: bool,
    penalty: float,
) -> bool:
    if collided and not already_applied and penalty:
        acc.add("collision_penalty", float(penalty))
        return True
    return already_applied


def apply_truncation_penalty(
    acc: RewardAccumulator,
    *,
    truncated: bool,
    already_applied: bool,
    penalty: float,
) -> bool:
    if truncated and not already_applied and penalty:
        acc.add("truncation_penalty", float(penalty))
        return True
    return already_applied


def apply_lap_completion_bonus(
    acc: RewardAccumulator,
    *,
    lap_progress: float,
    laps_rewarded: int,
    bonus: float,
) -> int:
    if bonus <= 0.0:
        return laps_rewarded
    lap_reward = 0.0
    next_threshold = float(laps_rewarded + 1)
    updated_laps = laps_rewarded
    while lap_progress >= next_threshold:
        lap_reward += bonus
        updated_laps += 1
        next_threshold = float(updated_laps + 1)
    if lap_reward:
        acc.add("lap_completion_bonus", lap_reward)
    return updated_laps


def apply_milestone_bonus(
    acc: RewardAccumulator,
    *,
    lap_progress: float,
    milestone_state: tuple[int, int],
    targets: tuple[float, ...],
    bonus: float,
) -> tuple[int, int]:
    if bonus <= 0.0 or not targets:
        return milestone_state
    total_laps = int(lap_progress // 1.0)
    lap_fraction = lap_progress - float(total_laps)
    prev_lap, milestone_idx = milestone_state
    if total_laps > prev_lap:
        milestone_idx = 0
    milestone_reward = 0.0
    while milestone_idx < len(targets) and lap_fraction >= targets[milestone_idx]:
        milestone_reward += bonus
        milestone_idx += 1
    if milestone_reward:
        acc.add("milestone_bonus", milestone_reward)
    return total_laps, milestone_idx


def apply_waypoint_bonus(
    acc: RewardAccumulator,
    *,
    cumulative_progress: float,
    next_threshold: float,
    step: float,
    bonus: float,
) -> float:
    if bonus <= 0.0 or step <= 0.0:
        return next_threshold
    threshold = next_threshold or step
    bonuses = 0
    while cumulative_progress >= threshold:
        bonuses += 1
        threshold += step
    if bonuses:
        acc.add("waypoint_bonus", bonus * bonuses)
    return threshold


def apply_event_reward(
    acc: RewardAccumulator,
    *,
    event_flag: Optional[bool],
    reward: float,
    key: str,
) -> None:
    if event_flag:
        acc.add(key, reward)


_SECTOR_DEGREES = (
    ("front", -22.5, 22.5),
    ("front_right", 22.5, 67.5),
    ("right", 67.5, 112.5),
    ("back_right", 112.5, 157.5),
    ("back", 157.5, -157.5),  # wrap-around handled separately
    ("back_left", -157.5, -112.5),
    ("left", -112.5, -67.5),
    ("front_left", -67.5, -22.5),
)


def _wrap_deg(angle: float) -> float:
    wrapped = (angle + 180.0) % 360.0 - 180.0
    return wrapped


def _classify_sector(angle_deg: float) -> str:
    angle_deg = _wrap_deg(angle_deg)
    for name, start, end in _SECTOR_DEGREES:
        if name == "back":
            if angle_deg >= 157.5 or angle_deg < -157.5:
                return name
        elif start <= angle_deg < end:
            return name
    return "front"


def _radial_gain(distance: float, preferred: float, inner_tol: float, outer_tol: float, falloff: str) -> float:
    if preferred <= 0.0:
        return 1.0
    inner_tol = max(0.0, inner_tol)
    outer_tol = max(0.0, outer_tol)
    lower = max(0.0, preferred - inner_tol)
    upper = preferred + outer_tol
    if distance < lower:
        if inner_tol == 0.0:
            return 0.0
        ratio = (distance - (preferred - inner_tol)) / (inner_tol if inner_tol else preferred)
        ratio = max(0.0, min(1.0, ratio))
    elif distance > upper:
        if outer_tol == 0.0:
            return 0.0
        ratio = (upper - distance) / outer_tol
        ratio = max(0.0, min(1.0, ratio))
    else:
        ratio = 1.0

    if falloff == "gaussian":
        sigma = (inner_tol + outer_tol) / 2.0 or 1.0
        ratio = math.exp(-((distance - preferred) ** 2) / (2.0 * sigma ** 2))
    elif falloff == "binary":
        ratio = 1.0 if lower <= distance <= upper else 0.0

    return max(0.0, min(1.0, ratio))


def apply_relative_sector_reward(
    acc: RewardAccumulator,
    *,
    relative_vector: np.ndarray,
    ego_heading: float,
    weights: Dict[str, float],
    preferred_radius: float,
    inner_tolerance: float,
    outer_tolerance: float,
    falloff: str = "linear",
    scale: float = 1.0,
) -> float:
    if relative_vector.size < 2:
        return 0.0

    dx = float(relative_vector[0])
    dy = float(relative_vector[1])
    if dx == 0.0 and dy == 0.0:
        return 0.0

    angle = math.degrees(math.atan2(dy, dx) - ego_heading)
    sector = _classify_sector(angle)
    weight = float(weights.get(sector, 0.0))
    if weight == 0.0:
        return 0.0

    distance = float(np.linalg.norm(relative_vector))
    radial_gain = _radial_gain(distance, preferred_radius, inner_tolerance, outer_tolerance, falloff.lower())
    if radial_gain <= 0.0:
        return 0.0

    reward = weight * radial_gain * float(scale)
    if reward:
        acc.add("relative_position", reward)
    return reward
