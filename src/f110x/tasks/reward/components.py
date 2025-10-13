"""Reusable reward component helpers shared across strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


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
