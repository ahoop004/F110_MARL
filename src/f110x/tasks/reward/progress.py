"""Progress-based reward task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

from f110x.utils.centerline import (
    centerline_arc_length,
    progress_from_spacing,
    project_to_centerline,
)

from .base import RewardRuntimeContext, RewardStep, RewardStrategy
from .registry import RewardTaskConfig, RewardTaskRegistry, RewardTaskSpec, register_reward_task


PROGRESS_PARAM_KEYS = (
    "progress_weight",
    "speed_weight",
    "lateral_penalty",
    "heading_penalty",
    "collision_penalty",
    "truncation_penalty",
    "reverse_penalty",
    "idle_penalty",
    "idle_penalty_steps",
    "waypoint_bonus",
    "waypoint_progress_step",
    "waypoint_spacing",
    "lap_completion_bonus",
    "milestone_progress",
    "milestone_spacing",
    "milestone_bonus",
)

PROGRESS_PARAM_DEFAULTS: Dict[str, Any] = {
    "progress_weight": 1.0,
    "speed_weight": 0.0,
    "lateral_penalty": 0.0,
    "heading_penalty": 0.0,
    "collision_penalty": 0.0,
    "truncation_penalty": 0.0,
    "reverse_penalty": 0.0,
    "idle_penalty": 0.0,
    "idle_penalty_steps": 5,
    "waypoint_bonus": 0.0,
    "waypoint_progress_step": 0.0,
    "waypoint_spacing": 0.0,
    "lap_completion_bonus": 0.0,
    "milestone_progress": (),
    "milestone_spacing": 0.0,
    "milestone_bonus": 0.0,
}


@dataclass
class ProgressAgentState:
    last_index: Optional[int] = None
    last_progress: float = 0.0
    has_progress: bool = False
    lap_progress: float = 0.0
    laps_rewarded: int = 0
    milestone_lap: int = 0
    milestone_idx: int = 0
    cumulative_progress: float = 0.0
    next_waypoint_progress: float = 0.0
    collision_applied: bool = False
    truncation_applied: bool = False
    idle_counter: int = 0
    last_speed: float = 0.0


class ProgressRewardStrategy(RewardStrategy):
    name = "progress"

    def __init__(
        self,
        *,
        centerline: Optional[np.ndarray],
        progress_weight: float = 1.0,
        speed_weight: float = 0.0,
        lateral_penalty: float = 0.0,
        heading_penalty: float = 0.0,
        collision_penalty: float = 0.0,
        truncation_penalty: float = 0.0,
        reverse_penalty: float = 0.0,
        idle_penalty: float = 0.0,
        idle_penalty_steps: int = 5,
        waypoint_bonus: float = 0.0,
        waypoint_progress_step: float = 0.0,
        waypoint_spacing: float = 0.0,
        lap_completion_bonus: float = 0.0,
        milestone_progress: Optional[Iterable[float]] = None,
        milestone_spacing: float = 0.0,
        milestone_bonus: float = 0.0,
    ) -> None:
        self.centerline = None if centerline is None else np.asarray(centerline, dtype=np.float32)
        self.progress_weight = float(progress_weight)
        self.speed_weight = float(speed_weight)
        self.lateral_penalty = float(lateral_penalty)
        self.heading_penalty = float(heading_penalty)
        self.collision_penalty = float(collision_penalty)
        self.truncation_penalty = float(truncation_penalty)
        self.reverse_penalty = float(reverse_penalty)
        self.idle_penalty = float(idle_penalty)
        self.idle_penalty_steps = max(int(idle_penalty_steps), 0)
        self.waypoint_bonus = float(waypoint_bonus)
        self.waypoint_spacing = float(waypoint_spacing)
        self.lap_completion_bonus = float(lap_completion_bonus)
        self.milestone_spacing = float(milestone_spacing)
        self.milestone_bonus = float(milestone_bonus)
        waypoint_progress_step = float(waypoint_progress_step)
        if (
            waypoint_progress_step <= 0.0
            and self.centerline is not None
            and self.centerline.shape[0] > 1
        ):
            if self.waypoint_spacing > 0.0:
                total_length = centerline_arc_length(self.centerline)
                if total_length > 0.0:
                    derived_step = self.waypoint_spacing / total_length
                    if derived_step < 1.0:
                        waypoint_progress_step = derived_step
            if waypoint_progress_step <= 0.0:
                waypoint_progress_step = 1.0 / float(self.centerline.shape[0] - 1)
        self._waypoint_step = waypoint_progress_step if waypoint_progress_step > 0.0 else 0.0
        if milestone_progress is None:
            milestone_iter: Iterable[float] = ()
        elif isinstance(milestone_progress, (float, int)):
            milestone_iter = (float(milestone_progress),)
        else:
            milestone_iter = milestone_progress
        processed = []
        for value in milestone_iter:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if 0.0 < numeric < 1.0:
                processed.append(numeric)
        if self.milestone_spacing > 0.0 and self.centerline is not None:
            spacing_targets = progress_from_spacing(self.centerline, self.milestone_spacing)
            if spacing_targets:
                processed.extend(spacing_targets)
        milestone_targets = tuple(sorted(set(processed)))
        self._milestone_targets = milestone_targets
        self._agent_state: Dict[str, ProgressAgentState] = {}

    def reset(self, episode_index: int) -> None:
        self._agent_state.clear()

    def compute(self, step: RewardStep) -> Tuple[float, Dict[str, float]]:
        if self.centerline is None or self.centerline.size == 0:
            return 0.0, {}

        pose = step.obs.get("pose")
        if pose is None or len(pose) < 3:
            return 0.0, {}

        position = np.asarray(pose[:2], dtype=np.float32)
        heading = float(pose[2])

        state = self._agent_state.setdefault(step.agent_id, ProgressAgentState())
        last_idx = state.last_index
        try:
            projection = project_to_centerline(
                self.centerline,
                position,
                heading,
                last_index=last_idx,
            )
        except ValueError:
            return 0.0, {}

        state.last_index = projection.index

        prev_progress = state.last_progress if state.has_progress else None
        progress = float(projection.progress)
        if prev_progress is None:
            delta = 0.0
        else:
            delta = progress - prev_progress
            if delta < -0.5:
                delta += 1.0
        state.last_progress = progress
        state.has_progress = True

        reward = 0.0
        components: Dict[str, float] = {}

        if delta:
            progress_term = self.progress_weight * delta
            reward += progress_term
            components["progress"] = progress_term

        increment = delta if delta > 0.0 else 0.0
        state.lap_progress += increment
        lap_progress = state.lap_progress

        if self.lap_completion_bonus:
            lap_reward = 0.0
            next_threshold = float(state.laps_rewarded + 1)
            while lap_progress >= next_threshold:
                lap_reward += self.lap_completion_bonus
                state.laps_rewarded += 1
                next_threshold = float(state.laps_rewarded + 1)
            if lap_reward:
                reward += lap_reward
                components["lap_completion_bonus"] = (
                    components.get("lap_completion_bonus", 0.0) + lap_reward
                )

        if self.milestone_bonus and self._milestone_targets:
            total_laps = int(np.floor(lap_progress))
            lap_fraction = lap_progress - float(total_laps)
            if total_laps > state.milestone_lap:
                state.milestone_lap = total_laps
                state.milestone_idx = 0
            milestone_reward = 0.0
            targets = self._milestone_targets
            while state.milestone_idx < len(targets) and lap_fraction >= targets[state.milestone_idx]:
                milestone_reward += self.milestone_bonus
                state.milestone_idx += 1
            if milestone_reward:
                reward += milestone_reward
                components["milestone_bonus"] = (
                    components.get("milestone_bonus", 0.0) + milestone_reward
                )

        if self.waypoint_bonus and self._waypoint_step > 0.0:
            state.cumulative_progress += increment
            cumulative = state.cumulative_progress

            next_threshold = state.next_waypoint_progress or self._waypoint_step
            bonuses = 0
            while cumulative >= next_threshold:
                bonuses += 1
                next_threshold += self._waypoint_step
            if bonuses:
                waypoint_reward = self.waypoint_bonus * bonuses
                reward += waypoint_reward
                components["waypoint_bonus"] = components.get("waypoint_bonus", 0.0) + waypoint_reward
            state.next_waypoint_progress = next_threshold

        velocity = np.asarray(step.obs.get("velocity", (0.0, 0.0)), dtype=np.float32)
        speed = float(np.linalg.norm(velocity))

        if self.speed_weight:
            speed_term = self.speed_weight * speed * step.timestep
            if speed_term:
                reward += speed_term
                components["speed"] = speed_term

        if self.lateral_penalty:
            penalty = -abs(projection.lateral_error) * self.lateral_penalty
            if penalty:
                reward += penalty
                components["lateral_penalty"] = penalty

        if self.heading_penalty:
            penalty = -abs(projection.heading_error) * self.heading_penalty
            if penalty:
                reward += penalty
                components["heading_penalty"] = penalty

        if self.reverse_penalty and delta < 0.0:
            reverse_term = -self.reverse_penalty * abs(delta)
            if reverse_term:
                reward += reverse_term
                components["reverse_penalty"] = (
                    components.get("reverse_penalty", 0.0) + reverse_term
                )

        if self.idle_penalty:
            speed = float(speed)
            if speed < 0.1:
                state.idle_counter += 1
            else:
                state.idle_counter = 0
            state.last_speed = speed
            threshold = self.idle_penalty_steps if self.idle_penalty_steps > 0 else 1
            if state.idle_counter >= threshold:
                idle_term = -self.idle_penalty
                reward += idle_term
                components["idle_penalty"] = components.get("idle_penalty", 0.0) + idle_term

        if self.collision_penalty:
            collision_flag = bool(step.obs.get("collision", False))
            if not collision_flag and step.info and "collision" in step.info:
                collision_flag = bool(step.info.get("collision", False))
            if collision_flag and not state.collision_applied:
                reward += self.collision_penalty
                components["collision_penalty"] = (
                    components.get("collision_penalty", 0.0) + self.collision_penalty
                )
                state.collision_applied = True

        if (
            self.truncation_penalty
            and step.info
            and bool(step.info.get("truncated", False))
            and not state.truncation_applied
        ):
            reward += self.truncation_penalty
            components["truncation_penalty"] = (
                components.get("truncation_penalty", 0.0) + self.truncation_penalty
            )
            state.truncation_applied = True

        return reward, components


def _build_progress_strategy(
    context: RewardRuntimeContext,
    config: RewardTaskConfig,
    registry: RewardTaskRegistry,
) -> RewardStrategy:
    raw_params = dict(PROGRESS_PARAM_DEFAULTS)
    raw_params.update(config.get("params", {}))
    raw_params.pop("centerline", None)  # provided by the runtime context

    params = {key: raw_params.get(key, PROGRESS_PARAM_DEFAULTS[key]) for key in PROGRESS_PARAM_KEYS}
    centerline = getattr(context.map_data, "centerline", None)
    return ProgressRewardStrategy(centerline=centerline, **params)


register_reward_task(
    RewardTaskSpec(
        name="progress",
        factory=_build_progress_strategy,
        legacy_sections=("progress",),
        param_keys=PROGRESS_PARAM_KEYS,
    )
)


__all__ = ["ProgressRewardStrategy", "PROGRESS_PARAM_KEYS", "PROGRESS_PARAM_DEFAULTS"]
