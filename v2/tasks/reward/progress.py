"""Progress-based reward task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

from v2.utils.centerline import (
    centerline_arc_length,
    progress_from_spacing,
    project_to_centerline,
)

from .base import PerAgentStateMixin, RewardRuntimeContext, RewardStep, RewardStrategy
from .components import (
    RewardAccumulator,
    apply_collision_penalty,
    apply_heading_penalty,
    apply_idle_penalty,
    apply_lap_completion_bonus,
    apply_lateral_penalty,
    apply_milestone_bonus,
    apply_progress,
    apply_reverse_penalty,
    apply_speed_bonus,
    apply_truncation_penalty,
    apply_waypoint_bonus,
)
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


@dataclass(slots=True)
class ProgressAgentState:
    """Mutable per-agent cache used by :class:`ProgressRewardStrategy`."""

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

    def advance_progress(self, progress: float) -> float:
        """Return delta progress since last call (wrapping around lap)."""

        progress = float(progress)
        if not self.has_progress:
            self.has_progress = True
            self.last_progress = progress
            return 0.0

        delta = progress - self.last_progress
        if delta < -0.5:
            delta += 1.0
        self.last_progress = progress
        return delta

    def accumulate_forward_progress(self, delta: float) -> float:
        """Track lap & cumulative progress for positive deltas only."""

        if delta <= 0.0:
            return 0.0
        self.lap_progress += delta
        self.cumulative_progress += delta
        return delta

    def register_idle(self, speed: float, threshold: float) -> int:
        """Update idle counter based on instantaneous speed."""

        self.last_speed = speed
        if speed < threshold:
            self.idle_counter += 1
        else:
            self.idle_counter = 0
        return self.idle_counter


class ProgressRewardStrategy(PerAgentStateMixin, RewardStrategy):
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
        PerAgentStateMixin.__init__(self, ProgressAgentState)
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

    def compute(self, step: RewardStep) -> Tuple[float, Dict[str, float]]:
        if self.centerline is None or self.centerline.size == 0:
            return 0.0, {}

        pose = step.obs.get("pose")
        if pose is None or len(pose) < 3:
            return 0.0, {}

        position = np.asarray(pose[:2], dtype=np.float32)
        heading = float(pose[2])

        state = self.state_for(step.agent_id)
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
        delta = state.advance_progress(projection.progress)

        acc = RewardAccumulator()

        if delta:
            apply_progress(acc, delta, self.progress_weight)

        increment = state.accumulate_forward_progress(delta)
        state.laps_rewarded = apply_lap_completion_bonus(
            acc,
            lap_progress=state.lap_progress,
            laps_rewarded=state.laps_rewarded,
            bonus=self.lap_completion_bonus,
        )

        if self.milestone_bonus and self._milestone_targets:
            state.milestone_lap, state.milestone_idx = apply_milestone_bonus(
                acc,
                lap_progress=state.lap_progress,
                milestone_state=(state.milestone_lap, state.milestone_idx),
                targets=self._milestone_targets,
                bonus=self.milestone_bonus,
            )

        if self.waypoint_bonus and self._waypoint_step > 0.0:
            state.next_waypoint_progress = apply_waypoint_bonus(
                acc,
                cumulative_progress=state.cumulative_progress,
                next_threshold=state.next_waypoint_progress,
                step=self._waypoint_step,
                bonus=self.waypoint_bonus,
            )

        velocity = np.asarray(step.obs.get("velocity", (0.0, 0.0)), dtype=np.float32)
        speed = float(np.linalg.norm(velocity))

        if self.speed_weight:
            apply_speed_bonus(acc, speed, step.timestep, self.speed_weight)

        if self.lateral_penalty:
            apply_lateral_penalty(acc, projection.lateral_error, self.lateral_penalty)

        if self.heading_penalty:
            apply_heading_penalty(acc, projection.heading_error, self.heading_penalty)

        if self.reverse_penalty:
            apply_reverse_penalty(acc, delta, self.reverse_penalty)

        if self.idle_penalty:
            counter = state.register_idle(speed, threshold=0.1)
            apply_idle_penalty(
                acc,
                idle_counter=counter,
                threshold=self.idle_penalty_steps,
                penalty=self.idle_penalty,
            )

        if self.collision_penalty:
            collision_flag = bool(step.events.get("collision")) if step.events else False
            if not collision_flag:
                collision_flag = bool(step.obs.get("collision", False))
            if not collision_flag and step.info and "collision" in step.info:
                collision_flag = bool(step.info.get("collision", False))
            state.collision_applied = apply_collision_penalty(
                acc,
                collided=collision_flag,
                already_applied=state.collision_applied,
                penalty=self.collision_penalty,
            )

        if self.truncation_penalty:
            truncated_flag = bool(step.events.get("truncated")) if step.events else False
            if not truncated_flag and step.info:
                truncated_flag = bool(step.info.get("truncated", False))
            state.truncation_applied = apply_truncation_penalty(
                acc,
                truncated=truncated_flag,
                already_applied=state.truncation_applied,
                penalty=self.truncation_penalty,
            )

        return acc.total, dict(acc.components)


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
