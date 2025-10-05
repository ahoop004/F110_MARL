"""Progress-based reward task."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from f110x.utils.centerline import project_to_centerline

from .base import RewardRuntimeContext, RewardStep, RewardStrategy
from .registry import RewardTaskConfig, RewardTaskRegistry, RewardTaskSpec, register_reward_task


PROGRESS_PARAM_KEYS = (
    "progress_weight",
    "speed_weight",
    "lateral_penalty",
    "heading_penalty",
)

PROGRESS_PARAM_DEFAULTS: Dict[str, float] = {
    "progress_weight": 1.0,
    "speed_weight": 0.0,
    "lateral_penalty": 0.0,
    "heading_penalty": 0.0,
}


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
    ) -> None:
        self.centerline = None if centerline is None else np.asarray(centerline, dtype=np.float32)
        self.progress_weight = float(progress_weight)
        self.speed_weight = float(speed_weight)
        self.lateral_penalty = float(lateral_penalty)
        self.heading_penalty = float(heading_penalty)
        self._last_index: Dict[str, Optional[int]] = {}
        self._last_progress: Dict[str, float] = {}

    def reset(self, episode_index: int) -> None:
        self._last_index.clear()
        self._last_progress.clear()

    def compute(self, step: RewardStep) -> Tuple[float, Dict[str, float]]:
        if self.centerline is None or self.centerline.size == 0:
            return 0.0, {}

        pose = step.obs.get("pose")
        if pose is None or len(pose) < 3:
            return 0.0, {}

        position = np.asarray(pose[:2], dtype=np.float32)
        heading = float(pose[2])

        last_idx = self._last_index.get(step.agent_id)
        try:
            projection = project_to_centerline(
                self.centerline,
                position,
                heading,
                last_index=last_idx,
            )
        except ValueError:
            return 0.0, {}

        self._last_index[step.agent_id] = projection.index

        prev_progress = self._last_progress.get(step.agent_id)
        progress = projection.progress
        if prev_progress is None:
            delta = 0.0
        else:
            delta = progress - prev_progress
            if delta < -0.5:
                delta += 1.0
        self._last_progress[step.agent_id] = progress

        reward = 0.0
        components: Dict[str, float] = {}

        if delta:
            progress_term = self.progress_weight * delta
            reward += progress_term
            components["progress"] = progress_term

        if self.speed_weight:
            velocity = np.asarray(step.obs.get("velocity", (0.0, 0.0)), dtype=np.float32)
            speed = float(np.linalg.norm(velocity))
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
