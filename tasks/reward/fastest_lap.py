"""Fastest-lap reward task."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from .base import RewardRuntimeContext, RewardStep, RewardStrategy
from .registry import RewardTaskConfig, RewardTaskRegistry, RewardTaskSpec, register_reward_task


FASTEST_LAP_PARAM_KEYS = (
    "step_penalty",
    "lap_bonus",
    "best_bonus",
    "collision_penalty",
    "truncation_penalty",
)


class FastestLapRewardStrategy(RewardStrategy):
    name = "fastest_lap"

    def __init__(
        self,
        *,
        step_penalty: float = 0.0,
        lap_bonus: float = 1.0,
        best_bonus: float = 0.5,
        collision_penalty: float = 0.0,
        truncation_penalty: float = 0.0,
    ) -> None:
        self.step_penalty = float(step_penalty)
        self.lap_bonus = float(lap_bonus)
        self.best_bonus = float(best_bonus)
        self.collision_penalty = float(collision_penalty)
        self.truncation_penalty = float(truncation_penalty)
        self._lap_count: Dict[str, int] = {}
        self._lap_start_time: Dict[str, float] = {}
        self._best_time: Dict[str, Optional[float]] = {}

    def reset(self, episode_index: int) -> None:
        self._lap_count.clear()
        self._lap_start_time.clear()
        self._best_time.clear()

    def compute(self, step: RewardStep) -> Tuple[float, Dict[str, float]]:
        components: Dict[str, float] = {}
        reward = 0.0

        if self.step_penalty:
            penalty = -self.step_penalty * step.timestep
            reward += penalty
            components["time_penalty"] = penalty

        lap_info = step.obs.get("lap")
        if lap_info is None or len(lap_info) == 0:
            return reward, components

        current_count = int(lap_info[0])
        last_count = self._lap_count.get(step.agent_id, current_count)

        if step.agent_id not in self._lap_start_time:
            self._lap_start_time[step.agent_id] = float(step.current_time)

        if current_count > last_count:
            lap_finish_time = float(step.current_time)
            lap_start = self._lap_start_time.get(step.agent_id, lap_finish_time - step.timestep)
            lap_duration = max(lap_finish_time - lap_start, step.timestep)
            self._lap_start_time[step.agent_id] = lap_finish_time
            self._lap_count[step.agent_id] = current_count

            lap_reward = self.lap_bonus
            reward += lap_reward
            components["lap_bonus"] = lap_reward

            best_time = self._best_time.get(step.agent_id)
            if best_time is None or lap_duration < best_time:
                improvement = 1.0
                if best_time is not None and best_time > 0.0:
                    improvement = max((best_time - lap_duration) / best_time, 0.0)
                best_reward = self.best_bonus * improvement
                reward += best_reward
                components["best_lap_bonus"] = best_reward
                self._best_time[step.agent_id] = lap_duration
            else:
                self._best_time[step.agent_id] = best_time
        else:
            self._lap_count[step.agent_id] = current_count

        collision_flag = False
        if isinstance(step.obs, dict):
            collision_flag = bool(step.obs.get("collision", False))
        if not collision_flag and step.info and "collision" in step.info:
            collision_flag = bool(step.info.get("collision", False))

        if self.collision_penalty and collision_flag:
            reward += self.collision_penalty
            components["collision_penalty"] = components.get("collision_penalty", 0.0) + self.collision_penalty

        if self.truncation_penalty and step.info and bool(step.info.get("truncated", False)):
            reward += self.truncation_penalty
            components["truncation_penalty"] = (
                components.get("truncation_penalty", 0.0) + self.truncation_penalty
            )

        return reward, components


def _build_fastest_lap_strategy(
    context: RewardRuntimeContext,
    config: RewardTaskConfig,
    registry: RewardTaskRegistry,
) -> RewardStrategy:
    params = dict(config.get("params", {}))
    return FastestLapRewardStrategy(**params)


register_reward_task(
    RewardTaskSpec(
        name="fastest_lap",
        factory=_build_fastest_lap_strategy,
        legacy_sections=("fastest_lap",),
        param_keys=FASTEST_LAP_PARAM_KEYS,
    )
)


__all__ = ["FastestLapRewardStrategy", "FASTEST_LAP_PARAM_KEYS"]
