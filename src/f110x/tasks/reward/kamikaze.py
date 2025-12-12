"""Kamikaze reward strategy that treats target collisions as success events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from .base import RewardRuntimeContext, RewardStep, RewardStrategy
from .components import RewardAccumulator
from .registry import RewardTaskConfig, RewardTaskRegistry, RewardTaskSpec, register_reward_task


KAMIKAZE_PARAM_KEYS = (
    "target_agent",
    "success_reward",
    "success_once",
    "step_penalty",
    "self_collision_penalty",
    "distance_reward_near",
    "distance_reward_near_distance",
    "distance_penalty_far",
    "distance_reward_far_distance",
    "radius_bonus_reward",
    "radius_bonus_distance",
    "radius_bonus_once",
    "distance_delta_reward",
    "idle_penalty",
    "idle_penalty_steps",
    "idle_speed_threshold",
)


@dataclass(slots=True)
class _AwardTracker:
    """Tracks which attacker/target pairs already received a success bonus."""

    awarded: set[Tuple[str, str]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.awarded is None:
            self.awarded = set()

    def reset(self) -> None:
        self.awarded.clear()

    def should_award(self, key: Tuple[str, str], *, success_once: bool) -> bool:
        if not success_once:
            return True
        if key in self.awarded:
            return False
        self.awarded.add(key)
        return True


class KamikazeRewardStrategy(RewardStrategy):
    """Reward attacker for forcing a specific target to crash, even if it self-destructs."""

    name = "kamikaze"

    def __init__(
        self,
        *,
        target_agent: Optional[str] = None,
        success_reward: float = 100.0,
        success_once: bool = True,
        step_penalty: float = 0.0,
        self_collision_penalty: float = 0.0,
        distance_reward_near: float = 0.0,
        distance_reward_near_distance: float = 0.0,
        distance_penalty_far: float = 0.0,
        distance_reward_far_distance: float = 0.0,
        radius_bonus_reward: float = 0.0,
        radius_bonus_distance: float = 0.0,
        radius_bonus_once: bool = False,
        distance_delta_reward: float = 0.0,
        idle_penalty: float = 0.0,
        idle_penalty_steps: int = 0,
        idle_speed_threshold: float = 0.0,
    ) -> None:
        self.target_agent = target_agent
        self.success_reward = float(success_reward)
        self.success_once = bool(success_once)
        self.step_penalty = float(step_penalty)
        self.self_collision_penalty = float(self_collision_penalty)
        self._tracker = _AwardTracker()
        self.distance_reward_near = float(distance_reward_near)
        self.distance_reward_near_distance = max(float(distance_reward_near_distance), 0.0)
        self.distance_penalty_far = float(distance_penalty_far)
        self.distance_reward_far_distance = max(float(distance_reward_far_distance), 0.0)
        self.radius_bonus_reward = float(radius_bonus_reward)
        self.radius_bonus_distance = max(float(radius_bonus_distance), 0.0)
        self.radius_bonus_once = bool(radius_bonus_once)
        self._radius_awarded: set[Tuple[str, str]] = set()
        self.distance_delta_reward = float(distance_delta_reward)
        self._distance_prev: Dict[Tuple[str, str], float] = {}
        self.idle_penalty = float(idle_penalty)
        self.idle_penalty_steps = max(int(idle_penalty_steps), 0)
        self.idle_speed_threshold = max(float(idle_speed_threshold), 0.0)
        self._idle_counters: Dict[str, int] = {}

    def reset(self, episode_index: int) -> None:
        self._tracker.reset()
        self._radius_awarded.clear()
        self._distance_prev.clear()
        self._idle_counters.clear()
        super().reset(episode_index)

    def _resolve_target_obs(
        self,
        step: RewardStep,
    ) -> Tuple[Optional[str], Optional[Dict[str, object]]]:
        all_obs = step.all_obs or {}
        target_id = self.target_agent
        if target_id is None:
            for candidate, obs in all_obs.items():
                if candidate != step.agent_id and obs is not None:
                    target_id = candidate
                    break
        if target_id is None:
            return None, None
        return target_id, all_obs.get(target_id)

    @staticmethod
    def _extract_pose(obs: Optional[Dict[str, Any]]) -> Optional[np.ndarray]:
        if not obs:
            return None
        pose_raw = obs.get("pose")
        if pose_raw is None:
            return None
        pose_arr = np.asarray(pose_raw, dtype=np.float32).flatten()
        if pose_arr.size < 2:
            return None
        return pose_arr

    @staticmethod
    def _extract_speed(obs: Optional[Dict[str, Any]]) -> Optional[float]:
        if not obs:
            return None
        speed = obs.get("speed")
        if isinstance(speed, (int, float)):
            return float(speed)
        velocity = obs.get("velocity")
        if isinstance(velocity, Mapping):
            candidate = velocity.get("speed")
            if isinstance(candidate, (int, float)):
                return float(candidate)
            vec = []
            for key in ("x", "y"):
                value = velocity.get(key)
                if isinstance(value, (int, float)):
                    vec.append(float(value))
            if vec:
                return float(np.linalg.norm(np.asarray(vec, dtype=np.float32)))
        if isinstance(velocity, (list, tuple, np.ndarray)):
            arr = np.asarray(velocity, dtype=np.float32).flatten()
            if arr.size >= 1:
                return float(np.linalg.norm(arr[: min(arr.size, 2)]))
        return None

    def compute(self, step: RewardStep):
        acc = RewardAccumulator()
        ego_obs = step.obs
        ego_collided = bool(ego_obs.get("collision", False))

        if self.step_penalty:
            acc.add("step_penalty", float(self.step_penalty))

        target_id, target_obs = self._resolve_target_obs(step)
        target_collided = bool(target_obs.get("collision", False)) if target_obs else False

        pose_ego = self._extract_pose(ego_obs)
        pose_target = self._extract_pose(target_obs)
        distance = None
        if pose_ego is not None and pose_target is not None:
            delta = pose_target[:2] - pose_ego[:2]
            distance = float(np.linalg.norm(delta))
        speed = self._extract_speed(ego_obs)

        timestep = float(step.timestep or 0.0)

        if (
            distance is not None
            and self.distance_reward_near
            and self.distance_reward_near_distance > 0.0
        ):
            near = self.distance_reward_near_distance
            far = self.distance_reward_far_distance
            bonus_value = 0.0
            if distance <= near:
                bonus_value = self.distance_reward_near
            elif far > near > 0.0 and distance < far:
                span = far - near
                weight = (far - distance) / span
                bonus_value = self.distance_reward_near * weight
            if bonus_value and timestep > 0.0:
                acc.add("distance_reward", bonus_value * timestep)

        if (
            distance is not None
            and self.distance_penalty_far
            and self.distance_reward_far_distance > 0.0
            and distance >= self.distance_reward_far_distance
            and timestep > 0.0
        ):
            acc.add("distance_penalty", -abs(self.distance_penalty_far) * timestep)

        if (
            distance is not None
            and target_id is not None
            and self.radius_bonus_reward
            and self.radius_bonus_distance > 0.0
            and distance <= self.radius_bonus_distance
        ):
            radius_key = (step.agent_id, target_id)
            if not self.radius_bonus_once or radius_key not in self._radius_awarded:
                acc.add("radius_bonus", self.radius_bonus_reward)
                if self.radius_bonus_once:
                    self._radius_awarded.add(radius_key)

        if (
            distance is not None
            and target_id is not None
            and self.distance_delta_reward
        ):
            delta_key = (step.agent_id, target_id)
            prev_distance = self._distance_prev.get(delta_key)
            if prev_distance is not None and distance < prev_distance:
                delta = prev_distance - distance
                acc.add("distance_delta_reward", self.distance_delta_reward * delta)
            self._distance_prev[delta_key] = distance

        if (
            self.idle_penalty
            and self.idle_penalty_steps > 0
            and self.idle_speed_threshold > 0.0
            and speed is not None
        ):
            counter = self._idle_counters.get(step.agent_id, 0)
            if speed < self.idle_speed_threshold:
                counter += 1
            else:
                counter = 0
            if counter >= self.idle_penalty_steps:
                acc.add("idle_penalty", float(self.idle_penalty))
                counter = 0
            self._idle_counters[step.agent_id] = counter

        if ego_collided and not target_collided and self.self_collision_penalty:
            acc.add("self_collision_penalty", float(self.self_collision_penalty))

        if target_collided and target_id is not None:
            key = (step.agent_id, target_id)
            if self._tracker.should_award(key, success_once=self.success_once):
                acc.add("kamikaze_success", float(self.success_reward))
                step.events.setdefault("kamikaze_success", True)

        return acc.total, acc.components


def _create_kamikaze(
    context: RewardRuntimeContext,
    config: RewardTaskConfig,
    registry: RewardTaskRegistry,
) -> RewardStrategy:
    params = dict(config.get("params") or {})
    return KamikazeRewardStrategy(**params)


register_reward_task(
    RewardTaskSpec(
        name="kamikaze",
        factory=_create_kamikaze,
        aliases=("suicide_attack",),
        param_keys=KAMIKAZE_PARAM_KEYS,
        description="Rewards the attacker for causing the specified target to crash.",
    )
)


__all__ = ["KAMIKAZE_PARAM_KEYS", "KamikazeRewardStrategy"]
