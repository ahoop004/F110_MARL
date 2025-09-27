from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .task import Task


class HerdingAttackTask(Task):
    """Reward function for an attacker that herds a target into walls."""

    def __init__(
        self,
        target_id: str,
        laps: int = 1,
        time_limit: float = 120.0,
        terminate_on_collision: bool = True,
        wall_pressure_reward: float = 1.0,
        collision_bonus: float = 100.0,
        distance_weight: float = 1.0,
        lateral_weight: float = 0.5,
    ) -> None:
        if not target_id:
            raise ValueError("HerdingAttackTask requires a non-empty target_id")
        self._target_id = str(target_id)
        self._time_limit = float(time_limit)
        self._laps = int(laps)
        self._terminate_on_collision = bool(terminate_on_collision)
        self._wall_pressure_reward = float(wall_pressure_reward)
        self._collision_bonus = float(collision_bonus)
        self._distance_weight = float(distance_weight)
        self._lateral_weight = float(lateral_weight)

    def reward(self, agent_id, state: Dict[str, Any], action) -> float:
        agent_state = state.get(agent_id, {}) if isinstance(state, dict) else {}
        target_state = state.get(self._target_id, {}) if isinstance(state, dict) else {}

        reward = 0.0
        if self._is_target_collided(target_state):
            reward += self._collision_bonus

        relative = self._relative_pose(agent_state, target_state)
        if relative is not None:
            dist, lateral = relative
            reward += self._wall_pressure_reward * self._wall_pressure(target_state)
            reward += self._distance_weight * (1.0 / (1.0 + dist))
            reward += self._lateral_weight * max(0.0, lateral)

        return reward

    def done(self, agent_id, state) -> bool:
        agent_state = state.get(agent_id, {}) if isinstance(state, dict) else {}
        target_state = state.get(self._target_id, {}) if isinstance(state, dict) else {}

        if self._terminate_on_collision and self._is_self_collided(agent_state):
            return True
        if self._is_target_collided(target_state):
            return True

        laps_done = float(agent_state.get("lap", 0.0)) >= self._laps if isinstance(agent_state, dict) else False
        timeout = float(agent_state.get("time", 0.0)) >= self._time_limit if isinstance(agent_state, dict) else False
        return laps_done or timeout

    def reset(self):
        pass

    def target_id(self) -> str:
        return self._target_id

    def _is_self_collided(self, agent_state: Dict[str, Any]) -> bool:
        if not isinstance(agent_state, dict):
            return False
        if agent_state.get("collision") or agent_state.get("wall_collision"):
            return True
        opp = agent_state.get("opponent_collisions")
        return bool(opp)

    def _is_target_collided(self, target_state: Dict[str, Any]) -> bool:
        if not isinstance(target_state, dict):
            return False
        if target_state.get("collision") or target_state.get("wall_collision"):
            return True
        opp = target_state.get("opponent_collisions")
        return bool(opp)

    def _relative_pose(
        self,
        agent_state: Dict[str, Any],
        target_state: Dict[str, Any],
    ) -> Optional[np.ndarray]:
        if not isinstance(agent_state, dict) or not isinstance(target_state, dict):
            return None
        agent_pose = agent_state.get("pose")
        target_pose = target_state.get("pose")
        if agent_pose is None or target_pose is None:
            return None
        agent_pose = np.asarray(agent_pose, dtype=np.float32)
        target_pose = np.asarray(target_pose, dtype=np.float32)
        if agent_pose.size < 3 or target_pose.size < 3:
            return None
        dx = target_pose[0] - agent_pose[0]
        dy = target_pose[1] - agent_pose[1]
        dist = float(np.hypot(dx, dy))
        heading = agent_pose[2]
        forward = np.array([np.cos(heading), np.sin(heading)], dtype=np.float32)
        to_target = np.array([dx, dy], dtype=np.float32)
        lateral = float(np.cross(forward, to_target))
        return np.array([dist, lateral], dtype=np.float32)

    def _wall_pressure(self, target_state: Dict[str, Any]) -> float:
        if not isinstance(target_state, dict):
            return 0.0
        obs = target_state.get("observations") or {}
        lidar = obs.get("lidar") or target_state.get("lidar")
        if lidar is None:
            return 0.0
        values = np.asarray(lidar, dtype=np.float32)
        if values.size == 0:
            return 0.0
        near_wall = np.count_nonzero(values < 1.0)
        return float(near_wall) / float(values.size)
