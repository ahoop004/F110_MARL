import math
from typing import Any, Dict, Optional, Tuple

import numpy as np


class SecondaryViconPolicy:
    """Heuristic that mirrors the ROS secondary_vicon controller."""

    CENTRAL_STATE_KEYS = (
        "poses_x",
        "poses_y",
        "poses_theta",
        "linear_vels_x",
        "linear_vels_y",
        "ang_vels_z",
        "collisions",
    )

    CONFIG_DEFAULTS: Dict[str, Any] = {
        "max_speed": 0.8,
        "warning_border": 0.35,
        "hard_border": 0.5,
        "safe_distance": 1.0,
        "max_turn": 1.0,
        "turn_gain": 1.5,
        "border_speed_scale": 0.5,
        "lane_center": 0.0,
        "target_slot": 0,
        "target_agent": None,
    }

    def __init__(
        self,
        max_speed: float = 0.8,
        warning_border: float = 0.35,
        hard_border: float = 0.5,
        safe_distance: float = 1.0,
        max_turn: float = 1.0,
        turn_gain: float = 1.5,
        border_speed_scale: float = 0.5,
        lane_center: float = 0.0,
        target_slot: int = 0,
        target_agent: Optional[str] = None,
    ) -> None:
        self.max_speed = float(max_speed)
        self.warning_border = float(warning_border)
        self.hard_border = float(hard_border)
        self.safe_distance = float(safe_distance)
        self.max_turn = float(max_turn)
        self.turn_gain = float(turn_gain)
        self.border_speed_scale = float(border_speed_scale)
        self.lane_center = float(lane_center)

        self.target_slot = int(target_slot)
        self.target_agent = target_agent.strip() if isinstance(target_agent, str) else None

        self.agent_slot: Optional[int] = None

    # ------------------------------------------------------------------ #
    # Configuration helpers
    @classmethod
    def from_config(cls, params: Optional[Dict[str, Any]]) -> "SecondaryViconPolicy":
        if not params:
            return cls()

        kwargs: Dict[str, Any] = {}
        for key, default in cls.CONFIG_DEFAULTS.items():
            if key not in params:
                continue
            value = params[key]
            if isinstance(default, bool):
                kwargs[key] = bool(value)
            elif isinstance(default, int) and not isinstance(default, bool):
                try:
                    kwargs[key] = int(value)
                except (TypeError, ValueError):
                    kwargs[key] = default
            elif isinstance(default, float):
                try:
                    kwargs[key] = float(value)
                except (TypeError, ValueError):
                    kwargs[key] = default
            else:
                kwargs[key] = value

        return cls(**kwargs)

    # ------------------------------------------------------------------ #
    def get_action(self, action_space, obs: Dict[str, Any]):
        state_vec, n_agents = self._decode_state(obs.get("state"))
        if state_vec is None or n_agents <= 0:
            return np.zeros(2, dtype=np.float32)

        ego_pose = np.asarray(obs.get("pose", ()), dtype=np.float32).flatten()
        if ego_pose.size < 2:
            return np.zeros(2, dtype=np.float32)

        agent_slot = self._resolve_agent_slot(ego_pose, state_vec, n_agents)
        target_slot = self._resolve_target_slot(agent_slot, n_agents)

        ego_pose_tuple = self._extract_pose_from_state(state_vec, agent_slot, n_agents)
        target_pose_tuple = self._extract_pose_from_state(state_vec, target_slot, n_agents)

        if ego_pose_tuple is None or target_pose_tuple is None:
            return np.zeros(2, dtype=np.float32)

        steering, speed = self._compute_command(ego_pose_tuple, target_pose_tuple)
        action = np.array([steering, speed], dtype=np.float32)
        if action_space is not None:
            action = np.clip(action, action_space.low, action_space.high)
        return action

    # ------------------------------------------------------------------ #
    def _decode_state(self, state_raw: Any) -> Tuple[Optional[np.ndarray], int]:
        if state_raw is None:
            return None, 0

        state_vec = np.asarray(state_raw, dtype=np.float32).flatten()
        key_count = len(self.CENTRAL_STATE_KEYS)
        if state_vec.size == 0 or state_vec.size % key_count != 0:
            return None, 0
        n_agents = state_vec.size // key_count
        return state_vec, n_agents

    def _resolve_agent_slot(self, ego_pose: np.ndarray, state_vec: np.ndarray, n_agents: int) -> int:
        slot = self.agent_slot
        if slot is None or not (0 <= slot < n_agents):
            slot = self._infer_agent_slot_from_state(ego_pose, state_vec, n_agents)
        if slot is None or not (0 <= slot < n_agents):
            slot = max(0, min(n_agents - 1, self.target_slot if 0 <= self.target_slot < n_agents else 0))
        self.agent_slot = slot
        return slot

    def _resolve_target_slot(self, agent_slot: int, n_agents: int) -> int:
        slot = int(self.target_slot)
        if slot == agent_slot or not (0 <= slot < n_agents):
            alternatives = [idx for idx in range(n_agents) if idx != agent_slot]
            slot = alternatives[0] if alternatives else agent_slot
        return slot

    def _extract_pose_from_state(
        self,
        state_vec: np.ndarray,
        slot: int,
        n_agents: int,
    ) -> Optional[Tuple[float, float, float]]:
        if not (0 <= slot < n_agents):
            return None
        px = state_vec[0:n_agents]
        py = state_vec[n_agents:2 * n_agents]
        headings = state_vec[2 * n_agents:3 * n_agents]
        return float(px[slot]), float(py[slot]), float(headings[slot])

    def _infer_agent_slot_from_state(
        self,
        ego_pose: np.ndarray,
        state_vec: np.ndarray,
        n_agents: int,
    ) -> Optional[int]:
        if ego_pose.size < 2 or n_agents <= 0:
            return None
        px = state_vec[0:n_agents]
        py = state_vec[n_agents:2 * n_agents]
        diffs = np.column_stack((px - ego_pose[0], py - ego_pose[1]))
        distances = np.sum(diffs ** 2, axis=1)
        return int(np.argmin(distances))

    # ------------------------------------------------------------------ #
    def _compute_command(
        self,
        ego_pose: Tuple[float, float, float],
        target_pose: Tuple[float, float, float],
    ) -> Tuple[float, float]:
        _, y_target, _ = target_pose
        x_self, y_self, _ = ego_pose

        y_offset = y_self - self.lane_center

        if abs(y_offset) > self.hard_border:
            return 0.0, 0.0

        if abs(y_offset) > self.warning_border:
            speed = self.max_speed * self.border_speed_scale
            turn_direction = -1.0 if y_offset > 0.0 else 1.0
            steering = turn_direction * self.max_turn
            return steering, max(0.0, speed)

        speed = self.max_speed

        dx = float(target_pose[0] - x_self)
        dy = float(target_pose[1] - y_self)
        distance = math.hypot(dx, dy)

        if distance < self.safe_distance:
            if y_target > 0.0:
                angle = math.atan2(-self.warning_border - y_target, dx)
            else:
                angle = math.atan2(self.warning_border - y_target, dx)
            angle = float(max(-self.max_turn, min(self.max_turn, angle)))
            steering = angle * self.turn_gain
        else:
            steering = 0.0

        return steering, max(0.0, speed)
