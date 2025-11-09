from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class BlockingControllerConfig:
    target_forward: float = 1.6
    target_lateral: float = -0.2
    speed_offset: float = -0.05
    forward_gain: float = 0.4
    lateral_gain: float = 1.2
    heading_gain: float = 0.6
    speed_gain: float = 0.9
    max_speed: float = 1.2
    max_steer: float = 0.35
    pressure_margin: float = 0.25
    pressure_gain: float = 0.5
    cruise_speed: float = 0.6


class BlockingPolicy:
    STATE_BLOCKS = 5

    CONFIG_DEFAULTS: Dict[str, Any] = {
        "target_forward": 1.6,
        "target_lateral": -0.2,
        "speed_offset": -0.05,
        "forward_gain": 0.4,
        "lateral_gain": 1.2,
        "heading_gain": 0.6,
        "speed_gain": 0.9,
        "max_speed": 1.2,
        "max_steer": 0.35,
        "pressure_margin": 0.25,
        "pressure_gain": 0.5,
        "cruise_speed": 0.6,
    }

    def __init__(self, **kwargs: Any) -> None:
        cfg = dict(self.CONFIG_DEFAULTS)
        for key, value in kwargs.items():
            if key in cfg:
                cfg[key] = value
        self.config = BlockingControllerConfig(**cfg)
        self.agent_slot: int = 0
        self.target_slot: Optional[int] = None
        self.total_agents: int = 0

    @classmethod
    def from_config(cls, params: Optional[Dict[str, Any]]) -> "BlockingPolicy":
        return cls(**(params or {}))

    def _resolve_state_component(self, state: np.ndarray, block_idx: int, slot: int) -> float:
        if self.total_agents <= 0 or not (0 <= slot < self.total_agents):
            raise IndexError("invalid agent slot")
        idx = block_idx * self.total_agents + slot
        if idx < 0 or idx >= state.size:
            raise IndexError("state vector too small for requested component")
        return float(state[idx])

    def _extract_pose(self, state: np.ndarray, slot: int) -> Optional[np.ndarray]:
        if self.total_agents <= 0 or not (0 <= slot < self.total_agents):
            return None
        required = self.STATE_BLOCKS * self.total_agents
        if state.size < required:
            return None
        try:
            components = [
                self._resolve_state_component(state, block_idx, slot)
                for block_idx in range(self.STATE_BLOCKS)
            ]
        except IndexError:
            return None
        return np.array(components, dtype=np.float32)

    def get_action(self, action_space, obs: Dict[str, Any]) -> np.ndarray:
        if self.total_agents <= 0 or self.target_slot is None:
            return np.zeros(2, dtype=np.float32)

        state_raw = obs.get("state")
        if state_raw is None:
            return np.zeros(2, dtype=np.float32)
        state_vec = np.asarray(state_raw, dtype=np.float32).flatten()

        target_pose = self._extract_pose(state_vec, self.target_slot)
        self_pose = self._extract_pose(state_vec, self.agent_slot)
        if target_pose is None or self_pose is None:
            return np.zeros(2, dtype=np.float32)

        forward_rel, lateral_rel = self._relative_components(self_pose, target_pose)
        forward_error = self.config.target_forward - forward_rel
        lateral_error = self.config.target_lateral - lateral_rel
        heading_error = self._wrap_angle(target_pose[2] - self_pose[2])

        steering = self.config.lateral_gain * lateral_error + self.config.heading_gain * heading_error
        steering = float(np.clip(steering, -self.config.max_steer, self.config.max_steer))

        target_speed = self._desired_speed(target_pose, forward_error, lateral_error)
        current_speed = self._current_speed(self_pose)
        speed_cmd = current_speed + self.config.speed_gain * (target_speed - current_speed)
        speed_cmd = float(np.clip(speed_cmd, 0.0, self.config.max_speed))

        action = np.array([steering, speed_cmd], dtype=np.float32)
        if action_space is not None:
            action = np.clip(action, action_space.low, action_space.high)
        return action

    def _desired_speed(
        self,
        target_pose: np.ndarray,
        forward_error: float,
        lateral_error: float,
    ) -> float:
        defender_speed = float(np.hypot(target_pose[3], target_pose[4]))
        speed = defender_speed + self.config.speed_offset + self.config.forward_gain * forward_error
        speed -= self._wall_pressure_penalty(lateral_error)
        if forward_error <= 0.0:
            speed = max(speed, self.config.cruise_speed)
        return float(np.clip(speed, 0.0, self.config.max_speed))

    def _relative_components(self, self_pose: np.ndarray, target_pose: np.ndarray) -> np.ndarray:
        dx = float(self_pose[0] - target_pose[0])
        dy = float(self_pose[1] - target_pose[1])
        sin_h = np.sin(target_pose[2])
        cos_h = np.cos(target_pose[2])
        forward = cos_h * dx + sin_h * dy
        lateral = -sin_h * dx + cos_h * dy
        return np.array([forward, lateral], dtype=np.float32)

    @staticmethod
    def _current_speed(pose: np.ndarray) -> float:
        if pose.size < 5:
            return 0.0
        return float(np.hypot(pose[3], pose[4]))

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    def _wall_pressure_penalty(self, lateral_error: float) -> float:
        if self.config.pressure_gain <= 0.0:
            return 0.0
        margin = max(self.config.pressure_margin, 1e-3)
        excess = abs(lateral_error) - margin
        if excess <= 0.0:
            return 0.0
        proximity = np.clip(excess / margin, 0.0, 1.0)
        return float(self.config.pressure_gain * proximity)
