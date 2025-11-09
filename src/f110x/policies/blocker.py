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
    predict_horizon: float = 0.6
    lateral_flip_threshold: float = 0.2
    lidar_flip_threshold: float = 0.2
    min_separation: float = 0.6
    accel_limit: float = 0.4
    decel_limit: float = 0.6
    steering_blend: float = 0.6
    catchup_limit: float = 0.2
    allow_side_switch: bool = False
    side_hysteresis_steps: int = 12


class BlockingPolicy:
    STATE_BLOCKS = 6

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
        "predict_horizon": 0.6,
        "lateral_flip_threshold": 0.2,
        "lidar_flip_threshold": 0.2,
        "min_separation": 0.6,
        "accel_limit": 0.4,
        "decel_limit": 0.6,
        "steering_blend": 0.6,
        "catchup_limit": 0.2,
        "allow_side_switch": False,
        "side_hysteresis_steps": 12,
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
        default_side = np.sign(self.config.target_lateral) or -1.0
        self._block_side: float = float(default_side)
        self._last_speed_cmd: float = 0.0
        self._side_switch_counter: int = 0

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
        lidar = self._extract_lidar(obs)

        block_side = self._resolve_block_side(target_pose, lateral_rel, lidar)
        predicted_pose = self._predict_target_pose(target_pose)
        block_point = self._blocking_point(predicted_pose, block_side)
        desired_lateral = self._desired_lateral(block_side)

        pp_steering, distance_to_point = self._steering_command(self_pose, block_point)
        lateral_error = desired_lateral - lateral_rel
        lateral_term = self.config.lateral_gain * lateral_error
        lateral_term = float(np.clip(lateral_term, -self.config.max_steer, self.config.max_steer))

        blend = float(np.clip(self.config.steering_blend, 0.0, 1.0))
        steering = (1.0 - blend) * pp_steering + blend * lateral_term
        steering = float(np.clip(steering, -self.config.max_steer, self.config.max_steer))

        separation = float(np.hypot(*(self_pose[:2] - target_pose[:2])))

        target_speed = self._desired_speed(
            target_pose,
            distance_to_point,
            separation,
            lateral_rel,
        )
        current_speed = self._current_speed(self_pose)
        speed_cmd = current_speed + self.config.speed_gain * (target_speed - current_speed)
        speed_cmd = self._limit_speed_change(speed_cmd)
        speed_cmd = float(np.clip(speed_cmd, 0.0, self.config.max_speed))

        action = np.array([steering, speed_cmd], dtype=np.float32)
        if action_space is not None:
            action = np.clip(action, action_space.low, action_space.high)
        return action

    def _desired_speed(
        self,
        target_pose: np.ndarray,
        distance_to_point: float,
        separation: float,
        lateral_rel: float,
    ) -> float:
        defender_speed = float(np.hypot(target_pose[3], target_pose[4]))
        gap_error = distance_to_point - self.config.target_forward
        catchup = self.config.forward_gain * gap_error
        catchup = float(np.clip(catchup, -self.config.catchup_limit, self.config.catchup_limit))
        speed = defender_speed + self.config.speed_offset + catchup
        speed -= self._wall_pressure_penalty(lateral_rel)
        if separation <= self.config.min_separation:
            speed = min(speed, defender_speed - 0.2)
        elif separation > self.config.target_forward * 1.5:
            speed = max(speed, defender_speed + 0.1)
        if gap_error <= 0.0:
            speed = max(speed, self.config.cruise_speed)
        return float(np.clip(speed, 0.0, self.config.max_speed))

    def _predict_target_pose(self, target_pose: np.ndarray) -> np.ndarray:
        horizon = max(self.config.predict_horizon, 0.0)
        vx, vy = target_pose[3], target_pose[4]
        omega = target_pose[5] if target_pose.size > 5 else 0.0
        px = target_pose[0] + vx * horizon
        py = target_pose[1] + vy * horizon
        heading = target_pose[2] + omega * horizon
        return np.array([px, py, heading], dtype=np.float32)

    def _blocking_point(self, predicted_pose: np.ndarray, block_side: float) -> np.ndarray:
        offset = np.array(
            [
                self.config.target_forward,
                block_side * max(abs(self.config.target_lateral), 1e-3),
            ],
            dtype=np.float32,
        )
        sin_h = np.sin(predicted_pose[2])
        cos_h = np.cos(predicted_pose[2])
        rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]], dtype=np.float32)
        return predicted_pose[:2] + rot @ offset

    def _desired_lateral(self, block_side: float) -> float:
        return block_side * max(abs(self.config.target_lateral), 1e-3)

    def _steering_command(self, self_pose: np.ndarray, block_point: np.ndarray) -> tuple[float, float]:
        error_vec = block_point - self_pose[:2]
        distance = float(np.hypot(error_vec[0], error_vec[1]))
        desired_heading = np.arctan2(error_vec[1], error_vec[0])
        heading_error = self._wrap_angle(desired_heading - self_pose[2])
        steering = self.config.heading_gain * heading_error
        steering = float(np.clip(steering, -self.config.max_steer, self.config.max_steer))
        return steering, distance

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

    def _resolve_block_side(
        self,
        target_pose: np.ndarray,
        lateral_rel: float,
        lidar: Optional[np.ndarray],
    ) -> float:
        default_side = np.sign(self.config.target_lateral) or -1.0
        current = self._block_side if self._block_side != 0.0 else default_side
        candidate = current

        if self.config.allow_side_switch:
            sensor_side = self._sensor_side_choice(target_pose, lidar)
            if sensor_side is not None:
                candidate = sensor_side

        return self._apply_side_hysteresis(candidate, default_side)

    def _wall_pressure_penalty(self, lateral_rel: float) -> float:
        if self.config.pressure_gain <= 0.0:
            return 0.0
        margin = max(self.config.pressure_margin, 1e-3)
        desired = self.config.target_lateral
        if desired < 0.0:
            limit = desired - margin
            excess = max(0.0, limit - lateral_rel)
        elif desired > 0.0:
            limit = desired + margin
            excess = max(0.0, lateral_rel - limit)
        else:
            excess = max(0.0, abs(lateral_rel) - margin)
        if excess <= 0.0:
            return 0.0
        proximity = np.clip(excess / margin, 0.0, 1.0)
        return float(self.config.pressure_gain * proximity)

    def _target_lateral_velocity(self, target_pose: np.ndarray) -> float:
        vx, vy = target_pose[3], target_pose[4]
        sin_h = np.sin(target_pose[2])
        cos_h = np.cos(target_pose[2])
        return float(-sin_h * vx + cos_h * vy)

    def _limit_speed_change(self, speed_cmd: float) -> float:
        prev = self._last_speed_cmd
        accel = max(self.config.accel_limit, 0.0)
        decel = max(self.config.decel_limit, 0.0)
        delta = speed_cmd - prev
        if delta > accel:
            speed_cmd = prev + accel
        elif delta < -decel:
            speed_cmd = prev - decel
        self._last_speed_cmd = speed_cmd
        return speed_cmd

    def _extract_lidar(self, obs: Dict[str, Any]) -> Optional[np.ndarray]:
        scan = obs.get("lidar")
        if scan is None:
            scan = obs.get("scans")
        if scan is None:
            return None
        arr = np.asarray(scan, dtype=np.float32).flatten()
        return arr if arr.size >= 4 else None

    def _sensor_side_choice(
        self,
        target_pose: np.ndarray,
        lidar: Optional[np.ndarray],
    ) -> Optional[float]:
        lateral_velocity = self._target_lateral_velocity(target_pose)
        velocity_thresh = max(self.config.lateral_flip_threshold, 0.0)
        if abs(lateral_velocity) > velocity_thresh:
            return float(np.sign(lateral_velocity))
        if lidar is None:
            return None
        half = lidar.size // 2
        if half == 0:
            return None
        right = float(np.mean(lidar[:half]))
        left = float(np.mean(lidar[half:]))
        diff = left - right
        lidar_thresh = max(self.config.lidar_flip_threshold, 0.0)
        if abs(diff) < lidar_thresh:
            return None
        return 1.0 if diff > 0.0 else -1.0

    def _apply_side_hysteresis(self, candidate: float, default_side: float) -> float:
        candidate = float(np.sign(candidate) or default_side)
        current = float(self._block_side if self._block_side != 0.0 else default_side)
        if np.sign(candidate) == np.sign(current):
            self._side_switch_counter = 0
            self._block_side = current
            return self._block_side
        self._side_switch_counter += 1
        threshold = max(int(self.config.side_hysteresis_steps), 1)
        if self._side_switch_counter >= threshold:
            self._block_side = candidate
            self._side_switch_counter = 0
        return self._block_side
