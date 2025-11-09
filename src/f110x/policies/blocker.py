from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class BlockingControllerConfig:
    target_forward: float = 1.8
    target_lateral: float = -0.25
    speed_offset: float = -0.08
    forward_gain: float = 0.35
    lateral_gain: float = 1.0
    heading_gain: float = 0.5
    speed_gain: float = 0.9
    max_speed: float = 1.2
    max_steer: float = 0.35
    pressure_margin: float = 0.2
    pressure_gain: float = 0.4
    cruise_speed: float = 0.6
    catchup_limit: float = 0.2
    damping_gain: float = 0.3
    accel_limit: float = 0.4
    decel_limit: float = 0.6
    guard_speed_penalty: float = 0.3
    min_block_speed: float = 0.15
    pressure_ramp_steps: int = 50
    pressure_ramp_delta: float = 0.01
    pressure_ramp_limit: float = 0.25
    band_forward_tol: float = 0.25
    band_lateral_tol: float = 0.05


class BlockingPolicy:
    """Relative-offset blocker that holds formation ahead of a target agent."""

    STATE_BLOCKS = 5

    CONFIG_DEFAULTS: Dict[str, Any] = {
        "target_forward": 1.8,
        "target_lateral": -0.25,
        "speed_offset": -0.05,
        "forward_gain": 0.45,
        "lateral_gain": 1.0,
        "heading_gain": 0.5,
        "speed_gain": 0.9,
        "max_speed": 1.2,
        "max_steer": 0.35,
        "pressure_margin": 0.2,
        "pressure_gain": 0.4,
        "cruise_speed": 0.6,
        "catchup_limit": 0.2,
        "damping_gain": 0.3,
        "accel_limit": 0.4,
        "decel_limit": 0.6,
        "guard_speed_penalty": 0.3,
        "min_block_speed": 0.15,
        "pressure_ramp_steps": 50,
        "pressure_ramp_delta": 0.01,
        "pressure_ramp_limit": 0.25,
        "band_forward_tol": 0.25,
        "band_lateral_tol": 0.05,
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
        self._prev_steer: float = 0.0
        self._last_speed_cmd: float = 0.0
        self._pressure_timer: int = 0
        self._pressure_offset: float = 0.0

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
        effective_lateral = self._effective_target_lateral()
        heading_error = self._heading_error_clamped(
            target_pose[2],
            self_pose[2],
            lateral_rel,
            effective_lateral,
        )
        forward_error = self.config.target_forward - forward_rel
        lateral_error = effective_lateral - lateral_rel
        self._update_pressure_ramp(forward_error, lateral_error)
        effective_lateral = self._effective_target_lateral()
        lateral_error = effective_lateral - lateral_rel

        steering = (
            self.config.lateral_gain * lateral_error
            + self.config.heading_gain * heading_error
        )
        steering = self._apply_damping(steering)
        steering = float(np.clip(steering, -self.config.max_steer, self.config.max_steer))

        defender_speed = float(np.hypot(target_pose[3], target_pose[4]))
        target_speed = self._desired_speed(
            target_pose,
            forward_error,
            lateral_rel,
            defender_speed,
            effective_lateral,
        )
        current_speed = self._current_speed(self_pose)
        speed_cmd = current_speed + self.config.speed_gain * (target_speed - current_speed)
        steering, speed_cmd = self._apply_guardrails(
            steering,
            speed_cmd,
            lateral_rel,
            defender_speed,
            effective_lateral,
        )
        speed_cmd = max(speed_cmd, self.config.min_block_speed)
        speed_cmd = self._limit_speed_change(speed_cmd)
        speed_cmd = float(np.clip(speed_cmd, 0.0, self.config.max_speed))

        action = np.array([steering, speed_cmd], dtype=np.float32)
        if action_space is not None:
            action = np.clip(action, action_space.low, action_space.high)
        return action

    def _desired_speed(
        self,
        target_pose: np.ndarray,
        forward_error: float,
        lateral_rel: float,
        defender_speed: float,
        effective_lateral: float,
    ) -> float:
        catchup = self.config.forward_gain * forward_error
        catchup = float(np.clip(catchup, -self.config.catchup_limit, self.config.catchup_limit))
        speed = defender_speed + self.config.speed_offset + catchup
        speed -= self._wall_pressure_penalty(lateral_rel, effective_lateral)
        if forward_error <= 0.0:
            speed = max(speed, self.config.cruise_speed)
        speed = max(speed, self.config.min_block_speed)
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

    def _heading_error_clamped(
        self,
        target_heading: float,
        self_heading: float,
        lateral_rel: float,
        effective_lateral: float,
    ) -> float:
        raw = self._wrap_angle(target_heading - self_heading)
        same_side = np.sign(lateral_rel) == np.sign(effective_lateral)
        if not same_side:
            limit = np.deg2rad(15.0)
            raw = float(np.clip(raw, -limit, limit))
        return raw

    def _wall_pressure_penalty(self, lateral_rel: float, effective_lateral: float) -> float:
        if self.config.pressure_gain <= 0.0:
            return 0.0
        margin = max(self.config.pressure_margin, 1e-3)
        desired = effective_lateral
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

    def _apply_damping(self, steering: float) -> float:
        alpha = float(np.clip(self.config.damping_gain, 0.0, 1.0))
        filtered = self._prev_steer + alpha * (steering - self._prev_steer)
        self._prev_steer = filtered
        return filtered

    def _apply_guardrails(
        self,
        steering: float,
        speed_cmd: float,
        lateral_rel: float,
        defender_speed: float,
        effective_lateral: float,
    ) -> tuple[float, float]:
        margin = max(self.config.pressure_margin, 0.05)
        desired = effective_lateral
        guard_penalty = max(self.config.guard_speed_penalty, 0.0)
        min_guard_speed = max(0.1, defender_speed * 0.25)

        if desired < 0.0 and lateral_rel < desired - margin:
            steering = max(steering, 0.0)
            allowed = max(min_guard_speed, defender_speed - guard_penalty)
            speed_cmd = min(max(speed_cmd, min_guard_speed), allowed)
        elif desired > 0.0 and lateral_rel > desired + margin:
            steering = min(steering, 0.0)
            allowed = max(min_guard_speed, defender_speed - guard_penalty)
            speed_cmd = min(max(speed_cmd, min_guard_speed), allowed)
        return steering, speed_cmd

    def _update_pressure_ramp(self, forward_error: float, lateral_error: float) -> None:
        tol_f = max(self.config.band_forward_tol, 1e-3)
        tol_l = max(self.config.band_lateral_tol, 1e-3)
        in_band = abs(forward_error) <= tol_f and abs(lateral_error) <= tol_l
        if not in_band:
            self._pressure_timer = 0
            self._pressure_offset = 0.0
            return
        self._pressure_timer += 1
        if self._pressure_timer < max(int(self.config.pressure_ramp_steps), 1):
            return
        delta = self.config.pressure_ramp_delta
        if self.config.target_lateral <= 0.0:
            delta = -abs(delta)
        else:
            delta = abs(delta)
        self._pressure_offset = float(
            np.clip(
                self._pressure_offset + delta,
                -abs(self.config.pressure_ramp_limit),
                abs(self.config.pressure_ramp_limit),
            )
        )
        self._pressure_timer = 0

    def _effective_target_lateral(self) -> float:
        base = self.config.target_lateral
        adjusted = base + self._pressure_offset
        # Ensure we do not cross over to opposite side
        if base <= 0.0:
            adjusted = min(adjusted, -1e-3)
        else:
            adjusted = max(adjusted, 1e-3)
        return adjusted
