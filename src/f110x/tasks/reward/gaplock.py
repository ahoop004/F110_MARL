"""Gaplock task reward strategy."""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

from f110x.utils.reward_utils import ScalingParams, apply_reward_scaling

from .base import RewardRuntimeContext, RewardStep, RewardStrategy
from .components import RewardAccumulator, apply_relative_sector_reward
from .registry import RewardTaskConfig, RewardTaskRegistry, RewardTaskSpec, register_reward_task


GAPLOCK_PARAM_KEYS = (
    "target_crash_reward",
    "self_collision_penalty",
    "truncation_penalty",
    "success_once",
    "reward_horizon",
    "reward_clip",
    "reward_weight",
    "reward_decay",
    "reward_smoothing",
    "step_reward",
    "idle_penalty",
    "idle_penalty_steps",
    "idle_speed_threshold",
    "idle_truncation_penalty",
    "pressure_distance",
    "pressure_timeout",
    "pressure_min_speed",
    "pressure_heading_tolerance",
    "ignored_agents",
    "pressure_bonus",
    "pressure_bonus_interval",
    "proximity_penalty_distance",
    "proximity_penalty_value",
    "speed_bonus_coef",
    "speed_bonus_target",
    "reverse_penalty",
    "reverse_speed_threshold",
    "brake_penalty",
    "brake_speed_threshold",
    "brake_drop_threshold",
    "heading_reward_coef",
    "distance_reward_near",
    "distance_reward_near_distance",
    "distance_reward_far_distance",
    "distance_penalty_far",
    "distance_gradient",
    "pressure_streak_bonus",
    "pressure_streak_cap",
    "commit_distance",
    "commit_heading_threshold",
    "commit_speed_threshold",
    "commit_bonus",
    "escape_distance",
    "escape_penalty",
    "success_border_radius",
    "success_border_lane_center",
    "success_border_requires_pressure",
    "success_requires_pressure",
    "target_offsets",
    "target_offset_radius",
    "target_offset_falloff",
    "target_offset_marker_radius",
    "target_offset_marker_segments",
)


class GaplockRewardStrategy(RewardStrategy):
    name = "gaplock"

    def __init__(
        self,
        *,
        target_crash_reward: float = 10.0,
        self_collision_penalty: float = -10.0,
        truncation_penalty: float = 0.0,
        success_once: bool = True,
        reward_horizon: Optional[float] = None,
        reward_clip: Optional[float] = None,
        reward_weight: Optional[float] = None,
        reward_decay: Optional[float] = None,
        reward_smoothing: Optional[float] = None,
        step_reward: float = 0.0,
        relative_reward: Optional[Dict[str, Any]] = None,
        idle_penalty: float = 0.0,
        idle_penalty_steps: int = 0,
        idle_speed_threshold: float = 0.1,
        idle_truncation_penalty: float = 0.0,
        target_resolver: Optional[Callable[[str], Optional[str]]] = None,
        reward_ring_callback: Optional[Callable[[str, Optional[str]], None]] = None,
        reward_ring_focus_agent: Optional[str] = None,
        pressure_distance: float = 0.75,
        pressure_timeout: float = 0.5,
        pressure_min_speed: float = 0.1,
        pressure_heading_tolerance: float = math.pi,
        ignored_agents: Optional[Iterable[Any]] = None,
        pressure_bonus: float = 0.0,
        pressure_bonus_interval: int = 1,
        proximity_penalty_distance: float = 0.0,
        proximity_penalty_value: float = 0.0,
        speed_bonus_coef: float = 0.0,
        speed_bonus_target: float = 0.0,
        reverse_penalty: float = 0.0,
        reverse_speed_threshold: float = 0.0,
        brake_penalty: float = 0.0,
        brake_speed_threshold: float = 0.0,
        brake_drop_threshold: float = 0.0,
        heading_reward_coef: float = 0.0,
        distance_reward_near: float = 0.0,
        distance_reward_near_distance: float = 0.0,
        distance_reward_far_distance: float = 0.0,
        distance_penalty_far: float = 0.0,
        distance_gradient: Optional[Dict[str, Any]] = None,
        pressure_streak_bonus: float = 0.0,
        pressure_streak_cap: int = 0,
        commit_distance: float = 0.0,
        commit_heading_threshold: float = 0.0,
        commit_speed_threshold: float = 0.0,
        commit_bonus: float = 0.0,
        escape_distance: float = 0.0,
        escape_penalty: float = 0.0,
        success_border_radius: float = 0.0,
        success_border_lane_center: float = 0.0,
        success_border_requires_pressure: bool = False,
        success_requires_pressure: bool = True,
        target_offsets: Optional[Iterable[Any]] = None,
        target_offset_radius: float = 0.0,
        target_offset_falloff: float = 0.0,
        target_offset_marker_radius: float = 0.0,
        target_offset_marker_segments: int = 16,
        reward_ring_state_callback: Optional[Callable[[str, Optional[List[bool]]], None]] = None,
    ) -> None:
        self.target_crash_reward = float(target_crash_reward)
        self.self_collision_penalty = float(self_collision_penalty)
        self.truncation_penalty = float(truncation_penalty)
        self.success_once = bool(success_once)
        self.scaling_params = ScalingParams(
            horizon=self._coerce_positive_float(reward_horizon),
            clip=self._coerce_positive_float(reward_clip),
            weight=self._coerce_positive_float(reward_weight),
            decay=self._coerce_positive_float(reward_decay),
            smoothing=self._coerce_positive_float(reward_smoothing),
        )
        self._success_awarded: Dict[str, set[Tuple[str, str]]] = {}
        self.step_reward = float(step_reward)
        self.idle_penalty = float(idle_penalty)
        self.idle_penalty_steps = max(int(idle_penalty_steps), 0)
        self.idle_speed_threshold = max(float(idle_speed_threshold), 0.0)
        self._idle_counters: Dict[str, int] = {}
        self._idle_penalty_applied: set[str] = set()
        self.idle_truncation_penalty = float(idle_truncation_penalty)
        self._idle_truncation_applied: set[str] = set()
        self._target_resolver = target_resolver
        self.relative_reward_cfg = self._prepare_relative_reward(relative_reward)
        self._reward_ring_callback = reward_ring_callback
        self._reward_ring_focus_agent = reward_ring_focus_agent
        self.pressure_distance = max(float(pressure_distance), 0.0)
        self.pressure_timeout = max(float(pressure_timeout), 0.0)
        self.pressure_min_speed = max(float(pressure_min_speed), 0.0)
        tolerance = float(pressure_heading_tolerance) if pressure_heading_tolerance is not None else math.pi
        self.pressure_heading_tolerance = min(max(tolerance, 0.0), math.pi)
        self._pressure_log: Dict[str, Dict[str, Tuple[float, int]]] = {}
        self._ignored_agents: set[str] = {
            str(agent_id) for agent_id in (ignored_agents or []) if agent_id is not None and str(agent_id)
        }
        self.pressure_bonus = max(float(pressure_bonus), 0.0)
        self.pressure_bonus_interval = max(int(pressure_bonus_interval), 1)
        self.proximity_penalty_distance = max(float(proximity_penalty_distance), 0.0)
        self.proximity_penalty_value = float(proximity_penalty_value)
        self._pressure_bonus_counters: Dict[str, int] = {}
        self._pressure_streak: Dict[str, int] = {}
        self._pressure_streak_levels: Dict[str, int] = {}
        self.speed_bonus_coef = float(speed_bonus_coef)
        self.speed_bonus_target = max(float(speed_bonus_target), 0.0)
        self.reverse_penalty = max(float(reverse_penalty), 0.0)
        self.reverse_speed_threshold = max(float(reverse_speed_threshold), 0.0)
        self.brake_penalty = float(brake_penalty)
        self.brake_speed_threshold = max(float(brake_speed_threshold), 0.0)
        self.brake_drop_threshold = max(float(brake_drop_threshold), 0.0)
        self.heading_reward_coef = float(heading_reward_coef)
        self.distance_reward_near = float(distance_reward_near)
        self.distance_reward_near_distance = max(float(distance_reward_near_distance), 0.0)
        self.distance_reward_far_distance = max(float(distance_reward_far_distance), 0.0)
        self.distance_penalty_far = max(float(distance_penalty_far), 0.0)
        self.distance_gradient_cfg = self._prepare_distance_gradient(distance_gradient)
        self.pressure_streak_bonus = max(float(pressure_streak_bonus), 0.0)
        self.pressure_streak_cap = max(int(pressure_streak_cap), 0)
        self.commit_distance = max(float(commit_distance), 0.0)
        self.commit_heading_threshold = float(commit_heading_threshold)
        self.commit_speed_threshold = max(float(commit_speed_threshold), 0.0)
        self.commit_bonus = float(commit_bonus)
        self.escape_distance = max(float(escape_distance), 0.0)
        self.escape_penalty = float(escape_penalty)
        self._last_speed: Dict[str, float] = {}
        self._pressure_streak: Dict[str, int] = {}
        self._commit_active: Dict[str, bool] = {}
        self._commit_awarded: set[str] = set()
        self.success_border_radius = max(float(success_border_radius), 0.0)
        self.success_border_lane_center = float(success_border_lane_center)
        self.success_border_requires_pressure = bool(success_border_requires_pressure)
        self.success_requires_pressure = bool(success_requires_pressure)
        offsets, offsets_raw = self._prepare_target_offsets(target_offsets)
        self._target_offsets = offsets
        self._target_offsets_config = offsets_raw
        self.target_offset_radius = max(float(target_offset_radius), 0.0)
        self.target_offset_falloff = max(float(target_offset_falloff), 0.0)
        self.target_offset_marker_radius = max(float(target_offset_marker_radius), 0.0)
        self.target_offset_marker_segments = max(int(target_offset_marker_segments), 4)
        self._reward_ring_state_callback = reward_ring_state_callback

    @staticmethod
    def _coerce_positive_float(value: Optional[Any]) -> Optional[float]:
        if value in (None, ""):
            return None
        try:
            val = float(value)
        except (TypeError, ValueError):
            return None
        return val if val > 0.0 else None

    @staticmethod
    def _prepare_target_offsets(cfg: Optional[Iterable[Any]]) -> Tuple[List[np.ndarray], List[Tuple[float, float]]]:
        if cfg is None:
            return [], []
        offsets: List[np.ndarray] = []
        raw_pairs: List[Tuple[float, float]] = []
        for entry in cfg:
            if entry is None:
                continue
            if isinstance(entry, dict):
                forward = entry.get("forward", entry.get("x", 0.0))
                left = entry.get("left", entry.get("y", 0.0))
            elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                forward, left = entry[0], entry[1]
            else:
                continue
            try:
                fx = float(forward)
                fy = float(left)
            except (TypeError, ValueError):
                continue
            offsets.append(np.array([fx, fy], dtype=np.float32))
            raw_pairs.append((fx, fy))
        return offsets, raw_pairs

    def _anchor_points(self, target_pose: Optional[np.ndarray]) -> List[np.ndarray]:
        if target_pose is None or target_pose.size < 3:
            return []
        if not self._target_offsets:
            return [np.asarray(target_pose[:2], dtype=np.float32)]
        heading = float(target_pose[2])
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)
        rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]], dtype=np.float32)
        origin = np.asarray(target_pose[:2], dtype=np.float32)
        anchors = []
        for offset in self._target_offsets:
            anchors.append(origin + rot @ offset)
        return anchors

    def _closest_anchor(
        self,
        ego_pose: Optional[np.ndarray],
        target_pose: Optional[np.ndarray],
    ) -> Tuple[Optional[np.ndarray], Optional[float], Optional[np.ndarray], List[np.ndarray], List[float]]:
        anchors = self._anchor_points(target_pose)
        if not anchors:
            return None, None, None, [], []
        if ego_pose is None or ego_pose.size < 2:
            anchor = anchors[0]
            return anchor, None, None, anchors, []
        ego_xy = np.asarray(ego_pose[:2], dtype=np.float32)
        dists = [float(np.linalg.norm(anchor - ego_xy)) for anchor in anchors]
        best_idx = int(np.argmin(dists))
        anchor = anchors[best_idx]
        distance = dists[best_idx]
        rel_vec = anchor - ego_xy
        return anchor, distance, rel_vec, anchors, dists

    def reset(self, episode_index: int) -> None:
        self._success_awarded.clear()
        self._idle_counters.clear()
        self._idle_penalty_applied.clear()
        self._idle_truncation_applied.clear()
        self._pressure_log.clear()
        self._pressure_bonus_counters.clear()
        self._pressure_streak.clear()
        self._pressure_streak_levels.clear()
        self._last_speed.clear()
        self._commit_active.clear()
        self._commit_awarded.clear()

    def _select_target_obs(
        self,
        step: RewardStep,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        if not step.all_obs:
            return None, None

        if callable(self._target_resolver):
            candidate_id = self._target_resolver(step.agent_id)
            if candidate_id and candidate_id in step.all_obs:
                candidate_obs = step.all_obs.get(candidate_id)
                if candidate_obs is not None:
                    return candidate_obs, candidate_id

        for other_id, other_obs in step.all_obs.items():
            if other_id != step.agent_id:
                return other_obs, other_id
        return None, None

    def _has_awarded(self, agent_id: str, target_id: str) -> bool:
        awarded = self._success_awarded.setdefault(agent_id, set())
        key = (agent_id, target_id)
        if key in awarded:
            return True
        if self.success_once:
            awarded.add(key)
        return False

    def _notify_reward_ring(self, agent_id: str, target_id: Optional[str]) -> None:
        if self._reward_ring_callback is None:
            return
        if self._reward_ring_focus_agent is not None and agent_id != self._reward_ring_focus_agent:
            return
        try:
            self._reward_ring_callback(agent_id, target_id)
        except Exception:
            # Rendering is auxiliary; ignore callback errors.
            pass

    @staticmethod
    def _coerce_speed_value(value: float) -> float:
        if np.isnan(value) or not np.isfinite(value):
            return 0.0
        return float(value)

    def _extract_speed(self, obs: Dict[str, Any]) -> Optional[float]:
        velocity = obs.get("velocity")
        speed_val: Optional[float] = None
        if velocity is not None:
            try:
                vector = np.asarray(velocity, dtype=np.float32)
                speed_val = float(np.linalg.norm(vector))
            except Exception:
                speed_val = None
        if speed_val is None:
            raw_speed = obs.get("speed")
            if raw_speed is None:
                return None
            try:
                speed_val = float(raw_speed)
            except (TypeError, ValueError):
                return None
        return self._coerce_speed_value(speed_val)

    def _extract_forward_speed(self, obs: Dict[str, Any]) -> Optional[float]:
        pose_arr = self._extract_pose(obs)
        if pose_arr is None or pose_arr.size < 3:
            return None
        velocity = obs.get("velocity")
        if velocity is None:
            return None
        try:
            vel_vec = np.asarray(velocity, dtype=np.float32).flatten()
        except Exception:
            return None
        if vel_vec.size < 2:
            return None
        heading = float(pose_arr[2])
        forward_dir = np.array([math.cos(heading), math.sin(heading)], dtype=np.float32)
        forward_component = float(np.dot(vel_vec[:2], forward_dir))
        if not np.isfinite(forward_component):
            return None
        return forward_component

    def _apply_idle_penalty(self, acc: RewardAccumulator, agent_id: str, speed: Optional[float]) -> None:
        if speed is None or self.idle_speed_threshold <= 0.0:
            self._idle_counters.pop(agent_id, None)
            return

        if speed < self.idle_speed_threshold:
            counter = self._idle_counters.get(agent_id, 0) + 1
            self._idle_counters[agent_id] = counter
            threshold_steps = max(1, self.idle_penalty_steps)
            if self.idle_penalty and counter >= threshold_steps:
                penalty = -abs(float(self.idle_penalty))
                acc.add("idle_penalty", penalty)
        else:
            self._idle_counters[agent_id] = 0

    def _is_idle_truncation(self, step: RewardStep) -> bool:
        if step.events and step.events.get("idle_triggered"):
            return True
        info = step.info if isinstance(step.info, dict) else None
        if info and info.get("idle_triggered"):
            return True
        return False

    def compute(self, step: RewardStep) -> Tuple[float, Dict[str, float]]:
        if self._ignored_agents and step.agent_id in self._ignored_agents:
            env_reward = float(step.env_reward)
            components: Dict[str, float] = {}
            if env_reward:
                components["env_reward"] = env_reward
            total, scaled_components = apply_reward_scaling(env_reward, components, self.scaling_params)
            return total, scaled_components

        acc = RewardAccumulator()

        env_reward = float(step.env_reward)
        if env_reward:
            acc.add("env_reward", env_reward)

        ego_obs = step.obs
        target_obs, explicit_target_id = self._select_target_obs(step)
        ego_crashed = bool(ego_obs.get("collision", False))
        overlay_target_id: Optional[str] = None

        timestep = float(step.timestep) if isinstance(step.timestep, (int, float)) else 0.0
        time_scale = timestep if timestep > 0.0 else 0.1
        if time_scale > 0.2:
            time_scale = 0.2

        speed = self._extract_speed(ego_obs)
        self._apply_idle_penalty(acc, step.agent_id, speed)

        prev_speed = self._last_speed.get(step.agent_id)
        if speed is not None:
            if self.speed_bonus_coef:
                capped = speed
                if self.speed_bonus_target > 0.0:
                    capped = min(speed, self.speed_bonus_target)
                acc.add("speed_bonus", self.speed_bonus_coef * max(capped, 0.0) * time_scale)
            if (
                self.brake_penalty
                and prev_speed is not None
                and self.brake_drop_threshold > 0.0
                and prev_speed - speed >= self.brake_drop_threshold
                and speed <= max(self.brake_speed_threshold, 0.0)
            ):
                acc.add("brake_penalty", float(self.brake_penalty))
        if speed is not None:
            self._last_speed[step.agent_id] = float(speed)
        else:
            self._last_speed.pop(step.agent_id, None)

        forward_speed = self._extract_forward_speed(ego_obs)
        if (
            self.reverse_penalty
            and forward_speed is not None
            and forward_speed < -self.reverse_speed_threshold
        ):
            penalty_value = -self.reverse_penalty * time_scale
            acc.add("reverse_penalty", penalty_value)

        if self.step_reward:
            idle_threshold = self.idle_speed_threshold
            is_idle_step = (
                speed is not None
                and idle_threshold > 0.0
                and speed < idle_threshold
            )
            if not is_idle_step:
                acc.add("step_reward", float(self.step_reward))

        if ego_crashed and self.self_collision_penalty:
            acc.add("self_collision_penalty", self.self_collision_penalty)

        pressure_recent = False
        distance: Optional[float] = None
        alignment: Optional[float] = None
        anchor_point: Optional[np.ndarray] = None
        if target_obs is not None:
            overlay_target_id = explicit_target_id or str(target_obs.get("agent_id", "target"))
            ego_pose_arr = self._extract_pose(ego_obs)
            target_pose_arr = self._extract_pose(target_obs)
            anchor_point, distance, rel_vec, anchors_all, anchors_dists = self._closest_anchor(ego_pose_arr, target_pose_arr)

            if overlay_target_id:
                if self._detect_pressure(ego_obs, target_obs, anchor_point=anchor_point):
                    self._store_pressure(
                        step.agent_id,
                        overlay_target_id,
                        step.current_time,
                        step.step_index,
                    )
                pressure_recent = self._has_recent_pressure(
                    step.agent_id,
                    overlay_target_id,
                    step.current_time,
                    step.step_index,
                    step.timestep,
                )

            if ego_pose_arr is not None and target_pose_arr is not None and anchor_point is not None:
                if rel_vec is None:
                    rel_vec = np.asarray(anchor_point, dtype=np.float32) - np.asarray(ego_pose_arr[:2], dtype=np.float32)
                    distance = float(np.linalg.norm(rel_vec))
                if distance is not None and np.isfinite(distance):
                    if distance > 1e-6:
                        rel_unit = rel_vec / distance
                        ego_heading = float(ego_pose_arr[2])
                        ego_dir = np.array([math.cos(ego_heading), math.sin(ego_heading)], dtype=np.float32)
                        alignment = float(np.dot(rel_unit, ego_dir))
                    else:
                        alignment = 1.0

                    near_dist = self.distance_reward_near_distance
                    far_dist = self.distance_reward_far_distance
                    if self.distance_reward_near and near_dist > 0.0:
                        reward_value = 0.0
                        if distance <= near_dist:
                            reward_value = self.distance_reward_near
                        elif far_dist > near_dist and distance < far_dist:
                            span = far_dist - near_dist
                            weight = (far_dist - distance) / span
                            reward_value = self.distance_reward_near * weight
                        if reward_value:
                            value = reward_value * time_scale
                            acc.add("distance_reward", min(max(value, -0.5), 0.5))
                    if far_dist > 0.0 and distance >= far_dist and self.distance_penalty_far:
                        penalty_value = -abs(self.distance_penalty_far) * time_scale
                        acc.add("distance_penalty", max(penalty_value, -0.5))

                    self._apply_distance_gradient(acc, distance, time_scale)

                    if self.proximity_penalty_distance > 0.0 and self.proximity_penalty_value:
                        if distance > self.proximity_penalty_distance:
                            acc.add("proximity_penalty", -abs(self.proximity_penalty_value))

                    if alignment is not None and self.heading_reward_coef:
                        heading_value = self.heading_reward_coef * alignment * time_scale
                        acc.add("heading_reward", min(max(heading_value, -0.3), 0.3))
                # marker state for renderer: active if within sweet spot of each offset
                if anchors_all and anchors_dists:
                    radius_th = self.target_offset_radius + self.target_offset_falloff
                    active_flags = [bool(dist <= max(radius_th, 1e-6)) for dist in anchors_dists]
                    if self._reward_ring_state_callback is not None:
                        try:
                            self._reward_ring_state_callback(step.agent_id, active_flags)
                        except Exception:
                            pass

            if pressure_recent:
                streak_value = self._pressure_streak.get(step.agent_id, 0) + 1
                self._pressure_streak[step.agent_id] = streak_value
                if self.pressure_streak_bonus > 0.0:
                    capped = streak_value if self.pressure_streak_cap <= 0 else min(streak_value, self.pressure_streak_cap)
                    prev_level = self._pressure_streak_levels.get(step.agent_id, 0)
                    delta = max(capped - prev_level, 0)
                    if delta > 0:
                        self._pressure_streak_levels[step.agent_id] = capped
                        streak_value_scaled = min(self.pressure_streak_bonus * delta, 1.0)
                        acc.add("pressure_streak", streak_value_scaled)
            else:
                self._pressure_streak.pop(step.agent_id, None)
                self._pressure_streak_levels.pop(step.agent_id, None)

            if pressure_recent and self.pressure_bonus > 0.0:
                counter = self._pressure_bonus_counters.get(step.agent_id, 0) + 1
                if counter >= self.pressure_bonus_interval:
                    acc.add("pressure_bonus", min(self.pressure_bonus, 0.5))
                    counter = 0
                self._pressure_bonus_counters[step.agent_id] = counter
            else:
                self._pressure_bonus_counters.pop(step.agent_id, None)

            commit_ready = False
            if (
                self.commit_distance > 0.0
                and distance is not None
                and distance <= self.commit_distance
                and alignment is not None
                and alignment >= self.commit_heading_threshold
                and speed is not None
                and speed >= self.commit_speed_threshold
            ):
                commit_ready = True

            if commit_ready and self.commit_bonus and step.agent_id not in self._commit_awarded:
                acc.add("commit_bonus", float(self.commit_bonus))
                self._commit_awarded.add(step.agent_id)
                self._commit_active[step.agent_id] = True
            elif (
                self._commit_active.get(step.agent_id)
                and self.escape_distance > 0.0
                and distance is not None
                and distance > self.escape_distance
            ):
                if self.escape_penalty:
                    acc.add("escape_penalty", -abs(self.escape_penalty))
                self._commit_active.pop(step.agent_id, None)
                self._commit_awarded.discard(step.agent_id)

            border_hit = False
            target_crashed = False
            if target_obs is not None:
                target_crashed = bool(target_obs.get("collision", False))
                if self.success_border_radius > 0.0:
                    pose_raw = target_obs.get("pose")
                    if pose_raw is not None:
                        pose_arr = np.asarray(pose_raw, dtype=np.float32).flatten()
                        if pose_arr.size >= 2:
                            lateral_pos = float(pose_arr[1])
                            if abs(lateral_pos - self.success_border_lane_center) >= self.success_border_radius:
                                border_hit = True

            if target_crashed and not ego_crashed:
                pressure_gate = pressure_recent or not self.success_requires_pressure
                if overlay_target_id and pressure_gate:
                    if not self.success_once or not self._has_awarded(step.agent_id, overlay_target_id):
                        acc.add("success_reward", self.target_crash_reward)
                        self._clear_pressure(step.agent_id, overlay_target_id)
                self._commit_active.pop(step.agent_id, None)
                self._commit_awarded.discard(step.agent_id)
            elif border_hit and not ego_crashed:
                border_pressure_gate = pressure_recent or not self.success_border_requires_pressure
                if overlay_target_id and border_pressure_gate:
                    if not self.success_once or not self._has_awarded(step.agent_id, overlay_target_id):
                        acc.add("success_reward", self.target_crash_reward)
                        self._clear_pressure(step.agent_id, overlay_target_id)
                if step.info is not None:
                    step.info.setdefault("border_success", True)
                step.events.setdefault("border_success", True)
                step.events.setdefault("terminated", True)

        truncated = bool(step.events.get("truncated")) if step.events else False
        if not truncated and isinstance(step.info, dict):
            truncated = bool(step.info.get("truncated", False))
        if truncated and self.truncation_penalty:
            acc.add("truncation_penalty", self.truncation_penalty)
        if self._is_idle_truncation(step) and self.idle_truncation_penalty and step.agent_id not in self._idle_truncation_applied:
            acc.add("idle_truncation_penalty", float(self.idle_truncation_penalty))
            self._idle_truncation_applied.add(step.agent_id)
        if self._is_idle_truncation(step) and self.idle_penalty and step.agent_id not in self._idle_penalty_applied:
            acc.add("idle_penalty", -abs(self.idle_penalty))
            self._idle_penalty_applied.add(step.agent_id)

        if self.relative_reward_cfg and target_obs is not None:
            self._apply_relative_reward(acc, ego_obs, target_obs, anchor_point=anchor_point)

        self._notify_reward_ring(step.agent_id, overlay_target_id)

        total, components = apply_reward_scaling(acc.total, acc.components, self.scaling_params)
        return total, components

    def _prepare_distance_gradient(self, cfg: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not cfg:
            return None
        points_raw = cfg.get("points") if isinstance(cfg, dict) else None
        if not points_raw:
            return None
        points: List[Tuple[float, float]] = []
        for entry in points_raw:
            if isinstance(entry, dict):
                dist = entry.get("distance")
                value = entry.get("value")
            elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                dist, value = entry[0], entry[1]
            else:
                continue
            try:
                dist_val = float(dist)
                value_val = float(value)
            except (TypeError, ValueError):
                continue
            points.append((max(dist_val, 0.0), value_val))
        if not points:
            return None
        points.sort(key=lambda item: item[0])
        clip_cfg = cfg.get("clip")
        clip_range: Optional[Tuple[float, float]] = None
        if isinstance(clip_cfg, (list, tuple)) and len(clip_cfg) >= 2:
            try:
                clip_min = float(clip_cfg[0])
                clip_max = float(clip_cfg[1])
                if clip_min > clip_max:
                    clip_min, clip_max = clip_max, clip_min
                clip_range = (clip_min, clip_max)
            except (TypeError, ValueError):
                clip_range = None
        scale = float(cfg.get("scale", 1.0))
        time_scaled = bool(cfg.get("time_scaled", True))
        label = str(cfg.get("label", "distance_gradient")) or "distance_gradient"
        return {
            "points": points,
            "clip": clip_range,
            "scale": scale,
            "time_scaled": time_scaled,
            "label": label,
        }

    def _apply_distance_gradient(
        self,
        acc: RewardAccumulator,
        distance: Optional[float],
        time_scale: float,
    ) -> None:
        cfg = self.distance_gradient_cfg
        if cfg is None or distance is None:
            return
        points: List[Tuple[float, float]] = cfg["points"]
        if not points:
            return
        if distance <= points[0][0]:
            value = points[0][1]
        elif distance >= points[-1][0]:
            value = points[-1][1]
        else:
            value = points[-1][1]
            for idx in range(1, len(points)):
                left = points[idx - 1]
                right = points[idx]
                if distance <= right[0]:
                    span = max(right[0] - left[0], 1e-6)
                    ratio = (distance - left[0]) / span
                    ratio = min(max(ratio, 0.0), 1.0)
                    value = left[1] + (right[1] - left[1]) * ratio
                    break
        value *= cfg["scale"]
        if cfg["time_scaled"]:
            value *= time_scale
        clip_range = cfg.get("clip")
        if clip_range is not None:
            value = min(max(value, clip_range[0]), clip_range[1])
        if abs(value) > 1e-9:
            acc.add(cfg.get("label", "distance_gradient"), float(value))

    def _prepare_relative_reward(self, cfg: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not cfg:
            return None
        weights_raw = cfg.get("weights", {})
        if not isinstance(weights_raw, Dict):
            weights_raw = {}
        weights = {str(key).lower(): float(value) for key, value in weights_raw.items()}
        if not any(abs(v) > 0.0 for v in weights.values()):
            return None
        return {
            "weights": weights,
            "preferred_radius": float(cfg.get("preferred_radius", 0.0)),
            "inner_tolerance": float(cfg.get("inner_tolerance", 0.0)),
            "outer_tolerance": float(cfg.get("outer_tolerance", 0.0)),
            "falloff": str(cfg.get("falloff", "linear")),
            "scale": float(cfg.get("scale", 1.0)),
        }

    def _apply_relative_reward(
        self,
        acc: RewardAccumulator,
        ego_obs: Dict[str, Any],
        target_obs: Dict[str, Any],
        anchor_point: Optional[np.ndarray] = None,
    ) -> None:
        cfg = self.relative_reward_cfg
        if not cfg:
            return
        ego_pose = ego_obs.get("pose")
        target_pose = target_obs.get("pose")
        if ego_pose is None or target_pose is None:
            return
        ego_pose = np.asarray(ego_pose, dtype=np.float32)
        target_pose = np.asarray(target_pose, dtype=np.float32)
        if ego_pose.size < 3 or target_pose.size < 2:
            return
        target_xy = anchor_point if anchor_point is not None else target_pose[:2]
        relative_vector = np.asarray(target_xy, dtype=np.float32) - ego_pose[:2]
        ego_heading = float(ego_pose[2])
        apply_relative_sector_reward(
            acc,
            relative_vector=relative_vector,
            ego_heading=ego_heading,
            weights=cfg["weights"],
            preferred_radius=cfg["preferred_radius"],
            inner_tolerance=cfg["inner_tolerance"],
            outer_tolerance=cfg["outer_tolerance"],
            falloff=cfg["falloff"],
            scale=cfg["scale"],
        )

    @staticmethod
    def _extract_pose(obs: Dict[str, Any]) -> Optional[np.ndarray]:
        pose = obs.get("pose")
        if pose is None:
            return None
        try:
            arr = np.asarray(pose, dtype=np.float32)
        except Exception:
            return None
        if arr.size < 3:
            return None
        return arr

    @staticmethod
    def _wrap_angle(value: float) -> float:
        return float(math.atan2(math.sin(value), math.cos(value)))

    def _detect_pressure(
        self,
        ego_obs: Dict[str, Any],
        target_obs: Dict[str, Any],
        *,
        anchor_point: Optional[np.ndarray] = None,
    ) -> bool:
        if self.pressure_distance <= 0.0:
            return False

        ego_pose = self._extract_pose(ego_obs)
        target_pose = self._extract_pose(target_obs)
        if ego_pose is None or target_pose is None:
            return False

        anchor_xy = anchor_point if anchor_point is not None else target_pose[:2]
        displacement = np.asarray(anchor_xy, dtype=np.float32) - ego_pose[:2]
        distance = float(np.linalg.norm(displacement))
        if not np.isfinite(distance) or distance > self.pressure_distance:
            return False

        if self.pressure_min_speed > 0.0:
            ego_speed = self._extract_speed(ego_obs)
            if ego_speed is None or ego_speed < self.pressure_min_speed:
                return False

        if self.pressure_heading_tolerance < math.pi:
            ego_heading = float(ego_pose[2])
            target_heading = float(target_pose[2])
            heading_delta = abs(self._wrap_angle(ego_heading - target_heading))
            if heading_delta > self.pressure_heading_tolerance:
                return False

        return True

    def _store_pressure(self, agent_id: str, target_id: str, current_time: float, step_index: int) -> None:
        entries = self._pressure_log.setdefault(agent_id, {})
        entries[target_id] = (float(current_time), int(step_index))

    def _clear_pressure(self, agent_id: str, target_id: str) -> None:
        entries = self._pressure_log.get(agent_id)
        if not entries:
            return
        entries.pop(target_id, None)
        if not entries:
            self._pressure_log.pop(agent_id, None)

    def _has_recent_pressure(
        self,
        agent_id: str,
        target_id: str,
        current_time: float,
        step_index: int,
        timestep: float,
    ) -> bool:
        if self.pressure_timeout <= 0.0:
            return True
        entries = self._pressure_log.get(agent_id)
        if not entries:
            return False
        record = entries.get(target_id)
        if record is None:
            return False

        last_time, last_step = record

        if np.isfinite(current_time) and np.isfinite(last_time):
            delta = float(current_time) - float(last_time)
            if delta >= 0.0 and delta <= self.pressure_timeout:
                return True

        if timestep <= 0.0:
            return False
        delta_steps = max(int(step_index) - int(last_step), 0)
        delta_time = float(delta_steps) * float(timestep)
        return delta_time <= self.pressure_timeout


def _normalise_role_members(roster: Any) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    if roster is None:
        return mapping

    raw_roles = getattr(roster, "roles", None)
    if isinstance(raw_roles, dict):
        for role, members in raw_roles.items():
            if isinstance(members, (list, tuple, set)):
                normalised = [str(member) for member in members]
            elif members is None:
                normalised = []
            else:
                normalised = [str(members)]
            if normalised:
                mapping[str(role)] = normalised
    return mapping


def _extract_agent_roles(roster: Any) -> Dict[str, str]:
    roles: Dict[str, str] = {}
    if roster is None:
        return roles

    assignments = getattr(roster, "assignments", None)
    if isinstance(assignments, Iterable):
        for assignment in assignments:
            agent_id = getattr(assignment, "agent_id", None)
            spec = getattr(assignment, "spec", None)
            role = getattr(spec, "role", None)
            if agent_id and role:
                roles[str(agent_id)] = str(role)
    return roles


def _find_trainable_agent(roster: Any) -> Optional[str]:
    assignments = getattr(roster, "assignments", None)
    if not isinstance(assignments, Iterable):
        return None
    for assignment in assignments:
        spec = getattr(assignment, "spec", None)
        trainable = getattr(spec, "trainable", None)
        if trainable is True:
            agent_id = getattr(assignment, "agent_id", None)
            if agent_id:
                return str(agent_id)
    for assignment in assignments:
        agent_id = getattr(assignment, "agent_id", None)
        if agent_id:
            return str(agent_id)
    return None


def _build_gaplock_target_resolver(context: RewardRuntimeContext) -> Optional[Callable[[str], Optional[str]]]:
    roster = getattr(context, "roster", None)
    role_members = _normalise_role_members(roster)
    agent_roles = _extract_agent_roles(roster)

    if not role_members and not agent_roles:
        return None

    attackers = role_members.get("attacker", [])
    defenders = role_members.get("defender", [])

    def resolver(agent_id: str) -> Optional[str]:
        role = agent_roles.get(agent_id)
        if role == "defender":
            for candidate in attackers:
                if candidate != agent_id:
                    return candidate
        else:
            for candidate in defenders:
                if candidate != agent_id:
                    return candidate

        for candidates in role_members.values():
            for candidate in candidates:
                if candidate != agent_id:
                    return candidate

        for candidate in agent_roles.keys():
            if candidate != agent_id:
                return candidate
        return None

    return resolver


def _build_gaplock_strategy(
    context: RewardRuntimeContext,
    config: RewardTaskConfig,
    registry: RewardTaskRegistry,
) -> RewardStrategy:
    params = dict(config.get("params", {}))
    if "target_resolver" not in params:
        resolver = _build_gaplock_target_resolver(context)
        if resolver is not None:
            params["target_resolver"] = resolver
    if "reward_ring_focus_agent" not in params:
        focus_agent = _find_trainable_agent(getattr(context, "roster", None))
        if focus_agent:
            params["reward_ring_focus_agent"] = focus_agent
    if "reward_ring_callback" not in params and hasattr(context.env, "update_reward_ring_target"):
        params["reward_ring_callback"] = context.env.update_reward_ring_target
    if "reward_ring_state_callback" not in params and hasattr(context.env, "update_reward_ring_markers"):
        params["reward_ring_state_callback"] = context.env.update_reward_ring_markers

    strategy = GaplockRewardStrategy(**params)

    if hasattr(context.env, "configure_reward_ring"):
        overlay_cfg = strategy.relative_reward_cfg
        offsets_cfg = getattr(strategy, "_target_offsets_config", [])
        if overlay_cfg or offsets_cfg:
            payload: Dict[str, Any] = {
                "preferred_radius": overlay_cfg.get("preferred_radius", 0.0) if overlay_cfg else 0.0,
                "inner_tolerance": overlay_cfg.get("inner_tolerance", 0.0) if overlay_cfg else 0.0,
                "outer_tolerance": overlay_cfg.get("outer_tolerance", 0.0) if overlay_cfg else 0.0,
                "segments": 96,
            }
            if overlay_cfg:
                payload["falloff"] = overlay_cfg.get("falloff", "linear")
                if "weights" in overlay_cfg:
                    payload["weights"] = overlay_cfg.get("weights")
            if offsets_cfg:
                payload["offsets"] = offsets_cfg
                payload["marker_radius"] = getattr(strategy, "target_offset_marker_radius", 0.0)
                payload["marker_segments"] = getattr(strategy, "target_offset_marker_segments", 12)
                # Only hide the ring when no radial overlay is configured
                payload["offsets_only"] = not bool(overlay_cfg)
            focus_agent = getattr(strategy, "_reward_ring_focus_agent", None)
            context.env.configure_reward_ring(payload, agent_id=focus_agent)
        else:
            context.env.configure_reward_ring(None)

        return strategy


register_reward_task(
    RewardTaskSpec(
        name="gaplock",
        factory=_build_gaplock_strategy,
        aliases=("basic", "sparse", "pursuit", "adversarial"),
        param_keys=GAPLOCK_PARAM_KEYS,
    )
)


__all__ = ["GaplockRewardStrategy", "GAPLOCK_PARAM_KEYS"]
