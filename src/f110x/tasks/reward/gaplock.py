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

    @staticmethod
    def _coerce_positive_float(value: Optional[Any]) -> Optional[float]:
        if value in (None, ""):
            return None
        try:
            val = float(value)
        except (TypeError, ValueError):
            return None
        return val if val > 0.0 else None

    def reset(self, episode_index: int) -> None:
        self._success_awarded.clear()
        self._idle_counters.clear()
        self._idle_penalty_applied.clear()
        self._idle_truncation_applied.clear()
        self._pressure_log.clear()
        self._pressure_bonus_counters.clear()

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

    def _apply_idle_penalty(self, acc: RewardAccumulator, agent_id: str, speed: Optional[float]) -> None:
        if speed is None or self.idle_speed_threshold <= 0.0:
            self._idle_counters.pop(agent_id, None)
            return

        if speed < self.idle_speed_threshold:
            counter = self._idle_counters.get(agent_id, 0) + 1
            self._idle_counters[agent_id] = counter
            threshold_steps = max(1, self.idle_penalty_steps)
            if self.idle_penalty and counter >= threshold_steps:
                acc.add("idle_penalty", float(self.idle_penalty))
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

        speed = self._extract_speed(ego_obs)
        self._apply_idle_penalty(acc, step.agent_id, speed)

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
        if target_obs is not None:
            overlay_target_id = explicit_target_id or str(target_obs.get("agent_id", "target"))
            if overlay_target_id:
                if self._detect_pressure(ego_obs, target_obs):
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

            if pressure_recent and self.pressure_bonus > 0.0:
                counter = self._pressure_bonus_counters.get(step.agent_id, 0) + 1
                if counter >= self.pressure_bonus_interval:
                    acc.add("pressure_bonus", self.pressure_bonus)
                    counter = 0
                self._pressure_bonus_counters[step.agent_id] = counter
            else:
                self._pressure_bonus_counters.pop(step.agent_id, None)

            if self.proximity_penalty_distance > 0.0 and self.proximity_penalty_value:
                ego_pose = self._extract_pose(ego_obs)
                target_pose = self._extract_pose(target_obs)
                if ego_pose is not None and target_pose is not None:
                    distance = float(np.linalg.norm(target_pose[:2] - ego_pose[:2]))
                    if np.isfinite(distance) and distance > self.proximity_penalty_distance:
                        acc.add("proximity_penalty", -abs(self.proximity_penalty_value))

            target_crashed = bool(target_obs.get("collision", False))
            if target_crashed and not ego_crashed:
                if overlay_target_id and pressure_recent:
                    if not self.success_once or not self._has_awarded(step.agent_id, overlay_target_id):
                        acc.add("success_reward", self.target_crash_reward)
                        self._clear_pressure(step.agent_id, overlay_target_id)

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
            self._apply_relative_reward(acc, ego_obs, target_obs)

        self._notify_reward_ring(step.agent_id, overlay_target_id)

        total, components = apply_reward_scaling(acc.total, acc.components, self.scaling_params)
        return total, components

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
        relative_vector = target_pose[:2] - ego_pose[:2]
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

    def _detect_pressure(self, ego_obs: Dict[str, Any], target_obs: Dict[str, Any]) -> bool:
        if self.pressure_distance <= 0.0:
            return False

        ego_pose = self._extract_pose(ego_obs)
        target_pose = self._extract_pose(target_obs)
        if ego_pose is None or target_pose is None:
            return False

        displacement = target_pose[:2] - ego_pose[:2]
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

    strategy = GaplockRewardStrategy(**params)

    if hasattr(context.env, "configure_reward_ring"):
        if strategy.relative_reward_cfg:
            overlay_cfg = strategy.relative_reward_cfg
            payload: Dict[str, Any] = {
                "preferred_radius": overlay_cfg.get("preferred_radius", 0.0),
                "inner_tolerance": overlay_cfg.get("inner_tolerance", 0.0),
                "outer_tolerance": overlay_cfg.get("outer_tolerance", 0.0),
                "segments": 96,
            }
            payload["falloff"] = overlay_cfg.get("falloff", "linear")
            if "weights" in overlay_cfg:
                payload["weights"] = overlay_cfg.get("weights")
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
