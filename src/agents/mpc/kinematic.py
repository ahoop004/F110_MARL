"""AgentFactory-compatible kinematic MPC controller."""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from agents.mpc.base import MPCConfig, evaluate_action_sequences, generate_action_sequences
from agents.mpc.costs import CostWeights


_DEFAULTS = {
    "horizon": 10,
    "dt": 0.1,
    "target_speed": 2.5,
    "min_speed": 0.5,
    "max_speed": 3.0,
    "max_steer": 0.42,
    "wheelbase": 0.3302,
    "num_steer_samples": 7,
    "num_speed_samples": 5,
    "max_candidate_sequences": 256,
    "path_weight": 2.0,
    "heading_weight": 0.5,
    "speed_weight": 0.5,
    "control_weight": 0.05,
    "smoothness_weight": 0.1,
    "progress_weight": 1.0,
    "search_window": 80,
}


class KinematicMPCAgent:
    """Short-horizon centerline-tracking MPC fixed-policy agent.

    The controller consumes observations with ``pose`` and optional
    ``velocity`` entries, reads ``env.centerline_points`` when an environment
    is injected, and returns physical actions in the repository's environment
    convention: ``[steering, speed]``.
    """

    def __init__(self, config: dict) -> None:
        params = dict(config.get("params", config))
        cfg = {**_DEFAULTS, **params}

        self.horizon = max(1, int(cfg["horizon"]))
        self.dt = max(float(cfg["dt"]), 1e-6)
        self.target_speed = float(cfg["target_speed"])
        self.min_speed = float(cfg["min_speed"])
        self.max_speed = float(cfg["max_speed"])
        self.max_steer = abs(float(cfg["max_steer"]))
        self.wheelbase = max(float(cfg["wheelbase"]), 1e-6)
        self.search_window = max(2, int(cfg["search_window"]))

        self._config = MPCConfig(
            horizon=self.horizon,
            dt=self.dt,
            wheelbase=self.wheelbase,
            steering_min=-self.max_steer,
            steering_max=self.max_steer,
            speed_min=self.min_speed,
            speed_max=self.max_speed,
            steering_samples=max(1, int(cfg["num_steer_samples"])),
            speed_samples=max(1, int(cfg["num_speed_samples"])),
            max_candidate_sequences=max(1, int(cfg["max_candidate_sequences"])),
        )
        self._weights = CostWeights(
            path_tracking=float(cfg["path_weight"]),
            heading_error=float(cfg["heading_weight"]),
            target_speed=float(cfg["speed_weight"]),
            control_effort=float(cfg["control_weight"]),
            steering_smoothness=float(cfg["smoothness_weight"]),
            progress=float(cfg["progress_weight"]),
        )
        self._speed_smoothness_weight = float(cfg["smoothness_weight"])
        self._env = None
        self._previous_action: Optional[np.ndarray] = None
        self._fallback_action = np.array([0.0, self.min_speed], dtype=np.float32)
        self._candidate_sequences = generate_action_sequences(self._config)

    def set_env(self, env) -> None:
        self._env = env

    def reset(self) -> None:
        self._previous_action = None

    def act(
        self,
        obs: Dict[str, Any],
        deterministic: bool = False,  # noqa: ARG002
        aid=None,  # noqa: ANN001, ARG002
    ) -> np.ndarray:
        pose = _extract_pose(obs)
        centerline = self._local_centerline(pose)
        if pose is None or centerline is None:
            return self._fallback_action.copy()

        target_speed = self._target_speed(obs)
        result = evaluate_action_sequences(
            pose,
            self._candidate_sequences,
            centerline=centerline,
            target_speed=target_speed,
            config=self._config,
            weights=self._weights,
        )
        action = result.first_action

        if self._previous_action is not None and self._candidate_sequences.shape[0] > 1:
            action = self._select_with_previous_action_smoothness(
                pose,
                centerline,
                target_speed,
            )

        action = self._clip_action(action)
        self._previous_action = action.copy()
        return action

    def store(self, *args, **kwargs) -> None:
        return None

    def finish_path(self, **kwargs) -> None:
        return None

    def update(self) -> None:
        return None

    def _select_with_previous_action_smoothness(
        self,
        pose: np.ndarray,
        centerline: np.ndarray,
        target_speed: float,
    ) -> np.ndarray:
        assert self._previous_action is not None
        best_cost = float("inf")
        best_action = self._fallback_action
        for sequence in self._candidate_sequences:
            result = evaluate_action_sequences(
                pose,
                sequence.reshape(1, sequence.shape[0], sequence.shape[1]),
                centerline=centerline,
                target_speed=target_speed,
                config=self._config,
                weights=self._weights,
            )
            first = result.first_action
            speed_delta = float(first[1] - self._previous_action[1])
            steer_delta = float(first[0] - self._previous_action[0])
            cost = result.cost + self._speed_smoothness_weight * (
                speed_delta * speed_delta + steer_delta * steer_delta
            )
            if cost < best_cost:
                best_cost = cost
                best_action = first
        return best_action.astype(np.float32, copy=True)

    def _target_speed(self, obs: Dict[str, Any]) -> float:
        speed = self.target_speed
        velocity = _extract_velocity(obs)
        if velocity is not None and velocity.size > 0:
            current = float(np.linalg.norm(velocity))
            if np.isfinite(current):
                # Keep the requested target as the primary objective while
                # avoiding abrupt jumps from a standing or slow start.
                speed = max(self.min_speed, min(self.target_speed, current + 1.0))
        return float(np.clip(speed, self.min_speed, self.max_speed))

    def _clip_action(self, action: np.ndarray) -> np.ndarray:
        clipped = np.asarray(action, dtype=np.float32).reshape(-1)
        result = self._fallback_action.copy()
        if clipped.size:
            result[: min(2, clipped.size)] = clipped[: min(2, clipped.size)]
        result[0] = float(np.clip(result[0], -self.max_steer, self.max_steer))
        result[1] = float(np.clip(result[1], self.min_speed, self.max_speed))
        return result

    def _local_centerline(self, pose: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if pose is None or self._env is None:
            return None
        centerline = getattr(self._env, "centerline_points", None)
        path = _normalize_centerline(centerline)
        if path is None:
            return None

        deltas = path[:, :2] - pose[:2]
        nearest = int(np.argmin(np.sum(deltas * deltas, axis=1)))
        end = min(path.shape[0], nearest + self.search_window)
        local = path[nearest:end]
        if local.shape[0] < 2 and path.shape[0] >= 2:
            start = max(0, path.shape[0] - self.search_window)
            local = path[start:]
        return local if local.shape[0] >= 2 else None


def _extract_pose(obs: Dict[str, Any]) -> Optional[np.ndarray]:
    if not isinstance(obs, dict) or "pose" not in obs:
        return None
    arr = np.asarray(obs.get("pose"), dtype=np.float32).reshape(-1)
    if arr.size < 3:
        return None
    pose = arr[:3].copy()
    if not np.isfinite(pose).all():
        return None
    return pose


def _extract_velocity(obs: Dict[str, Any]) -> Optional[np.ndarray]:
    if not isinstance(obs, dict) or "velocity" not in obs:
        return None
    arr = np.asarray(obs.get("velocity"), dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return None
    arr = arr[np.isfinite(arr)]
    return arr if arr.size else None


def _normalize_centerline(centerline: Any) -> Optional[np.ndarray]:
    if centerline is None:
        return None
    arr = np.asarray(centerline, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
        return None
    path = arr[:, :2].copy()
    mask = np.isfinite(path).all(axis=1)
    path = path[mask]
    return path if path.shape[0] >= 2 else None


__all__ = ["KinematicMPCAgent"]
