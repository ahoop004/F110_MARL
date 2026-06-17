"""Sample-based MPCC fixed-policy controller."""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from agents.mpc.base import MPCConfig, generate_action_sequences
from agents.mpc.costs import (
    MPCCWeights,
    control_effort_cost,
    mpcc_geometry_cost,
    steering_smoothness_cost,
    target_speed_cost,
)
from agents.mpc.rollout import rollout_kinematic_bicycle
from agents.mpc.track_geometry import CenterlineGeometry, prepare_centerline_geometry


_DEFAULTS = {
    "horizon": 12,
    "dt": 0.1,
    "target_speed": 3.0,
    "min_speed": 0.5,
    "max_speed": 3.0,
    "max_steer": 0.42,
    "wheelbase": 0.3302,
    "num_steer_samples": 9,
    "num_speed_samples": 5,
    "contour_weight": 4.0,
    "lag_weight": 0.5,
    "heading_weight": 0.5,
    "progress_weight": 2.0,
    "speed_weight": 0.3,
    "control_weight": 0.05,
    "smoothness_weight": 0.1,
    "track_boundary_weight": 0.0,
    "max_candidates": 256,
    "search_window": 100,
    "fallback_speed": 0.5,
    "closed_centerline": None,
}


class MPCCAgent:
    """Dependency-free, sample-based MPCC-style fixed-policy agent.

    This is not a nonlinear optimizer.  It rolls out short candidate
    ``[steering, speed]`` sequences with a kinematic bicycle model and selects
    the one with the lowest centerline contouring/progress cost.
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
        self.search_window = max(1, int(cfg["search_window"]))
        self.fallback_speed = float(cfg["fallback_speed"])
        self.track_boundary_weight = float(cfg["track_boundary_weight"])
        self.closed_centerline = cfg.get("closed_centerline")

        max_candidates = cfg.get("max_candidate_sequences", cfg.get("max_candidates", 256))
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
            max_candidate_sequences=max(1, int(max_candidates)),
        )
        self._mpcc_weights = MPCCWeights(
            contouring=float(cfg["contour_weight"]),
            lag=float(cfg["lag_weight"]),
            heading=float(cfg["heading_weight"]),
            progress=float(cfg["progress_weight"]),
        )
        self._speed_weight = float(cfg["speed_weight"])
        self._control_weight = float(cfg["control_weight"])
        self._smoothness_weight = float(cfg["smoothness_weight"])

        self._env = None
        self._previous_action: Optional[np.ndarray] = None
        self._centerline_cache_id: Optional[int] = None
        self._geometry_cache: Optional[CenterlineGeometry] = None
        self._candidate_sequences = generate_action_sequences(self._config)
        self._fallback_action = self._make_fallback_action()

    def set_env(self, env) -> None:
        self._env = env
        self._centerline_cache_id = None
        self._geometry_cache = None

    def reset(self) -> None:
        self._previous_action = None

    def act(
        self,
        obs: Dict[str, Any],
        deterministic: bool = False,  # noqa: ARG002
        aid=None,  # noqa: ANN001, ARG002
    ) -> np.ndarray:
        pose = _extract_pose(obs)
        geometry = self._centerline_geometry()
        if pose is None or geometry is None or not geometry.valid:
            return self._fallback_action.copy()

        target_speed = self._target_speed(obs)
        current_speed = _extract_speed(obs)
        best_action = self._select_action(pose, geometry, target_speed, current_speed)
        action = self._clip_action(best_action)
        self._previous_action = action.copy()
        return action

    def store(self, *args, **kwargs) -> None:
        return None

    def finish_path(self, **kwargs) -> None:
        return None

    def update(self) -> None:
        return None

    def _select_action(
        self,
        pose: np.ndarray,
        geometry: CenterlineGeometry,
        target_speed: float,
        current_speed: Optional[float],
    ) -> np.ndarray:
        best_cost = float("inf")
        best_action = self._fallback_action

        for sequence in self._candidate_sequences:
            trajectory = rollout_kinematic_bicycle(
                pose,
                sequence,
                dt=self.dt,
                wheelbase=self.wheelbase,
                horizon=self.horizon,
            )
            cost = self._sequence_cost(
                trajectory,
                sequence,
                geometry,
                target_speed,
                current_speed,
            )
            if cost < best_cost:
                best_cost = cost
                best_action = sequence[0]

        return np.asarray(best_action, dtype=np.float32)

    def _sequence_cost(
        self,
        trajectory: np.ndarray,
        sequence: np.ndarray,
        geometry: CenterlineGeometry,
        target_speed: float,
        current_speed: Optional[float],
    ) -> float:
        cost = mpcc_geometry_cost(trajectory, geometry, weights=self._mpcc_weights)
        cost += self._speed_weight * target_speed_cost(sequence, target_speed)
        cost += self._control_weight * control_effort_cost(sequence)
        cost += self._smoothness_weight * steering_smoothness_cost(sequence)

        if sequence.shape[0] > 1:
            speed_deltas = np.diff(sequence[:, 1])
            cost += self._smoothness_weight * float(np.mean(speed_deltas * speed_deltas))

        previous = self._previous_action
        if previous is None and current_speed is not None:
            previous = np.array([0.0, current_speed], dtype=np.float32)
        if previous is not None:
            first_delta = sequence[0] - previous
            cost += self._smoothness_weight * float(np.dot(first_delta, first_delta))

        # Boundary facts are not part of the current MPC utility contract.  The
        # weight is accepted for scenario compatibility and remains a no-op
        # until wall/track-boundary geometry is exposed.
        _ = self.track_boundary_weight
        return float(cost)

    def _centerline_geometry(self) -> Optional[CenterlineGeometry]:
        if self._env is None:
            return None
        centerline = getattr(self._env, "centerline_points", None)
        if centerline is None:
            self._centerline_cache_id = None
            self._geometry_cache = None
            return None

        cache_id = id(centerline)
        if cache_id == self._centerline_cache_id and self._geometry_cache is not None:
            return self._geometry_cache

        geometry = prepare_centerline_geometry(
            centerline,
            closed=self.closed_centerline if self.closed_centerline is not None else None,
        )
        self._centerline_cache_id = cache_id
        self._geometry_cache = geometry
        return geometry if geometry.valid else None

    def _target_speed(self, obs: Dict[str, Any]) -> float:
        speed = float(np.clip(self.target_speed, self.min_speed, self.max_speed))
        current = _extract_speed(obs)
        if current is not None and np.isfinite(current):
            speed = max(self.min_speed, min(speed, current + 1.0))
        return float(np.clip(speed, self.min_speed, self.max_speed))

    def _clip_action(self, action: np.ndarray) -> np.ndarray:
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        result = self._fallback_action.copy()
        if arr.size:
            result[: min(2, arr.size)] = arr[: min(2, arr.size)]
        result[0] = float(np.clip(result[0], -self.max_steer, self.max_steer))
        result[1] = float(np.clip(result[1], self.min_speed, self.max_speed))
        return result.astype(np.float32, copy=False)

    def _make_fallback_action(self) -> np.ndarray:
        speed = float(np.clip(self.fallback_speed, self.min_speed, self.max_speed))
        return np.array([0.0, speed], dtype=np.float32)


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


def _extract_speed(obs: Dict[str, Any]) -> Optional[float]:
    if not isinstance(obs, dict) or "velocity" not in obs:
        return None
    arr = np.asarray(obs.get("velocity"), dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return None
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    speed = float(np.linalg.norm(arr))
    return speed if np.isfinite(speed) else None


__all__ = ["MPCCAgent"]
