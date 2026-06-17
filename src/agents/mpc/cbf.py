"""Soft-barrier CBF-MPC fixed-policy controller."""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from agents.mpc.mpcc import MPCCAgent, _extract_pose, _extract_speed
from agents.mpc.obstacles import extract_lidar_scan, lidar_points_world
from agents.mpc.rollout import rollout_kinematic_bicycle
from agents.mpc.track_geometry import CenterlineGeometry


_DEFAULTS = {
    "base_controller": "mpcc",
    "barrier_weight": 8.0,
    "lidar_barrier_weight": None,
    "opponent_barrier_weight": 6.0,
    "wall_barrier_weight": 0.0,
    "safe_distance": 0.8,
    "hard_stop_distance": 0.35,
    "slow_distance": 1.5,
    "min_barrier_speed": 0.5,
    "hard_violation_penalty": 100.0,
    "use_lidar": True,
    "use_target_pose": True,
    "use_global_state": False,
    "target_id": None,
    "lidar_fov": 4.71238898,
    "max_obstacle_points": 96,
}


class CBFMPCAgent(MPCCAgent):
    """MPCC-style controller with soft control-barrier safety penalties.

    This controller is a fixed policy and uses only factual geometry from
    observations/env state.  Safety terms increase candidate cost and cap speed
    near unsafe regions; they never reward crashes or opponent failures.
    """

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        params = dict(config.get("params", config))
        cfg = {**_DEFAULTS, **params}

        self.base_controller = str(cfg["base_controller"]).strip().lower()
        self.barrier_weight = max(0.0, float(cfg["barrier_weight"]))
        lidar_weight = cfg["lidar_barrier_weight"]
        self.lidar_barrier_weight = (
            self.barrier_weight if lidar_weight is None else max(0.0, float(lidar_weight))
        )
        self.opponent_barrier_weight = max(0.0, float(cfg["opponent_barrier_weight"]))
        self.wall_barrier_weight = max(0.0, float(cfg["wall_barrier_weight"]))
        self.safe_distance = max(0.0, float(cfg["safe_distance"]))
        self.hard_stop_distance = max(0.0, float(cfg["hard_stop_distance"]))
        self.slow_distance = max(self.hard_stop_distance, float(cfg["slow_distance"]))
        self.min_barrier_speed = float(np.clip(
            float(cfg["min_barrier_speed"]),
            self.min_speed,
            self.max_speed,
        ))
        self.hard_violation_penalty = max(0.0, float(cfg["hard_violation_penalty"]))
        self.use_lidar = bool(cfg["use_lidar"])
        self.use_target_pose = bool(cfg["use_target_pose"])
        self.use_global_state = bool(cfg["use_global_state"])
        self.target_id = cfg.get("target_id")
        self.lidar_fov = max(float(cfg["lidar_fov"]), 1e-6)
        self.max_obstacle_points = max(1, int(cfg["max_obstacle_points"]))

    def act(
        self,
        obs: Dict[str, Any],
        deterministic: bool = False,  # noqa: ARG002
        aid=None,  # noqa: ANN001
    ) -> np.ndarray:
        pose = _extract_pose(obs)
        geometry = self._centerline_geometry()
        if pose is None or geometry is None or not geometry.valid:
            return self._fallback_action.copy()

        lidar_points, nearest_lidar = self._lidar_obstacles(obs, pose)
        target_pose = self._target_pose(obs, aid=aid)
        target_points = _target_points(target_pose)
        if lidar_points.shape[0] == 0 and target_points.shape[0] == 0:
            return super().act(obs, deterministic=deterministic, aid=aid)

        target_speed = self._target_speed(obs)
        speed_cap = self._safety_speed_cap(nearest_lidar, target_pose, pose)
        target_speed = min(target_speed, speed_cap)

        sequences = self._candidate_sequences.copy()
        sequences[:, :, 1] = np.clip(sequences[:, :, 1], self.min_speed, speed_cap)

        current_speed = _extract_speed(obs)
        best_action = self._select_cbf_action(
            pose,
            geometry,
            target_speed,
            current_speed,
            sequences,
            lidar_points,
            target_points,
        )
        action = self._clip_action(best_action)
        action[1] = float(np.clip(action[1], self.min_speed, speed_cap))
        self._previous_action = action.copy()
        return action

    def _select_cbf_action(
        self,
        pose: np.ndarray,
        geometry: CenterlineGeometry,
        target_speed: float,
        current_speed: Optional[float],
        sequences: np.ndarray,
        lidar_points: np.ndarray,
        target_points: np.ndarray,
    ) -> np.ndarray:
        best_cost = float("inf")
        best_action = self._fallback_action
        for sequence in sequences:
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
            if lidar_points.shape[0] > 0:
                cost += self.lidar_barrier_weight * soft_barrier_cost(
                    trajectory,
                    lidar_points,
                    safe_distance=self.safe_distance,
                    hard_stop_distance=self.hard_stop_distance,
                    hard_violation_penalty=self.hard_violation_penalty,
                )
            if target_points.shape[0] > 0:
                cost += self.opponent_barrier_weight * soft_barrier_cost(
                    trajectory,
                    target_points,
                    safe_distance=self.safe_distance,
                    hard_stop_distance=self.hard_stop_distance,
                    hard_violation_penalty=self.hard_violation_penalty,
                )

            # Wall/boundary distance facts are not stable yet.  Accept the
            # weight for forward-compatible scenario configs, but keep it a
            # no-op until a public boundary-distance API exists.
            _ = self.wall_barrier_weight

            if cost < best_cost:
                best_cost = cost
                best_action = sequence[0]
        return np.asarray(best_action, dtype=np.float32)

    def _lidar_obstacles(self, obs: Dict[str, Any], pose: np.ndarray) -> tuple[np.ndarray, Optional[float]]:
        if not self.use_lidar:
            return np.zeros((0, 2), dtype=np.float32), None
        scan = extract_lidar_scan(obs)
        if scan is None:
            return np.zeros((0, 2), dtype=np.float32), None
        valid = scan[np.isfinite(scan) & (scan > 0.0)]
        nearest = float(np.min(valid)) if valid.size else None
        points = lidar_points_world(
            scan,
            pose,
            lidar_fov=self.lidar_fov,
            max_range=self.slow_distance,
            max_points=self.max_obstacle_points,
        )
        return points, nearest

    def _target_pose(self, obs: Dict[str, Any], *, aid=None) -> Optional[np.ndarray]:  # noqa: ANN001
        if self.use_target_pose:
            pose = extract_target_pose(obs)
            if pose is not None:
                return pose
        if not self.use_global_state or self._env is None or self.target_id is None:
            return None
        getter = getattr(self._env, "get_agent_state", None)
        if not callable(getter):
            return None
        try:
            state = getter(str(self.target_id))
        except (KeyError, TypeError, ValueError):
            return None
        pose = np.asarray(getattr(state, "pose", None), dtype=np.float32).reshape(-1)
        if pose.size < 3 or not np.isfinite(pose[:3]).all():
            return None
        return pose[:3].copy()

    def _safety_speed_cap(
        self,
        nearest_lidar: Optional[float],
        target_pose: Optional[np.ndarray],
        ego_pose: np.ndarray,
    ) -> float:
        nearest = nearest_lidar
        if target_pose is not None:
            target_dist = float(np.linalg.norm(np.asarray(target_pose, dtype=np.float32)[:2] - ego_pose[:2]))
            if np.isfinite(target_dist):
                nearest = target_dist if nearest is None else min(nearest, target_dist)
        if nearest is None or not np.isfinite(nearest):
            return self.max_speed
        if nearest <= self.hard_stop_distance:
            return self.min_barrier_speed
        if nearest >= self.slow_distance:
            return self.max_speed
        span = max(self.slow_distance - self.hard_stop_distance, 1e-6)
        alpha = (nearest - self.hard_stop_distance) / span
        return float(self.min_barrier_speed + alpha * (self.max_speed - self.min_barrier_speed))


def extract_target_pose(obs: Dict[str, Any]) -> Optional[np.ndarray]:
    if not isinstance(obs, dict) or "target_pose" not in obs:
        return None
    arr = np.asarray(obs.get("target_pose"), dtype=np.float32).reshape(-1)
    if arr.size < 3:
        return None
    pose = arr[:3].copy()
    if not np.isfinite(pose).all():
        return None
    return pose


def soft_barrier_cost(
    trajectory: np.ndarray,
    obstacle_points: np.ndarray,
    *,
    safe_distance: float,
    hard_stop_distance: float,
    hard_violation_penalty: float = 100.0,
) -> float:
    traj = np.asarray(trajectory, dtype=np.float32)
    obstacles = np.asarray(obstacle_points, dtype=np.float32)
    if traj.ndim != 2 or traj.shape[0] < 2 or obstacles.ndim != 2:
        return 0.0
    if traj.shape[1] < 2 or obstacles.shape[0] == 0 or obstacles.shape[1] < 2:
        return 0.0

    points = traj[1:, :2]
    deltas = points[:, None, :] - obstacles[None, :, :2]
    distances = np.sqrt(np.sum(deltas * deltas, axis=2))
    nearest = np.min(distances, axis=1)
    if nearest.size == 0:
        return 0.0

    safe_distance = max(0.0, float(safe_distance))
    hard_stop_distance = max(0.0, float(hard_stop_distance))
    soft_margin = np.maximum(0.0, safe_distance - nearest)
    cost = float(np.mean(soft_margin * soft_margin))

    hard_mask = nearest < hard_stop_distance
    if np.any(hard_mask):
        hard_margin = np.maximum(0.0, hard_stop_distance - nearest[hard_mask])
        cost += float(hard_violation_penalty * np.mean(1.0 + hard_margin * hard_margin))
    return float(cost)


def _target_points(target_pose: Optional[np.ndarray]) -> np.ndarray:
    if target_pose is None:
        return np.zeros((0, 2), dtype=np.float32)
    arr = np.asarray(target_pose, dtype=np.float32).reshape(-1)
    if arr.size < 2 or not np.isfinite(arr[:2]).all():
        return np.zeros((0, 2), dtype=np.float32)
    return arr[:2].reshape(1, 2).astype(np.float32)


__all__ = [
    "CBFMPCAgent",
    "extract_target_pose",
    "soft_barrier_cost",
]
