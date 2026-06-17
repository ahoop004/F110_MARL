"""Defensive MPC fixed-policy controller."""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from agents.mpc.costs import trajectory_cost
from agents.mpc.kinematic import KinematicMPCAgent, _extract_pose
from agents.mpc.obstacles import (
    ObstacleAwareMPCAgent,
    extract_lidar_scan,
    lidar_points_world,
    obstacle_proximity_cost,
)
from agents.mpc.rollout import rollout_kinematic_bicycle


_DEFAULTS = {
    "base_controller": "obstacle_aware_mpc",
    "defend_distance": 2.5,
    "lateral_block_offset": 0.4,
    "max_block_offset": 0.7,
    "defensive_weight": 1.0,
    "safety_distance": 0.8,
    "target_id": None,
    "allow_blocking": True,
}


class DefensiveMPCAgent(ObstacleAwareMPCAgent):
    """Racing MPC with a conservative defensive line bias.

    Defensive behavior is activated only when ``obs["target_pose"]`` is
    available and the target is behind and close.  The target contributes only
    cost and safety constraints; this policy never rewards or seeks target
    crashes.
    """

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        params = dict(config.get("params", config))
        cfg = {**_DEFAULTS, **params}
        requested_base = str(cfg["base_controller"]).strip().lower()
        self.base_controller = (
            requested_base
            if requested_base in {"kinematic_mpc", "obstacle_aware_mpc"}
            else "obstacle_aware_mpc"
        )
        self.defend_distance = max(0.0, float(cfg["defend_distance"]))
        self.lateral_block_offset = float(cfg["lateral_block_offset"])
        self.max_block_offset = max(0.0, float(cfg["max_block_offset"]))
        self.defensive_weight = max(0.0, float(cfg["defensive_weight"]))
        self.safety_distance = max(0.0, float(cfg["safety_distance"]))
        self.target_id = cfg.get("target_id")
        self.allow_blocking = bool(cfg["allow_blocking"])

    def act(
        self,
        obs: Dict[str, Any],
        deterministic: bool = False,
        aid=None,
    ) -> np.ndarray:
        pose = _extract_pose(obs)
        target_pose = extract_target_pose(obs)
        centerline = self._local_centerline(pose)
        if (
            pose is None
            or target_pose is None
            or centerline is None
            or not defensive_bias_active(
                pose,
                target_pose,
                defend_distance=self.defend_distance,
                allow_blocking=self.allow_blocking,
            )
        ):
            return self._base_action(obs, deterministic=deterministic, aid=aid)

        target_speed = self._target_speed(obs)
        sequences = self._candidate_sequences.copy()
        obstacle_points = np.zeros((0, 2), dtype=np.float32)
        speed_cap = self.max_speed
        if self.base_controller == "obstacle_aware_mpc" and self.use_lidar:
            scan = extract_lidar_scan(obs)
            if scan is not None:
                obstacle_points = lidar_points_world(
                    scan,
                    pose,
                    lidar_fov=self.lidar_fov,
                    max_range=self.slow_distance,
                    max_points=self.max_obstacle_points,
                )
                if obstacle_points.shape[0] > 0:
                    speed_cap = self._obstacle_speed_cap(scan)
                    target_speed = min(target_speed, speed_cap)
                    sequences[:, :, 1] = np.clip(sequences[:, :, 1], self.min_speed, speed_cap)

        defensive_path = build_defensive_centerline(
            centerline,
            ego_pose=pose,
            target_pose=target_pose,
            lateral_block_offset=self.lateral_block_offset,
            max_block_offset=self.max_block_offset,
        )
        action = self._select_defensive_action(
            pose,
            target_pose,
            defensive_path,
            obstacle_points,
            target_speed,
            sequences,
        )
        action = self._clip_action(action)
        action[1] = float(np.clip(action[1], self.min_speed, speed_cap))
        self._previous_action = action.copy()
        return action

    def _base_action(self, obs: Dict[str, Any], *, deterministic: bool = False, aid=None) -> np.ndarray:
        if self.base_controller == "kinematic_mpc":
            return KinematicMPCAgent.act(self, obs, deterministic=deterministic, aid=aid)
        return ObstacleAwareMPCAgent.act(self, obs, deterministic=deterministic, aid=aid)

    def _select_defensive_action(
        self,
        pose: np.ndarray,
        target_pose: np.ndarray,
        defensive_path: np.ndarray,
        obstacle_points: np.ndarray,
        target_speed: float,
        sequences: np.ndarray,
    ) -> np.ndarray:
        best_cost = float("inf")
        best_action = self._fallback_action
        for sequence in sequences:
            trajectory = rollout_kinematic_bicycle(
                pose,
                sequence,
                dt=self.dt,
                wheelbase=self.wheelbase,
                horizon=sequence.shape[0],
            )
            cost = trajectory_cost(
                trajectory,
                sequence,
                centerline=defensive_path,
                target_speed=target_speed,
                weights=self._weights,
            )
            cost += self.defensive_weight * target_safety_cost(
                trajectory,
                target_pose,
                safety_distance=self.safety_distance,
            )
            if obstacle_points.shape[0] > 0:
                cost += self.obstacle_weight * obstacle_proximity_cost(
                    trajectory,
                    obstacle_points,
                    safe_distance=self.safe_distance,
                    hard_stop_distance=self.hard_stop_distance,
                )
            if self._previous_action is not None:
                first = sequence[0]
                speed_delta = float(first[1] - self._previous_action[1])
                steer_delta = float(first[0] - self._previous_action[0])
                cost += self._speed_smoothness_weight * (
                    speed_delta * speed_delta + steer_delta * steer_delta
                )
            if cost < best_cost:
                best_cost = cost
                best_action = sequence[0]
        return np.asarray(best_action, dtype=np.float32)


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


def defensive_bias_active(
    ego_pose: np.ndarray,
    target_pose: np.ndarray,
    *,
    defend_distance: float,
    allow_blocking: bool = True,
) -> bool:
    if not allow_blocking:
        return False
    ego = np.asarray(ego_pose, dtype=np.float32).reshape(-1)
    target = np.asarray(target_pose, dtype=np.float32).reshape(-1)
    if ego.size < 3 or target.size < 2:
        return False
    relative = target[:2] - ego[:2]
    distance = float(np.linalg.norm(relative))
    if not np.isfinite(distance) or distance > max(0.0, float(defend_distance)):
        return False
    forward = np.array([np.cos(float(ego[2])), np.sin(float(ego[2]))], dtype=np.float32)
    return float(np.dot(relative, forward)) < 0.0


def build_defensive_centerline(
    centerline: np.ndarray,
    *,
    ego_pose: np.ndarray,
    target_pose: np.ndarray,
    lateral_block_offset: float,
    max_block_offset: float,
) -> np.ndarray:
    path = np.asarray(centerline, dtype=np.float32)
    if path.ndim != 2 or path.shape[0] < 2 or path.shape[1] < 2:
        return np.zeros((0, 2), dtype=np.float32)
    offset = defensive_lateral_offset(
        path,
        ego_pose=ego_pose,
        target_pose=target_pose,
        lateral_block_offset=lateral_block_offset,
        max_block_offset=max_block_offset,
    )
    normals = _centerline_normals(path[:, :2])
    return (path[:, :2] + normals * offset).astype(np.float32)


def defensive_lateral_offset(
    centerline: np.ndarray,
    *,
    ego_pose: np.ndarray,
    target_pose: np.ndarray,
    lateral_block_offset: float,
    max_block_offset: float,
) -> float:
    path = np.asarray(centerline, dtype=np.float32)
    if path.ndim != 2 or path.shape[0] < 2:
        return 0.0
    deltas = path[:, :2] - np.asarray(ego_pose, dtype=np.float32)[:2]
    nearest = int(np.argmin(np.sum(deltas * deltas, axis=1)))
    tangent = _path_tangent(path[:, :2], nearest)
    normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
    target_rel = np.asarray(target_pose, dtype=np.float32)[:2] - np.asarray(ego_pose, dtype=np.float32)[:2]
    side = float(np.sign(np.dot(target_rel, normal)))
    if side == 0.0:
        side = 1.0
    raw_offset = side * float(lateral_block_offset)
    return float(np.clip(raw_offset, -max_block_offset, max_block_offset))


def target_safety_cost(
    trajectory: np.ndarray,
    target_pose: np.ndarray,
    *,
    safety_distance: float,
) -> float:
    traj = np.asarray(trajectory, dtype=np.float32)
    target = np.asarray(target_pose, dtype=np.float32).reshape(-1)
    if traj.ndim != 2 or traj.shape[0] < 2 or target.size < 2:
        return 0.0
    points = traj[1:, :2]
    distances = np.linalg.norm(points - target[:2], axis=1)
    margin = np.maximum(0.0, float(safety_distance) - distances)
    if margin.size == 0:
        return 0.0
    return float(100.0 * np.mean(margin * margin))


def _centerline_normals(path: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(path, dtype=np.float32)
    for idx in range(path.shape[0]):
        tangent = _path_tangent(path, idx)
        normals[idx] = np.array([-tangent[1], tangent[0]], dtype=np.float32)
    return normals


def _path_tangent(path: np.ndarray, index: int) -> np.ndarray:
    if path.shape[0] < 2:
        return np.array([1.0, 0.0], dtype=np.float32)
    if index >= path.shape[0] - 1:
        delta = path[index] - path[index - 1]
    else:
        delta = path[index + 1] - path[index]
    norm = float(np.linalg.norm(delta))
    if norm <= 1e-6 or not np.isfinite(norm):
        return np.array([1.0, 0.0], dtype=np.float32)
    return (delta / norm).astype(np.float32)


__all__ = [
    "DefensiveMPCAgent",
    "build_defensive_centerline",
    "defensive_bias_active",
    "defensive_lateral_offset",
    "extract_target_pose",
    "target_safety_cost",
]
