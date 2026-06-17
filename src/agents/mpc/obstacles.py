"""Obstacle-aware MPC fixed-policy controller."""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from agents.mpc.costs import trajectory_cost
from agents.mpc.kinematic import KinematicMPCAgent, _extract_pose
from agents.mpc.rollout import rollout_kinematic_bicycle


_DEFAULTS = {
    "obstacle_weight": 5.0,
    "safe_distance": 0.8,
    "hard_stop_distance": 0.35,
    "slow_distance": 1.5,
    "min_obstacle_speed": 0.5,
    "use_lidar": True,
    "lidar_fov": 4.71238898,
    "max_obstacle_points": 96,
}


class ObstacleAwareMPCAgent(KinematicMPCAgent):
    """Kinematic MPC with LiDAR obstacle proximity costs.

    This controller remains a fixed policy.  It does not use reward signals and
    does not assign value to opponent crashes; obstacle data only increases
    cost and caps speed near nearby scan returns.
    """

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        params = dict(config.get("params", config))
        cfg = {**_DEFAULTS, **params}
        self.obstacle_weight = float(cfg["obstacle_weight"])
        self.safe_distance = max(0.0, float(cfg["safe_distance"]))
        self.hard_stop_distance = max(0.0, float(cfg["hard_stop_distance"]))
        self.slow_distance = max(self.hard_stop_distance, float(cfg["slow_distance"]))
        self.min_obstacle_speed = float(np.clip(
            float(cfg["min_obstacle_speed"]),
            self.min_speed,
            self.max_speed,
        ))
        self.use_lidar = bool(cfg["use_lidar"])
        self.lidar_fov = max(float(cfg["lidar_fov"]), 1e-6)
        self.max_obstacle_points = max(1, int(cfg["max_obstacle_points"]))

    def act(
        self,
        obs: Dict[str, Any],
        deterministic: bool = False,
        aid=None,
    ) -> np.ndarray:
        if not self.use_lidar:
            return super().act(obs, deterministic=deterministic, aid=aid)

        pose = _extract_pose(obs)
        scan = extract_lidar_scan(obs)
        if pose is None or scan is None:
            return super().act(obs, deterministic=deterministic, aid=aid)

        centerline = self._local_centerline(pose)
        if centerline is None:
            return self._fallback_action.copy()

        obstacle_points = lidar_points_world(
            scan,
            pose,
            lidar_fov=self.lidar_fov,
            max_range=self.slow_distance,
            max_points=self.max_obstacle_points,
        )
        if obstacle_points.shape[0] == 0:
            return super().act(obs, deterministic=deterministic, aid=aid)

        target_speed = self._target_speed(obs)
        speed_cap = self._obstacle_speed_cap(scan)
        target_speed = min(target_speed, speed_cap)
        sequences = self._candidate_sequences.copy()
        sequences[:, :, 1] = np.clip(sequences[:, :, 1], self.min_speed, speed_cap)

        action = self._select_obstacle_aware_action(
            pose,
            centerline,
            obstacle_points,
            target_speed,
            sequences,
        )
        action = self._clip_action(action)
        action[1] = float(np.clip(action[1], self.min_speed, speed_cap))
        self._previous_action = action.copy()
        return action

    def _select_obstacle_aware_action(
        self,
        pose: np.ndarray,
        centerline: np.ndarray,
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
            base_cost = trajectory_cost(
                trajectory,
                sequence,
                centerline=centerline,
                target_speed=target_speed,
                weights=self._weights,
            )
            obstacle_cost = obstacle_proximity_cost(
                trajectory,
                obstacle_points,
                safe_distance=self.safe_distance,
                hard_stop_distance=self.hard_stop_distance,
            )
            cost = base_cost + self.obstacle_weight * obstacle_cost
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

    def _obstacle_speed_cap(self, scan: np.ndarray) -> float:
        valid = scan[np.isfinite(scan) & (scan > 0.0)]
        if valid.size == 0:
            return self.max_speed
        nearest = float(np.min(valid))
        if nearest <= self.hard_stop_distance:
            return self.min_obstacle_speed
        if nearest >= self.slow_distance:
            return self.max_speed
        span = max(self.slow_distance - self.hard_stop_distance, 1e-6)
        alpha = (nearest - self.hard_stop_distance) / span
        return float(self.min_obstacle_speed + alpha * (self.max_speed - self.min_obstacle_speed))


def extract_lidar_scan(obs: Dict[str, Any]) -> Optional[np.ndarray]:
    if not isinstance(obs, dict):
        return None
    for key in ("lidar", "scan", "scans"):
        if key in obs:
            arr = np.asarray(obs[key], dtype=np.float32).reshape(-1)
            if arr.size == 0:
                return None
            return arr
    return None


def lidar_points_world(
    scan: np.ndarray,
    pose: np.ndarray,
    *,
    lidar_fov: float,
    max_range: float,
    max_points: int,
) -> np.ndarray:
    ranges = np.asarray(scan, dtype=np.float32).reshape(-1)
    if ranges.size == 0:
        return np.zeros((0, 2), dtype=np.float32)

    valid = np.isfinite(ranges) & (ranges > 0.0) & (ranges <= max_range)
    if not np.any(valid):
        return np.zeros((0, 2), dtype=np.float32)

    indices = np.nonzero(valid)[0]
    if indices.size > max_points:
        sample_idx = np.linspace(0, indices.size - 1, max_points, dtype=np.int32)
        indices = indices[sample_idx]

    if ranges.size == 1:
        angles = np.zeros(1, dtype=np.float32)
    else:
        angles = np.linspace(
            -0.5 * float(lidar_fov),
            0.5 * float(lidar_fov),
            ranges.size,
            dtype=np.float32,
        )
    selected_ranges = ranges[indices]
    selected_angles = angles[indices]
    local = np.stack(
        [
            selected_ranges * np.cos(selected_angles),
            selected_ranges * np.sin(selected_angles),
        ],
        axis=1,
    ).astype(np.float32)

    theta = float(pose[2])
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
    return pose[:2].astype(np.float32) + local @ rot.T


def obstacle_proximity_cost(
    trajectory: np.ndarray,
    obstacle_points: np.ndarray,
    *,
    safe_distance: float = 0.8,
    hard_stop_distance: float = 0.35,
) -> float:
    traj = np.asarray(trajectory, dtype=np.float32)
    obstacles = np.asarray(obstacle_points, dtype=np.float32)
    if traj.ndim != 2 or traj.shape[0] < 2 or obstacles.ndim != 2 or obstacles.shape[0] == 0:
        return 0.0
    points = traj[1:, :2]
    if points.shape[0] == 0 or obstacles.shape[1] < 2:
        return 0.0

    deltas = points[:, None, :] - obstacles[None, :, :2]
    distances = np.sqrt(np.sum(deltas * deltas, axis=2))
    nearest = np.min(distances, axis=1)

    safe_distance = max(float(safe_distance), 0.0)
    hard_stop_distance = max(float(hard_stop_distance), 0.0)
    soft = np.maximum(0.0, safe_distance - nearest)
    cost = float(np.mean(soft * soft)) if soft.size else 0.0

    hard = nearest[nearest < hard_stop_distance]
    if hard.size:
        margin = np.maximum(0.0, hard_stop_distance - hard)
        cost += float(100.0 * np.mean(1.0 + margin * margin))
    return cost


__all__ = [
    "ObstacleAwareMPCAgent",
    "extract_lidar_scan",
    "lidar_points_world",
    "obstacle_proximity_cost",
]
