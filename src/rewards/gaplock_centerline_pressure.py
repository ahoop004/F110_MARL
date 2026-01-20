"""Gaplock reward that pressures target away from centerline and toward walls."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from rewards.base import RewardStrategy
from utils.centerline import project_to_centerline


class GaplockCenterlinePressureReward(RewardStrategy):
    """Reward shaping for gaplock that penalizes target centerline adherence."""

    def __init__(self, config: Dict[str, Any]) -> None:
        cfg = config.get("gaplock_centerline_pressure", config) if isinstance(config, dict) else {}

        self.centerline_weight = float(cfg.get("centerline_weight", 1.0))
        self.centerline_scale = float(cfg.get("centerline_scale", 1.0))
        self.wall_weight = float(cfg.get("wall_weight", 1.0))
        self.wall_distance_scale = float(cfg.get("wall_distance_scale", 1.0))

        self.success_reward = float(cfg.get("success_reward", 200.0))
        self.self_crash_penalty = float(cfg.get("self_crash_penalty", -100.0))
        self.collision_penalty = float(cfg.get("collision_penalty", -100.0))
        self.timeout_penalty = float(cfg.get("timeout_penalty", -50.0))
        self.target_finish_penalty = float(cfg.get("target_finish_penalty", -50.0))

        self._last_centerline_index: Optional[int] = None
        self._wall_points: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._last_centerline_index = None
        self._wall_points = None

    def compute(self, step_info: dict) -> Tuple[float, Dict[str, float]]:
        info = step_info.get("info") or {}
        done = bool(step_info.get("done", False))
        truncated = bool(step_info.get("truncated", False))

        target_obs = step_info.get("target_obs")
        target_pose = self._extract_pose(target_obs)

        centerline = step_info.get("centerline")
        walls = step_info.get("walls")

        centerline_term = 0.0
        if centerline is not None and target_pose is not None:
            centerline_term = self._centerline_deviation(target_pose, centerline)

        wall_term = 0.0
        if walls is not None and target_pose is not None:
            wall_term = self._wall_closeness(target_pose, walls)

        total = (
            self.centerline_weight * centerline_term
            + self.wall_weight * wall_term
        )

        if done:
            terminal_reward = self._terminal_reward(info, truncated)
            total += terminal_reward

        components = {
            "gaplock_pressure/centerline": float(centerline_term),
            "gaplock_pressure/wall": float(wall_term),
        }
        if done:
            components["gaplock_pressure/terminal"] = float(terminal_reward)
        return float(total), components

    def _centerline_deviation(self, pose: np.ndarray, centerline: Any) -> float:
        if self.centerline_scale <= 0.0:
            return 0.0
        projection = project_to_centerline(
            np.asarray(centerline, dtype=np.float32),
            pose[:2].astype(np.float32),
            float(pose[2]),
            last_index=self._last_centerline_index,
        )
        self._last_centerline_index = projection.index
        deviation = abs(float(projection.lateral_error))
        return float(np.clip(deviation / self.centerline_scale, 0.0, 1.0))

    def _wall_closeness(self, pose: np.ndarray, walls: Any) -> float:
        if self.wall_distance_scale <= 0.0:
            return 0.0
        wall_points = self._get_wall_points(walls)
        if wall_points is None or wall_points.size == 0:
            return 0.0
        diffs = wall_points - pose[:2].astype(np.float32)
        distances = np.linalg.norm(diffs, axis=1)
        min_dist = float(np.min(distances)) if distances.size else self.wall_distance_scale
        closeness = 1.0 - min_dist / self.wall_distance_scale
        return float(np.clip(closeness, 0.0, 1.0))

    def _terminal_reward(self, info: Dict[str, Any], truncated: bool) -> float:
        target_finished = bool(info.get("target_finished", False) or info.get("car_1/finished", False))
        if target_finished:
            return self.target_finish_penalty

        attacker_crashed = bool(
            info.get("collision", False)
            or info.get("self_crash", False)
            or info.get("attacker_collision", False)
        )
        target_crashed = bool(
            info.get("target_collision", False)
            or info.get("target_crash", False)
            or info.get("success", False)
        )

        if attacker_crashed and target_crashed:
            return self.collision_penalty
        if target_crashed:
            return self.success_reward
        if attacker_crashed:
            return self.self_crash_penalty
        if truncated or info.get("truncated", False):
            return self.timeout_penalty
        return 0.0

    @staticmethod
    def _extract_pose(obs: Any) -> Optional[np.ndarray]:
        if not isinstance(obs, dict):
            return None
        pose = obs.get("pose")
        if pose is None:
            pose = obs.get("target_pose")
        if pose is None:
            return None
        arr = np.asarray(pose, dtype=np.float32).reshape(-1)
        if arr.size < 3:
            return None
        return arr[:3]

    def _get_wall_points(self, walls: Any) -> Optional[np.ndarray]:
        if self._wall_points is not None:
            return self._wall_points
        if isinstance(walls, dict):
            points = [np.asarray(points, dtype=np.float32)[:, :2] for points in walls.values() if points is not None]
            if points:
                self._wall_points = np.vstack(points)
        elif isinstance(walls, np.ndarray):
            arr = np.asarray(walls, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                self._wall_points = arr[:, :2]
        return self._wall_points


__all__ = ["GaplockCenterlinePressureReward"]
