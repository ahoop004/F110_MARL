"""Centerline reward strategy for single-agent racing."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from rewards.base import RewardStrategy
from utils.centerline import project_to_centerline


class CenterlineReward(RewardStrategy):
    """Reward strategy encouraging speed along the centerline with smooth steering."""

    def __init__(self, config: Dict[str, Any]) -> None:
        center_cfg = config.get("centerline", config) if isinstance(config, dict) else {}
        self.vs_weight = float(center_cfg.get("vs_weight", 1.0))
        self.vd_weight = float(center_cfg.get("vd_weight", 0.01))
        self.d_weight = float(center_cfg.get("d_weight", 0.02))
        self.steer_weight = float(center_cfg.get("steer_weight", 0.1))
        self.collision_penalty = float(center_cfg.get("collision_penalty", -1000.0))
        self.steer_index = int(center_cfg.get("steer_index", 0))
        self._last_centerline_index: Optional[int] = None

    def reset(self) -> None:
        self._last_centerline_index = None

    def compute(self, step_info: dict) -> Tuple[float, Dict[str, float]]:
        obs = step_info.get("next_obs") or step_info.get("obs") or {}
        info = step_info.get("info") or {}

        metrics = {}
        if isinstance(info, dict):
            metrics = info.get("centerline", {}) if isinstance(info.get("centerline"), dict) else {}

        vs = metrics.get("vs")
        vd = metrics.get("vd")
        d = metrics.get("d")

        if vs is None or vd is None or d is None:
            centerline = step_info.get("centerline")
            if centerline is None:
                centerline = step_info.get("env_centerline")
            if centerline is None:
                centerline = getattr(step_info.get("env", None), "centerline_points", None)

            vs, vd, d = self._compute_centerline_terms(obs, centerline)

        steer = self._extract_steer(step_info, obs)

        total = (
            self.vs_weight * float(vs)
            - self.vd_weight * abs(float(vd))
            - self.d_weight * abs(float(d))
            - self.steer_weight * abs(float(steer))
        )

        collision = bool(info.get("collision", False)) if isinstance(info, dict) else False
        if collision:
            total += self.collision_penalty

        components = {
            "centerline/vs": float(vs),
            "centerline/vd": float(vd),
            "centerline/d": float(d),
            "centerline/steer": float(steer),
            "centerline/collision": float(self.collision_penalty if collision else 0.0),
        }
        return float(total), components

    def _compute_centerline_terms(
        self,
        obs: Dict[str, Any],
        centerline: Optional[np.ndarray],
    ) -> Tuple[float, float, float]:
        if centerline is None or not isinstance(obs, dict):
            return 0.0, 0.0, 0.0

        pose = obs.get("pose")
        velocity = obs.get("velocity")
        if pose is None or velocity is None:
            return 0.0, 0.0, 0.0

        pose_arr = np.asarray(pose, dtype=np.float32).reshape(-1)
        vel_arr = np.asarray(velocity, dtype=np.float32).reshape(-1)
        if pose_arr.size < 3 or vel_arr.size < 2:
            return 0.0, 0.0, 0.0

        position = pose_arr[:2]
        heading = float(pose_arr[2])
        projection = project_to_centerline(
            np.asarray(centerline, dtype=np.float32),
            position.astype(np.float32),
            heading,
            last_index=self._last_centerline_index,
        )
        self._last_centerline_index = projection.index

        tangent_theta = self._centerline_theta(centerline, projection.index)
        tangent_cos = float(np.cos(tangent_theta))
        tangent_sin = float(np.sin(tangent_theta))

        vx = float(vel_arr[0])
        vy = float(vel_arr[1])
        vs = vx * tangent_cos + vy * tangent_sin
        vd = -vx * tangent_sin + vy * tangent_cos

        return float(vs), float(vd), float(projection.lateral_error)

    @staticmethod
    def _centerline_theta(centerline: np.ndarray, index: int) -> float:
        if centerline.ndim == 2 and centerline.shape[1] >= 3:
            theta = float(centerline[index, 2])
            if np.isfinite(theta):
                return theta

        points = centerline[:, :2].astype(np.float32, copy=False)
        idx_prev = max(0, index - 1)
        idx_next = min(points.shape[0] - 1, index + 1)
        delta = points[idx_next] - points[idx_prev]
        if np.allclose(delta, 0.0):
            return 0.0
        return float(np.arctan2(delta[1], delta[0]))

    def _extract_steer(self, step_info: dict, obs: Dict[str, Any]) -> float:
        action = step_info.get("action")
        if action is None:
            action = step_info.get("prev_action")
        if action is None and isinstance(obs, dict):
            action = obs.get("prev_action")
        if action is None:
            return 0.0
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_arr.size == 0:
            return 0.0
        idx = min(max(self.steer_index, 0), action_arr.size - 1)
        return float(action_arr[idx])


__all__ = ["CenterlineReward"]
