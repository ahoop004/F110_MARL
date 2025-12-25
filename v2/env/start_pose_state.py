"""Start pose state tracking and lap progression helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np


def _rotation_from_angles(angles: np.ndarray) -> np.ndarray:
    if angles.size == 0:
        return np.eye(2, dtype=np.float32)
    if angles.size == 1 or np.allclose(angles, angles[0]):
        angle = float(angles[0])
        cos_a = float(np.cos(angle))
        sin_a = float(np.sin(angle))
        return np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)

    cos_vals = np.cos(angles).astype(np.float32)
    sin_vals = np.sin(angles).astype(np.float32)
    rot = np.empty((angles.shape[0], 2, 2), dtype=np.float32)
    rot[:, 0, 0] = cos_vals
    rot[:, 0, 1] = -sin_vals
    rot[:, 1, 0] = sin_vals
    rot[:, 1, 1] = cos_vals
    return rot


@dataclass
class StartPoseState:
    agent_ids: Sequence[str]
    start_xs: np.ndarray
    start_ys: np.ndarray
    start_thetas: np.ndarray
    start_rot: np.ndarray
    start_heading: np.ndarray
    near_starts: np.ndarray
    toggle_list: np.ndarray
    lap_counts: np.ndarray
    lap_times: np.ndarray
    left_start_forward: np.ndarray
    lap_forward_eps: float
    left_threshold: float = 2.0
    right_threshold: float = 2.0

    @classmethod
    def build(
        cls,
        agent_ids: Sequence[str],
        start_poses: np.ndarray,
        lap_forward_eps: float,
    ) -> "StartPoseState":
        n_agents = len(agent_ids)
        start_xs = np.zeros((n_agents,), dtype=np.float32)
        start_ys = np.zeros((n_agents,), dtype=np.float32)
        start_thetas = np.zeros((n_agents,), dtype=np.float32)

        if start_poses.size > 0:
            pose_array = np.atleast_2d(np.asarray(start_poses, dtype=np.float32))
            count = min(n_agents, pose_array.shape[0])
            if count:
                start_xs[:count] = pose_array[:count, 0]
                if pose_array.shape[1] >= 2:
                    start_ys[:count] = pose_array[:count, 1]
                if pose_array.shape[1] >= 3:
                    start_thetas[:count] = pose_array[:count, 2]
        angles = start_thetas.copy()
        start_rot = _rotation_from_angles(angles[:n_agents])
        start_heading = np.zeros((n_agents, 2), dtype=np.float32)
        if n_agents:
            start_heading[:, 0] = np.cos(start_thetas).astype(np.float32)
            start_heading[:, 1] = np.sin(start_thetas).astype(np.float32)

        return cls(
            agent_ids=list(agent_ids),
            start_xs=start_xs,
            start_ys=start_ys,
            start_thetas=start_thetas,
            start_rot=start_rot,
            start_heading=start_heading,
            near_starts=np.ones((n_agents,), dtype=bool),
            toggle_list=np.zeros((n_agents,), dtype=np.float32),
            lap_counts=np.zeros((n_agents,), dtype=np.float32),
            lap_times=np.zeros((n_agents,), dtype=np.float32),
            left_start_forward=np.zeros((n_agents,), dtype=bool),
            lap_forward_eps=float(lap_forward_eps),
        )

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.near_starts.fill(True)
        self.toggle_list.fill(0.0)
        self.lap_counts.fill(0.0)
        self.lap_times.fill(0.0)
        self.left_start_forward.fill(False)

    def apply_start_poses(self, poses: np.ndarray) -> None:
        if poses is None or poses.size == 0:
            return
        pose_array = np.atleast_2d(np.asarray(poses, dtype=np.float32))
        count = min(len(self.agent_ids), pose_array.shape[0])
        if count == 0:
            return
        self.start_xs[:count] = pose_array[:count, 0]
        if pose_array.shape[1] >= 2:
            self.start_ys[:count] = pose_array[:count, 1]
        if pose_array.shape[1] >= 3:
            self.start_thetas[:count] = pose_array[:count, 2]
        angles = self.start_thetas[:count]
        self.start_rot = _rotation_from_angles(angles if count == len(self.agent_ids) else self.start_thetas)
        self.start_heading[:, 0] = np.cos(self.start_thetas).astype(np.float32)
        self.start_heading[:, 1] = np.sin(self.start_thetas).astype(np.float32)

    def update_progress(
        self,
        poses_x: np.ndarray,
        poses_y: np.ndarray,
        linear_vels_x: np.ndarray,
        linear_vels_y: np.ndarray,
        current_time: float,
        target_laps: int,
    ) -> Dict[str, bool]:
        offsets = np.stack(
            (np.asarray(poses_x, dtype=np.float32) - self.start_xs,
             np.asarray(poses_y, dtype=np.float32) - self.start_ys),
            axis=-1,
        )

        rotation = self.start_rot
        if rotation.ndim == 2:
            delta = rotation @ offsets.T
        else:
            delta = np.einsum("aij,aj->ai", rotation, offsets).T

        temp_y = delta[1, :].copy()
        left_t = self.left_threshold
        right_t = self.right_threshold
        idx_left = temp_y > left_t
        idx_right = temp_y < -right_t
        temp_y[idx_left] -= left_t
        temp_y[idx_right] = -right_t - temp_y[idx_right]
        keep_mask = ~(idx_left | idx_right)
        temp_y[keep_mask] = 0.0

        dist2 = delta[0, :] ** 2 + temp_y ** 2
        closes = dist2 <= 0.1

        forward = (
            np.asarray(linear_vels_x, dtype=np.float32) * self.start_heading[:, 0]
            + np.asarray(linear_vels_y, dtype=np.float32) * self.start_heading[:, 1]
        )
        forward_ok = forward > self.lap_forward_eps

        for idx, _ in enumerate(self.agent_ids):
            if closes[idx]:
                if not self.near_starts[idx]:
                    self.near_starts[idx] = True
                    if self.left_start_forward[idx] and forward_ok[idx]:
                        self.toggle_list[idx] += 1
                        self.left_start_forward[idx] = False
                    elif not forward_ok[idx]:
                        self.left_start_forward[idx] = False
                else:
                    if forward_ok[idx]:
                        self.left_start_forward[idx] = True
            else:
                if self.near_starts[idx]:
                    self.near_starts[idx] = False
                    forward_exit = forward_ok[idx]
                    self.left_start_forward[idx] = forward_exit
                    if forward_exit:
                        self.toggle_list[idx] += 1
                elif forward_ok[idx]:
                    self.left_start_forward[idx] = True

        prev_counts = self.lap_counts.copy()
        self.lap_counts[:] = self.toggle_list // 2
        improved = {}
        for idx, agent_id in enumerate(self.agent_ids):
            if self.lap_counts[idx] > prev_counts[idx]:
                self.lap_times[idx] = float(current_time)
            improved[agent_id] = self.lap_counts[idx] >= target_laps
        return improved
