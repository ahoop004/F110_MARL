"""State buffer helpers for `F110ParallelEnv`."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class StateBuffers:
    """Container tracking per-agent kinematics across simulator steps."""

    poses_x: np.ndarray
    poses_y: np.ndarray
    poses_theta: np.ndarray
    collisions: np.ndarray
    linear_vels_x_prev: np.ndarray
    linear_vels_y_prev: np.ndarray
    angular_vels_prev: np.ndarray
    linear_vels_x_curr: np.ndarray
    linear_vels_y_curr: np.ndarray
    angular_vels_curr: np.ndarray
    velocity_initialized: bool = False

    @classmethod
    def build(cls, n_agents: int) -> "StateBuffers":
        zeros = np.zeros((n_agents,), dtype=np.float32)
        return cls(
            poses_x=zeros.copy(),
            poses_y=zeros.copy(),
            poses_theta=zeros.copy(),
            collisions=zeros.copy(),
            linear_vels_x_prev=zeros.copy(),
            linear_vels_y_prev=zeros.copy(),
            angular_vels_prev=zeros.copy(),
            linear_vels_x_curr=zeros.copy(),
            linear_vels_y_curr=zeros.copy(),
            angular_vels_curr=zeros.copy(),
            velocity_initialized=False,
        )

    # ------------------------------------------------------------------
    def reset(self) -> None:
        for array in (
            self.poses_x,
            self.poses_y,
            self.poses_theta,
            self.collisions,
            self.linear_vels_x_prev,
            self.linear_vels_y_prev,
            self.angular_vels_prev,
            self.linear_vels_x_curr,
            self.linear_vels_y_curr,
            self.angular_vels_curr,
        ):
            array.fill(0.0)
        self.velocity_initialized = False

    def update(self, obs_dict: Dict[str, np.ndarray]) -> None:
        self._assign(self.poses_x, obs_dict.get("poses_x"))
        self._assign(self.poses_y, obs_dict.get("poses_y"))
        self._assign(self.poses_theta, obs_dict.get("poses_theta"))
        self._assign(self.collisions, obs_dict.get("collisions"))

        self.linear_vels_x_prev[:] = self.linear_vels_x_curr
        self.linear_vels_y_prev[:] = self.linear_vels_y_curr
        self.angular_vels_prev[:] = self.angular_vels_curr

        self._assign(self.linear_vels_x_curr, obs_dict.get("linear_vels_x"))
        self._assign(self.linear_vels_y_curr, obs_dict.get("linear_vels_y"))
        self._assign(self.angular_vels_curr, obs_dict.get("ang_vels_z"))

        self.velocity_initialized = True

    # ------------------------------------------------------------------
    @staticmethod
    def _assign(target: np.ndarray, source: np.ndarray | None) -> None:
        if source is None:
            target.fill(0.0)
            return
        arr = np.asarray(source, dtype=np.float32)
        count = min(target.shape[0], arr.shape[0])
        target[:count] = arr[:count]
        if count < target.shape[0]:
            target[count:] = 0.0
