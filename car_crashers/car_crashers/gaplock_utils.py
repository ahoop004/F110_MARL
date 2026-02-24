#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Shared observation/action helpers for gaplock ROS2 actors.

Observation layout (65 dims) — confirmed from ppo_model.zip policy.pth weight shapes:
  [0:54]   LiDAR beams, normalized to [0, 1] with max_range=12.0 m
  [54:57]  Ego velocity [vx, vy, omega], clipped to [-1, 1] / SPEED_SCALE=1.0
  [57:60]  Target velocity [vx, vy, omega], clipped to [-1, 1] / SPEED_SCALE=1.0
  [60:65]  Relative pose in ego-centric frame:
              [rel_x, rel_y, sin(Δθ), cos(Δθ), distance], clipped to [-1, 1] / 12.0 m
"""

import math
from typing import Any, Dict, Optional

import numpy as np

LIDAR_BEAMS = 54
MAX_LIDAR_RANGE = 12.0        # metres — matches training lidar.max_range
LIDAR_POSITION_SCALE = 12.0   # metres — position normalisation for relative pose
SPEED_SCALE = 1.0             # m/s  — velocity normalisation
OBS_DIM = 65                  # 54 + 3 + 3 + 5 (confirmed from policy.pth weight shapes)
ACTION_LOW = np.array([-0.46, -1.0], dtype=np.float32)
ACTION_HIGH = np.array([0.46, 1.0], dtype=np.float32)


def init_agent_state() -> Dict[str, Any]:
    # vel is [vx, vy, omega]
    return {"pose": None, "vel": np.zeros(3, dtype=np.float32), "stamp": None}


def quat_to_yaw(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def update_agent_state(state: Dict[str, Any], msg) -> None:
    """Update pose/velocity from a VICON TransformStamped message."""
    t = msg.transform.translation
    yaw = quat_to_yaw(msg.transform.rotation)
    pose = np.array([float(t.x), float(t.y), float(yaw)], dtype=np.float32)

    s = msg.header.stamp
    stamp_sec = float(s.sec) + float(s.nanosec) * 1e-9

    vx = vy = omega = 0.0
    if state["pose"] is not None and state["stamp"] is not None:
        dt = 0.01  # fixed 100 Hz dt (VICON rate)
        vx = (pose[0] - state["pose"][0]) / dt
        vy = (pose[1] - state["pose"][1]) / dt
        dyaw = (yaw - float(state["pose"][2]) + math.pi) % (2.0 * math.pi) - math.pi
        omega = dyaw / dt

    state["pose"] = pose
    state["vel"] = np.array([vx, vy, omega], dtype=np.float32)
    state["stamp"] = stamp_sec


def downsample_lidar_to_108(
    ranges: Optional[np.ndarray],
    replace_inf: float = MAX_LIDAR_RANGE,
) -> np.ndarray:
    if ranges is None:
        return np.full(LIDAR_BEAMS, replace_inf, dtype=np.float32)
    arr = np.asarray(ranges, dtype=np.float32)
    arr = np.where(np.isfinite(arr) & (arr > 0.0), arr, replace_inf)
    n = arr.shape[0]
    if n == LIDAR_BEAMS:
        return arr
    if n <= 1:
        return np.full(LIDAR_BEAMS, replace_inf, dtype=np.float32)
    idx = np.linspace(0, n - 1, LIDAR_BEAMS).astype(np.int32)
    return arr[idx]


def build_observation(
    last_scan: np.ndarray,
    primary_state: Dict[str, Any],
    secondary_state: Dict[str, Any],
) -> np.ndarray:
    """Build a 119-dim observation vector matching the training flattener.

    primary_state  = attacker (ego)
    secondary_state = target (defender)
    """
    # --- LiDAR: 108 dims, [0, 1] ---
    lidar = downsample_lidar_to_108(last_scan)
    lidar_norm = np.clip(lidar / MAX_LIDAR_RANGE, 0.0, 1.0)

    # --- Ego velocity: [vx, vy, omega], 3 dims, [-1, 1] ---
    ego_vel = np.clip(primary_state["vel"] / SPEED_SCALE, -1.0, 1.0).astype(np.float32)

    # --- Target velocity: [vx, vy, omega], 3 dims, [-1, 1] ---
    target_vel = np.clip(secondary_state["vel"] / SPEED_SCALE, -1.0, 1.0).astype(np.float32)

    # --- Relative pose in ego-centric frame: 5 dims ---
    x, y, theta = map(float, primary_state["pose"])
    tx, ty, ttheta = map(float, secondary_state["pose"])
    dx = tx - x
    dy = ty - y
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    rel_x = float(np.clip((cos_t * dx + sin_t * dy) / LIDAR_POSITION_SCALE, -1.0, 1.0))
    rel_y = float(np.clip((-sin_t * dx + cos_t * dy) / LIDAR_POSITION_SCALE, -1.0, 1.0))
    dtheta = (ttheta - theta + math.pi) % (2.0 * math.pi) - math.pi
    sin_dth = math.sin(dtheta)
    cos_dth = math.cos(dtheta)
    distance = float(np.clip(math.sqrt(dx**2 + dy**2) / LIDAR_POSITION_SCALE, 0.0, 1.0))
    relative = np.array([rel_x, rel_y, sin_dth, cos_dth, distance], dtype=np.float32)

    obs = np.concatenate([lidar_norm, ego_vel, target_vel, relative], axis=0)
    assert obs.shape[0] == OBS_DIM, f"obs dim {obs.shape[0]} != {OBS_DIM}"
    return obs.astype(np.float32, copy=False)
