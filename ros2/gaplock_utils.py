#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Shared observation/action helpers for gaplock ROS actors."""

#from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np

# These defaults are aligned with the gaplock scenarios in this repo:
# - wrappers.obs.params.max_scan: 12.0
# - wrappers.obs.components.lidar.params.beams: 720
# - wrappers.obs.components.lidar.params.max_range: 12.0
LIDAR_BEAMS = 720
MAX_LIDAR_RANGE = 12.0
EGO_POSE_NORM = 12.0
VEL_NORM = 2.0
OBS_DIM = LIDAR_BEAMS + 18  # lidar + ego/target pose/vel + relative pose
ACTION_LOW = np.array([-0.41890001297, -1.0], dtype=np.float32)
ACTION_HIGH = np.array([0.41890001297, 2.0], dtype=np.float32)


def init_agent_state() -> Dict[str, Any]:
    return {"pose": None, "vel": np.zeros(2, dtype=np.float32), "stamp": None}


def quat_to_yaw(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def update_agent_state(state: Dict[str, Any], msg) -> None:
    t = msg.transform.translation
    yaw = quat_to_yaw(msg.transform.rotation)
    pose = np.array([float(t.x), float(t.y), float(yaw)], dtype=np.float32)

    stamp = getattr(getattr(msg, "header", None), "stamp", None)
    stamp_sec = None
    if stamp is not None:
        nanoseconds = getattr(stamp, "nanoseconds", None)
        if nanoseconds is not None:
            stamp_sec = float(nanoseconds) * 1e-9
        else:
            sec = getattr(stamp, "sec", None)
            nanosec = getattr(stamp, "nanosec", None)
            if sec is not None and nanosec is not None:
                stamp_sec = float(sec) + float(nanosec) * 1e-9

    vx = vy = 0.0
    if state["pose"] is not None and state["stamp"] is not None and stamp_sec is not None:
        dt = float(stamp_sec - float(state["stamp"]))
        if 1e-6 < dt < 1.0:
            vx = float((pose[0] - state["pose"][0]) / dt)
            vy = float((pose[1] - state["pose"][1]) / dt)
    state["pose"] = pose
    state["vel"] = np.array([vx, vy], dtype=np.float32)
    state["stamp"] = stamp_sec


def downsample_lidar(
    ranges: Optional[np.ndarray],
    *,
    target_beams: int = LIDAR_BEAMS,
    max_range: float = MAX_LIDAR_RANGE,
    replace_inf: Optional[float] = None,
) -> np.ndarray:
    if replace_inf is None:
        replace_inf = float(max_range)

    target_beams = int(target_beams)
    if target_beams <= 0:
        raise ValueError("target_beams must be positive")

    if ranges is None:
        return np.full(target_beams, replace_inf, dtype=np.float32)
    arr = np.asarray(ranges, dtype=np.float32)
    arr = np.where(np.isfinite(arr) & (arr > 0.0), arr, replace_inf)
    arr = np.clip(arr, 0.0, float(max_range))
    n = arr.shape[0]
    if n == target_beams:
        return arr
    if n <= 1:
        return np.full(target_beams, replace_inf, dtype=np.float32)
    idx = np.linspace(0, n - 1, target_beams).astype(np.int32)
    return arr[idx]


def downsample_lidar_to_108(ranges: Optional[np.ndarray], replace_inf: float = MAX_LIDAR_RANGE) -> np.ndarray:
    """Backwards-compatible alias (older scripts expected 108 beams)."""

    return downsample_lidar(ranges, target_beams=108, max_range=MAX_LIDAR_RANGE, replace_inf=replace_inf)


def encode_pose(pose: np.ndarray, *, normalize_xy: Optional[float]) -> np.ndarray:
    x = float(pose[0])
    y = float(pose[1])
    yaw = pose[2]
    if normalize_xy:
        scale = float(normalize_xy)
        if scale != 0.0:
            x /= scale
            y /= scale
    return np.array([x, y, math.sin(yaw), math.cos(yaw)], dtype=np.float32)


def encode_velocity(vel: np.ndarray) -> np.ndarray:
    vx = vel[0] / VEL_NORM
    vy = vel[1] / VEL_NORM
    speed = math.sqrt(vx * vx + vy * vy)
    return np.array([vx, vy, speed], dtype=np.float32)


def encode_relative(attacker_pose: np.ndarray, target_pose: np.ndarray) -> np.ndarray:
    dx = target_pose[0] - attacker_pose[0]
    dy = target_pose[1] - attacker_pose[1]
    dtheta = target_pose[2] - attacker_pose[2]
    return np.array([dx, dy, math.sin(dtheta), math.cos(dtheta)], dtype=np.float32)


def build_observation(last_scan: np.ndarray, primary_state: Dict[str, Any], secondary_state: Dict[str, Any]) -> np.ndarray:
    lidar = downsample_lidar(last_scan)
    lidar = np.clip(lidar / MAX_LIDAR_RANGE, 0.0, 1.0)

    # Matches the training wrapper defaults: ego pose x/y normalised by max_scan (12.0),
    # target pose left unnormalised unless explicitly configured.
    attacker_pose = encode_pose(primary_state["pose"], normalize_xy=EGO_POSE_NORM)
    attacker_vel = encode_velocity(primary_state["vel"])
    target_pose = encode_pose(secondary_state["pose"], normalize_xy=None)
    target_vel = encode_velocity(secondary_state["vel"])
    relative = encode_relative(primary_state["pose"], secondary_state["pose"])
    obs = np.concatenate([lidar, attacker_pose, attacker_vel, target_pose, target_vel, relative], axis=0)
    return obs.astype(np.float32, copy=False)


def scale_continuous_action(raw_action: np.ndarray) -> np.ndarray:
    clipped = np.clip(raw_action, -1.0, 1.0)
    range_half = (ACTION_HIGH - ACTION_LOW) / 2.0
    mid = (ACTION_HIGH + ACTION_LOW) / 2.0
    return clipped * range_half + mid
