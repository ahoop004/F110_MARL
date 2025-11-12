#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Shared observation/action helpers for gaplock ROS actors."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np
import rospy

LIDAR_BEAMS = 108
MAX_LIDAR_RANGE = 30.0
POSE_NORM = 30.0
VEL_NORM = 2.0
OBS_DIM = 126  # 108 lidar + 18 pose/velocity features
ACTION_LOW = np.array([-0.41890001297, -1.0], dtype=np.float32)
ACTION_HIGH = np.array([0.41890001297, 1.0], dtype=np.float32)


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
    if msg.header and msg.header.stamp:
        stamp = msg.header.stamp.to_sec()
    else:
        stamp = rospy.Time.now().to_sec()
    vx = vy = 0.0
    if state["pose"] is not None and state["stamp"] is not None and stamp:
        dt = max(stamp - state["stamp"], 1e-6)
        vx = (pose[0] - state["pose"][0]) / dt
        vy = (pose[1] - state["pose"][1]) / dt
    state["pose"] = pose
    state["vel"] = np.array([vx, vy], dtype=np.float32)
    state["stamp"] = stamp


def downsample_lidar_to_108(ranges: Optional[np.ndarray], replace_inf: float = MAX_LIDAR_RANGE) -> np.ndarray:
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


def encode_pose(pose: np.ndarray) -> np.ndarray:
    x = pose[0] / POSE_NORM
    y = pose[1] / POSE_NORM
    yaw = pose[2]
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
    lidar = downsample_lidar_to_108(last_scan)
    lidar = np.clip(lidar / MAX_LIDAR_RANGE, 0.0, 1.0)
    attacker_pose = encode_pose(primary_state["pose"])
    attacker_vel = encode_velocity(primary_state["vel"])
    target_pose = encode_pose(secondary_state["pose"])
    target_vel = encode_velocity(secondary_state["vel"])
    relative = encode_relative(primary_state["pose"], secondary_state["pose"])
    obs = np.concatenate([lidar, attacker_pose, attacker_vel, target_pose, target_vel, relative], axis=0)
    return obs.astype(np.float32, copy=False)


def scale_continuous_action(raw_action: np.ndarray) -> np.ndarray:
    clipped = np.clip(raw_action, -1.0, 1.0)
    range_half = (ACTION_HIGH - ACTION_LOW) / 2.0
    mid = (ACTION_HIGH + ACTION_LOW) / 2.0
    return clipped * range_half + mid
