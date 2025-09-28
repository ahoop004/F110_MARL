"""Helpers for parsing and validating start pose configurations."""
from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np


def parse_start_pose_options(options: Optional[Iterable]) -> Optional[List[np.ndarray]]:
    if not options:
        return None
    processed: List[np.ndarray] = []
    for option in options:
        arr = np.asarray(option, dtype=np.float32)
        if arr.ndim == 1:
            arr = np.expand_dims(arr, axis=0)
        processed.append(arr)
    return processed


def validate_pose_on_map(pose: np.ndarray, track_mask: np.ndarray) -> bool:
    x, y = pose[:2].astype(int)
    if x < 0 or y < 0 or x >= track_mask.shape[1] or y >= track_mask.shape[0]:
        return False
    return bool(track_mask[y, x])


def adjust_start_poses(
    poses: np.ndarray,
    back_gap: float = 0.0,
    min_spacing: float = 0.0,
    leader_index: int = 0,
) -> np.ndarray:
    if poses.shape[0] < 2:
        return poses

    adjusted = poses.copy()
    leader = adjusted[leader_index]
    heading = np.array([np.cos(leader[2]), np.sin(leader[2])], dtype=np.float32)

    if back_gap > 0.0:
        for idx in range(adjusted.shape[0]):
            if idx == leader_index:
                continue
            rel = adjusted[idx, :2] - leader[:2]
            proj = float(np.dot(rel, heading))
            if proj < 0 and abs(proj) < back_gap:
                delta = back_gap + proj
                adjusted[idx, :2] -= heading * delta

    if min_spacing > 0.0:
        for idx in range(adjusted.shape[0]):
            if idx == leader_index:
                continue
            rel = adjusted[idx, :2] - leader[:2]
            dist = float(np.linalg.norm(rel))
            if 0 < dist < min_spacing:
                direction = heading if np.dot(rel, heading) >= 0 else -heading
                adjusted[idx, :2] += direction * (min_spacing - dist)

    return adjusted


def reset_with_start_poses(
    env,
    options: Optional[List[np.ndarray]],
    back_gap: float = 0.0,
    min_spacing: float = 0.0,
    max_attempts: int = 20,
):
    if not options:
        return env.reset()

    indices = np.random.permutation(len(options))
    for idx in indices[:max_attempts]:
        poses = adjust_start_poses(options[idx], back_gap, min_spacing)
        obs, infos = env.reset(options={"poses": poses})
        collisions = [obs.get(aid, {}).get("collision", False) for aid in obs.keys()]
        if not any(collisions):
            return obs, infos

    return env.reset()
