"""Helpers for parsing and validating start pose configurations."""
from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import numpy as np

from f110x.utils.map_loader import MapData


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


def _world_to_pixel(map_data: MapData, x: float, y: float) -> Tuple[int, int]:
    resolution = float(map_data.metadata.get("resolution", 0.05))
    origin = map_data.metadata.get("origin", (0.0, 0.0, 0.0))
    px = int((x - origin[0]) / resolution)
    py = int((y - origin[1]) / resolution)
    return px, py


def _pixel_to_world(map_data: MapData, px: int, py: int) -> Tuple[float, float]:
    resolution = float(map_data.metadata.get("resolution", 0.05))
    origin = map_data.metadata.get("origin", (0.0, 0.0, 0.0))
    x = px * resolution + origin[0]
    y = py * resolution + origin[1]
    return float(x), float(y)

def _project_to_track(pose: np.ndarray, map_data: MapData, max_radius: int = 20) -> np.ndarray:
    if map_data.track_mask is None:
        return pose
    px, py = _world_to_pixel(map_data, float(pose[0]), float(pose[1]))
    mask = map_data.track_mask
    if 0 <= px < mask.shape[1] and 0 <= py < mask.shape[0] and mask[py, px]:
        return pose
    for radius in range(1, max_radius + 1):
        xs = range(max(0, px - radius), min(mask.shape[1], px + radius + 1))
        ys = range(max(0, py - radius), min(mask.shape[0], py + radius + 1))
        for x in xs:
            for y in ys:
                if mask[y, x]:
                    wx, wy = _pixel_to_world(map_data, x, y)
                    adjusted = pose.copy()
                    adjusted[0] = wx
                    adjusted[1] = wy
                    return adjusted
    return pose


def adjust_start_poses(
    poses: np.ndarray,
    back_gap: float = 0.0,
    min_spacing: float = 0.0,
    leader_index: int = 0,
    map_data: MapData | None = None,
) -> np.ndarray:
    if poses.shape[0] < 2:
        if map_data is not None:
            return _project_to_track(poses.copy(), map_data)
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

    if map_data is not None:
        for idx in range(adjusted.shape[0]):
            adjusted[idx] = _project_to_track(adjusted[idx], map_data)

    return adjusted


def reset_with_start_poses(
    env,
    options: Optional[List[np.ndarray]],
    back_gap: float = 0.0,
    min_spacing: float = 0.0,
    map_data: MapData | None = None,
    max_attempts: int = 20,
):
    if not options:
        return env.reset()

    indices = np.random.permutation(len(options))
    for idx in indices[:max_attempts]:
        poses = adjust_start_poses(options[idx], back_gap, min_spacing, map_data=map_data)
        obs, infos = env.reset(options={"poses": poses})
        collisions = [obs.get(aid, {}).get("collision", False) for aid in obs.keys()]
        if not any(collisions):
            return obs, infos

    return env.reset()
