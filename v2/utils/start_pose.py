"""Helpers for parsing and validating start pose configurations."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from v2.utils.map_loader import MapData


@dataclass
class StartPoseOption:
    """Container describing a start pose selection with optional metadata."""

    poses: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


def parse_start_pose_options(options: Optional[Iterable]) -> Optional[List[StartPoseOption]]:
    if not options:
        return None
    processed: List[StartPoseOption] = []
    for option in options:
        if isinstance(option, StartPoseOption):
            processed.append(option)
            continue

        metadata: Dict[str, Any] = {}
        payload = option

        if isinstance(option, dict):
            payload = option.get("poses", option.get("pose"))
            metadata = option.get("metadata") or {}
            if not metadata:
                metadata = {
                    key: value
                    for key, value in option.items()
                    if key not in {"poses", "pose"}
                }

        if payload is None:
            continue

        arr = np.asarray(payload, dtype=np.float32)
        if arr.ndim == 1:
            arr = np.expand_dims(arr, axis=0)
        processed.append(StartPoseOption(poses=arr, metadata=dict(metadata)))
    return processed or None


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
        return poses.copy()

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
    options: Optional[List[StartPoseOption]],
    back_gap: float = 0.0,
    min_spacing: float = 0.0,
    map_data: MapData | None = None,
    max_attempts: int = 20,
):
    random_enabled = bool(getattr(env, "_random_spawn_enabled", False))
    sampler = getattr(env, "_sample_random_spawn", None)
    if random_enabled and callable(sampler):
        attempts = max_attempts if max_attempts and max_attempts > 0 else 1
        attempt_idx = 0
        while attempt_idx < attempts:
            sampled = sampler()
            if sampled is None:
                break
            spawn_mapping, poses = sampled
            adjusted = adjust_start_poses(
                poses,
                back_gap,
                min_spacing,
                map_data=map_data,
            )
            obs, infos = env.reset(options={"poses": adjusted})
            if spawn_mapping:
                for agent_id, spawn_name in spawn_mapping.items():
                    infos.setdefault(agent_id, {})["spawn_point"] = spawn_name
                if hasattr(env, "_last_spawn_selection"):
                    env._last_spawn_selection = dict(spawn_mapping)
            collisions = [obs.get(aid, {}).get("collision", False) for aid in obs.keys()]
            if not any(collisions):
                return obs, infos
            attempt_idx += 1
        return env.reset()

    if not options:
        return env.reset()

    option_count = len(options)
    if option_count == 0:
        return env.reset()

    cycle_enabled = bool(getattr(env, "_spawn_cycle_enabled", False))
    if cycle_enabled and not hasattr(env, "_spawn_cycle_index"):
        setattr(env, "_spawn_cycle_index", 0)

    agent_ids = list(getattr(env, "possible_agents", []))

    def _agent_id(idx: int) -> str:
        if idx < len(agent_ids):
            return agent_ids[idx]
        return f"car_{idx}"

    if cycle_enabled:
        cycle_start = int(getattr(env, "_spawn_cycle_index", 0)) % option_count
        ordered = list(range(option_count))
        indices = ordered[cycle_start:] + ordered[:cycle_start]
    else:
        indices = np.random.permutation(option_count).tolist()

    def _sample_option(option: StartPoseOption) -> Optional[Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]]:
        metadata = dict(option.metadata or {})
        poses = option.poses
        spawn_mapping: Dict[str, Any] = dict(metadata.get("spawn_points", {}))

        random_pool = metadata.get("spawn_random_pool")
        if random_pool and map_data is not None:
            pool_names = [name for name in random_pool if name in map_data.spawn_points]
            if not pool_names:
                return None

            allow_reuse = bool(metadata.get("spawn_random_allow_reuse", False))
            count = int(metadata.get("spawn_random_count", len(agent_ids) or len(poses)))
            if count <= 0:
                count = len(agent_ids) or len(poses)

            if not allow_reuse and len(pool_names) < count:
                return None

            rng = getattr(env, "rng", None)
            if rng is None:
                rng = np.random.default_rng()

            if allow_reuse:
                selected = rng.choice(pool_names, size=count, replace=True)
            else:
                selected = rng.choice(pool_names, size=count, replace=False)

            pose_stack = [map_data.spawn_points[str(name)] for name in selected]
            poses = np.asarray(pose_stack, dtype=np.float32)
            spawn_mapping = {_agent_id(idx): str(name) for idx, name in enumerate(selected)}

        return poses, spawn_mapping, metadata

    max_tries = max_attempts if max_attempts and max_attempts > 0 else option_count
    attempt_idx = 0
    cycle_start_index = indices[0] if cycle_enabled and indices else 0
    while attempt_idx < max_tries:
        option = options[indices[attempt_idx % option_count]]
        sampled = _sample_option(option)
        if sampled is None:
            attempt_idx += 1
            continue
        poses, spawn_mapping, metadata = sampled

        adjusted = adjust_start_poses(poses, back_gap, min_spacing, map_data=map_data)
        obs, infos = env.reset(options={"poses": adjusted})

        if spawn_mapping:
            for agent_id, spawn_name in spawn_mapping.items():
                infos.setdefault(agent_id, {})["spawn_point"] = spawn_name

        option_id = metadata.get("spawn_option_id")
        if option_id is not None:
            for agent_id in infos:
                infos[agent_id].setdefault("spawn_option", option_id)

        collisions = [obs.get(aid, {}).get("collision", False) for aid in obs.keys()]
        if not any(collisions):
            if cycle_enabled:
                next_index = (cycle_start_index + 1) % option_count
                setattr(env, "_spawn_cycle_index", next_index)
            return obs, infos

        attempt_idx += 1

    return env.reset()
