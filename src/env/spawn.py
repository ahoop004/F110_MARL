"""Spawn point parsing and sampling helpers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple
from pathlib import Path

import numpy as np
import yaml

from src.utils.centerline import centerline_arc_length, centerline_pose


CenterlineSpawnFn = Callable[[], Optional[Tuple[np.ndarray, Dict[str, float], Dict[str, float]]]]


@dataclass(frozen=True)
class SpawnRequest:
    """Inputs needed to resolve reset poses for one env reset."""

    agent_ids: Sequence[str]
    n_agents: int
    agent_index: Mapping[str, int]
    options: Optional[Mapping[str, Any]] = None
    current_start_poses: Optional[np.ndarray] = None
    random_spawn_enabled: bool = False
    random_spawn_allow_reuse: bool = False
    spawn_points: Mapping[str, np.ndarray] = field(default_factory=dict)
    spawn_point_names: Sequence[str] = field(default_factory=tuple)
    rng: Optional[np.random.Generator] = None
    seed: int = 0


@dataclass(frozen=True)
class SpawnResult:
    """Resolved reset spawn payload consumed by the env."""

    poses: Optional[np.ndarray]
    velocities: Optional[np.ndarray]
    spawn_mapping: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    locked_velocities: Dict[str, float] = field(default_factory=dict)
    lock_speed_steps: int = 0
    update_start_poses: bool = False


@dataclass(frozen=True)
class CenterlineSpawnResult:
    """Resolved centerline-relative spawn payload."""

    poses: np.ndarray
    velocities: Dict[str, float]
    metadata: Dict[str, float]
    next_index: int


def extract_spawn_points(
    map_data: Optional[Any],
    metadata: Mapping[str, Any],
) -> Dict[str, np.ndarray]:
    """Extract named spawn poses from preloaded map data or map metadata."""

    spawn_points: Dict[str, np.ndarray] = {}
    if map_data is not None:
        candidate = getattr(map_data, "spawn_points", None)
        if isinstance(candidate, Mapping):
            for name, value in candidate.items():
                arr = np.asarray(value, dtype=np.float32)
                if arr.ndim == 1 and arr.shape[0] >= 2:
                    spawn_points[str(name)] = arr
    if spawn_points:
        return spawn_points

    annotations = metadata.get("annotations", {})
    if isinstance(annotations, Mapping):
        points = annotations.get("spawn_points")
    else:
        points = None
    if isinstance(points, Mapping):
        iterable = points.items()
    elif isinstance(points, list):
        iterable = (
            (entry.get("name"), entry.get("pose"))
            for entry in points
            if isinstance(entry, Mapping)
        )
    else:
        iterable = []

    for name, value in iterable:
        if name is None:
            continue
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 1 and arr.shape[0] >= 2:
            spawn_points[str(name)] = arr
    return spawn_points


def load_spawn_points_from_map(map_path: str, spawn_names: Sequence[str]) -> np.ndarray:
    """Load named spawn point poses from a map YAML file."""

    map_yaml_path = Path(map_path)
    if not map_yaml_path.exists():
        raise FileNotFoundError(f"Map YAML not found: {map_path}")

    with open(map_yaml_path, "r") as handle:
        map_data = yaml.safe_load(handle) or {}

    spawn_points = map_data.get("annotations", {}).get("spawn_points", [])
    spawn_dict = {sp["name"]: sp["pose"] for sp in spawn_points}

    poses = []
    for name in spawn_names:
        if name not in spawn_dict:
            available = list(spawn_dict.keys())
            raise ValueError(
                f"Spawn point '{name}' not found in map. Available: {available}"
            )
        poses.append(spawn_dict[name])

    return np.array(poses, dtype=np.float64)


def pose3_from_spawn(value: np.ndarray) -> np.ndarray:
    """Normalize a spawn pose row to [x, y, theta]."""

    raw = np.asarray(value, dtype=np.float32).reshape(-1)
    pose = np.zeros(3, dtype=np.float32)
    count = min(raw.shape[0], 3)
    if count > 0:
        pose[:count] = raw[:count]
    return pose


def sample_random_spawn(
    *,
    spawn_points: Mapping[str, np.ndarray],
    spawn_point_names: Sequence[str],
    agent_ids: Sequence[str],
    rng: np.random.Generator,
    allow_reuse: bool,
) -> Optional[Tuple[Dict[str, str], np.ndarray]]:
    """Sample one named spawn point per agent."""

    count = len(agent_ids)
    if count == 0 or not spawn_point_names:
        return None

    pool = list(spawn_point_names)
    replace = bool(allow_reuse) or len(pool) < count
    selected_names = rng.choice(pool, size=count, replace=replace)
    if isinstance(selected_names, np.ndarray):
        selected_list = [str(name) for name in selected_names.tolist()]
    else:
        selected_list = [str(name) for name in selected_names]

    pose_rows = []
    spawn_mapping: Dict[str, str] = {}
    for idx, name in enumerate(selected_list):
        raw = spawn_points.get(name)
        if raw is None:
            continue
        pose_rows.append(pose3_from_spawn(raw))
        if idx < count:
            spawn_mapping[str(agent_ids[idx])] = name

    if len(pose_rows) != count:
        return None
    return spawn_mapping, np.stack(pose_rows, axis=0)


def sample_centerline_relative_spawn(
    *,
    spawn_policy: Optional[Any],
    centerline: Optional[np.ndarray],
    map_split_mode: str,
    spawn_centerline_cfg: Mapping[str, Any],
    spawn_offsets_cfg: Mapping[str, Any],
    spawn_target_cfg: Mapping[str, Any],
    spawn_ego_cfg: Mapping[str, Any],
    agent_ids: Sequence[str],
    rng: np.random.Generator,
    current_index: int,
    walls: Optional[Mapping[str, np.ndarray]] = None,
) -> Optional[CenterlineSpawnResult]:
    """Generate legacy centerline-relative reset poses."""

    if str(spawn_policy or "").lower() != "centerline_relative":
        return None
    if centerline is None or centerline.shape[0] == 0:
        return None

    mode = str(spawn_centerline_cfg.get("mode", "random")).lower()
    if str(map_split_mode).lower() == "eval":
        mode_eval = spawn_centerline_cfg.get("mode_eval")
        if mode_eval is not None:
            mode = str(mode_eval).lower()
        elif mode == "random":
            mode = "round_robin"

    min_progress = float(spawn_centerline_cfg.get("min_progress", 0.05))
    max_progress = float(spawn_centerline_cfg.get("max_progress", 0.95))
    avoid_finish = bool(spawn_centerline_cfg.get("avoid_finish", True))
    if avoid_finish:
        min_progress = max(min_progress, 0.05)
        max_progress = min(max_progress, 0.95)

    n_points = centerline.shape[0]
    min_idx = int(max(0, min_progress * (n_points - 1)))
    max_idx = int(min(n_points - 1, max_progress * (n_points - 1)))
    if max_idx <= min_idx:
        min_idx = 0
        max_idx = n_points - 1

    next_index = int(current_index)
    if mode == "round_robin":
        idx = next_index
        if idx < min_idx or idx > max_idx:
            idx = min_idx
        next_index = idx + 1
        if next_index > max_idx:
            next_index = min_idx
    else:
        idx = int(rng.integers(min_idx, max_idx + 1))

    s_offset = float(spawn_offsets_cfg.get("s_offset", 0.0))
    d_offset = float(spawn_offsets_cfg.get("d_offset", 0.0))
    d_max = spawn_offsets_cfg.get("d_max")
    d_max = float(d_max) if d_max is not None else None

    spacing = centerline_arc_length(centerline) / max(n_points - 1, 1)
    offset_idx = int(round(s_offset / spacing)) if spacing > 0.0 else 0
    ego_idx = (idx + offset_idx) % n_points

    target_enabled = bool(spawn_target_cfg.get("enabled", True))
    target_pose = centerline_pose(centerline, idx)
    ego_pose = centerline_pose(centerline, ego_idx)
    d_offset = clamp_lateral_offset(centerline, ego_idx, d_offset, d_max, walls)

    normal = np.array([-np.sin(ego_pose[2]), np.cos(ego_pose[2])], dtype=np.float32)
    ego_pose[:2] = ego_pose[:2] + normal * d_offset

    poses = np.zeros((len(agent_ids), 3), dtype=np.float32)
    for index, agent_id in enumerate(agent_ids):
        if target_enabled and agent_id == "car_1":
            poses[index] = target_pose
        else:
            poses[index] = ego_pose

    speeds: Dict[str, float] = {}
    target_speed = float(spawn_target_cfg.get("speed", 0.0))
    ego_speed = float(spawn_ego_cfg.get("speed", target_speed))
    for agent_id in agent_ids:
        if target_enabled and agent_id == "car_1":
            speeds[str(agent_id)] = target_speed
        else:
            speeds[str(agent_id)] = ego_speed

    metadata = {
        "spawn_s": float(idx) / max(n_points - 1, 1),
        "spawn_d": float(d_offset),
    }
    return CenterlineSpawnResult(
        poses=poses,
        velocities=speeds,
        metadata=metadata,
        next_index=next_index,
    )


def clamp_lateral_offset(
    centerline: np.ndarray,
    index: int,
    d_offset: float,
    d_max: Optional[float],
    walls: Optional[Mapping[str, np.ndarray]] = None,
) -> float:
    limit = d_max if d_max is not None else None
    if walls:
        point = centerline[index, :2].astype(np.float32)
        distances = []
        for wall_points in walls.values():
            if wall_points is None or len(wall_points) == 0:
                continue
            diffs = wall_points[:, :2] - point
            dist = float(np.min(np.linalg.norm(diffs, axis=1)))
            distances.append(dist)
        if distances:
            wall_limit = max(min(distances) - 0.1, 0.0)
            limit = wall_limit if limit is None else min(limit, wall_limit)
    if limit is None:
        return float(d_offset)
    return float(np.clip(d_offset, -limit, limit))


def _normalize_option_poses(options: Mapping[str, Any]) -> Optional[np.ndarray]:
    if "poses" not in options:
        return None
    poses = np.array(options["poses"], dtype=np.float32)
    if poses.ndim == 1:
        poses = np.expand_dims(poses, axis=0)
    return poses


def _normalize_option_velocities(
    options: Mapping[str, Any],
    *,
    n_agents: int,
    agent_index: Mapping[str, int],
) -> Tuple[Optional[np.ndarray], Dict[str, float]]:
    if "velocities" not in options:
        return None, {}

    raw = options["velocities"]
    if isinstance(raw, Mapping):
        velocities = np.full(n_agents, np.nan, dtype=np.float32)
        locked: Dict[str, float] = {}
        for agent_id, value in raw.items():
            if agent_id in agent_index:
                velocity = float(value)
                velocities[agent_index[agent_id]] = velocity
                locked[str(agent_id)] = velocity
        return velocities, locked

    velocities = np.array(raw, dtype=np.float32)
    if velocities.ndim == 0:
        velocities = np.array([velocities], dtype=np.float32)
    return velocities, {}


def resolve_reset_spawn(
    request: SpawnRequest,
    *,
    centerline_spawn_fn: Optional[CenterlineSpawnFn] = None,
) -> SpawnResult:
    """Resolve reset poses while preserving legacy precedence.

    Precedence:
    1. explicit options["poses"]
    2. centerline-relative spawn callback
    3. random named spawn points
    4. configured current_start_poses
    """

    options = request.options if isinstance(request.options, Mapping) else None
    velocities: Optional[np.ndarray] = None
    locked_velocities: Dict[str, float] = {}
    lock_speed_steps = 0

    if options is not None:
        poses = _normalize_option_poses(options)
        velocities, locked_velocities = _normalize_option_velocities(
            options,
            n_agents=request.n_agents,
            agent_index=request.agent_index,
        )
        if "lock_speed_steps" in options:
            lock_speed_steps = max(0, int(options["lock_speed_steps"]))
        if poses is not None:
            return SpawnResult(
                poses=poses,
                velocities=velocities,
                locked_velocities=locked_velocities,
                lock_speed_steps=lock_speed_steps,
                update_start_poses=True,
            )

    if centerline_spawn_fn is not None:
        centerline_spawn = centerline_spawn_fn()
        if centerline_spawn is not None:
            poses, centerline_velocities, metadata = centerline_spawn
            velocity_array = _velocity_dict_to_array(
                centerline_velocities,
                n_agents=request.n_agents,
                agent_index=request.agent_index,
            )
            return SpawnResult(
                poses=poses,
                velocities=velocity_array,
                metadata=dict(metadata),
                locked_velocities=locked_velocities,
                lock_speed_steps=lock_speed_steps,
                update_start_poses=True,
            )

    if request.random_spawn_enabled:
        rng = request.rng or np.random.default_rng(request.seed)
        sampled = sample_random_spawn(
            spawn_points=request.spawn_points,
            spawn_point_names=request.spawn_point_names,
            agent_ids=request.agent_ids,
            rng=rng,
            allow_reuse=request.random_spawn_allow_reuse,
        )
        if sampled is not None:
            spawn_mapping, sampled_poses = sampled
            return SpawnResult(
                poses=sampled_poses,
                velocities=velocities,
                spawn_mapping=spawn_mapping,
                locked_velocities=locked_velocities,
                lock_speed_steps=lock_speed_steps,
                update_start_poses=True,
            )

    start_poses = request.current_start_poses
    if start_poses is not None and len(start_poses) > 0:
        return SpawnResult(
            poses=np.asarray(start_poses, dtype=np.float32),
            velocities=velocities,
            locked_velocities=locked_velocities,
            lock_speed_steps=lock_speed_steps,
            update_start_poses=False,
        )

    return SpawnResult(
        poses=None,
        velocities=velocities,
        locked_velocities=locked_velocities,
        lock_speed_steps=lock_speed_steps,
    )


def _velocity_dict_to_array(
    velocity_map: Mapping[str, float],
    *,
    n_agents: int,
    agent_index: Mapping[str, int],
) -> np.ndarray:
    velocities = np.full(n_agents, np.nan, dtype=np.float32)
    for agent_id, value in velocity_map.items():
        if agent_id in agent_index:
            velocities[agent_index[agent_id]] = float(value)
    return velocities
