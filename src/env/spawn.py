"""Spawn point parsing and sampling helpers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple
from pathlib import Path

import numpy as np
import yaml

from src.env.types import SpawnPlan, SpawnState  # noqa: F401  (re-exported for tests)
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
    random_spawn_min_distance: float = 0.5
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
    min_distance: float = 0.5,
) -> Optional[Tuple[Dict[str, str], np.ndarray]]:
    """Sample non-overlapping named spawn points using rejection sampling."""

    count = len(agent_ids)
    if count == 0 or not spawn_point_names:
        return None

    min_distance = max(float(min_distance), 0.0)
    pool = list(spawn_point_names)

    # If reuse is disabled, there must be enough candidate points.
    if not allow_reuse and len(pool) < count:
        return None

    selected_names = []
    selected_poses = []

    # Candidates that have not yet been consumed when reuse is disabled.
    available = pool.copy()

    max_attempts = max(100, count * len(pool) * 10)

    for _ in range(max_attempts):
        if len(selected_names) == count:
            break

        candidates = pool if allow_reuse else available
        if not candidates:
            break

        name = str(rng.choice(candidates))
        raw = spawn_points.get(name)

        if raw is None:
            if not allow_reuse and name in available:
                available.remove(name)
            continue

        pose = pose3_from_spawn(raw)

        # Rejection test: reject candidate if it is too close
        # to any vehicle already accepted.
        too_close = any(
            np.linalg.norm(pose[:2] - existing[:2]) < min_distance
            for existing in selected_poses
        )

        if too_close:
            if not allow_reuse and name in available:
                available.remove(name)
            continue

        selected_names.append(name)
        selected_poses.append(pose)

        if not allow_reuse and name in available:
            available.remove(name)

    if len(selected_names) != count:
        raise ValueError(
            "Unable to sample non-overlapping vehicle spawn points: "
            f"requested {count} agents with "
            f"min_distance={min_distance:.3f} m "
            f"from {len(pool)} available spawn points."
        )

    spawn_mapping = {
        str(agent_ids[i]): selected_names[i]
        for i in range(count)
    }

    return spawn_mapping, np.stack(selected_poses, axis=0)
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


def _apply_spawn_plan(plan: SpawnPlan, request: SpawnRequest) -> SpawnResult:
    """Convert a deterministic *plan* into a :class:`SpawnResult`.

    Agents present in *plan.states* receive their stored pose; agents absent
    from the plan receive ``[0, 0, 0]``.  ``plan_id`` and per-agent
    ``spawn_id`` values are forwarded in the result metadata.

    ``update_start_poses`` is ``False`` so replay plans do not overwrite the
    live start-pose state used by subsequent random/centerline spawns.
    """
    plan_map: Dict[str, SpawnState] = {s.agent_id: s for s in plan.states}
    poses = np.zeros((request.n_agents, 3), dtype=np.float32)
    spawn_mapping: Dict[str, str] = {}
    spawn_ids: Dict[str, Optional[str]] = {}

    for agent_id, idx in request.agent_index.items():
        state = plan_map.get(agent_id)
        if state is not None:
            raw = np.asarray(state.pose, dtype=np.float32).reshape(-1)
            n = min(raw.shape[0], 3)
            poses[idx, :n] = raw[:n]
            if state.spawn_id is not None:
                spawn_mapping[agent_id] = state.spawn_id
                spawn_ids[agent_id] = state.spawn_id

    metadata: Dict[str, Any] = {}
    if plan.plan_id is not None:
        metadata["plan_id"] = plan.plan_id
    if spawn_ids:
        metadata["spawn_ids"] = spawn_ids

    return SpawnResult(
        poses=poses,
        velocities=None,
        spawn_mapping=spawn_mapping,
        metadata=metadata,
        update_start_poses=False,
    )


def resolve_reset_spawn(
    request: SpawnRequest,
    *,
    centerline_spawn_fn: Optional[CenterlineSpawnFn] = None,
    spawn_plan: Optional[SpawnPlan] = None,
) -> SpawnResult:
    """Resolve reset poses while preserving legacy precedence.

    Precedence (highest to lowest):
    0. deterministic *spawn_plan* (eval / offline replay)
    1. explicit options["poses"]
    2. centerline-relative spawn callback
    3. random named spawn points
    4. configured current_start_poses
    """
    # Precedence 0: deterministic plan overrides everything.
    if spawn_plan is not None:
        return _apply_spawn_plan(spawn_plan, request)

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
            min_distance=request.random_spawn_min_distance,
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
