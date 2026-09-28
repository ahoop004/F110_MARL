"""Per-event recovery configuration and controller-state synchronization."""
from collections.abc import Mapping
import math

import numpy as np

from utils.centerline import project_to_centerline


def validate_respawn(config, agent_ids):
    if config is None:
        return {}
    if not isinstance(config, Mapping) or set(config) - {
        "boundary_agents", "collision_agents", "collision_placement", "random_ahead"
    }:
        raise ValueError("respawn accepts boundary_agents, collision_agents, collision_placement and random_ahead")
    for key in ("boundary_agents", "collision_agents"):
        ids = config.get(key, [])
        if (not isinstance(ids, list) or any(not isinstance(a, str) for a in ids)
                or len(ids) != len(set(ids)) or not set(ids) <= set(agent_ids)):
            raise ValueError(f"respawn.{key} must list unique known agent IDs")
    placement = config.get("collision_placement", "nearest_centerline")
    if placement not in {"nearest_centerline", "random_ahead"}:
        raise ValueError("respawn.collision_placement must be nearest_centerline or random_ahead")
    if placement == "random_ahead":
        ahead = config.get("random_ahead")
        if not isinstance(ahead, Mapping) or set(ahead) != {"ego_id", "min_gap", "max_gap", "speed", "clearance"}:
            raise ValueError("respawn.random_ahead requires ego_id, min_gap, max_gap, speed and clearance")
        recovered = set(config.get("boundary_agents", [])) | set(config.get("collision_agents", []))
        if ahead["ego_id"] not in agent_ids or ahead["ego_id"] in recovered or not recovered:
            raise ValueError("respawn.random_ahead.ego_id must name a non-respawning agent")
        for key in ("min_gap", "max_gap", "speed", "clearance"):
            v = ahead[key]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
                raise ValueError(f"respawn.random_ahead.{key} must be finite and nonnegative")
        if not 0 < ahead["min_gap"] < ahead["max_gap"] or ahead["clearance"] <= 0:
            raise ValueError("respawn.random_ahead requires 0 < min_gap < max_gap and positive clearance")
    elif "random_ahead" in config:
        raise ValueError("respawn.random_ahead requires random_ahead placement")
    return dict(config)


def sample_ahead_pose(*, geometry, ego_pose, other_positions, rng, config, walls, length, width):
    """Uniform forward arc-distance draws, rejecting occupied or wall-adjacent poses."""
    if not geometry.closed or config["max_gap"] >= geometry.total_length / 2:
        raise ValueError("random_ahead respawn requires a closed track and max_gap < half its length")
    origin = project_to_centerline(geometry, ego_pose[:2], ego_pose[2]).arc_length
    radius = math.hypot(length, width) / 2 + .05
    wall_segments = []
    for points in (walls or {}).values():
        points = np.asarray(points)[:, :2]
        if len(points) > 1:
            wall_segments.append((points[:-1], np.diff(points, axis=0)))
    for _ in range(256):
        s = (origin + rng.uniform(config["min_gap"], config["max_gap"])) % geometry.total_length
        k = min(np.searchsorted(geometry.arc_lengths, s, side="right") - 1, len(geometry.segment_lengths) - 1)
        vector = geometry.segment_vectors[k]
        xy = geometry.segment_starts[k] + vector * ((s - geometry.arc_lengths[k]) / geometry.segment_lengths[k])
        if len(other_positions) and np.any(np.linalg.norm(other_positions - xy, axis=1) <= max(config["clearance"], 2 * length)):
            continue
        clear = True
        for starts, segments in wall_segments:
            fractions = np.clip(np.sum((xy - starts) * segments, axis=1) /
                                np.maximum(np.sum(segments * segments, axis=1), 1e-12), 0., 1.)
            if np.any(np.linalg.norm(xy - starts - fractions[:, None] * segments, axis=1) <= radius):
                clear = False
                break
        if clear:
            return np.array([*xy, np.arctan2(vector[1], vector[0])])
    raise RuntimeError("No safe random-ahead respawn within configured gap bounds")


def reset_respawned(infos, *, controllers, actions, observations):
    """Reset physical command memory, keeping policy weights and reward history."""
    respawned = {aid for aid, info in infos.items() if info.get("respawned")}
    for aid in respawned:
        for components in (controllers, actions, observations):
            component = components.get(aid)
            if component is not None and hasattr(component, "reset"):
                component.reset()
    return respawned
