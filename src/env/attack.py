"""Attack events for fixed or switching targets, measured before relocation.

Recent proximity is an interaction proxy, not causal attribution. Both reward
and evaluation consume these same facts, including the post-crash survival gate.
"""
from collections.abc import Mapping
import math

import numpy as np

from utils.track_preview import _distance_to_wall


def attack_geometry(ego_pose, target_pose, walls, length, width):
    """Footprint clearances before respawn; intersecting cars are handled by physics."""
    footprint = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]]) * [length / 2, width / 2]

    def rotation(pose):
        c, s = math.cos(pose[2]), math.sin(pose[2])
        return np.array([[c, -s], [s, c]])

    def clearance(pose, polylines):
        rot = rotation(pose)
        corners = footprint @ rot.T + pose[:2]
        result = math.inf
        for wall in polylines:
            wall = np.asarray(wall)[:, :2]
            # For disjoint polygons the closest pair includes a vertex of one.
            local = (wall - pose[:2]) @ rot
            endpoint_distance = np.linalg.norm(
                np.maximum(np.abs(local) - [length / 2, width / 2], 0.), axis=1).min()
            result = min(result, float(endpoint_distance), float(_distance_to_wall(corners, wall).min()))
        return result

    target_corners = footprint @ rotation(target_pose).T + target_pose[:2]
    return dict(ego_clearance=clearance(ego_pose, walls.values()),
                target_clearance=clearance(target_pose, walls.values()),
                vehicle_clearance=clearance(ego_pose, [target_corners]), width=width)


def validate_attack(config, agent_ids):
    if config is None:
        return None
    dynamic = isinstance(config, Mapping) and "target_ids" in config
    fields = {"ego_id", "interaction_distance", "interaction_window_s", "survival_s"} | (
        {"target_ids", "selection"} if dynamic else {"target_id"})
    if (not isinstance(config, Mapping) or not fields <= set(config)
            or set(config) - fields - {"survival_min_progress"}):
        raise ValueError(f"attack_task requires {sorted(fields)}; optional survival_min_progress")
    ego = config["ego_id"]
    targets = config["target_ids"] if dynamic else [config["target_id"]]
    if (not isinstance(targets, list) or not targets or any(not isinstance(a, str) for a in targets)
            or len(set(targets)) != len(targets) or ego not in agent_ids
            or any(t not in agent_ids or t == ego for t in targets)):
        raise ValueError("attack_task requires distinct known ego_id and target_id")
    if dynamic and config["selection"] != "nearest_ahead":
        raise ValueError("attack_task.selection must be nearest_ahead")
    for key in ("interaction_distance", "interaction_window_s", "survival_s"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"attack_task.{key} must be finite and positive")
    progress = config.get("survival_min_progress", 0.)
    if isinstance(progress, bool) or not isinstance(progress, (int, float)) or not math.isfinite(progress) or progress < 0:
        raise ValueError("attack_task.survival_min_progress must be finite and nonnegative")
    return dict(config)


class AttackTracker:
    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        self._last_interaction = -math.inf
        self._pending = []
        self._previous_distance = None
        self._previous_edge = None
        self._last_time = None
        self._forward_distance = 0.

    def update(self, *, time, infos, collisions, selected=True, allow_missing=False, geometry=None):
        cfg = self.config
        ego, target = infos[cfg["ego_id"]], infos[cfg["target_id"]]
        failed = bool(collisions[cfg["ego_id"]] or ego.get("track_limits", {}).get("exceeded"))
        crashed = bool(collisions[cfg["target_id"]] or target.get("track_limits", {}).get("exceeded"))
        relative = ego.get("target_frenet")
        if relative is None and not allow_missing:
            raise ValueError("attack_task requires the ego's designated target_frenet facts")
        distance = math.hypot(relative["delta_s"], relative["delta_d"]) if relative else math.inf
        speed = ego.get("centerline", {}).get("vs", 0.)
        moving = speed > .5
        if self._last_time is not None:
            self._forward_distance += speed * max(0., time - self._last_time)
        self._last_time = time
        engaged = selected and moving and distance <= cfg["interaction_distance"]
        if engaged and not failed:
            self._last_interaction = time
        eligible = crashed and not failed and time - self._last_interaction <= cfg["interaction_window_s"]
        if eligible:
            self._pending.append((time + cfg["survival_s"], self._forward_distance))
        if failed:
            self._pending.clear()
        min_progress = cfg.get("survival_min_progress", 0.)
        confirmed = sum(deadline <= time + 1e-9 and
                        (min_progress == 0. or (moving and self._forward_distance - start >= min_progress))
                        for deadline, start in self._pending)
        self._pending = [(deadline, start) for deadline, start in self._pending if deadline > time + 1e-9]

        # Signed changes avoid paying indefinitely for following or parked cars.
        limits = target.get("track_limits", {})
        edge = min(abs(limits.get("lateral_error", 0.)) / max(limits.get("half_width", 1.), 1e-6), 1.)
        approach = (max(-1., min(1., self._previous_distance - distance))
                    if selected and moving and self._previous_distance is not None and math.isfinite(distance) else 0.)
        pressure = edge - self._previous_edge if engaged and self._previous_edge is not None else 0.
        if crashed or failed:
            approach = pressure = 0.
            self._previous_distance = self._previous_edge = None
            self._last_interaction = -math.inf
        else:
            self._previous_distance = distance if selected and math.isfinite(distance) else None
            self._previous_edge = edge if engaged else None
        ego["attack"] = dict(target_crash=crashed, eligible_crash=eligible,
                             success=confirmed, ego_failed=failed,
                             approach_delta=approach, edge_delta=pressure)
        if geometry is not None:
            ego["attack"]["shaping"] = dict(geometry, distance=distance, moving=moving,
                delta_s=relative["delta_s"], delta_d=relative["delta_d"],
                target_d=limits.get("lateral_error", 0.))


class MultiTargetAttackTracker:
    """Keep interaction and pending survival credit attached to opponent IDs."""

    def __init__(self, config):
        self.config = config
        self.trackers = {aid: AttackTracker({k: v for k, v in config.items()
                         if k not in {"target_ids", "selection"}} | {"target_id": aid})
                         for aid in config["target_ids"]}
        self.reset()

    def reset(self):
        for tracker in self.trackers.values():
            tracker.reset()
        self.previous_target = None

    def update(self, *, time, infos, collisions, active_target):
        ego_id = self.config["ego_id"]
        ego = infos[ego_id]
        neighbors = {n["agent_id"]: n for n in ego.get("frenet_neighbors", [])}
        results = {}
        for aid, tracker in self.trackers.items():
            if active_target != self.previous_target:
                tracker._previous_distance = tracker._previous_edge = None
            local = dict(ego, target_frenet=neighbors.get(aid))
            tracker.update(time=time, infos={**infos, ego_id: local}, collisions=collisions,
                           selected=aid == active_target, allow_missing=True)
            results[aid] = local["attack"]
        self.previous_target = active_target
        ego["attack"] = {key: sum(row[key] for row in results.values()) for key in (
            "target_crash", "eligible_crash", "success", "approach_delta", "edge_delta")}
        ego["attack"]["ego_failed"] = any(row["ego_failed"] for row in results.values())
        ego["attack"]["per_target"] = results
        ego["attack"]["target_id"] = active_target
