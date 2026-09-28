"""Repeated 1v1 attack events, measured before target relocation.

Recent proximity is an interaction proxy, not causal attribution. Both reward
and evaluation consume these same facts, including the post-crash survival gate.
"""
from collections.abc import Mapping
import math


def validate_attack(config, agent_ids):
    if config is None:
        return None
    fields = {"ego_id", "target_id", "interaction_distance", "interaction_window_s", "survival_s"}
    if not isinstance(config, Mapping) or set(config) != fields:
        raise ValueError(f"attack_task requires exactly {sorted(fields)}")
    ego, target = config["ego_id"], config["target_id"]
    if ego not in agent_ids or target not in agent_ids or ego == target:
        raise ValueError("attack_task requires distinct known ego_id and target_id")
    for key in fields - {"ego_id", "target_id"}:
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"attack_task.{key} must be finite and positive")
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

    def update(self, *, time, infos, collisions):
        cfg = self.config
        ego, target = infos[cfg["ego_id"]], infos[cfg["target_id"]]
        failed = bool(collisions[cfg["ego_id"]] or ego.get("track_limits", {}).get("exceeded"))
        crashed = bool(collisions[cfg["target_id"]] or target.get("track_limits", {}).get("exceeded"))
        relative = ego.get("target_frenet")
        if relative is None:
            raise ValueError("attack_task requires the ego's designated target_frenet facts")
        distance = math.hypot(relative["delta_s"], relative["delta_d"])
        moving = ego.get("centerline", {}).get("vs", 0.) > .5
        engaged = moving and distance <= cfg["interaction_distance"]
        if engaged and not failed:
            self._last_interaction = time
        eligible = crashed and not failed and time - self._last_interaction <= cfg["interaction_window_s"]
        if eligible:
            self._pending.append(time + cfg["survival_s"])
        if failed:
            self._pending.clear()
        confirmed = sum(deadline <= time + 1e-9 for deadline in self._pending)
        self._pending = [deadline for deadline in self._pending if deadline > time + 1e-9]

        # Signed changes avoid paying indefinitely for following or parked cars.
        limits = target.get("track_limits", {})
        edge = min(abs(limits.get("lateral_error", 0.)) / max(limits.get("half_width", 1.), 1e-6), 1.)
        approach = (max(-1., min(1., self._previous_distance - distance))
                    if moving and self._previous_distance is not None and math.isfinite(distance) else 0.)
        pressure = edge - self._previous_edge if engaged and self._previous_edge is not None else 0.
        if crashed or failed:
            approach = pressure = 0.
            self._previous_distance = self._previous_edge = None
            self._last_interaction = -math.inf
        else:
            self._previous_distance = distance if math.isfinite(distance) else None
            self._previous_edge = edge if engaged else None
        ego["attack"] = dict(target_crash=crashed, eligible_crash=eligible,
                             success=confirmed, ego_failed=failed,
                             approach_delta=approach, edge_delta=pressure)
