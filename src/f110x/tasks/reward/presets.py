"""Reward preset helpers providing reusable parameter bundles."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple


REWARD_FEATURE_PRESETS: Dict[str, Dict[str, Any]] = {
    "progress_basic": {
        "progress_weight": 1.0,
        "speed_weight": 0.0,
    },
    "progress_navigation": {
        "progress_weight": 4.0,
        "lateral_penalty": 0.01,
        "heading_penalty": 0.01,
        "idle_penalty": 0.2,
        "idle_penalty_steps": 40,
    },
    "collision_penalty": {
        "collision_penalty": -2.0,
    },
    "idle_guard": {
        "idle_penalty": 0.5,
        "idle_penalty_steps": 25,
    },
    "waypoint_bonus": {
        "waypoint_bonus": 1.0,
        "waypoint_spacing": 5.0,
    },
    "milestone_bonus": {
        "milestone_spacing": 2.0,
        "milestone_bonus": 2.0,
    },
    "gaplock_offense": {
        "target_crash_reward": 10.0,
        "self_collision_penalty": -5.0,
        "truncation_penalty": -1.0,
    },
}


def resolve_reward_presets(feature_names: Iterable[Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Resolve a list of named reward presets into a merged parameter dictionary."""

    merged: Dict[str, Any] = {}
    notes: List[str] = []
    for raw_name in feature_names:
        if raw_name is None:
            continue
        name = str(raw_name).strip().lower()
        if not name:
            continue
        preset = REWARD_FEATURE_PRESETS.get(name)
        if not preset:
            notes.append(f"Unknown reward preset '{raw_name}' ignored")
            continue
        merged.update(preset)
        notes.append(f"Applied reward preset '{name}'")
    return merged, notes


__all__ = ["REWARD_FEATURE_PRESETS", "resolve_reward_presets"]
