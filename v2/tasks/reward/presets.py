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
    "gaplock_pinch_pocket": {
        "pocket_reward_weight": 0.3,
        "pinch_anchor_dx": 1.2,
        "pinch_anchor_dy": 0.7,
        "pinch_sigma": 0.5,
        "pinch_in_front_x_min": 0.4,
        "pinch_pressure_distance": 1.5,
        "pinch_pressure_heading_tol_deg": 30.0,
        "pinch_pressure_recent_seconds": 1.5,
    },
    "gaplock_potential_field": {
        "potential_field_weight": 0.05,
        "potential_field_sigma": 0.5,
        "potential_field_peak": 1.0,
        "potential_field_floor": -1.0,
        "potential_field_power": 2.0,
        "potential_field_time_scaled": True,
    },
    "gaplock_force_clearance": {
        "force_reward_weight": 1.0,
        "force_reward_enable_ema": True,
        "force_reward_ema_alpha": 0.4,
        "force_reward_band_min": 0.4,
        "force_reward_band_max": 3.0,
        "force_reward_clip": 0.2,
        "force_reward_time_scaled": True,
        "lidar_filter_range_min": 0.3,
        "target_neighborhood_r_min": 0.3,
        "target_neighborhood_r_max": 6.0,
        "target_neighborhood_x_band": 2.0,
        "target_side_y_epsilon": 0.05,
    },
    "gaplock_force_turn": {
        "turn_reward_weight": 0.2,
        "turn_reward_clip": 0.2,
        "turn_reward_time_scaled": True,
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
