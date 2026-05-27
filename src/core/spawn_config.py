"""Spawn config normalization for environment scenario configs.

Translates the new nested ``environment.spawn:`` block (or legacy flat spawn
keys) into canonical flat form so the rest of the code never needs to know
which format the YAML used.

Canonical flat output keys match what ``build_env_kwargs`` already passes
through to ``F110ParallelEnv``, so the env constructor needs no changes.

Nested scenario form (``environment.spawn:``)
--------------------------------------------
::

    environment:
      spawn:
        policy: centerline_relative   # or: random_named, fixed, replay_plan
        enabled: true                 # enables random named-point spawn
        allow_reuse: false            # allow the same point for multiple agents
        centerline:                   # → spawn_centerline sub-config
          mode: random
          min_progress: 0.05
          max_progress: 0.95
          avoid_finish: true
        offsets:                      # → spawn_offsets
          s_offset: 0.0
          d_offset: 0.0
        target:                       # → spawn_target
          enabled: true
          speed: 0.0
        ego:                          # → spawn_ego
          speed: 0.0
        points: [spawn_1, spawn_2]    # → spawn_points (named lookup list)
        start_poses:                  # → start_poses
          - [x, y, theta]

Legacy flat form (all existing scenarios continue to work)
----------------------------------------------------------
::

    environment:
      random_spawn:
        enabled: true
        allow_reuse: true
      spawn_policy: centerline_relative
      spawn_centerline: {mode: random}
      spawn_offsets: {s_offset: 0.0}
      spawn_target: {enabled: true, speed: 0.0}
      spawn_ego: {speed: 0.0}
      spawn_points: [spawn_1, spawn_2]
      start_poses: [[x, y, theta]]
"""
from __future__ import annotations

from typing import Any, Dict, Mapping

# All spawn-related flat keys that the env builder / env constructor recognize.
_SPAWN_FLAT_KEYS = (
    "spawn_policy",
    "random_spawn",
    "random_spawn_allow_reuse",
    "spawn_centerline",
    "spawn_offsets",
    "spawn_target",
    "spawn_ego",
    "spawn_points",
    "start_poses",
)

# Mapping from nested block keys to canonical flat output keys.
_NESTED_SUB_CONFIG_MAP = {
    "centerline": "spawn_centerline",
    "offsets": "spawn_offsets",
    "target": "spawn_target",
    "ego": "spawn_ego",
}

_SENTINEL = object()


def normalize_spawn_config(env_config: Mapping[str, Any]) -> Dict[str, Any]:
    """Return canonical flat spawn keys from *env_config*.

    If *env_config* contains a nested ``spawn:`` block its keys are translated
    to canonical flat equivalents.  Legacy flat keys are passed through
    unchanged when no nested block is present.  When both exist, nested keys
    take precedence field-by-field; legacy flat keys fill in any gaps.

    Only spawn-related keys appear in the returned dict — all other env config
    keys are left to the caller.
    """
    nested = env_config.get("spawn")
    if not isinstance(nested, Mapping):
        # Fast path: no nested block — pass legacy flat keys through as-is.
        return {k: env_config[k] for k in _SPAWN_FLAT_KEYS if k in env_config}

    out: Dict[str, Any] = {}

    # --- policy ---
    if "policy" in nested:
        out["spawn_policy"] = str(nested["policy"])
    elif "spawn_policy" in env_config:
        out["spawn_policy"] = env_config["spawn_policy"]

    # --- random spawn enabled / allow_reuse ---
    rs: Dict[str, Any] = {}
    enabled = nested.get("enabled", _SENTINEL)
    if enabled is not _SENTINEL:
        rs["enabled"] = bool(enabled)
    allow_reuse = nested.get("allow_reuse", _SENTINEL)
    if allow_reuse is not _SENTINEL:
        rs["allow_reuse"] = bool(allow_reuse)

    if rs:
        out["random_spawn"] = rs
    elif "random_spawn" in env_config:
        out["random_spawn"] = env_config["random_spawn"]

    # legacy allow_reuse fallback
    if "random_spawn_allow_reuse" in env_config and "random_spawn" not in out:
        out["random_spawn_allow_reuse"] = env_config["random_spawn_allow_reuse"]

    # --- sub-configs (centerline / offsets / target / ego) ---
    for nested_key, flat_key in _NESTED_SUB_CONFIG_MAP.items():
        if nested_key in nested:
            out[flat_key] = nested[nested_key]
        elif flat_key in nested:
            # allow flat key within the nested block too
            out[flat_key] = nested[flat_key]
        elif flat_key in env_config:
            out[flat_key] = env_config[flat_key]

    # --- spawn_points: use "points" in nested block, "spawn_points" as fallback ---
    if "points" in nested:
        out["spawn_points"] = nested["points"]
    elif "spawn_points" in nested:
        out["spawn_points"] = nested["spawn_points"]
    elif "spawn_points" in env_config:
        out["spawn_points"] = env_config["spawn_points"]

    # --- start_poses ---
    if "start_poses" in nested:
        out["start_poses"] = nested["start_poses"]
    elif "start_poses" in env_config:
        out["start_poses"] = env_config["start_poses"]

    return out
