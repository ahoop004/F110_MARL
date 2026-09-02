"""Build F110 environments from expanded scenario config."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping
import re

import numpy as np

from src.core.map_selection import relative_yaml_name
from src.core.spawn_config import normalize_spawn_config
from src.env import F110ParallelEnv
from src.env.spawn import load_spawn_points_from_map
from src.utils.map_loader import MapLoader


def derive_num_agents(
    env_config: Mapping[str, Any],
    agent_configs: Mapping[str, Any],
) -> int:
    """Derive contiguous car count from configured agent IDs."""

    indices = []
    for agent_id in agent_configs or {}:
        match = re.search(r"(\d+)$", str(agent_id))
        if match:
            indices.append(int(match.group(1)))
    if indices:
        return max(indices) + 1
    return int(env_config.get("num_agents", env_config.get("n_agents", 1)))


def build_env_kwargs(
    env_config: Mapping[str, Any],
    agent_configs: Mapping[str, Any],
    seed: Any = None,
) -> Dict[str, Any]:
    """Translate scenario env config into F110ParallelEnv constructor kwargs.

    Handles both the new nested ``environment.spawn:`` block and legacy flat
    spawn keys via :func:`normalize_spawn_config`.  All other env config keys
    are read directly from *env_config*.
    """
    # Merge normalized spawn keys over the raw env config so passthrough logic
    # below picks up canonical flat keys regardless of which YAML form was used.
    spawn_norm = normalize_spawn_config(env_config)
    # Build a shallow merged view: spawn_norm overrides env_config for spawn keys.
    effective_config: Dict[str, Any] = dict(env_config)
    effective_config.update(spawn_norm)

    num_agents = derive_num_agents(effective_config, agent_configs)
    env_seed = env_config.get("seed", seed)
    env_kwargs: Dict[str, Any] = {
        "map": env_config["map"],
        "n_agents": num_agents,
        "timestep": env_config.get("timestep", 0.01),
        "max_steps": env_config.get("max_steps", 5000),
    }
    if "map_dir" in env_config:
        env_kwargs["map_dir"] = env_config["map_dir"]
    if "map_yaml" in env_config:
        env_kwargs["map_yaml"] = env_config["map_yaml"]
    if "map_ext" in env_config:
        env_kwargs["map_ext"] = env_config["map_ext"]
    if env_seed is not None:
        env_kwargs["seed"] = env_seed

    if "lidar_beams" in env_config:
        env_kwargs["lidar_beams"] = env_config["lidar_beams"]
    if "lidar_range" in env_config:
        env_kwargs["lidar_range"] = env_config["lidar_range"]
    if "render" in env_config:
        env_kwargs["render_mode"] = "human" if env_config["render"] else None
    if "vehicle_params" in env_config:
        env_kwargs["vehicle_params"] = env_config["vehicle_params"]

    passthrough_keys = [
        "map_root",
        "map_bundle",
        "map_bundle_active",
        "map_bundles",
        "map_bundles_train",
        "map_bundles_eval",
        "map_split_mode",
        "map_cycle",
        "map_pick",
        "epoch_shuffle",
        "centerline_autoload",
        "centerline_csv",
        "centerline_render",
        "centerline_features",
        "feature_requirements",
        "track_preview",
        "action_repeat",
        "walls_autoload",
        "walls_csv",
        "track_threshold",
        "track_inverted",
        "spawn_policy",
        "spawn_centerline",
        "spawn_offsets",
        "spawn_target",
        "spawn_ego",
        "random_spawn",
        "random_spawn_allow_reuse",
        "controlled_agents",
        "trainable_agents",
        "fixed_policy_agents",
        "episode_termination",
        "terminate_on_any_done",
        "terminate_on_collision",
        "target_laps",
        "lap_counting",
        "terminal_agents",
        "finish_line",
        "info_level",
        "rendering",        # nested render config: vehicle_colors, hud, etc.
    ]
    for key in passthrough_keys:
        if key in effective_config and key not in env_kwargs:
            env_kwargs[key] = effective_config[key]

    return env_kwargs


def maybe_load_map_data(env_config: Mapping[str, Any]) -> Any:
    centerline_requested = bool(
        env_config.get("centerline_autoload")
        or env_config.get("centerline_csv")
        or env_config.get("centerline_render")
        or env_config.get("centerline_features")
    )
    if not centerline_requested:
        return None

    map_loader_cfg = dict(env_config)
    map_loader_cfg["centerline_autoload"] = bool(
        env_config.get("centerline_autoload", False)
        or env_config.get("centerline_csv")
        or env_config.get("centerline_render")
        or env_config.get("centerline_features")
    )
    map_value = map_loader_cfg.get("map")
    if isinstance(map_value, str):
        map_path = Path(map_value)
        if map_path.parent != Path(".") and not map_loader_cfg.get("map_dir"):
            map_file = map_path if map_path.suffix else map_path.with_suffix(".yaml")
            map_loader_cfg["map_dir"] = str(map_file.parent)
            if not map_loader_cfg.get("map_yaml"):
                map_loader_cfg["map_yaml"] = map_file.name
            map_loader_cfg["map"] = map_file.name
    try:
        map_loader = MapLoader(base_dir=Path.cwd())
        return map_loader.load(map_loader_cfg)
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("Failed to load centerline data: %s", exc)
        return None


def create_environment(
    env_config: Mapping[str, Any],
    agent_configs: Mapping[str, Any],
    seed: Any = None,
) -> F110ParallelEnv:
    env_kwargs = build_env_kwargs(env_config, agent_configs, seed)
    map_data = maybe_load_map_data(env_config)

    if map_data is not None:
        env_kwargs["map_data"] = map_data
        map_dir_value = env_kwargs.get("map_dir")
        if map_dir_value:
            env_kwargs["map"] = relative_yaml_name(Path(map_dir_value), map_data.yaml_path)
            env_kwargs["map_yaml"] = env_kwargs["map"]
        else:
            env_kwargs["map"] = str(map_data.yaml_path)

    # Resolve spawn_points / start_poses using the normalized config so that
    # the nested spawn.points form is also handled.
    spawn_norm = normalize_spawn_config(env_config)
    if "spawn_points" in spawn_norm:
        spawn_names = spawn_norm["spawn_points"]
        map_path = env_config["map"]
        env_kwargs["start_poses"] = load_spawn_points_from_map(map_path, spawn_names)
    elif "start_poses" in spawn_norm and "start_poses" not in env_kwargs:
        env_kwargs["start_poses"] = np.array(spawn_norm["start_poses"], dtype=np.float64)

    env = F110ParallelEnv(**env_kwargs)
    if map_data is not None and map_data.centerline is not None:
        env.set_centerline(map_data.centerline, path=map_data.centerline_path)
        env.register_centerline_usage(
            require_render=bool(env_config.get("centerline_render")),
            require_features=bool(env_config.get("centerline_features")),
        )
    requirements = env_config.get("feature_requirements")
    if isinstance(requirements, Mapping):
        preview_agents = tuple(requirements.get("track_preview_agents", ()))
        neighbor_agents = tuple(requirements.get("frenet_neighbor_agents", ()))
        if preview_agents and env._track_preview_geometry is None:
            raise ValueError(
                "Frenet track-preview observation requires available centerline "
                f"geometry; requested by agents {preview_agents}."
            )
        if neighbor_agents and (
            env.centerline_points is None or not env.centerline_features_enabled
        ):
            raise ValueError(
                "Frenet-neighbor observation requires enabled centerline facts; "
                f"requested by agents {neighbor_agents}."
            )
    return env
