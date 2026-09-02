"""Derive environment-side feature work from observation and reward configs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Set, Tuple

import yaml


_CENTERLINE_OBSERVATIONS = {
    "centerline_ego_state",
    "progress",
    "frenet_vehicle_track",
    "frenet_neighbors",
}
_CENTERLINE_REWARDS = {
    "centerline",
    "centerline_progress",
    "centerline_lateral_velocity_penalty",
    "centerline_deviation_penalty",
    "progress_delta_bonus",
    "relative_progress_bonus",
    "team_progress_bonus",
    "team_relative_progress_bonus",
    "wrong_way_penalty",
    "reverse_progress_penalty",
    "offtrack_penalty",
    "progress_safety",
}


@dataclass(frozen=True)
class EnvironmentFeatureRequirements:
    """Immutable aggregate of simulator facts required by configured consumers."""

    centerline_progress_agents: Tuple[str, ...] = ()
    frenet_vehicle_state_agents: Tuple[str, ...] = ()
    track_preview_agents: Tuple[str, ...] = ()
    frenet_neighbor_agents: Tuple[str, ...] = ()
    centerline_render: bool = False

    @property
    def requires_centerline_facts(self) -> bool:
        return bool(self.centerline_progress_agents)

    @property
    def requires_track_preview(self) -> bool:
        return bool(self.track_preview_agents)

    @property
    def requires_frenet_neighbors(self) -> bool:
        return bool(self.frenet_neighbor_agents)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "centerline_progress_agents": list(self.centerline_progress_agents),
            "frenet_vehicle_state_agents": list(self.frenet_vehicle_state_agents),
            "track_preview_agents": list(self.track_preview_agents),
            "frenet_neighbor_agents": list(self.frenet_neighbor_agents),
            "centerline_render": self.centerline_render,
        }


def _load_config(path: Path, visited: Optional[Set[Path]] = None) -> Dict[str, Any]:
    """Load and merge a component config, including reward-style includes."""

    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Feature consumer config not found: {resolved}")
    active = visited or set()
    if resolved in active:
        raise ValueError(f"Feature consumer include cycle detected at: {resolved}")
    active.add(resolved)
    with resolved.open() as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Feature consumer config must be a YAML mapping: {resolved}")

    includes = data.pop("includes", None)
    merged: Dict[str, Any] = {}
    if includes:
        include_paths = [includes] if isinstance(includes, (str, Path)) else includes
        if not isinstance(include_paths, list):
            raise ValueError(f"Feature consumer 'includes' must be a list: {resolved}")
        for include_path in include_paths:
            child = _load_config(resolved.parent / str(include_path), active)
            merged = _deep_merge(merged, child)
    active.remove(resolved)
    return _deep_merge(merged, data)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_config(reference: Any, scenario_dir: Path) -> Dict[str, Any]:
    if isinstance(reference, Mapping):
        return dict(reference)
    if isinstance(reference, str):
        return _load_config(scenario_dir / reference)
    return {}


def _enabled_keys(config: Mapping[str, Any], section: str) -> Set[str]:
    block = config.get(section, config)
    if not isinstance(block, Mapping):
        return set()
    return {
        str(key)
        for key, value in block.items()
        if isinstance(value, Mapping) and bool(value.get("enabled", False))
    }


def derive_environment_feature_requirements(
    agent_configs: Mapping[str, Mapping[str, Any]],
    *,
    scenario_dir: Path,
    centerline_render: bool = False,
) -> EnvironmentFeatureRequirements:
    """Aggregate feature requirements across every configured agent consumer."""

    centerline: Set[str] = set()
    vehicle_state: Set[str] = set()
    preview: Set[str] = set()
    neighbors: Set[str] = set()

    for raw_agent_id, agent_config in agent_configs.items():
        agent_id = str(raw_agent_id)
        observation = _resolve_config(agent_config.get("observation"), scenario_dir)
        observation_keys = _enabled_keys(observation, "observation")
        reward = _resolve_config(agent_config.get("reward"), scenario_dir)
        reward_keys = _enabled_keys(reward, "reward")

        if observation_keys & _CENTERLINE_OBSERVATIONS or reward_keys & _CENTERLINE_REWARDS:
            centerline.add(agent_id)
        if "frenet_vehicle_track" in observation_keys:
            vehicle_state.add(agent_id)
            preview.add(agent_id)
        if "frenet_neighbors" in observation_keys:
            neighbors.add(agent_id)

    return EnvironmentFeatureRequirements(
        centerline_progress_agents=tuple(sorted(centerline)),
        frenet_vehicle_state_agents=tuple(sorted(vehicle_state)),
        track_preview_agents=tuple(sorted(preview)),
        frenet_neighbor_agents=tuple(sorted(neighbors)),
        centerline_render=bool(centerline_render),
    )


__all__ = [
    "EnvironmentFeatureRequirements",
    "derive_environment_feature_requirements",
]
