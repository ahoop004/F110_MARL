"""RewardComposer — assembles RewardComponents from config."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import yaml

from wrappers.rewards.base import RewardComponent
from wrappers.rewards.centerline import (
    CenterlineDeviationPenaltyComponent,
    CenterlineLateralVelocityPenaltyComponent,
    CenterlineProgressComponent,
    CenterlineRewardComponent,
    SteeringPenaltyComponent,
)
from wrappers.rewards.collision import CollisionRewardComponent, SelfCrashPenaltyComponent
from wrappers.rewards.speed import SpeedRewardComponent
from wrappers.rewards.proximity import TargetProximityComponent
from wrappers.rewards.track_edge_pressure import TargetEdgePressureComponent
from wrappers.rewards.lap_completion import LapCompletionComponent
from wrappers.rewards.progress_safety import (
    OfftrackPenaltyComponent,
    ProgressSafetyComponent,
    ReverseProgressPenaltyComponent,
    WrongWayPenaltyComponent,
)
from wrappers.rewards.target_finish import TargetFinishComponent
from wrappers.rewards.target_crash import TargetCrashBonusComponent
from wrappers.rewards.track_completion import (
    FinishAheadBonusComponent,
    ProgressDeltaBonusComponent,
    RelativeProgressBonusComponent,
    StepTimePenaltyComponent,
    TeamProgressBonusComponent,
    TeamRelativeProgressBonusComponent,
)
from wrappers.rewards.timeout import TimeoutPenaltyComponent


COMPONENT_REGISTRY: Dict[str, Type[RewardComponent]] = {
    "centerline": CenterlineRewardComponent,
    "centerline_progress": CenterlineProgressComponent,
    "centerline_lateral_velocity_penalty": CenterlineLateralVelocityPenaltyComponent,
    "centerline_deviation_penalty": CenterlineDeviationPenaltyComponent,
    "steering_penalty": SteeringPenaltyComponent,
    "collision": CollisionRewardComponent,
    "speed": SpeedRewardComponent,
    "target_proximity": TargetProximityComponent,
    "target_edge_pressure": TargetEdgePressureComponent,
    "progress_delta_bonus": ProgressDeltaBonusComponent,
    "relative_progress_bonus": RelativeProgressBonusComponent,
    "finish_ahead_bonus": FinishAheadBonusComponent,
    "team_progress_bonus": TeamProgressBonusComponent,
    "team_relative_progress_bonus": TeamRelativeProgressBonusComponent,
    "step_time_penalty": StepTimePenaltyComponent,
    "lap_completion": LapCompletionComponent,
    "target_finish": TargetFinishComponent,
    "progress_safety": ProgressSafetyComponent,
    "wrong_way_penalty": WrongWayPenaltyComponent,
    "reverse_progress_penalty": ReverseProgressPenaltyComponent,
    "offtrack_penalty": OfftrackPenaltyComponent,
    "target_crash_bonus": TargetCrashBonusComponent,
    "timeout_penalty": TimeoutPenaltyComponent,
    "self_crash_penalty": SelfCrashPenaltyComponent,
}

COMPONENT_ALIASES: Dict[str, str] = {
    "gaplock_pressure": "target_proximity",
    "gaplock_forcing": "target_edge_pressure",
    "terminal_success": "target_crash_bonus",
    "terminal_timeout": "timeout_penalty",
    "terminal_self_crash": "self_crash_penalty",
}


def canonical_component_key(key: str) -> str:
    """Return the preferred generic component key for a config key."""

    return COMPONENT_ALIASES.get(str(key), str(key))


def _component_config(cfg: Dict, canonical_key: str) -> Optional[Dict]:
    comp_cfg = cfg.get(canonical_key)
    if isinstance(comp_cfg, dict):
        return comp_cfg
    for alias, target in COMPONENT_ALIASES.items():
        if target != canonical_key:
            continue
        alias_cfg = cfg.get(alias)
        if isinstance(alias_cfg, dict):
            return alias_cfg
    return None


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_reward_config_file(path: Path, visited: Optional[set[Path]] = None) -> Dict[str, Any]:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Reward config not found: {path}")

    visited = visited or set()
    if path in visited:
        raise ValueError(f"Reward config include cycle detected at: {path}")
    visited.add(path)

    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Reward config must be a YAML mapping: {path}")

    includes = data.pop("includes", None)
    merged: Dict[str, Any] = {}
    if includes:
        if isinstance(includes, (str, Path)):
            includes = [includes]
        if not isinstance(includes, list):
            raise ValueError(f"Reward config 'includes' must be a list: {path}")
        for include_path in includes:
            if not isinstance(include_path, (str, Path)):
                raise ValueError(f"Reward config include entries must be paths: {path}")
            include_obj = (path.parent / include_path).resolve()
            merged = _deep_merge(merged, _load_reward_config_file(include_obj, visited))

    merged = _deep_merge(merged, data)
    visited.remove(path)
    return merged


class RewardComposer:
    """Sums enabled RewardComponents into a total scalar + breakdown dict.

    Built from a config dict (loaded from configs/reward/<name>.yaml).
    """

    def __init__(self, components: List[RewardComponent]) -> None:
        self._components = components

    def reset(self) -> None:
        for c in self._components:
            c.reset()

    def compute(self, step_info: dict) -> Tuple[float, Dict[str, float]]:
        breakdown: Dict[str, float] = {}
        for component in self._components:
            breakdown.update(component.compute(step_info))
        total = sum(breakdown.values())
        return total, breakdown

    @classmethod
    def from_config(cls, reward_config: Dict) -> "RewardComposer":
        """Build from a parsed reward config dict."""
        cfg = reward_config.get("reward", reward_config)
        components: List[RewardComponent] = []

        for key, component_cls in COMPONENT_REGISTRY.items():
            comp_cfg = _component_config(cfg, key) or {}
            if isinstance(comp_cfg, dict) and comp_cfg.get("enabled", False):
                components.append(component_cls(comp_cfg))

        if not components:
            raise ValueError("RewardComposer: no components enabled in reward config.")

        return cls(components)

    @classmethod
    def from_file(cls, path: str) -> "RewardComposer":
        """Load from a YAML reward config file path."""
        reward_config = _load_reward_config_file(Path(path))
        return cls.from_config(reward_config)
