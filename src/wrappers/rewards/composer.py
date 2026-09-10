"""RewardComposer — assembles RewardComponents from config."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

from core.scenario import load_yaml_config

from wrappers.rewards.base import RewardComponent
from wrappers.rewards.motion import (
    CenterlineDeviationPenaltyComponent,
    CenterlineLateralVelocityPenaltyComponent,
    CenterlineProgressComponent,
    CenterlineRewardComponent,
    SteeringPenaltyComponent,
    ReverseVelocityPenaltyComponent,
    SpeedRewardComponent,
    OfftrackPenaltyComponent,
    ProgressSafetyComponent,
    ReverseProgressPenaltyComponent,
    WrongWayPenaltyComponent,
)
from wrappers.rewards.events import (
    CollisionRewardComponent,
    SelfCrashPenaltyComponent,
    TargetFinishComponent,
    TargetCrashBonusComponent,
    TimeoutPenaltyComponent,
)
from wrappers.rewards.interaction import TargetProximityComponent, TargetEdgePressureComponent
from wrappers.rewards.completion import (
    LapCompletionComponent,
    PerLapBonusComponent,
    FinishAheadBonusComponent,
    ProgressDeltaBonusComponent,
    RelativeProgressBonusComponent,
    StepTimePenaltyComponent,
    TeamProgressBonusComponent,
    TeamRelativeProgressBonusComponent,
)


COMPONENT_REGISTRY: Dict[str, Type[RewardComponent]] = {
    "centerline": CenterlineRewardComponent,
    "centerline_progress": CenterlineProgressComponent,
    "centerline_lateral_velocity_penalty": CenterlineLateralVelocityPenaltyComponent,
    "centerline_deviation_penalty": CenterlineDeviationPenaltyComponent,
    "steering_penalty": SteeringPenaltyComponent,
    "collision": CollisionRewardComponent,
    "speed": SpeedRewardComponent,
    "reverse_velocity_penalty": ReverseVelocityPenaltyComponent,
    "target_proximity": TargetProximityComponent,
    "target_edge_pressure": TargetEdgePressureComponent,
    "progress_delta_bonus": ProgressDeltaBonusComponent,
    "relative_progress_bonus": RelativeProgressBonusComponent,
    "finish_ahead_bonus": FinishAheadBonusComponent,
    "team_progress_bonus": TeamProgressBonusComponent,
    "team_relative_progress_bonus": TeamRelativeProgressBonusComponent,
    "step_time_penalty": StepTimePenaltyComponent,
    "lap_completion": LapCompletionComponent,
    "per_lap_bonus": PerLapBonusComponent,
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
        reward_config = load_yaml_config(Path(path))
        return cls.from_config(reward_config)
