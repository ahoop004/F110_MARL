"""RewardComposer — assembles RewardComponents from config."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import yaml

from wrappers.rewards.base import RewardComponent
from wrappers.rewards.centerline import CenterlineRewardComponent
from wrappers.rewards.collision import CollisionRewardComponent
from wrappers.rewards.speed import SpeedRewardComponent
from wrappers.rewards.gaplock_pressure import GaplockPressureComponent
from wrappers.rewards.gaplock_forcing import GaplockForcingComponent
from wrappers.rewards.lap_completion import LapCompletionComponent
from wrappers.rewards.progress_safety import ProgressSafetyComponent
from wrappers.rewards.target_finish import TargetFinishComponent
from wrappers.rewards.terminal_success import TerminalSuccessComponent
from wrappers.rewards.terminal_timeout import TerminalTimeoutComponent
from wrappers.rewards.terminal_self_crash import TerminalSelfCrashComponent


_COMPONENT_MAP = {
    "centerline": CenterlineRewardComponent,
    "collision": CollisionRewardComponent,
    "speed": SpeedRewardComponent,
    "gaplock_pressure": GaplockPressureComponent,
    "gaplock_forcing": GaplockForcingComponent,
    "lap_completion": LapCompletionComponent,
    "target_finish": TargetFinishComponent,
    "progress_safety": ProgressSafetyComponent,
    "terminal_success": TerminalSuccessComponent,
    "terminal_timeout": TerminalTimeoutComponent,
    "terminal_self_crash": TerminalSelfCrashComponent,
}


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

        for key, component_cls in _COMPONENT_MAP.items():
            comp_cfg = cfg.get(key, {})
            if isinstance(comp_cfg, dict) and comp_cfg.get("enabled", False):
                components.append(component_cls(comp_cfg))

        if not components:
            raise ValueError("RewardComposer: no components enabled in reward config.")

        return cls(components)

    @classmethod
    def from_file(cls, path: str) -> "RewardComposer":
        """Load from a YAML reward config file path."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Reward config not found: {path}")
        with open(p) as f:
            reward_config = yaml.safe_load(f) or {}
        return cls.from_config(reward_config)
