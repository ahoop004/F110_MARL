"""RewardComposer — assembles RewardComponents from config."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from wrappers.rewards.base import RewardComponent
from wrappers.rewards.centerline import CenterlineRewardComponent
from wrappers.rewards.collision import CollisionRewardComponent
from wrappers.rewards.terminal import TerminalRewardComponent
from wrappers.rewards.speed import SpeedRewardComponent
from wrappers.rewards.gaplock import GaplockRewardComponent


_COMPONENT_MAP = {
    "centerline": CenterlineRewardComponent,
    "collision": CollisionRewardComponent,
    "terminal": TerminalRewardComponent,
    "speed": SpeedRewardComponent,
    "gaplock": GaplockRewardComponent,
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
        """Build from a parsed reward config dict.

        reward_config: the 'reward:' block from the YAML file
        """
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
