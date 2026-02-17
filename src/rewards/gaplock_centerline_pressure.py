"""Gaplock centerline pressure reward strategy.

Composes terminal, centerline deviation, and wall closeness components
to create a reward signal that encourages forcing the target off the
racing line and into walls.
"""

from typing import Dict, List, Tuple

from rewards.base import RewardComponent, RewardStrategy
from rewards.composer import ComposedReward
from rewards.gaplock.terminal import TerminalReward
from rewards.gaplock.centerline_deviation import CenterlineDeviationReward
from rewards.gaplock.wall_closeness import WallClosenessReward


class GaplockCenterlinePressureReward(RewardStrategy):
    """Composed reward strategy for gaplock centerline pressure task.

    Combines:
    - Terminal: Episode outcome rewards (success, crash, timeout)
    - Centerline deviation: Rewards target being pushed off centerline
    - Wall closeness: Rewards target being forced near walls

    Example:
        >>> from rewards import load_preset
        >>> config = load_preset('gaplock_centerline_pressure')
        >>> reward = GaplockCenterlinePressureReward(config)
        >>> reward.reset()
        >>> total, components = reward.compute(step_info)
    """

    def __init__(self, config: dict) -> None:
        components: List[RewardComponent] = []

        # Terminal rewards (always included)
        components.append(TerminalReward(config.get("terminal", {})))

        # Centerline deviation shaping
        if config.get("centerline_deviation", {}).get("enabled", True):
            components.append(
                CenterlineDeviationReward(config.get("centerline_deviation", {}))
            )

        # Wall closeness shaping
        if config.get("wall_closeness", {}).get("enabled", True):
            components.append(
                WallClosenessReward(config.get("wall_closeness", {}))
            )

        self.composer = ComposedReward(components)

    def reset(self) -> None:
        self.composer.reset()

    def compute(self, step_info: dict) -> Tuple[float, Dict[str, float]]:
        return self.composer.compute(step_info)


__all__ = ["GaplockCenterlinePressureReward"]
