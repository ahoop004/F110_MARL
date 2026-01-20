"""V2 Reward System.

Component-based reward architecture with presets for common tasks.

Example usage:
    >>> from rewards import load_preset, GaplockReward
    >>> config = load_preset('gaplock_full')
    >>> reward = GaplockReward(config)
    >>> reward.reset()
    >>> total, components = reward.compute(step_info)
"""

from rewards.base import RewardComponent, RewardStrategy
from rewards.composer import ComposedReward
from rewards.presets import (
    GAPLOCK_FULL,
    GAPLOCK_SIMPLE,
    CENTERLINE_RACING,
    PRESETS,
    load_preset,
    merge_config,
)
from rewards.centerline import CenterlineReward
from rewards.builder import build_reward_strategy

__all__ = [
    'RewardComponent',
    'RewardStrategy',
    'ComposedReward',
    'GAPLOCK_FULL',
    'GAPLOCK_SIMPLE',
    'CENTERLINE_RACING',
    'PRESETS',
    'load_preset',
    'merge_config',
    'build_reward_strategy',
    'CenterlineReward',
]
