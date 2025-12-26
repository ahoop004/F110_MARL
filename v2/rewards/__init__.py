"""V2 Reward System.

Component-based reward architecture with presets for common tasks.

Example usage:
    >>> from v2.rewards import load_preset, GaplockReward
    >>> config = load_preset('gaplock_full')
    >>> reward = GaplockReward(config)
    >>> reward.reset()
    >>> total, components = reward.compute(step_info)
"""

from v2.rewards.base import RewardComponent, RewardStrategy
from v2.rewards.composer import ComposedReward
from v2.rewards.presets import (
    GAPLOCK_FULL,
    GAPLOCK_SIMPLE,
    PRESETS,
    load_preset,
    merge_config,
)
from v2.rewards.builder import build_reward_strategy

__all__ = [
    'RewardComponent',
    'RewardStrategy',
    'ComposedReward',
    'GAPLOCK_FULL',
    'GAPLOCK_SIMPLE',
    'PRESETS',
    'load_preset',
    'merge_config',
    'build_reward_strategy',
]
