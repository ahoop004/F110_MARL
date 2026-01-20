"""Reward strategy builder - creates reward strategies from configuration.

This module provides factory functions to build reward strategies from
scenario configuration dictionaries.
"""

from typing import Dict, Any, Optional
from rewards.base import RewardStrategy
from rewards.presets import load_preset, merge_config
from rewards.gaplock import GaplockReward
from rewards.centerline import CenterlineReward
from rewards.gaplock_centerline_pressure import GaplockCenterlinePressureReward


def build_reward_strategy(
    config: Dict[str, Any],
    agent_id: str,
    target_id: Optional[str] = None,
) -> RewardStrategy:
    """Build a reward strategy from configuration.

    Args:
        config: Reward configuration dict with:
            - preset: (optional) Preset name ('gaplock_full', 'gaplock_simple')
            - overrides: (optional) Override values for preset
            - type: (optional) Reward type ('gaplock'). Default: 'gaplock'
            - direct config keys: terminal, pressure, distance, etc.
        agent_id: Agent ID this reward is for
        target_id: Target agent ID (for adversarial tasks)

    Returns:
        RewardStrategy instance

    Example with preset:
        >>> config = {'preset': 'gaplock_full'}
        >>> reward = build_reward_strategy(config, 'car_0', 'car_1')

    Example with preset + overrides:
        >>> config = {
        ...     'preset': 'gaplock_full',
        ...     'overrides': {
        ...         'terminal': {'target_crash': 100.0},  # Increase success reward
        ...     }
        ... }
        >>> reward = build_reward_strategy(config, 'car_0', 'car_1')

    Example with direct config:
        >>> config = {
        ...     'terminal': {'target_crash': 60.0, 'self_crash': -90.0},
        ...     'pressure': {'enabled': True, 'distance_threshold': 1.30},
        ... }
        >>> reward = build_reward_strategy(config, 'car_0', 'car_1')
    """
    # Determine reward type
    reward_type = config.get('type', 'gaplock')

    # Build final config
    if 'preset' in config:
        # Load preset and apply overrides
        preset_name = config['preset']
        reward_config = load_preset(preset_name)

        if 'overrides' in config:
            reward_config = merge_config(reward_config, config['overrides'])
    else:
        # Use config directly
        reward_config = {k: v for k, v in config.items() if k not in ['type']}

    # Create reward strategy based on type
    if reward_type == 'gaplock':
        return GaplockReward(reward_config)
    if reward_type == 'gaplock_centerline_pressure':
        return GaplockCenterlinePressureReward(reward_config)
    if reward_type == 'centerline':
        return CenterlineReward(reward_config)
    raise ValueError(
        f"Unknown reward type: {reward_type}. "
        f"Available types: gaplock, gaplock_centerline_pressure, centerline"
    )


__all__ = ['build_reward_strategy']
