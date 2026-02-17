"""Reward strategy builder - creates reward strategies from configuration.

This module provides a registry-based factory for building reward strategies
from scenario configuration dictionaries.
"""

from typing import Any, Dict, Optional, Type

from rewards.base import RewardStrategy
from rewards.presets import load_preset, merge_config
from rewards.gaplock import GaplockReward
from rewards.centerline import CenterlineReward
from rewards.gaplock_centerline_pressure import GaplockCenterlinePressureReward


# Registry mapping reward type names to strategy classes.
# To add a new reward type, register it here.
REWARD_REGISTRY: Dict[str, Type[RewardStrategy]] = {
    'gaplock': GaplockReward,
    'gaplock_centerline_pressure': GaplockCenterlinePressureReward,
    'centerline': CenterlineReward,
}


def register_reward(name: str, cls: Type[RewardStrategy]) -> None:
    """Register a reward strategy class under the given name.

    Args:
        name: Type name used in scenario configs
        cls: RewardStrategy class to register
    """
    REWARD_REGISTRY[name] = cls


def build_reward_strategy(
    config: Dict[str, Any],
    agent_id: str,
    target_id: Optional[str] = None,
) -> RewardStrategy:
    """Build a reward strategy from configuration.

    Args:
        config: Reward configuration dict with:
            - type: Reward type name (default: 'gaplock')
            - preset: (optional) Preset name to load defaults from
            - overrides: (optional) Override values for preset
            - direct config keys: component-level dicts
        agent_id: Agent ID this reward is for
        target_id: Target agent ID (for adversarial tasks)

    Returns:
        RewardStrategy instance

    Raises:
        ValueError: If reward type is not registered
    """
    reward_type = config.get('type', 'gaplock')

    if reward_type not in REWARD_REGISTRY:
        available = ', '.join(sorted(REWARD_REGISTRY.keys()))
        raise ValueError(
            f"Unknown reward type: {reward_type}. Available: {available}"
        )

    # Build final config from preset + overrides (or use config directly)
    if 'preset' in config:
        reward_config = load_preset(config['preset'])
        if 'overrides' in config:
            reward_config = merge_config(reward_config, config['overrides'])
    else:
        reward_config = {k: v for k, v in config.items() if k not in ['type']}

    return REWARD_REGISTRY[reward_type](reward_config)


__all__ = ['REWARD_REGISTRY', 'register_reward', 'build_reward_strategy']
