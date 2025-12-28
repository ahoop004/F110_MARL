"""Reward composition system.

Combines multiple reward components into a unified reward strategy.
"""

from typing import Dict, List, Tuple
from rewards.base import RewardComponent, RewardStrategy


class ComposedReward(RewardStrategy):
    """Composes multiple reward components into a single strategy.

    This class implements the composite pattern, combining independent
    reward components (terminal, pressure, distance, etc.) into a unified
    reward signal.

    Example:
        >>> components = [
        ...     TerminalReward(config['terminal']),
        ...     PressureReward(config['pressure']),
        ...     DistanceReward(config['distance']),
        ... ]
        >>> reward = ComposedReward(components)
        >>> total, breakdown = reward.compute(step_info)
        >>> # total = sum of all components
        >>> # breakdown = {'terminal/success': 60.0, 'pressure/bonus': 0.02, ...}
    """

    def __init__(self, components: List[RewardComponent]):
        """Initialize composed reward.

        Args:
            components: List of reward components to compose.
        """
        self.components = components

    def reset(self) -> None:
        """Reset all components that have state."""
        for component in self.components:
            if hasattr(component, 'reset'):
                component.reset()

    def compute(self, step_info: dict) -> Tuple[float, Dict[str, float]]:
        """Compute total reward by summing all components.

        Args:
            step_info: Step information dict (see RewardComponent.compute)

        Returns:
            Tuple of (total_reward, components_dict)
            - total_reward: Sum of all component values
            - components_dict: Individual component contributions
        """
        all_components = {}

        # Collect rewards from each component
        for component in self.components:
            component_rewards = component.compute(step_info)
            all_components.update(component_rewards)

        # Sum all components
        total = sum(all_components.values())

        return total, all_components


__all__ = ['ComposedReward']
