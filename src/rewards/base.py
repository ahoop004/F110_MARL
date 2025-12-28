"""Base protocols for reward system.

This module defines the core interfaces for the V2 reward system.
Rewards are composed of multiple components that each compute a portion
of the total reward signal.
"""

from typing import Protocol, Dict, Tuple, runtime_checkable


@runtime_checkable
class RewardComponent(Protocol):
    """Protocol for individual reward components.

    Components compute specific aspects of the reward (e.g., terminal rewards,
    pressure bonuses, distance shaping). Each component returns a dict of
    named reward values that can be logged for analysis.
    """

    def compute(self, step_info: dict) -> Dict[str, float]:
        """Compute reward components for this step.

        Args:
            step_info: Dictionary containing:
                - obs: Agent observation dict
                - target_obs: Target agent observation dict (if available)
                - done: Whether episode is done
                - truncated: Whether episode was truncated
                - info: Additional info dict from environment
                - timestep: Simulation timestep (dt)

        Returns:
            Dict mapping component names to reward values.
            Example: {'terminal/success': 60.0, 'pressure/bonus': 0.02}
        """
        ...


@runtime_checkable
class RewardStrategy(Protocol):
    """Protocol for complete reward strategies.

    A strategy combines multiple components and manages episode-level state
    (e.g., streak tracking, commit flags).
    """

    def reset(self) -> None:
        """Reset internal state for new episode."""
        ...

    def compute(self, step_info: dict) -> Tuple[float, Dict[str, float]]:
        """Compute total reward and individual components.

        Args:
            step_info: Same as RewardComponent.compute()

        Returns:
            Tuple of (total_reward, components_dict)
            - total_reward: Sum of all component rewards
            - components_dict: Individual component contributions (for logging)
        """
        ...


__all__ = ['RewardComponent', 'RewardStrategy']
