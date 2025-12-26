"""Main gaplock reward strategy.

Integrates all reward components into a unified strategy.
"""

from typing import Dict, Tuple, List
from v2.rewards.base import RewardStrategy, RewardComponent
from v2.rewards.composer import ComposedReward
from v2.rewards.gaplock.terminal import TerminalReward
from v2.rewards.gaplock.pressure import PressureReward
from v2.rewards.gaplock.distance import DistanceReward
from v2.rewards.gaplock.heading import HeadingReward
from v2.rewards.gaplock.speed import SpeedReward
from v2.rewards.gaplock.forcing import ForcingReward
from v2.rewards.gaplock.penalties import BehaviorPenalties


class GaplockReward(RewardStrategy):
    """Complete reward strategy for gaplock adversarial task.

    Combines all reward components:
    - Terminal: Episode outcome rewards
    - Pressure: Proximity and sustained pressure bonuses
    - Distance: Distance-based shaping
    - Heading: Alignment toward target
    - Speed: Movement bonuses
    - Forcing: Advanced forcing mechanics (pinch, clearance, turn)
    - Penalties: Discouraging bad behaviors

    Example:
        >>> from v2.rewards import load_preset
        >>> from v2.rewards.gaplock import GaplockReward
        >>>
        >>> config = load_preset('gaplock_full')
        >>> reward = GaplockReward(config)
        >>>
        >>> reward.reset()
        >>> total, components = reward.compute(step_info)
        >>> # total = sum of all components
        >>> # components = {'terminal/success': 60.0, 'pressure/bonus': 0.12, ...}
    """

    def __init__(self, config: dict):
        """Initialize gaplock reward.

        Args:
            config: Configuration dict with keys for each component:
                - terminal: Terminal reward config
                - pressure: Pressure reward config
                - distance: Distance reward config
                - heading: Heading reward config
                - speed: Speed reward config
                - forcing: Forcing reward config (optional)
                - penalties: Penalty config

        Example config:
            {
                'terminal': {'target_crash': 60.0, ...},
                'pressure': {'enabled': True, 'distance_threshold': 1.30, ...},
                'distance': {'enabled': True, ...},
                'heading': {'enabled': True, 'coefficient': 0.08},
                'speed': {'enabled': True, 'bonus_coef': 0.05},
                'forcing': {'enabled': False, ...},
                'penalties': {'enabled': True, ...},
            }
        """
        components: List[RewardComponent] = []

        # Terminal rewards (always included)
        if 'terminal' in config:
            components.append(TerminalReward(config['terminal']))

        # Pressure rewards
        if config.get('pressure', {}).get('enabled', True):
            components.append(PressureReward(config.get('pressure', {})))

        # Distance shaping
        if config.get('distance', {}).get('enabled', True):
            components.append(DistanceReward(config.get('distance', {})))

        # Heading alignment
        if config.get('heading', {}).get('enabled', True):
            components.append(HeadingReward(config.get('heading', {})))

        # Speed bonus
        if config.get('speed', {}).get('enabled', True):
            components.append(SpeedReward(config.get('speed', {})))

        # Forcing rewards (optional, disabled by default)
        if config.get('forcing', {}).get('enabled', False):
            components.append(ForcingReward(config.get('forcing', {})))

        # Behavior penalties
        if config.get('penalties', {}).get('enabled', True):
            components.append(BehaviorPenalties(config.get('penalties', {})))

        # Compose all components
        self.composer = ComposedReward(components)

    def reset(self) -> None:
        """Reset all stateful components for new episode."""
        self.composer.reset()

    def compute(self, step_info: dict) -> Tuple[float, Dict[str, float]]:
        """Compute total reward and component breakdown.

        Args:
            step_info: Dict containing:
                - obs: Attacker observation dict
                - target_obs: Target observation dict
                - done: Whether episode ended
                - truncated: Whether episode was truncated
                - info: Additional info from environment
                - timestep: Simulation timestep (dt)

        Returns:
            Tuple of (total_reward, components_dict)
            - total_reward: Sum of all component rewards
            - components_dict: Individual component contributions for logging
        """
        return self.composer.compute(step_info)


__all__ = ['GaplockReward']
