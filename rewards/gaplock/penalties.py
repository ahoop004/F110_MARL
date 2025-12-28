"""Behavior penalties for gaplock task.

Discourages idle, reverse, and hard braking behaviors.
"""

import numpy as np
from typing import Dict


class BehaviorPenalties:
    """Penalties for undesirable behaviors.

    Discourages:
    - Idle: Staying still (speed too low)
    - Reverse: Moving backwards
    - Brake: Hard braking (large speed drop)
    """

    def __init__(self, config: dict):
        """Initialize behavior penalties.

        Args:
            config: Dict with keys:
                - enabled: Whether enabled (default: True)
                - idle: Idle penalty config with:
                    - penalty: Penalty value (default: 0.05)
                    - speed_threshold: Max speed for idle (default: 0.12)
                    - patience_steps: Steps before penalty (default: 25)
                - reverse: Reverse penalty config with:
                    - penalty: Penalty value (default: 0.10)
                    - speed_threshold: Min reverse speed (default: 0.02)
                - brake: Brake penalty config with:
                    - penalty: Penalty value (default: 0.05)
                    - speed_threshold: Speed below which braking detected (default: 0.40)
                    - drop_threshold: Min speed drop for penalty (default: 0.25)
        """
        self.enabled = bool(config.get('enabled', True))

        # Idle config
        idle_config = config.get('idle', {})
        self.idle_penalty = float(idle_config.get('penalty', 0.05))
        self.idle_speed_threshold = float(idle_config.get('speed_threshold', 0.12))
        self.idle_patience_steps = int(idle_config.get('patience_steps', 25))

        # Reverse config
        reverse_config = config.get('reverse', {})
        self.reverse_penalty = float(reverse_config.get('penalty', 0.10))
        self.reverse_speed_threshold = float(reverse_config.get('speed_threshold', 0.02))

        # Brake config
        brake_config = config.get('brake', {})
        self.brake_penalty = float(brake_config.get('penalty', 0.05))
        self.brake_speed_threshold = float(brake_config.get('speed_threshold', 0.40))
        self.brake_drop_threshold = float(brake_config.get('drop_threshold', 0.25))

        # State
        self.idle_steps = 0
        self.last_speed = 0.0

    def reset(self) -> None:
        """Reset penalty state for new episode."""
        self.idle_steps = 0
        self.last_speed = 0.0

    def compute(self, step_info: dict) -> Dict[str, float]:
        """Compute behavior penalties.

        Args:
            step_info: Must contain:
                - obs: Dict with 'velocity' key [vx, vy]

        Returns:
            Dict with penalties (all negative or zero)
        """
        if not self.enabled:
            return {}

        obs = step_info.get('obs', {})
        velocity = obs.get('velocity', [0, 0])

        if len(velocity) < 2:
            return {}

        components = {}

        # Compute speed
        vx, vy = velocity[0], velocity[1]
        speed = np.sqrt(vx * vx + vy * vy)

        # Idle penalty
        if speed < self.idle_speed_threshold:
            self.idle_steps += 1
            if self.idle_steps >= self.idle_patience_steps:
                components['penalties/idle'] = -self.idle_penalty
        else:
            self.idle_steps = 0

        # Reverse penalty (if moving backwards in velocity frame)
        if vx < -self.reverse_speed_threshold:
            components['penalties/reverse'] = -self.reverse_penalty

        # Brake penalty (large speed drop)
        if self.last_speed > self.brake_speed_threshold:
            speed_drop = self.last_speed - speed
            if speed_drop > self.brake_drop_threshold:
                components['penalties/brake'] = -self.brake_penalty

        # Update last speed
        self.last_speed = speed

        return components


__all__ = ['BehaviorPenalties']
