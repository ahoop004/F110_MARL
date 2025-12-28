"""Speed bonus rewards for gaplock task.

Encourages aggressive, fast movement.
"""

import numpy as np
from typing import Dict


class SpeedReward:
    """Reward for moving at target speed.

    Encourages the attacker to move quickly and aggressively,
    which is necessary for effective pursuit.
    """

    def __init__(self, config: dict):
        """Initialize speed reward.

        Args:
            config: Dict with keys:
                - enabled: Whether enabled (default: True)
                - bonus_coef: Reward per m/s (default: 0.05)
                - target_speed: Speed to aim for in m/s (default: 0.60)
        """
        self.enabled = bool(config.get('enabled', True))
        self.bonus_coef = float(config.get('bonus_coef', 0.05))
        self.target_speed = float(config.get('target_speed', 0.60))

    def compute(self, step_info: dict) -> Dict[str, float]:
        """Compute speed bonus.

        Args:
            step_info: Must contain:
                - obs: Dict with 'velocity' key [vx, vy]

        Returns:
            Dict with speed bonus (0 to bonus_coef * target_speed)
        """
        if not self.enabled:
            return {}

        obs = step_info.get('obs', {})
        velocity = obs.get('velocity', [0, 0])

        if len(velocity) < 2:
            return {}

        # Compute speed
        vx, vy = velocity[0], velocity[1]
        speed = np.sqrt(vx * vx + vy * vy)

        # Cap speed at target
        capped_speed = min(speed, self.target_speed)

        reward = self.bonus_coef * capped_speed

        if reward > 0.0:
            return {'speed/bonus': float(reward)}

        return {}


__all__ = ['SpeedReward']
