"""Heading alignment rewards for gaplock task.

Rewards pointing toward the target.
"""

import numpy as np
from typing import Dict


class HeadingReward:
    """Reward for pointing toward target.

    Encourages the attacker to orient toward the target, which is
    necessary for effective pursuit and pressure.
    """

    def __init__(self, config: dict):
        """Initialize heading reward.

        Args:
            config: Dict with keys:
                - enabled: Whether enabled (default: True)
                - coefficient: Reward multiplier (default: 0.08)
        """
        self.enabled = bool(config.get('enabled', True))
        self.coefficient = float(config.get('coefficient', 0.08))

    def compute(self, step_info: dict) -> Dict[str, float]:
        """Compute heading alignment reward.

        Args:
            step_info: Must contain:
                - obs: Dict with 'pose' key [x, y, theta]
                - target_obs: Dict with 'pose' key

        Returns:
            Dict with heading reward (0 to coefficient)
        """
        if not self.enabled:
            return {}

        obs = step_info.get('obs', {})
        target_obs = step_info.get('target_obs')

        if target_obs is None:
            return {}

        # Extract poses
        ego_pose = np.array(obs.get('pose', [0, 0, 0]))
        target_pose = np.array(target_obs.get('pose', [0, 0, 0]))

        if ego_pose.size < 3 or target_pose.size < 2:
            return {}

        # Compute heading error
        dx = target_pose[0] - ego_pose[0]
        dy = target_pose[1] - ego_pose[1]

        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            # At same position, no heading reward
            return {}

        ego_theta = ego_pose[2]
        target_angle = np.arctan2(dy, dx)
        heading_error = abs(self._normalize_angle(target_angle - ego_theta))

        # Alignment ranges from 0 (opposite) to 1 (perfectly aligned)
        alignment = 1.0 - (heading_error / np.pi)

        reward = self.coefficient * alignment

        if reward > 0.0:
            return {'heading/alignment': float(reward)}

        return {}

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle


__all__ = ['HeadingReward']
