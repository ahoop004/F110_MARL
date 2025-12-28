"""Distance shaping rewards for gaplock task.

Provides smooth reward gradient based on distance to target.
"""

import numpy as np
from typing import Dict, List, Tuple


class DistanceReward:
    """Reward shaping based on distance to target.

    Provides:
    - Near reward: Bonus for being within near distance
    - Far penalty: Penalty for being too far away
    - Gradient: Smooth interpolated reward based on distance points
    """

    def __init__(self, config: dict):
        """Initialize distance reward.

        Args:
            config: Dict with keys:
                - enabled: Whether enabled (default: True)
                - reward_near: Bonus when close (default: 0.12)
                - near_distance: Threshold for near bonus (default: 1.00)
                - far_distance: Threshold for far penalty (default: 2.50)
                - penalty_far: Penalty when far (default: 0.08)
                - gradient: Optional gradient config with:
                    - enabled: Whether to use gradient (default: False)
                    - scale: Multiplier for gradient values (default: 0.20)
                    - time_scaled: Scale by timestep (default: True)
                    - clip: [min, max] clip values (default: None)
                    - points: List of [distance, reward] pairs
        """
        self.enabled = bool(config.get('enabled', True))
        self.reward_near = float(config.get('reward_near', 0.12))
        self.near_distance = float(config.get('near_distance', 1.00))
        self.far_distance = float(config.get('far_distance', 2.50))
        self.penalty_far = float(config.get('penalty_far', 0.08))

        # Gradient configuration
        grad_config = config.get('gradient', {})
        self.gradient_enabled = bool(grad_config.get('enabled', False))
        self.gradient_scale = float(grad_config.get('scale', 0.20))
        self.gradient_time_scaled = bool(grad_config.get('time_scaled', True))
        self.gradient_clip = grad_config.get('clip')  # [min, max] or None

        # Parse gradient points
        self.gradient_points: List[Tuple[float, float]] = []
        if self.gradient_enabled and 'points' in grad_config:
            for point in grad_config['points']:
                if len(point) >= 2:
                    self.gradient_points.append((float(point[0]), float(point[1])))
            # Sort by distance
            self.gradient_points.sort(key=lambda x: x[0])

    def compute(self, step_info: dict) -> Dict[str, float]:
        """Compute distance-based rewards.

        Args:
            step_info: Must contain:
                - obs: Dict with 'pose' key
                - target_obs: Dict with 'pose' key
                - timestep: Simulation timestep (if using time_scaled gradient)

        Returns:
            Dict with distance rewards
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

        if ego_pose.size < 2 or target_pose.size < 2:
            return {}

        # Compute distance
        dx = target_pose[0] - ego_pose[0]
        dy = target_pose[1] - ego_pose[1]
        distance = np.sqrt(dx * dx + dy * dy)

        components = {}

        # Near/far rewards
        if distance < self.near_distance:
            components['distance/near'] = self.reward_near
        elif distance > self.far_distance:
            components['distance/far'] = -self.penalty_far

        # Gradient reward
        if self.gradient_enabled and self.gradient_points:
            gradient_value = self._interpolate_gradient(distance)

            # Scale
            gradient_value *= self.gradient_scale

            # Time scale
            if self.gradient_time_scaled:
                timestep = step_info.get('timestep', 0.01)
                gradient_value *= timestep

            # Clip
            if self.gradient_clip:
                gradient_value = np.clip(gradient_value, self.gradient_clip[0], self.gradient_clip[1])

            if gradient_value != 0.0:
                components['distance/gradient'] = float(gradient_value)

        return components

    def _interpolate_gradient(self, distance: float) -> float:
        """Interpolate gradient value for given distance.

        Args:
            distance: Distance to target

        Returns:
            Interpolated reward value
        """
        if not self.gradient_points:
            return 0.0

        # Find surrounding points
        if distance <= self.gradient_points[0][0]:
            return self.gradient_points[0][1]

        if distance >= self.gradient_points[-1][0]:
            return self.gradient_points[-1][1]

        # Linear interpolation between points
        for i in range(len(self.gradient_points) - 1):
            d1, r1 = self.gradient_points[i]
            d2, r2 = self.gradient_points[i + 1]

            if d1 <= distance <= d2:
                # Linear interpolation
                t = (distance - d1) / (d2 - d1)
                return r1 + t * (r2 - r1)

        return 0.0


__all__ = ['DistanceReward']
