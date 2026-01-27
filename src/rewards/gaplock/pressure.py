"""Pressure rewards for gaplock task.

Rewards the attacker for maintaining close proximity to the target.
Includes streak bonuses for sustained pressure.
"""

import numpy as np
from typing import Dict


class PressureReward:
    """Reward for applying pressure to target by staying close.

    Pressure encourages the attacker to:
    - Get close to the target (within distance threshold)
    - Maintain proximity over time (streak bonuses)
    - Move with sufficient speed
    - Point toward the target

    This provides partial credit even in failed episodes, teaching
    the agent to approach and pressure the target.
    """

    def __init__(self, config: dict):
        """Initialize pressure reward.

        Args:
            config: Dict with keys:
                - enabled: Whether pressure is enabled (default: True)
                - distance_threshold: Max distance for pressure (default: 1.30)
                - timeout: Max time since last pressure (default: 1.20)
                - min_speed: Minimum speed required (default: 0.30)
                - heading_tolerance: Max heading error in radians (default: 1.57)
                - bonus: Reward per pressure step (default: 0.12)
                - bonus_interval: Steps between bonuses (default: 5)
                - streak_bonus: Additional reward per streak step (default: 0.10)
                - streak_cap: Max streak multiplier (default: 40)
        """
        self.enabled = bool(config.get('enabled', True))
        self.distance_threshold = float(config.get('distance_threshold', 1.30))
        self.timeout = float(config.get('timeout', 1.20))
        self.min_speed = float(config.get('min_speed', 0.30))
        self.heading_tolerance = float(config.get('heading_tolerance', 1.57))
        self.bonus = float(config.get('bonus', 0.12))
        self.bonus_interval = int(config.get('bonus_interval', 5))
        self.streak_bonus = float(config.get('streak_bonus', 0.10))
        self.streak_cap = int(config.get('streak_cap', 40))

        # State
        self.pressure_streak = 0
        self.steps_since_bonus = 0
        self.time_since_pressure = 0.0

    def reset(self) -> None:
        """Reset pressure state for new episode."""
        self.pressure_streak = 0
        self.steps_since_bonus = 0
        self.time_since_pressure = 0.0

    def compute(self, step_info: dict) -> Dict[str, float]:
        """Compute pressure reward.

        Args:
            step_info: Must contain:
                - obs: Dict with 'pose' and 'velocity' keys
                - target_obs: Dict with 'pose' key (if available)

        Returns:
            Dict with pressure rewards (empty if not in pressure state)
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

        if ego_pose.size < 3 or target_pose.size < 3:
            return {}

        # Compute distance
        dx = target_pose[0] - ego_pose[0]
        dy = target_pose[1] - ego_pose[1]
        distance = np.sqrt(dx * dx + dy * dy)

        # Check if in pressure state
        in_pressure = distance < self.distance_threshold

        # Check speed requirement
        if in_pressure:
            velocity = obs.get('velocity', [0, 0])
            if len(velocity) >= 2:
                speed = np.sqrt(velocity[0]**2 + velocity[1]**2)
                if speed < self.min_speed:
                    in_pressure = False

        # Check heading requirement
        if in_pressure and self.heading_tolerance < np.pi:
            ego_theta = ego_pose[2]
            target_angle = np.arctan2(dy, dx)
            heading_error = abs(self._normalize_angle(target_angle - ego_theta))
            if heading_error > self.heading_tolerance:
                in_pressure = False

        components = {}
        if in_pressure:
            # Update streak
            self.pressure_streak += 1
            self.steps_since_bonus += 1
            self.time_since_pressure = 0.0

            # Base pressure bonus (every N steps)
            if self.steps_since_bonus >= self.bonus_interval:
                components['pressure/bonus'] = self.bonus
                self.steps_since_bonus = 0

            # Streak bonus (increases with consecutive pressure)
            if self.streak_bonus > 0 and self.pressure_streak > 1:
                capped_streak = min(self.pressure_streak, self.streak_cap)
                components['pressure/streak'] = self.streak_bonus * (capped_streak / self.streak_cap)

        else:
            timestep = float(step_info.get('timestep', 0.01))
            if timestep < 0.0:
                timestep = 0.0
            self.time_since_pressure += timestep
            if self.timeout <= 0.0 or self.time_since_pressure > self.timeout:
                # Lost pressure for too long; reset streak tracking
                self.pressure_streak = 0
                self.steps_since_bonus = 0

        return components

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle


__all__ = ['PressureReward']
