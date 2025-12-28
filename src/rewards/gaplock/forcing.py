"""Forcing rewards for gaplock task.

Advanced rewards for forcing the target toward walls and obstacles.
Includes pinch pockets, clearance reduction, and turn shaping.
"""

import numpy as np
from typing import Dict, Optional


class ForcingReward:
    """Rewards for forcing target into dangerous positions.

    Implements three forcing mechanisms:
    1. Pinch Pockets: Gaussian potential field rewards at optimal attack positions
    2. Clearance Reduction: Reward when target's wall clearance decreases
    3. Turn Shaping: Reward when target turns away from wall (being forced)

    These are advanced shaping signals that teach the attacker how to
    effectively force the target into crashing.
    """

    def __init__(self, config: dict):
        """Initialize forcing rewards.

        Args:
            config: Dict with keys:
                - enabled: Whether forcing is enabled (default: False)
                - pinch_pockets: Config for pinch pocket Gaussian potential field:
                    - enabled: Whether enabled (default: True)
                    - anchor_forward: Distance ahead of target (default: 1.20)
                    - anchor_lateral: Distance to side of target (default: 0.70)
                    - sigma: Gaussian width (default: 0.50)
                    - weight: Reward multiplier (default: 0.30)
                    - peak: Max reward at optimal position (default: None, uses simple Gaussian)
                    - floor: Min reward when far from optimal (default: None)
                    - power: Field decay exponent (default: 2.0)
                - clearance: Config for clearance reduction:
                    - enabled: Whether enabled (default: True)
                    - weight: Reward multiplier (default: 0.80)
                    - band_min: Min distance for reward (default: 0.30)
                    - band_max: Max distance for reward (default: 3.20)
                    - clip: Max reward per step (default: 0.25)
                    - time_scaled: Scale by timestep (default: True)
                - turn: Config for turn shaping:
                    - enabled: Whether enabled (default: True)
                    - weight: Reward multiplier (default: 2.0)
                    - clip: Max reward per step (default: 0.35)
                    - time_scaled: Scale by timestep (default: True)
        """
        self.enabled = bool(config.get('enabled', False))

        # Pinch pockets config (unified with potential field)
        pinch_config = config.get('pinch_pockets', {})
        self.pinch_enabled = bool(pinch_config.get('enabled', True))
        self.pinch_anchor_forward = float(pinch_config.get('anchor_forward', 1.20))
        self.pinch_anchor_lateral = float(pinch_config.get('anchor_lateral', 0.70))
        self.pinch_sigma = float(pinch_config.get('sigma', 0.50))
        self.pinch_weight = float(pinch_config.get('weight', 0.30))

        # Potential field parameters (optional, if None uses simple Gaussian)
        self.pinch_peak = pinch_config.get('peak', None)
        self.pinch_floor = pinch_config.get('floor', None)
        self.pinch_power = float(pinch_config.get('power', 2.0)) if self.pinch_peak is not None else None

        # Convert to float if provided
        if self.pinch_peak is not None:
            self.pinch_peak = float(self.pinch_peak)
        if self.pinch_floor is not None:
            self.pinch_floor = float(self.pinch_floor)

        # Clearance config
        clearance_config = config.get('clearance', {})
        self.clearance_enabled = bool(clearance_config.get('enabled', True))
        self.clearance_weight = float(clearance_config.get('weight', 0.80))
        self.clearance_band_min = float(clearance_config.get('band_min', 0.30))
        self.clearance_band_max = float(clearance_config.get('band_max', 3.20))
        self.clearance_clip = float(clearance_config.get('clip', 0.25))
        self.clearance_time_scaled = bool(clearance_config.get('time_scaled', True))

        # Turn config
        self.turn_enabled = bool(config.get('turn', {}).get('enabled', True))
        self.turn_weight = float(config.get('turn', {}).get('weight', 2.0))
        self.turn_clip = float(config.get('turn', {}).get('clip', 0.35))
        self.turn_time_scaled = bool(config.get('turn', {}).get('time_scaled', True))

        # State
        self.last_min_clearance: Optional[float] = None
        self.last_target_heading: Optional[float] = None

    def reset(self) -> None:
        """Reset forcing state for new episode."""
        self.last_min_clearance = None
        self.last_target_heading = None

    def compute(self, step_info: dict) -> Dict[str, float]:
        """Compute forcing rewards.

        Args:
            step_info: Must contain:
                - obs: Dict with 'pose' key
                - target_obs: Dict with 'pose', 'scans' keys
                - timestep: Simulation timestep (if using time_scaled)

        Returns:
            Dict with forcing rewards
        """
        if not self.enabled:
            return {}

        obs = step_info.get('obs', {})
        target_obs = step_info.get('target_obs')

        if target_obs is None:
            return {}

        components = {}

        # Pinch pocket rewards
        if self.pinch_enabled:
            pinch_reward = self._compute_pinch_pockets(obs, target_obs)
            if pinch_reward != 0.0:
                components['forcing/pinch'] = pinch_reward

        # Clearance reduction
        if self.clearance_enabled:
            clearance_reward = self._compute_clearance_reduction(target_obs, step_info.get('timestep', 0.01))
            if clearance_reward != 0.0:
                components['forcing/clearance'] = clearance_reward

        # Turn shaping
        if self.turn_enabled:
            turn_reward = self._compute_turn_shaping(target_obs, step_info.get('timestep', 0.01))
            if turn_reward != 0.0:
                components['forcing/turn'] = turn_reward

        return components

    def _compute_pinch_pockets(self, obs: dict, target_obs: dict) -> float:
        """Compute pinch pocket Gaussian potential field reward.

        Supports two modes:
        1. Simple Gaussian (default): Sum of Gaussians at pinch pockets
        2. Potential Field: Gaussian field with peak/floor mapping

        Args:
            obs: Attacker observation
            target_obs: Target observation

        Returns:
            Pinch pocket reward
        """
        ego_pose = np.array(obs.get('pose', [0, 0, 0]))
        target_pose = np.array(target_obs.get('pose', [0, 0, 0]))

        if ego_pose.size < 3 or target_pose.size < 3:
            return 0.0

        # Get relative position in target's frame
        target_x, target_y, target_theta = target_pose[0], target_pose[1], target_pose[2]
        ego_x, ego_y = ego_pose[0], ego_pose[1]

        # Transform to target's local frame
        dx = ego_x - target_x
        dy = ego_y - target_y

        # Rotate to target's heading frame
        cos_theta = np.cos(-target_theta)
        sin_theta = np.sin(-target_theta)
        local_x = dx * cos_theta - dy * sin_theta
        local_y = dx * sin_theta + dy * cos_theta

        # Compute distance to each pinch pocket
        anchor_x = self.pinch_anchor_forward

        # Right pocket
        anchor_y_right = -self.pinch_anchor_lateral
        dist_right = np.sqrt((local_x - anchor_x)**2 + (local_y - anchor_y_right)**2)

        # Left pocket
        anchor_y_left = self.pinch_anchor_lateral
        dist_left = np.sqrt((local_x - anchor_x)**2 + (local_y - anchor_y_left)**2)

        # Use minimum distance to either pocket
        d_min = min(dist_right, dist_left)

        # Compute reward based on mode
        if self.pinch_peak is not None and self.pinch_floor is not None:
            # Potential field mode: Map distance to [floor, peak] range
            ratio = d_min / max(self.pinch_sigma, 1e-6)
            shaped = np.exp(-0.5 * (ratio ** self.pinch_power))

            # Map shaped value (0 to 1) to [floor, peak] range
            value = self.pinch_floor + (self.pinch_peak - self.pinch_floor) * shaped
            reward = value
        else:
            # Simple Gaussian mode: Sum of Gaussians at each pocket
            gauss_right = np.exp(-(dist_right**2) / (2 * self.pinch_sigma**2))
            gauss_left = np.exp(-(dist_left**2) / (2 * self.pinch_sigma**2))
            reward = gauss_right + gauss_left

        return float(self.pinch_weight * reward)

    def _compute_clearance_reduction(self, target_obs: dict, timestep: float) -> float:
        """Compute clearance reduction reward.

        Rewards when target's minimum wall clearance decreases,
        indicating the attacker is successfully forcing the target
        closer to walls.

        Args:
            target_obs: Target observation with 'scans' key
            timestep: Simulation timestep

        Returns:
            Clearance reduction reward
        """
        scans = target_obs.get('scans')
        if scans is None:
            scans = target_obs.get('lidar')

        if scans is None:
            return 0.0

        # Find minimum clearance
        scans_array = np.array(scans)
        min_clearance = float(np.min(scans_array))

        # Check if in reward band
        if min_clearance < self.clearance_band_min or min_clearance > self.clearance_band_max:
            self.last_min_clearance = min_clearance
            return 0.0

        reward = 0.0

        if self.last_min_clearance is not None:
            # Reward for reduction in clearance
            clearance_delta = self.last_min_clearance - min_clearance

            if clearance_delta > 0:
                reward = self.clearance_weight * clearance_delta

                # Time scale
                if self.clearance_time_scaled:
                    reward *= timestep

                # Clip
                reward = min(reward, self.clearance_clip)

        self.last_min_clearance = min_clearance
        return float(reward)

    def _compute_turn_shaping(self, target_obs: dict, timestep: float) -> float:
        """Compute turn shaping reward.

        Rewards when target turns away from wall, indicating the
        target is reacting to the attacker's pressure.

        Args:
            target_obs: Target observation with 'pose' key
            timestep: Simulation timestep

        Returns:
            Turn shaping reward
        """
        target_pose = np.array(target_obs.get('pose', [0, 0, 0]))

        if target_pose.size < 3:
            return 0.0

        target_theta = target_pose[2]

        reward = 0.0

        if self.last_target_heading is not None:
            # Compute heading change
            heading_delta = self._normalize_angle(target_theta - self.last_target_heading)

            # Reward for turning (any direction)
            turn_magnitude = abs(heading_delta)

            if turn_magnitude > 0:
                reward = self.turn_weight * turn_magnitude

                # Time scale
                if self.turn_time_scaled:
                    reward *= timestep

                # Clip
                reward = min(reward, self.turn_clip)

        self.last_target_heading = target_theta
        return float(reward)

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle


__all__ = ['ForcingReward']
