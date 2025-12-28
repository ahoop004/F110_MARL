"""Simple observation flattening for Dict observations.

Converts F110ParallelEnv Dict observations to flat numpy arrays for agents.
"""

from typing import Dict, Any, Optional
import numpy as np


def flatten_gaplock_obs(obs_dict: Dict[str, Any], target_id: Optional[str] = None) -> np.ndarray:
    """Flatten gaplock observation dict to 738-dim vector.

    Expected observation structure (from F110ParallelEnv):
    - scans: (720,) LiDAR scan
    - poses_x, poses_y, poses_theta: ego pose scalars
    - linear_vels_x, linear_vels_y: ego velocity scalars
    - ang_vels_z: ego angular velocity scalar

    Output structure (738 dims total):
    - LiDAR: 720 dims (normalized to [0, 1])
    - Ego state: 7 dims (x, y, theta, sin_theta, cos_theta, vx, vy, omega)
    - Target state: 7 dims (x, y, theta, sin_theta, cos_theta, vx, vy, omega)
    - Relative: 4 dims (dx, dy, dtheta, distance)

    Args:
        obs_dict: Observation dict from environment
        target_id: Target agent ID (for adversarial obs). If None, uses zeros.

    Returns:
        Flat observation array (738,)
    """
    components = []

    # LiDAR scan (720 dims) - normalize to [0, 1] with max range 12.0m
    lidar = np.asarray(obs_dict.get('scans', np.zeros(720)), dtype=np.float32)
    lidar_norm = np.clip(lidar / 12.0, 0.0, 1.0)
    components.append(lidar_norm)

    # Ego pose (4 dims: x, y, sin(theta), cos(theta))
    x = float(obs_dict.get('poses_x', 0.0))
    y = float(obs_dict.get('poses_y', 0.0))
    theta = float(obs_dict.get('poses_theta', 0.0))
    components.append(np.array([x, y, np.sin(theta), np.cos(theta)], dtype=np.float32))

    # Ego velocity (3 dims: vx, vy, omega)
    vx = float(obs_dict.get('linear_vels_x', 0.0))
    vy = float(obs_dict.get('linear_vels_y', 0.0))
    omega = float(obs_dict.get('ang_vels_z', 0.0))
    components.append(np.array([vx, vy, omega], dtype=np.float32))

    # Target state (7 dims) - from central_state if available
    if target_id and 'central_state' in obs_dict:
        central = obs_dict['central_state']
        target_x = float(central.get('poses_x', [0.0])[0] if target_id else 0.0)
        target_y = float(central.get('poses_y', [0.0])[0] if target_id else 0.0)
        target_theta = float(central.get('poses_theta', [0.0])[0] if target_id else 0.0)
        target_vx = float(central.get('linear_vels_x', [0.0])[0] if target_id else 0.0)
        target_vy = float(central.get('linear_vels_y', [0.0])[0] if target_id else 0.0)
        target_omega = float(central.get('ang_vels_z', [0.0])[0] if target_id else 0.0)

        components.append(np.array([
            target_x, target_y, np.sin(target_theta), np.cos(target_theta),
            target_vx, target_vy, target_omega
        ], dtype=np.float32))

        # Relative pose (4 dims: dx, dy, dtheta, distance)
        dx = target_x - x
        dy = target_y - y
        dtheta = target_theta - theta
        # Wrap dtheta to [-pi, pi]
        dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
        distance = np.sqrt(dx**2 + dy**2)

        components.append(np.array([dx, dy, dtheta, distance], dtype=np.float32))
    else:
        # No target - use zeros
        components.append(np.zeros(7, dtype=np.float32))  # Target state
        components.append(np.zeros(4, dtype=np.float32))  # Relative pose

    # Concatenate all components
    flat_obs = np.concatenate(components)

    return flat_obs


def flatten_observation(
    obs_dict: Dict[str, Any],
    preset: str = 'gaplock',
    target_id: Optional[str] = None
) -> np.ndarray:
    """Flatten observation dict based on preset.

    Args:
        obs_dict: Observation dict from environment
        preset: Observation preset name
        target_id: Target agent ID (for adversarial tasks)

    Returns:
        Flat observation array

    Raises:
        ValueError: If preset not supported
    """
    if preset == 'gaplock':
        return flatten_gaplock_obs(obs_dict, target_id)
    else:
        raise ValueError(
            f"Unsupported observation preset: {preset}. "
            f"Supported: gaplock"
        )


__all__ = ['flatten_observation', 'flatten_gaplock_obs']
