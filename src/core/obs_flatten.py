"""Simple observation flattening for Dict observations.

Converts F110ParallelEnv Dict observations to flat numpy arrays for agents.
"""

from typing import Dict, Any, Optional
import numpy as np


def flatten_gaplock_obs(obs_dict: Dict[str, Any], target_id: Optional[str] = None) -> np.ndarray:
    """Flatten gaplock observation dict to 738-dim vector.

    Expected observation structure (from F110ParallelEnv):
    - scans/lidar: (N,) LiDAR scan
    - pose: (3,) [x, y, theta]
    - velocity: (2,) [vx, vy]
    - angular_velocity: scalar
    - state: central-state vector (optional)

    Output structure (738 dims total for 720-beam LiDAR):
    - LiDAR: N dims (normalized to [0, 1] using 12.0m max range)
    - Ego state: 7 dims (x, y, sin_theta, cos_theta, vx, vy, omega)
    - Target state: 7 dims (x, y, sin_theta, cos_theta, vx, vy, omega)
    - Relative: 4 dims (dx, dy, dtheta, distance)

    Args:
        obs_dict: Observation dict from environment
        target_id: Target agent ID (for adversarial obs). If None, uses zeros.

    Returns:
        Flat observation array
    """
    def _extract_pose(data: Any) -> np.ndarray:
        if isinstance(data, dict):
            pose = data.get('pose')
            if pose is None:
                pose = data.get('target_pose')
            if pose is not None:
                arr = np.asarray(pose, dtype=np.float32).reshape(-1)
                if arr.size >= 3:
                    return arr[:3]
            if any(k in data for k in ('poses_x', 'poses_y', 'poses_theta')):
                return np.array([
                    float(data.get('poses_x', 0.0)),
                    float(data.get('poses_y', 0.0)),
                    float(data.get('poses_theta', 0.0)),
                ], dtype=np.float32)
            return np.zeros(3, dtype=np.float32)
        arr = np.asarray(data, dtype=np.float32).reshape(-1)
        if arr.size >= 3:
            return arr[:3]
        return np.zeros(3, dtype=np.float32)

    def _extract_velocity(data: Any) -> np.ndarray:
        vx = 0.0
        vy = 0.0
        omega = 0.0
        if isinstance(data, dict):
            velocity = data.get('velocity')
            if velocity is not None:
                arr = np.asarray(velocity, dtype=np.float32).reshape(-1)
                if arr.size >= 2:
                    vx, vy = float(arr[0]), float(arr[1])
            else:
                vx = float(data.get('linear_vels_x', 0.0))
                vy = float(data.get('linear_vels_y', 0.0))

            if 'angular_velocity' in data:
                omega_val = data.get('angular_velocity')
                try:
                    omega = float(np.asarray(omega_val, dtype=np.float32).reshape(-1)[0])
                except Exception:
                    omega = 0.0
            else:
                omega = float(data.get('ang_vels_z', 0.0))
        return np.array([vx, vy, omega], dtype=np.float32)

    def _parse_agent_index(agent_name: Optional[str]) -> Optional[int]:
        if not agent_name:
            return None
        if isinstance(agent_name, str) and agent_name.startswith("car_"):
            try:
                return int(agent_name.split("_", 1)[1])
            except (ValueError, IndexError):
                return None
        return None

    def _extract_from_state_vector(state_vec: np.ndarray, agent_idx: int) -> Optional[np.ndarray]:
        keys_per_agent = 7  # poses_x, poses_y, poses_theta, linear_vels_x, linear_vels_y, ang_vels_z, collisions
        if state_vec.ndim != 1 or state_vec.size < keys_per_agent:
            return None
        if state_vec.size % keys_per_agent != 0:
            return None
        n_agents = state_vec.size // keys_per_agent
        if agent_idx < 0 or agent_idx >= n_agents:
            return None

        def _slice(key_index: int) -> float:
            offset = key_index * n_agents + agent_idx
            return float(state_vec[offset])

        return np.array([
            _slice(0),  # poses_x
            _slice(1),  # poses_y
            _slice(2),  # poses_theta
            _slice(3),  # linear_vels_x
            _slice(4),  # linear_vels_y
            _slice(5),  # ang_vels_z
        ], dtype=np.float32)

    components = []

    # LiDAR scan (normalize to [0, 1] with max range 12.0m)
    scan = obs_dict.get('scans')
    if scan is None:
        scan = obs_dict.get('lidar')
    if scan is None:
        lidar = np.zeros(720, dtype=np.float32)
    else:
        lidar = np.asarray(scan, dtype=np.float32).reshape(-1)
    lidar_norm = np.clip(lidar / 12.0, 0.0, 1.0)
    components.append(lidar_norm)

    # Ego pose (4 dims: x, y, sin(theta), cos(theta))
    ego_pose = _extract_pose(obs_dict)
    x, y, theta = map(float, ego_pose[:3])
    components.append(np.array([x, y, np.sin(theta), np.cos(theta)], dtype=np.float32))

    # Ego velocity (3 dims: vx, vy, omega)
    components.append(_extract_velocity(obs_dict))

    # Target state (7 dims)
    target_state = None
    if target_id:
        central = obs_dict.get('central_state')
        if central is None:
            central = obs_dict.get('state')

        if isinstance(central, dict):
            target_pose = _extract_pose(central)
            target_vel = _extract_velocity(central)
            target_state = np.concatenate([target_pose, target_vel], axis=0)
        elif central is not None:
            state_vec = np.asarray(central, dtype=np.float32).reshape(-1)
            target_idx = _parse_agent_index(target_id)
            if target_idx is not None:
                parsed = _extract_from_state_vector(state_vec, target_idx)
                if parsed is not None:
                    target_state = parsed

    if target_state is not None:
        target_x, target_y, target_theta, target_vx, target_vy, target_omega = map(float, target_state[:6])
        components.append(np.array([
            target_x, target_y, np.sin(target_theta), np.cos(target_theta),
            target_vx, target_vy, target_omega
        ], dtype=np.float32))

        # Relative pose (4 dims: dx, dy, dtheta, distance)
        dx = target_x - x
        dy = target_y - y
        dtheta = target_theta - theta
        dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
        distance = np.sqrt(dx**2 + dy**2)

        components.append(np.array([dx, dy, dtheta, distance], dtype=np.float32))
    else:
        components.append(np.zeros(7, dtype=np.float32))  # Target state
        components.append(np.zeros(4, dtype=np.float32))  # Relative pose

    return np.concatenate(components)


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
