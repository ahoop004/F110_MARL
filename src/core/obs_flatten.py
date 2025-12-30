"""Simple observation flattening for Dict observations.

Converts F110ParallelEnv Dict observations to flat numpy arrays for agents.
"""

from typing import Dict, Any, Optional
import numpy as np


def flatten_gaplock_obs(
    obs_dict: Dict[str, Any],
    target_id: Optional[str] = None,
    scales: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """Flatten gaplock observation dict to a normalized feature vector.

    Expected observation structure (from F110ParallelEnv):
    - scans/lidar: (N,) LiDAR scan
    - pose: (3,) [x, y, theta]
    - velocity: (2,) [vx, vy]
    - angular_velocity: scalar
    - state: central-state vector (optional)

    Output structure (N + 11 dims):
    - LiDAR: N dims (normalized to [0, 1] using lidar range)
    - Ego velocity: 3 dims (vx, vy, omega) scaled by speed scale
    - Target velocity: 3 dims (vx, vy, omega) scaled by speed scale
    - Relative pose: 5 dims (rel_x, rel_y, sin(rel_theta), cos(rel_theta), distance)
      to the target vehicle in ego frame, scaled by map scale.

    Args:
        obs_dict: Observation dict from environment
        target_id: Target agent ID (for adversarial obs). If None, uses zeros.
        scales: Optional normalization scales dict with keys:
            - lidar_range
            - position
            - speed

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

    scales = scales or {}
    lidar_range = float(scales.get("lidar_range", 12.0))
    position_scale = float(scales.get("position", lidar_range))
    speed_scale = float(scales.get("speed", 1.0))
    if lidar_range <= 0.0:
        lidar_range = 12.0
    if position_scale <= 0.0:
        position_scale = lidar_range
    if speed_scale <= 0.0:
        speed_scale = 1.0
    # LiDAR scan (normalize to [0, 1] with max range 12.0m)
    scan = obs_dict.get('scans')
    if scan is None:
        scan = obs_dict.get('lidar')
    if scan is None:
        lidar = np.zeros(108, dtype=np.float32)
    else:
        lidar = np.asarray(scan, dtype=np.float32).reshape(-1)
    lidar_norm = np.clip(lidar / lidar_range, 0.0, 1.0)
    components.append(lidar_norm)

    # Ego pose (used for relative computation only; not emitted)
    ego_pose = _extract_pose(obs_dict)
    x, y, theta = map(float, ego_pose[:3])

    # Ego velocity (3 dims: vx, vy, omega) - scaled by speed
    ego_vel = _extract_velocity(obs_dict)
    ego_vel_norm = np.clip(ego_vel.astype(np.float32) / speed_scale, -1.0, 1.0)
    components.append(ego_vel_norm)

    # Target state (pose + velocity)
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
        target_vel = np.array([target_vx, target_vy, target_omega], dtype=np.float32)
        target_vel_norm = np.clip(target_vel / speed_scale, -1.0, 1.0)
        components.append(target_vel_norm)

        # Relative pose to target vehicle in ego frame (5 dims)
        dx = target_x - x
        dy = target_y - y
        cos_t = float(np.cos(theta))
        sin_t = float(np.sin(theta))
        rel_x = cos_t * dx + sin_t * dy
        rel_y = -sin_t * dx + cos_t * dy
        rel_x_norm = float(np.clip(rel_x / position_scale, -1.0, 1.0))
        rel_y_norm = float(np.clip(rel_y / position_scale, -1.0, 1.0))
        dtheta = float((target_theta - theta + np.pi) % (2.0 * np.pi) - np.pi)
        sin_theta = float(np.sin(dtheta))
        cos_theta = float(np.cos(dtheta))
        distance = float(np.sqrt(dx**2 + dy**2))
        distance_norm = float(np.clip(distance / position_scale, 0.0, 1.0))

        components.append(
            np.array(
                [rel_x_norm, rel_y_norm, sin_theta, cos_theta, distance_norm],
                dtype=np.float32,
            )
        )
    else:
        components.append(np.zeros(3, dtype=np.float32))  # Target velocity
        components.append(np.zeros(5, dtype=np.float32))  # Relative pose

    return np.concatenate(components)


def flatten_observation(
    obs_dict: Dict[str, Any],
    preset: str = 'gaplock',
    target_id: Optional[str] = None,
    scales: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """Flatten observation dict based on preset.

    Args:
        obs_dict: Observation dict from environment
        preset: Observation preset name
        target_id: Target agent ID (for adversarial tasks)
        scales: Optional normalization scales for preset-specific flattening

    Returns:
        Flat observation array

    Raises:
        ValueError: If preset not supported
    """
    if preset == 'gaplock':
        return flatten_gaplock_obs(obs_dict, target_id, scales=scales)
    else:
        raise ValueError(
            f"Unsupported observation preset: {preset}. "
            f"Supported: gaplock"
        )


__all__ = ['flatten_observation', 'flatten_gaplock_obs']
