"""Observation flattening for F110 Dict observations.

This module converts structured observations (dicts with LiDAR, pose, velocity)
into flat numpy arrays suitable for neural network input.

Key Features:
- Flattens multi-agent observations into agent-centric representations
- Handles adversarial tasks by including relative state to target agent
- Normalizes features to bounded ranges for stable learning
- Uses sin/cos encoding for angles to avoid discontinuities
"""

from typing import Dict, Any, Optional
import numpy as np


def flatten_gaplock_obs(
    obs_dict: Dict[str, Any],
    target_id: Optional[str] = None,
    scales: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """Flatten gaplock observation dict to a normalized feature vector.

    The gaplock task is an adversarial racing scenario where an attacker
    tries to overtake a defender. Observations must include both ego state
    and relative state to the target for effective learning.

    Input Structure (from F110ParallelEnv):
    - scans/lidar: (108,) LiDAR range measurements in meters
    - pose: (3,) [x, y, theta] in world frame
    - velocity: (2,) [vx, vy] in world frame (m/s)
    - angular_velocity: scalar omega (rad/s)
    - central_state: (14,) flattened multi-agent state (optional)

    Output Structure (119 dims total for gaplock with 108-beam LiDAR):
    - LiDAR:           108 dims (normalized to [0, 1], max_range=12m)
    - Ego velocity:      3 dims (vx, vy, omega) clipped to [-1, 1]
    - Target velocity:   3 dims (vx, vy, omega) clipped to [-1, 1]
    - Relative pose:     5 dims (rel_x, rel_y, sin(Δθ), cos(Δθ), dist) clipped to [-1, 1]

    Normalization Strategy:
    - LiDAR: Divide by max_range (12m), clip to [0, 1]
    - Velocities: Divide by speed_scale (1.0 m/s), clip to [-1, 1]
    - Positions: Divide by position_scale (12m), clip to [-1, 1]
    - Angles: Use sin/cos encoding to avoid 2π discontinuity

    Args:
        obs_dict: Observation dict from environment (ego agent's view)
        target_id: Target agent ID for adversarial tasks (e.g., 'car_1')
                   If None, target features are zeros
        scales: Optional normalization scales:
            - lidar_range: Max LiDAR range in meters (default: 12.0)
            - position: Position scale for relative coords (default: 12.0)
            - speed: Speed scale for velocity normalization (default: 1.0)

    Returns:
        Flattened observation array (shape: 119,) normalized to bounded ranges

    Example:
        >>> obs = {'scans': np.array([...]),  # 108 LiDAR rays
        ...        'pose': [0.0, 0.0, 0.0],
        ...        'velocity': [0.5, 0.0],
        ...        'angular_velocity': 0.1,
        ...        'central_state': np.array([...])}  # Multi-agent state
        >>> flat_obs = flatten_gaplock_obs(obs, target_id='car_1')
        >>> flat_obs.shape
        (119,)
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

    # ========================================
    # OBSERVATION FLATTENING
    # ========================================

    components = []  # List of feature vectors to concatenate

    # Extract normalization scales (with validation)
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

    # ========================================
    # COMPONENT 1: LiDAR (108 dims)
    # ========================================

    # Extract LiDAR scan (handles both 'scans' and 'lidar' keys)
    scan = obs_dict.get('scans')
    if scan is None:
        scan = obs_dict.get('lidar')
    if scan is None:
        lidar = np.zeros(108, dtype=np.float32)  # Default if missing
    else:
        lidar = np.asarray(scan, dtype=np.float32).reshape(-1)

    # Normalize LiDAR: ranges in meters → [0, 1]
    # Closer obstacles = smaller values, free space = 1.0
    lidar_norm = np.clip(lidar / lidar_range, 0.0, 1.0)
    components.append(lidar_norm)

    # ========================================
    # COMPONENT 2: Ego Velocity (3 dims)
    # ========================================

    # Extract ego pose (NOT included in observation, used for relative computation)
    ego_pose = _extract_pose(obs_dict)
    x, y, theta = map(float, ego_pose[:3])

    # Extract ego velocity: [vx, vy, omega]
    # vx/vy: Linear velocity in world frame (m/s)
    # omega: Angular velocity (rad/s)
    ego_vel = _extract_velocity(obs_dict)
    ego_vel_norm = np.clip(ego_vel.astype(np.float32) / speed_scale, -1.0, 1.0)
    components.append(ego_vel_norm)

    # ========================================
    # COMPONENT 3: Target Velocity (3 dims)
    # ========================================

    # For adversarial tasks, extract target agent's state from central_state
    target_state = None
    if target_id:
        central = obs_dict.get('central_state')
        if central is None:
            central = obs_dict.get('state')

        # Parse target state from central observation
        if isinstance(central, dict):
            # Dict format: directly extract target's pose and velocity
            target_pose = _extract_pose(central)
            target_vel = _extract_velocity(central)
            target_state = np.concatenate([target_pose, target_vel], axis=0)
        elif central is not None:
            # Flattened format: extract from state vector by index
            # State vector: [poses_x[0], poses_x[1], ..., poses_y[0], ...]
            state_vec = np.asarray(central, dtype=np.float32).reshape(-1)
            target_idx = _parse_agent_index(target_id)
            if target_idx is not None:
                parsed = _extract_from_state_vector(state_vec, target_idx)
                if parsed is not None:
                    target_state = parsed

    if target_state is not None:
        # Extract and normalize target velocity
        target_x, target_y, target_theta, target_vx, target_vy, target_omega = map(float, target_state[:6])
        target_vel = np.array([target_vx, target_vy, target_omega], dtype=np.float32)
        target_vel_norm = np.clip(target_vel / speed_scale, -1.0, 1.0)
        components.append(target_vel_norm)

        # ========================================
        # COMPONENT 4: Relative Pose to Target (5 dims)
        # ========================================

        # Transform target position to ego-centric frame
        # Rotation matrix: R(θ) = [[cos(θ), sin(θ)], [-sin(θ), cos(θ)]]
        dx = target_x - x  # World-frame displacement
        dy = target_y - y
        cos_t = float(np.cos(theta))
        sin_t = float(np.sin(theta))
        rel_x = cos_t * dx + sin_t * dy  # Ego-frame x (forward/backward)
        rel_y = -sin_t * dx + cos_t * dy  # Ego-frame y (left/right)
        rel_x_norm = float(np.clip(rel_x / position_scale, -1.0, 1.0))
        rel_y_norm = float(np.clip(rel_y / position_scale, -1.0, 1.0))

        # Relative heading: sin/cos encoding to avoid 2π discontinuity
        dtheta = float((target_theta - theta + np.pi) % (2.0 * np.pi) - np.pi)  # Wrap to [-π, π]
        sin_theta = float(np.sin(dtheta))
        cos_theta = float(np.cos(dtheta))

        # Euclidean distance to target
        distance = float(np.sqrt(dx**2 + dy**2))
        distance_norm = float(np.clip(distance / position_scale, 0.0, 1.0))

        # Relative pose features: [rel_x, rel_y, sin(Δθ), cos(Δθ), distance]
        components.append(
            np.array(
                [rel_x_norm, rel_y_norm, sin_theta, cos_theta, distance_norm],
                dtype=np.float32,
            )
        )
    else:
        # No target: use zero features
        components.append(np.zeros(3, dtype=np.float32))  # Target velocity
        components.append(np.zeros(5, dtype=np.float32))  # Relative pose

    # ========================================
    # FINAL CONCATENATION
    # ========================================

    # Concatenate all components into single flat vector
    # Total dims: 108 (LiDAR) + 3 (ego vel) + 3 (target vel) + 5 (rel pose) = 119
    return np.concatenate(components)


def flatten_centerline_obs(
    obs_dict: Dict[str, Any],
    scales: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """Flatten centerline observation dict to a normalized feature vector.

    Input features:
    - LiDAR scans clipped to 10m and normalized to [0, 1]
    - Speed magnitude normalized to [0, 1]
    - Previous action clipped to [-1, 1]
    """
    scales = scales or {}

    lidar_range = float(scales.get("lidar_range", 10.0))
    if lidar_range <= 0.0:
        lidar_range = 10.0

    scan = obs_dict.get("scans")
    if scan is None:
        scan = obs_dict.get("lidar")
    if scan is None:
        lidar = np.zeros(108, dtype=np.float32)
    else:
        lidar = np.asarray(scan, dtype=np.float32).reshape(-1)
    lidar_norm = np.clip(lidar, 0.0, lidar_range) / lidar_range

    velocity = obs_dict.get("velocity")
    if velocity is None:
        vx = 0.0
        vy = 0.0
    else:
        vel_arr = np.asarray(velocity, dtype=np.float32).reshape(-1)
        vx = float(vel_arr[0]) if vel_arr.size > 0 else 0.0
        vy = float(vel_arr[1]) if vel_arr.size > 1 else 0.0
    speed = float(np.sqrt(vx * vx + vy * vy))
    speed_scale = float(scales.get("speed", 1.0))
    if speed_scale <= 0.0:
        speed_scale = 1.0
    speed_norm = np.clip(speed / speed_scale, 0.0, 1.0)

    prev_action = obs_dict.get("prev_action")
    if prev_action is None:
        prev_arr = np.zeros(2, dtype=np.float32)
    else:
        prev_arr = np.asarray(prev_action, dtype=np.float32).reshape(-1)
        if prev_arr.size < 2:
            padded = np.zeros(2, dtype=np.float32)
            padded[: prev_arr.size] = prev_arr
            prev_arr = padded
    prev_norm = np.clip(prev_arr, -1.0, 1.0)

    return np.concatenate(
        [lidar_norm.astype(np.float32), np.array([speed_norm], dtype=np.float32), prev_norm.astype(np.float32)],
        axis=0,
    )


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
    elif preset == 'centerline':
        return flatten_centerline_obs(obs_dict, scales=scales)
    else:
        raise ValueError(
            f"Unsupported observation preset: {preset}. "
            f"Supported: gaplock, centerline"
        )

__all__ = ['flatten_observation', 'flatten_gaplock_obs', 'flatten_centerline_obs']
