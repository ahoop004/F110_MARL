"""Observation flattening for F110 Dict observations.

This module converts structured observations (dicts with LiDAR, pose, velocity)
into flat numpy arrays suitable for neural network input.

Key Features:
- Config-driven: YAML preset settings control which features are included
- Flattens multi-agent observations into agent-centric representations
- Handles adversarial tasks by including relative state to target agent
- Normalizes features to bounded ranges for stable learning
- Uses sin/cos encoding for angles to avoid discontinuities
"""

from typing import Dict, Any, Optional
import numpy as np


def _get_nested(config: Dict[str, Any], *keys, default: Any = None) -> Any:
    """Get nested config value with default."""
    value = config
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
        if value is None:
            return default
    return value


def _extract_pose(data: Any) -> np.ndarray:
    """Extract pose [x, y, theta] from observation data."""
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
    """Extract velocity [vx, vy, omega] from observation data."""
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
    """Parse agent index from agent name (e.g., 'car_1' -> 1)."""
    if not agent_name:
        return None
    if isinstance(agent_name, str) and agent_name.startswith("car_"):
        try:
            return int(agent_name.split("_", 1)[1])
        except (ValueError, IndexError):
            return None
    return None


def _extract_from_state_vector(state_vec: np.ndarray, agent_idx: int) -> Optional[np.ndarray]:
    """Extract agent state from flattened multi-agent state vector."""
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


def flatten_gaplock_obs(
    obs_dict: Dict[str, Any],
    target_id: Optional[str] = None,
    scales: Optional[Dict[str, float]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Flatten gaplock observation dict to a normalized feature vector.

    The observation components are controlled by the config parameter:
    - lidar.enabled: Include LiDAR scan (default: True)
    - lidar.normalize: Normalize LiDAR to [0,1] (default: True)
    - lidar.max_range: Max range for normalization (default: 10.0)
    - ego_state.velocity: Include ego velocity (default: True)
    - target_state.enabled: Include target features (default: True)
    - target_state.velocity: Include target velocity (default: True)
    - relative_pose.enabled: Include relative pose to target (default: True)
    - relative_pose.dim: Dimension of relative pose features (default: 5)

    Args:
        obs_dict: Observation dict from environment (ego agent's view)
        target_id: Target agent ID for adversarial tasks (e.g., 'car_1')
        scales: Optional normalization scales from environment
        config: Observation preset config controlling which features to include

    Returns:
        Flattened observation array normalized to bounded ranges
    """
    config = config or {}
    components = []

    # Extract normalization scales
    scales = scales or {}
    lidar_range = float(_get_nested(config, 'lidar', 'max_range', default=10.0))
    if lidar_range <= 0.0:
        lidar_range = 10.0
    # Allow environment to override if provided
    if 'lidar_range' in scales:
        lidar_range = float(scales['lidar_range'])

    position_scale = float(scales.get('position', lidar_range))
    speed_scale = float(scales.get('speed', 1.0))
    if position_scale <= 0.0:
        position_scale = lidar_range
    if speed_scale <= 0.0:
        speed_scale = 1.0

    # ========================================
    # COMPONENT: LiDAR
    # ========================================
    lidar_enabled = _get_nested(config, 'lidar', 'enabled', default=True)
    if lidar_enabled:
        scan = obs_dict.get('scans')
        if scan is None:
            scan = obs_dict.get('lidar')
        if scan is None:
            lidar = np.zeros(54, dtype=np.float32)  # Default beam count
        else:
            lidar = np.asarray(scan, dtype=np.float32).reshape(-1)

        lidar_normalize = _get_nested(config, 'lidar', 'normalize', default=True)
        if lidar_normalize:
            lidar = np.clip(lidar / lidar_range, 0.0, 1.0)

        components.append(lidar.astype(np.float32))

    # ========================================
    # COMPONENT: Ego Velocity
    # ========================================
    ego_velocity_enabled = _get_nested(config, 'ego_state', 'velocity', default=True)

    # Always extract ego pose for relative computations (even if not in output)
    ego_pose = _extract_pose(obs_dict)
    x, y, theta = map(float, ego_pose[:3])

    if ego_velocity_enabled:
        ego_vel = _extract_velocity(obs_dict)
        ego_vel_norm = np.clip(ego_vel / speed_scale, -1.0, 1.0)
        components.append(ego_vel_norm.astype(np.float32))

    # ========================================
    # COMPONENT: Target State (velocity + relative pose)
    # ========================================
    target_enabled = _get_nested(config, 'target_state', 'enabled', default=True)
    target_velocity_enabled = _get_nested(config, 'target_state', 'velocity', default=True)
    relative_pose_enabled = _get_nested(config, 'relative_pose', 'enabled', default=True)

    # Only process target if at least one target feature is enabled
    if target_id and (target_enabled or relative_pose_enabled):
        # Extract target state from central_state
        target_state = None
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

            # Target velocity (if enabled)
            if target_enabled and target_velocity_enabled:
                target_vel = np.array([target_vx, target_vy, target_omega], dtype=np.float32)
                target_vel_norm = np.clip(target_vel / speed_scale, -1.0, 1.0)
                components.append(target_vel_norm)

            # Relative pose (if enabled)
            if relative_pose_enabled:
                rel_dim = int(_get_nested(config, 'relative_pose', 'dim', default=5))

                # Transform to ego-centric frame
                dx = target_x - x
                dy = target_y - y
                cos_t = float(np.cos(theta))
                sin_t = float(np.sin(theta))
                rel_x = cos_t * dx + sin_t * dy
                rel_y = -sin_t * dx + cos_t * dy
                rel_x_norm = float(np.clip(rel_x / position_scale, -1.0, 1.0))
                rel_y_norm = float(np.clip(rel_y / position_scale, -1.0, 1.0))

                # Relative heading with sin/cos encoding
                dtheta = float((target_theta - theta + np.pi) % (2.0 * np.pi) - np.pi)
                sin_theta = float(np.sin(dtheta))
                cos_theta = float(np.cos(dtheta))

                # Distance
                distance = float(np.sqrt(dx**2 + dy**2))
                distance_norm = float(np.clip(distance / position_scale, 0.0, 1.0))

                if rel_dim >= 5:
                    rel_features = np.array(
                        [rel_x_norm, rel_y_norm, sin_theta, cos_theta, distance_norm],
                        dtype=np.float32,
                    )
                elif rel_dim == 4:
                    rel_features = np.array(
                        [rel_x_norm, rel_y_norm, sin_theta, cos_theta],
                        dtype=np.float32,
                    )
                elif rel_dim == 3:
                    rel_features = np.array(
                        [rel_x_norm, rel_y_norm, distance_norm],
                        dtype=np.float32,
                    )
                else:
                    rel_features = np.array([rel_x_norm, rel_y_norm], dtype=np.float32)

                components.append(rel_features)
        else:
            # No target state available - add zeros for enabled features
            if target_enabled and target_velocity_enabled:
                components.append(np.zeros(3, dtype=np.float32))
            if relative_pose_enabled:
                rel_dim = int(_get_nested(config, 'relative_pose', 'dim', default=5))
                components.append(np.zeros(rel_dim, dtype=np.float32))

    if not components:
        raise ValueError("No observation components enabled in config")

    return np.concatenate(components)


def flatten_centerline_obs(
    obs_dict: Dict[str, Any],
    scales: Optional[Dict[str, float]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Flatten centerline observation dict to a normalized feature vector.

    Components controlled by config:
    - lidar.enabled: Include LiDAR scan (default: True)
    - lidar.normalize: Normalize LiDAR (default: True)
    - lidar.max_range: Max range for normalization (default: 10.0)
    - speed.enabled: Include speed magnitude (default: True)
    - prev_action.enabled: Include previous action (default: True)
    - prev_action.dim: Dimension of prev_action (default: 2)

    Args:
        obs_dict: Observation dict from environment
        scales: Optional normalization scales from environment
        config: Observation preset config controlling which features to include

    Returns:
        Flattened observation array
    """
    config = config or {}
    scales = scales or {}
    components = []

    # LiDAR
    lidar_enabled = _get_nested(config, 'lidar', 'enabled', default=True)
    if lidar_enabled:
        lidar_range = float(_get_nested(config, 'lidar', 'max_range', default=10.0))
        if lidar_range <= 0.0:
            lidar_range = 10.0
        if 'lidar_range' in scales:
            lidar_range = float(scales['lidar_range'])

        scan = obs_dict.get("scans")
        if scan is None:
            scan = obs_dict.get("lidar")
        if scan is None:
            lidar = np.zeros(108, dtype=np.float32)
        else:
            lidar = np.asarray(scan, dtype=np.float32).reshape(-1)

        lidar_normalize = _get_nested(config, 'lidar', 'normalize', default=True)
        if lidar_normalize:
            lidar = np.clip(lidar, 0.0, lidar_range) / lidar_range

        components.append(lidar.astype(np.float32))

    # Speed magnitude
    speed_enabled = _get_nested(config, 'speed', 'enabled', default=True)
    if speed_enabled:
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
        components.append(np.array([speed_norm], dtype=np.float32))

    # Previous action
    prev_action_enabled = _get_nested(config, 'prev_action', 'enabled', default=True)
    if prev_action_enabled:
        prev_dim = int(_get_nested(config, 'prev_action', 'dim', default=2))
        prev_action = obs_dict.get("prev_action")
        if prev_action is None:
            prev_arr = np.zeros(prev_dim, dtype=np.float32)
        else:
            prev_arr = np.asarray(prev_action, dtype=np.float32).reshape(-1)
            if prev_arr.size < prev_dim:
                padded = np.zeros(prev_dim, dtype=np.float32)
                padded[: prev_arr.size] = prev_arr
                prev_arr = padded
            elif prev_arr.size > prev_dim:
                prev_arr = prev_arr[:prev_dim]
        prev_norm = np.clip(prev_arr, -1.0, 1.0)
        components.append(prev_norm.astype(np.float32))

    if not components:
        raise ValueError("No observation components enabled in config")

    return np.concatenate(components)


def flatten_observation(
    obs_dict: Dict[str, Any],
    preset: str = 'gaplock',
    target_id: Optional[str] = None,
    scales: Optional[Dict[str, float]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Flatten observation dict based on preset configuration.

    If config is not provided, loads the preset config from observation presets.
    Config values override defaults, allowing YAML settings to control behavior.

    Args:
        obs_dict: Observation dict from environment
        preset: Observation preset name ('gaplock', 'centerline')
        target_id: Target agent ID (for adversarial tasks)
        scales: Optional normalization scales from environment
        config: Optional preset config (if None, loads from preset registry)

    Returns:
        Flat observation array

    Raises:
        ValueError: If preset not supported
    """
    # Load config from preset if not provided
    if config is None:
        try:
            from src.core.observations import load_observation_preset
            config = load_observation_preset(preset)
        except (ImportError, ValueError):
            config = {}

    if preset == 'gaplock':
        return flatten_gaplock_obs(obs_dict, target_id, scales=scales, config=config)
    elif preset == 'centerline':
        return flatten_centerline_obs(obs_dict, scales=scales, config=config)
    else:
        raise ValueError(
            f"Unsupported observation preset: {preset}. "
            f"Supported: gaplock, centerline"
        )


def compute_obs_dim_from_config(
    config: Dict[str, Any],
    preset: str = 'gaplock',
    has_target: bool = True,
    lidar_beams: int = 54,
    frame_stack: int = 1,
) -> int:
    """Compute observation dimension from config settings.

    Args:
        config: Observation preset config
        preset: Preset name for preset-specific logic
        has_target: Whether a target agent exists
        lidar_beams: Number of LiDAR beams from environment
        frame_stack: Frame stacking multiplier

    Returns:
        Total observation dimension (with frame stacking applied)
    """
    dim = 0

    if preset == 'gaplock':
        # LiDAR
        if _get_nested(config, 'lidar', 'enabled', default=True):
            dim += lidar_beams

        # Ego velocity
        if _get_nested(config, 'ego_state', 'velocity', default=True):
            dim += 3

        # Target features (only if target exists)
        if has_target:
            target_enabled = _get_nested(config, 'target_state', 'enabled', default=True)
            target_velocity = _get_nested(config, 'target_state', 'velocity', default=True)
            if target_enabled and target_velocity:
                dim += 3

            if _get_nested(config, 'relative_pose', 'enabled', default=True):
                rel_dim = int(_get_nested(config, 'relative_pose', 'dim', default=5))
                dim += rel_dim

    elif preset == 'centerline':
        # LiDAR
        if _get_nested(config, 'lidar', 'enabled', default=True):
            dim += lidar_beams

        # Speed
        if _get_nested(config, 'speed', 'enabled', default=True):
            dim += 1

        # Previous action
        if _get_nested(config, 'prev_action', 'enabled', default=True):
            prev_dim = int(_get_nested(config, 'prev_action', 'dim', default=2))
            dim += prev_dim

    # Apply frame stacking
    if frame_stack > 1:
        dim *= frame_stack

    return dim


__all__ = [
    'flatten_observation',
    'flatten_gaplock_obs',
    'flatten_centerline_obs',
    'compute_obs_dim_from_config',
]
