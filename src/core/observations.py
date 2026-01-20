"""Observation configuration and presets for F110 agents.

Provides preset observation configurations that define what information
agents receive from the environment. Configurations specify LiDAR settings,
state components, and normalization options.
"""

from typing import Dict, Any, Optional
import copy


# Observation dimension calculations
def compute_obs_dim(config: Dict[str, Any]) -> int:
    """Compute total observation dimension from config.

    Args:
        config: Observation configuration dict

    Returns:
        Total observation dimension

    Example:
        >>> config = {'lidar_beams': 720, 'ego_pose': True, 'ego_velocity': True}
        >>> compute_obs_dim(config)
        738
    """
    dim = 0

    # LiDAR beams
    if config.get('lidar', {}).get('enabled', True):
        dim += config.get('lidar', {}).get('beams', 720)

    # Ego state
    if config.get('ego_state', {}).get('pose', True):
        dim += 4  # x, y, theta, sin(theta), cos(theta) -> actually 4 in practice

    if config.get('ego_state', {}).get('velocity', True):
        dim += 3  # vx, vy, angular_velocity

    # Target state (for adversarial tasks)
    if config.get('target_state', {}).get('enabled', False):
        if config.get('target_state', {}).get('pose', True):
            dim += 4

        if config.get('target_state', {}).get('velocity', True):
            dim += 3

    # Relative pose (for adversarial tasks)
    if config.get('relative_pose', {}).get('enabled', False):
        rel_dim = config.get('relative_pose', {}).get('dim', 4)
        try:
            rel_dim = int(rel_dim)
        except (TypeError, ValueError):
            rel_dim = 4
        dim += max(rel_dim, 0)

    # Centerline extras (single-agent)
    if config.get('speed', {}).get('enabled', False):
        dim += 1  # speed magnitude
    if config.get('prev_action', {}).get('enabled', False):
        prev_dim = config.get('prev_action', {}).get('dim', 2)
        try:
            prev_dim = int(prev_dim)
        except (TypeError, ValueError):
            prev_dim = 2
        dim += max(prev_dim, 0)

    return dim


# Preset observation configurations

# Gaplock observation configuration (119 dims with 108-beam LiDAR).
# Flattening uses:
# - LiDAR: 108 beams, 12.0m max range, normalized
# - Ego velocity: 3 dims (vx, vy, omega)
# - Target velocity: 3 dims (vx, vy, omega)
# - Relative pose: 5 dims (rel_x, rel_y, sin(Δθ), cos(Δθ), distance)
# - Total: beams + 11
GAPLOCK_OBS: Dict[str, Any] = {
    'lidar': {
        'enabled': True,
        'beams': 108,
        'max_range': 12.0,
        'normalize': True,
    },
    'ego_state': {
        'pose': False,
        'velocity': True,   # vx, vy, angular_vel (3 dims)
    },
    'target_state': {
        'enabled': True,
        'pose': False,
        'velocity': True,   # vx, vy, angular_vel (3 dims)
    },
    'relative_pose': {
        'enabled': True,
        'dim': 5,
    },
    'normalization': {
        'enabled': True,
        'trainable_only': True,  # Only normalize trainable agents, not FTG
    },
}


# Minimal observation configuration (115 dims).
# Reduced observation space for faster training:
# - LiDAR: 108 beams (every 10°), 12.0m max range
# - Ego state: pose + velocity = 7 dims
# - No target state
# - Total: 108 + 7 = 115 dims
MINIMAL_OBS: Dict[str, Any] = {
    'lidar': {
        'enabled': True,
        'beams': 108,
        'max_range': 12.0,
        'normalize': True,
    },
    'ego_state': {
        'pose': True,
        'velocity': True,
    },
    'target_state': {
        'enabled': False,
    },
    'relative_pose': {
        'enabled': False,
    },
    'normalization': {
        'enabled': True,
        'trainable_only': True,
    },
}


# Full observation configuration (1098 dims).
# Maximum observation space with all features:
# - LiDAR: 1080 beams (every 0.25°), 12.0m max range
# - Ego state: pose + velocity = 7 dims
# - Target state: pose + velocity = 7 dims
# - Relative pose: 4 dims
# - Total: 1080 + 7 + 7 + 4 = 1098 dims
FULL_OBS: Dict[str, Any] = {
    'lidar': {
        'enabled': True,
        'beams': 1080,
        'max_range': 12.0,
        'normalize': True,
    },
    'ego_state': {
        'pose': True,
        'velocity': True,
    },
    'target_state': {
        'enabled': True,
        'pose': True,
        'velocity': True,
    },
    'relative_pose': {
        'enabled': True,
    },
    'normalization': {
        'enabled': True,
        'trainable_only': True,
    },
}


# Centerline observation configuration (1083 dims).
# - LiDAR: 1080 beams, 10.0m max range
# - Speed magnitude: 1 dim
# - Previous action: 2 dims
# - Total: 1080 + 1 + 2 = 1083 dims
CENTERLINE_OBS: Dict[str, Any] = {
    'lidar': {
        'enabled': True,
        'beams': 1080,
        'max_range': 10.0,
        'normalize': True,
    },
    'ego_state': {
        'pose': False,
        'velocity': False,
    },
    'target_state': {
        'enabled': False,
    },
    'relative_pose': {
        'enabled': False,
    },
    'speed': {
        'enabled': True,
    },
    'prev_action': {
        'enabled': True,
        'dim': 2,
    },
    'normalization': {
        'enabled': True,
        'trainable_only': True,
    },
}


# Registry of all presets
OBSERVATION_PRESETS: Dict[str, Dict[str, Any]] = {
    'gaplock': GAPLOCK_OBS,
    'minimal': MINIMAL_OBS,
    'full': FULL_OBS,
    'centerline': CENTERLINE_OBS,
}


def load_observation_preset(name: str) -> Dict[str, Any]:
    """Load an observation preset by name.

    Args:
        name: Preset name ('gaplock', 'minimal', 'full', or 'centerline')

    Returns:
        Deep copy of preset configuration

    Raises:
        ValueError: If preset name is unknown

    Example:
        >>> config = load_observation_preset('gaplock')
        >>> config['lidar']['beams']
        720
    """
    if name not in OBSERVATION_PRESETS:
        available = ', '.join(OBSERVATION_PRESETS.keys())
        raise ValueError(
            f"Unknown observation preset '{name}'. "
            f"Available presets: {available}"
        )

    return copy.deepcopy(OBSERVATION_PRESETS[name])


def merge_observation_config(
    preset: Dict[str, Any],
    overrides: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge observation config overrides into preset.

    Args:
        preset: Base preset configuration
        overrides: Override values (can be nested)

    Returns:
        Merged configuration (deep copy)

    Example:
        >>> preset = load_observation_preset('gaplock')
        >>> overrides = {'lidar': {'beams': 360}}
        >>> config = merge_observation_config(preset, overrides)
        >>> config['lidar']['beams']
        360
    """
    result = copy.deepcopy(preset)

    def merge_dict(base: dict, override: dict):
        """Recursively merge dicts."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                merge_dict(base[key], value)
            else:
                base[key] = value

    merge_dict(result, overrides)
    return result


def get_observation_config(
    preset: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Get observation configuration from preset, overrides, or full config.

    Provides flexible configuration loading:
    - Use preset name with optional overrides
    - Or provide complete config dict

    Args:
        preset: Preset name (if using preset)
        overrides: Optional overrides to preset
        config: Complete config dict (alternative to preset)

    Returns:
        Observation configuration dict

    Raises:
        ValueError: If neither preset nor config provided

    Example:
        >>> # Using preset
        >>> config = get_observation_config(preset='gaplock')
        >>>
        >>> # Using preset with overrides
        >>> config = get_observation_config(
        ...     preset='gaplock',
        ...     overrides={'lidar': {'beams': 360}}
        ... )
        >>>
        >>> # Using complete config
        >>> config = get_observation_config(config={...})
    """
    if config is not None:
        return copy.deepcopy(config)

    if preset is not None:
        base = load_observation_preset(preset)
        if overrides:
            return merge_observation_config(base, overrides)
        return base

    raise ValueError("Must provide either 'preset' or 'config'")


__all__ = [
    'compute_obs_dim',
    'load_observation_preset',
    'merge_observation_config',
    'get_observation_config',
    'OBSERVATION_PRESETS',
    'GAPLOCK_OBS',
    'MINIMAL_OBS',
    'FULL_OBS',
    'CENTERLINE_OBS',
]
