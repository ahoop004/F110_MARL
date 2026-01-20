"""Reward presets for common tasks.

Presets provide default configurations that can be overridden.
"""

from typing import Dict, Any
import copy


# Gaplock Full: Complete v1 configuration
GAPLOCK_FULL: Dict[str, Any] = {
    'terminal': {
        'target_crash': 60.0,
        'self_crash': -90.0,
        'collision': -90.0,
        'timeout': -20.0,
        'idle_stop': -5.0,
        'target_finish': -20.0,
    },
    'pressure': {
        'enabled': True,
        'distance_threshold': 1.30,
        'timeout': 1.20,
        'min_speed': 0.30,
        'heading_tolerance': 1.57,
        'bonus': 0.12,
        'bonus_interval': 5,
        'streak_bonus': 0.10,
        'streak_cap': 40,
    },
    'distance': {
        'enabled': True,
        'reward_near': 0.12,
        'near_distance': 1.00,
        'far_distance': 2.50,
        'penalty_far': 0.08,
        'gradient': {
            'enabled': True,
            'scale': 0.20,
            'time_scaled': True,
            'clip': [-0.20, 0.20],
            'points': [
                [0.25, -1.00],
                [0.50, -0.50],
                [0.75, 0.00],
                [1.00, 0.30],
                [1.50, 0.50],
                [2.00, 0.20],
                [2.50, 0.00],
                [3.50, -0.30],
            ],
        },
    },
    'heading': {
        'enabled': True,
        'coefficient': 0.08,
    },
    'speed': {
        'enabled': True,
        'bonus_coef': 0.05,
        'target_speed': 0.60,
    },
    'forcing': {
        'enabled': True,
        'pinch_pockets': {
            'enabled': True,
            'anchor_forward': 1.20,
            'anchor_lateral': 0.70,
            'sigma': 0.50,
            'weight': 0.30,
        },
        'clearance': {
            'enabled': True,
            'weight': 0.80,
            'band_min': 0.30,
            'band_max': 3.20,
            'clip': 0.25,
            'time_scaled': True,
        },
        'turn': {
            'enabled': True,
            'weight': 2.0,
            'clip': 0.35,
            'time_scaled': True,
        },
    },
    'penalties': {
        'enabled': True,
        'idle': {
            'penalty': 0.05,
            'speed_threshold': 0.12,
            'patience_steps': 25,
        },
        'reverse': {
            'penalty': 0.10,
            'speed_threshold': 0.02,
        },
        'brake': {
            'penalty': 0.05,
            'speed_threshold': 0.40,
            'drop_threshold': 0.25,
        },
    },
}


# Gaplock Simple: Terminal + basic shaping (no forcing)
GAPLOCK_SIMPLE: Dict[str, Any] = {
    'terminal': GAPLOCK_FULL['terminal'].copy(),
    'pressure': GAPLOCK_FULL['pressure'].copy(),
    'distance': {
        'enabled': True,
        'reward_near': 0.10,
        'near_distance': 1.00,
        'far_distance': 2.50,
        'penalty_far': 0.05,
        'gradient': {'enabled': False},
    },
    'heading': GAPLOCK_FULL['heading'].copy(),
    'speed': GAPLOCK_FULL['speed'].copy(),
    'forcing': {'enabled': False},
    'penalties': GAPLOCK_FULL['penalties'].copy(),
}


# Preset registry
PRESETS = {
    'gaplock_full': GAPLOCK_FULL,
    'gaplock_simple': GAPLOCK_SIMPLE,
}


# Centerline racing preset
CENTERLINE_RACING: Dict[str, Any] = {
    'centerline': {
        'vs_weight': 1.0,
        'vd_weight': 0.01,
        'd_weight': 0.02,
        'steer_weight': 0.1,
        'collision_penalty': -1000.0,
        'steer_index': 0,
    },
}

PRESETS['centerline_racing'] = CENTERLINE_RACING


def load_preset(name: str) -> Dict[str, Any]:
    """Load a reward preset by name.

    Args:
        name: Preset name ('gaplock_full', 'gaplock_simple')

    Returns:
        Copy of preset configuration dict

    Raises:
        ValueError: If preset name not found
    """
    if name not in PRESETS:
        available = ', '.join(PRESETS.keys())
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")

    return copy.deepcopy(PRESETS[name])


def merge_config(base: dict, overrides: dict) -> dict:
    """Deep merge overrides into base config.

    Args:
        base: Base configuration dict
        overrides: Override values

    Returns:
        Merged configuration (new dict)
    """
    result = base.copy()

    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dicts
            result[key] = merge_config(result[key], value)
        else:
            # Override value
            result[key] = value

    return result


__all__ = [
    'GAPLOCK_FULL',
    'GAPLOCK_SIMPLE',
    'CENTERLINE_RACING',
    'PRESETS',
    'load_preset',
    'merge_config',
]
