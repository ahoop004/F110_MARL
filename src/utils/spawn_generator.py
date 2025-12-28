"""Utilities for generating spawn points from reward parameters.

Automatically creates spawn point configurations that align with
reward shaping, particularly pinch pocket Gaussian positions.
"""

from typing import Dict, List, Tuple
import numpy as np


def generate_pinch_pocket_spawns(
    defender_pose: Tuple[float, float, float],
    anchor_forward: float = 1.20,
    anchor_lateral: float = 0.70,
) -> Dict[str, Dict[str, List[float]]]:
    """Generate spawn points at pinch pocket positions.

    Creates spawn configurations where the attacker starts at optimal
    pinch pocket positions relative to the defender, based on forcing
    reward parameters.

    Args:
        defender_pose: Defender (x, y, theta) pose
        anchor_forward: Distance ahead of defender for pinch pocket (m)
        anchor_lateral: Lateral offset from defender centerline (m)

    Returns:
        Dict of spawn point configurations:
            {
                'spawn_pinch_right': {
                    'car_0': [x, y, theta],  # Attacker
                    'car_1': [x, y, theta],  # Defender
                },
                'spawn_pinch_left': { ... },
                'spawn_pinch_ahead': { ... },
            }

    Example:
        >>> defender = (0.0, 0.0, 0.0)  # At origin, facing +x
        >>> spawns = generate_pinch_pocket_spawns(defender)
        >>> spawns['spawn_pinch_right']['car_0']  # Attacker at right pocket
        [1.2, -0.7, 0.0]  # 1.2m ahead, 0.7m to right
    """
    def_x, def_y, def_theta = defender_pose

    # Defender's heading vector
    cos_theta = np.cos(def_theta)
    sin_theta = np.sin(def_theta)

    # Transform pinch positions to world frame
    spawns = {}

    # Right pinch pocket (ahead and to defender's right)
    local_x_right = anchor_forward
    local_y_right = -anchor_lateral  # Right in defender's frame
    world_x_right = def_x + local_x_right * cos_theta - local_y_right * sin_theta
    world_y_right = def_y + local_x_right * sin_theta + local_y_right * cos_theta

    spawns['spawn_pinch_right'] = {
        'car_0': [float(world_x_right), float(world_y_right), float(def_theta)],
        'car_1': [float(def_x), float(def_y), float(def_theta)],
        'metadata': {
            'difficulty': 'easy',
            'curriculum_stage': 0,
            'description': 'Attacker at right pinch pocket'
        }
    }

    # Left pinch pocket (ahead and to defender's left)
    local_x_left = anchor_forward
    local_y_left = anchor_lateral  # Left in defender's frame
    world_x_left = def_x + local_x_left * cos_theta - local_y_left * sin_theta
    world_y_left = def_y + local_x_left * sin_theta + local_y_left * cos_theta

    spawns['spawn_pinch_left'] = {
        'car_0': [float(world_x_left), float(world_y_left), float(def_theta)],
        'car_1': [float(def_x), float(def_y), float(def_theta)],
        'metadata': {
            'difficulty': 'easy',
            'curriculum_stage': 0,
            'description': 'Attacker at left pinch pocket'
        }
    }

    # Directly ahead (centerline)
    local_x_ahead = anchor_forward
    local_y_ahead = 0.0
    world_x_ahead = def_x + local_x_ahead * cos_theta - local_y_ahead * sin_theta
    world_y_ahead = def_y + local_x_ahead * sin_theta + local_y_ahead * cos_theta

    spawns['spawn_pinch_ahead'] = {
        'car_0': [float(world_x_ahead), float(world_y_ahead), float(def_theta)],
        'car_1': [float(def_x), float(def_y), float(def_theta)],
        'metadata': {
            'difficulty': 'easy',
            'curriculum_stage': 0,
            'description': 'Attacker directly ahead on centerline'
        }
    }

    return spawns


def generate_approach_spawns(
    defender_pose: Tuple[float, float, float],
    approach_distance: float = 2.5,
    lateral_offsets: List[float] = None,
) -> Dict[str, Dict[str, List[float]]]:
    """Generate spawn points at approach positions (further back).

    Creates spawn configurations where attacker must approach and position
    themselves, requiring more skill than starting at pinch pockets.

    Args:
        defender_pose: Defender (x, y, theta) pose
        approach_distance: Distance behind defender to start (m)
        lateral_offsets: List of lateral offsets to generate (m)

    Returns:
        Dict of spawn point configurations

    Example:
        >>> defender = (5.0, 3.0, np.pi/4)
        >>> spawns = generate_approach_spawns(defender, approach_distance=3.0)
    """
    if lateral_offsets is None:
        lateral_offsets = [0.0, 0.5, -0.5, 1.0, -1.0]

    def_x, def_y, def_theta = defender_pose

    cos_theta = np.cos(def_theta)
    sin_theta = np.sin(def_theta)

    spawns = {}

    for idx, lateral_offset in enumerate(lateral_offsets):
        # Position behind defender
        local_x = -approach_distance  # Behind (negative forward)
        local_y = lateral_offset

        world_x = def_x + local_x * cos_theta - local_y * sin_theta
        world_y = def_y + local_x * sin_theta + local_y * cos_theta

        offset_str = f"{abs(lateral_offset):.1f}{'L' if lateral_offset > 0 else 'R' if lateral_offset < 0 else 'C'}"
        spawn_name = f'spawn_approach_{offset_str}'

        spawns[spawn_name] = {
            'car_0': [float(world_x), float(world_y), float(def_theta)],
            'car_1': [float(def_x), float(def_y), float(def_theta)],
            'metadata': {
                'difficulty': 'medium',
                'curriculum_stage': 1,
                'description': f'Attacker {approach_distance}m behind, {offset_str} offset'
            }
        }

    return spawns


def print_spawn_yaml(spawns: Dict[str, Dict[str, List[float]]]) -> None:
    """Print spawn points in YAML format for easy copy-paste.

    Args:
        spawns: Dict of spawn configurations

    Example:
        >>> spawns = generate_pinch_pocket_spawns((0, 0, 0))
        >>> print_spawn_yaml(spawns)
    """
    print("spawn_points:")
    for spawn_name, spawn_data in spawns.items():
        print(f"  {spawn_name}:")
        for agent_id, pose in spawn_data.items():
            if agent_id == 'metadata':
                print(f"    metadata:")
                for key, value in pose.items():
                    if isinstance(value, str):
                        print(f"      {key}: \"{value}\"")
                    else:
                        print(f"      {key}: {value}")
            else:
                print(f"    {agent_id}: [{pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}]")


__all__ = [
    'generate_pinch_pocket_spawns',
    'generate_approach_spawns',
    'print_spawn_yaml',
]
