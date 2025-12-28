#!/usr/bin/env python3
"""Test heatmap reward query functionality."""

import numpy as np
from src.rewards.presets import load_preset
from src.rewards.composer import ComposedReward
from src.rewards.gaplock.forcing import ForcingReward
from src.rewards.gaplock.distance import DistanceReward
from src.rewards.gaplock.heading import HeadingReward

# Load gaplock config
config = load_preset('gaplock_full')

# Create forcing reward (pinch pockets)
forcing_config = {
    'enabled': True,
    'pinch_pockets': {
        'enabled': True,
        'anchor_forward': 0.55,
        'anchor_lateral': 0.45,
        'sigma': 0.35,
        'weight': 0.12,
    }
}

forcing_reward = ForcingReward(forcing_config)

# Test pinch pocket rewards at various positions
target_obs = {
    'pose': np.array([0.0, 0.0, 0.0]),  # Target at origin
    'scans': np.ones(720) * 5.0,
}

print("Testing Pinch Pocket Rewards:")
print("=" * 60)

# Test positions around the target
test_positions = [
    (0.0, 0.0, "At target"),
    (0.55, 0.45, "Right pocket center"),
    (0.55, -0.45, "Left pocket center"),
    (0.55, 0.0, "Ahead of target"),
    (1.0, 0.0, "Far ahead"),
    (0.0, 1.0, "To side"),
]

for x, y, desc in test_positions:
    # Attacker position
    theta = np.arctan2(-y, -x)  # Face toward target
    attacker_obs = {
        'pose': np.array([x, y, theta]),
        'scans': np.zeros(720),
    }

    step_info = {
        'obs': attacker_obs,
        'target_obs': target_obs,
        'done': False,
        'truncated': False,
        'info': {},
        'timestep': 0.01,
    }

    # Compute forcing reward
    components = forcing_reward.compute(step_info)
    pinch_reward = components.get('forcing/pinch', 0.0)

    print(f"{desc:20s} ({x:5.2f}, {y:5.2f}): {pinch_reward:6.4f}")

print("\n✓ Test complete!")
print("Expected: Highest rewards at pocket centers (±0.55, ±0.45)")
