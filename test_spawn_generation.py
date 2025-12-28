#!/usr/bin/env python3
"""Test spawn point generation for pinch pockets."""

import numpy as np
from utils.spawn_generator import (
    generate_pinch_pocket_spawns,
    generate_approach_spawns,
    print_spawn_yaml
)

# Use spawn_2 as baseline defender position (from line2 map)
defender_base_pose = (-46.769, 0.0768, 0.0)

print("=" * 70)
print("PINCH POCKET SPAWN POINTS")
print("=" * 70)
print(f"\nDefender base position: {defender_base_pose}")
print(f"Using reward parameters: anchor_forward=1.20m, anchor_lateral=0.70m\n")

# Generate pinch pocket spawns (easy - curriculum stage 0)
pinch_spawns = generate_pinch_pocket_spawns(
    defender_base_pose,
    anchor_forward=1.20,
    anchor_lateral=0.70
)

print_spawn_yaml(pinch_spawns)

print("\n" + "=" * 70)
print("APPROACH SPAWN POINTS")
print("=" * 70)
print(f"\nAttacker starts further back, must approach and position\n")

# Generate approach spawns (medium - curriculum stage 1)
approach_spawns = generate_approach_spawns(
    defender_base_pose,
    approach_distance=2.5,
    lateral_offsets=[0.0, 0.5, -0.5]
)

print_spawn_yaml(approach_spawns)

print("\n" + "=" * 70)
print("SPAWN CURRICULUM CONFIGURATION")
print("=" * 70)
print("""
Add this to your scenario YAML:

environment:
  spawn_curriculum:
    enabled: true
    window: 200
    activation_samples: 50
    min_episode: 100
    enable_patience: 5
    disable_patience: 3
    cooldown: 20

    stages:
      # Stage 0: Easy - Fixed optimal positions, fixed speed
      - name: "optimal_fixed"
        spawn_points: [spawn_pinch_right, spawn_pinch_left, spawn_pinch_ahead]
        speed_range: [0.5, 0.5]  # Fixed 0.5 m/s for both agents
        enable_rate: 0.70  # Advance when success >= 70%

      # Stage 1: Medium - Optimal positions, variable speed
      - name: "optimal_varied_speed"
        spawn_points: [spawn_pinch_right, spawn_pinch_left, spawn_pinch_ahead, spawn_approach_0.0C, spawn_approach_0.5L, spawn_approach_0.5R]
        speed_range: [0.3, 1.0]  # Random speed 0.3-1.0 m/s
        enable_rate: 0.65
        disable_rate: 0.50  # Regress if success < 50%

      # Stage 2: Hard - All spawns, variable speed
      - name: "full_random"
        spawn_points: "all"  # Use all available spawn points
        speed_range: [0.3, 1.0]
        disable_rate: 0.45
""")

print("\n✓ Spawn generation complete!")
print("Copy the spawn points above into your scenario YAML or map configuration.")
