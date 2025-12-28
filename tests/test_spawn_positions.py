#!/usr/bin/env python3
"""Visual inspection tool for spawn point positions.

Cycles through all spawn configurations so you can verify they're properly centered
on the track and not spawning outside boundaries.
"""

import numpy as np
import yaml
import time
from src.env.f110ParallelEnv import F110ParallelEnv

def main():
    # Load scenario
    with open('scenarios/v2/gaplock_sac.yaml', 'r') as f:
        scenario = yaml.safe_load(f)

    # Get spawn configs
    spawn_curriculum = scenario['environment']['spawn_curriculum']
    spawn_configs = spawn_curriculum['spawn_configs']

    print("=" * 70)
    print("SPAWN POINT VISUAL INSPECTION")
    print("=" * 70)
    print(f"\nFound {len(spawn_configs)} spawn configurations")
    print("\nControls:")
    print("  - Window will show each spawn point for 3 seconds")
    print("  - Press Ctrl+C to exit early")
    print("  - Check that cars are centered on the lane")
    print("\n" + "=" * 70)

    # Create environment with rendering
    env_config = scenario['environment'].copy()
    env_config['render'] = True
    env = F110ParallelEnv(**env_config, render_mode='human')

    # Cycle through each spawn point
    spawn_names = list(spawn_configs.keys())

    try:
        while True:  # Continuous loop
            for i, spawn_name in enumerate(spawn_names, 1):
                spawn_data = spawn_configs[spawn_name]

                # Extract poses
                poses = []
                for agent_id in ['car_0', 'car_1']:
                    if agent_id in spawn_data:
                        poses.append(spawn_data[agent_id])

                poses_array = np.array(poses, dtype=np.float32)

                # Reset environment with this spawn point
                print(f"\n[{i}/{len(spawn_names)}] Showing: {spawn_name}")
                print(f"  car_0: x={poses[0][0]:.3f}, y={poses[0][1]:.3f}, theta={poses[0][2]:.3f}")
                print(f"  car_1: x={poses[1][0]:.3f}, y={poses[1][1]:.3f}, theta={poses[1][2]:.3f}")

                obs, info = env.reset(options={
                    'poses': poses_array,
                    'velocities': {'car_0': 0.0}  # Stationary for inspection
                })

                # Render for 3 seconds
                for _ in range(30):  # 30 frames at ~10 FPS
                    env.render()
                    time.sleep(0.1)

            print("\n" + "=" * 70)
            print("Completed one cycle. Repeating...")
            print("=" * 70)

    except KeyboardInterrupt:
        print("\n\nInspection complete!")
    finally:
        env.close()

if __name__ == "__main__":
    main()
