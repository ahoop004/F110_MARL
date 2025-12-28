"""Demo script showing how to use rendering extensions for training visualization.

This script demonstrates how to configure the renderer with various visualization
extensions including telemetry HUD, reward rings, and reward heatmaps.

Usage:
    PYTHONPATH=/home/aaron/F110_MARL:$PYTHONPATH python3 examples/visualization_demo.py

Keyboard controls:
    T - Cycle telemetry display mode (off → minimal → basic → detailed → full)
    R - Toggle reward ring visualization
    H - Toggle reward heatmap
    F - Toggle camera follow mode
    1-9 - Focus telemetry on specific agent
    0 - Show all agents in telemetry

    Mouse scroll - Zoom
    Mouse drag - Pan camera
"""

import numpy as np
from src.env.f110ParallelEnv import F110ParallelEnv
from src.render import EnvRenderer, TelemetryHUD, RewardRingExtension, RewardHeatmap
from src.agents.ftg import FollowTheGap
from src.rewards.presets import load_preset

def main():
    print("=== Visualization Demo ===")
    print("Press T, R, H to toggle visualizations")
    print("Press F to toggle camera follow mode")
    print("Press 1-2 to focus on specific agent, 0 for all")
    print()

    # Create environment with rendering
    env = F110ParallelEnv(
        map_name='maps/line2/line2',
        num_agents=2,
        timestep=0.01,
        render_mode='human'
    )

    # Create renderer with extensions
    renderer = env.renderer

    # === Configure Telemetry HUD ===
    telemetry = TelemetryHUD(renderer)
    telemetry.configure(
        enabled=True,
        mode=TelemetryHUD.MODE_BASIC  # Start in basic mode
    )
    renderer.add_extension(telemetry)

    # === Configure Reward Ring ===
    # Load gaplock reward config to get distance thresholds
    reward_config = load_preset('gaplock_full')
    distance_config = reward_config['distance']

    ring = RewardRingExtension(renderer)
    ring.configure(
        enabled=True,
        target_agent='car_1',  # Defender (target)
        inner_radius=distance_config['near_distance'],  # 1.0m
        outer_radius=distance_config['far_distance'],   # 2.5m
        preferred_radius=1.5,  # Optimal distance (between inner and outer)
    )
    renderer.add_extension(ring)

    # === Configure Reward Heatmap ===
    heatmap = RewardHeatmap(renderer)
    heatmap.configure(
        enabled=False,  # Start disabled (toggle with H)
        target_agent='car_1',
        attacker_agent='car_0',
        extent_m=6.0,  # 6m radius around target
        cell_size_m=0.25,  # 25cm cells
        alpha=0.22,  # Transparency
        near_distance=distance_config['near_distance'],
        far_distance=distance_config['far_distance'],
        reward_near=distance_config['reward_near'],
        penalty_far=distance_config['penalty_far'],
        update_frequency=5  # Update every 5 frames for performance
    )
    renderer.add_extension(heatmap)

    # Create FTG agents for demonstration
    agents = {
        'car_0': FollowTheGap(agent_id='car_0'),  # Will be "attacker" in viz
        'car_1': FollowTheGap(agent_id='car_1'),  # Defender/target
    }

    # Reset environment
    observations, infos = env.reset()

    print("Running demonstration episode...")
    print("Visualization extensions configured:")
    print("  - Telemetry HUD: ENABLED (mode: BASIC)")
    print("  - Reward Ring: ENABLED (around car_1)")
    print("  - Reward Heatmap: DISABLED (press H to enable)")
    print()

    # Run episode
    episode = 1
    step = 0
    done = False

    # Mock reward components for demonstration
    # In real training, these would come from the reward strategy
    reward_components = {
        'car_0': {
            'proximity': 0.0,
            'heading': 0.0,
            'speed': 0.0,
            'forcing': 0.0,
        }
    }

    while not done:
        step += 1

        # Get actions from agents
        actions = {}
        for agent_id, agent in agents.items():
            obs = observations[agent_id]
            action = agent.get_action(obs)
            actions[agent_id] = action

        # Step environment
        observations, rewards, terminations, truncations, infos = env.step(actions)

        # Update telemetry with episode info
        telemetry.update_episode_info(episode, step)

        # Update rewards for each agent
        for agent_id, reward in rewards.items():
            # In real training, extract components from reward strategy
            # For demo, create simple mock components
            components = reward_components.get(agent_id, {})

            # Update telemetry
            telemetry.update_rewards(
                agent_id=agent_id,
                reward=reward,
                components=components,
                reset=False
            )

            # Update collision status
            collision = infos.get(agent_id, {}).get('collision', False)
            telemetry.update_collision_status(agent_id, collision)

        # Render (this will display all extensions)
        env.render()

        # Check if episode is done
        done = all(terminations.values()) or all(truncations.values())

        # Print progress every 100 steps
        if step % 100 == 0:
            print(f"Step {step}: Running...")

    print(f"Episode completed in {step} steps")
    print()
    print("Demonstration complete!")
    print("Close the window to exit.")

    # Keep window open
    try:
        import pyglet
        pyglet.app.run()
    except KeyboardInterrupt:
        pass
    finally:
        env.close()

if __name__ == '__main__':
    main()
