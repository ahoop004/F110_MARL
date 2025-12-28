"""Simple TD3 training example using v2 architecture.

This demonstrates off-policy training with the v2 system.

Usage:
    python v2/examples/train_td3_simple.py
"""
import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core import (
    AgentFactory,
    EnvironmentFactory,
    TrainingLoop,
    save_checkpoint,
    SimpleLogger,
    set_random_seeds,
)


def main():
    """Run simple TD3 training."""
    print("="*60)
    print("v2 Simple TD3 Training Example")
    print("="*60)

    # Set random seeds
    set_random_seeds(42)

    # 1. Create Environment
    print("\n[1/5] Creating environment...")
    env_config = {
        'map': 'maps/line_map.yaml',
        'num_agents': 1,
        'timestep': 0.01,
        'integrator': 'rk4',
    }
    env = EnvironmentFactory.create(env_config)
    print(f"✓ Created environment")

    # 2. Create Agent
    print("\n[2/5] Creating TD3 agent...")
    agent_config = {
        'obs_dim': 370,
        'act_dim': 2,
        'action_low': np.array([-1.0, -1.0]),  # TD3 needs action bounds as arrays
        'action_high': np.array([1.0, 1.0]),
        'device': 'cpu',
        'lr': 3e-4,
        'gamma': 0.99,
        'tau': 0.005,
        'policy_delay': 2,
        'exploration_noise': 0.1,
        'target_noise': 0.2,
        'noise_clip': 0.5,
        'buffer_size': 100000,
        'warmup_steps': 1000,
        'batch_size': 256,
    }
    agent = AgentFactory.create('td3', agent_config)
    agents = {'agent_0': agent}
    print(f"✓ Created TD3 agent with replay buffer (size={agent_config['buffer_size']})")

    # 3. Create Logger
    print("\n[3/5] Setting up logging...")
    logger = SimpleLogger(log_dir='logs/td3_simple', verbose=True)
    print("✓ Logger initialized")

    # 4. Create Training Loop
    print("\n[4/5] Creating training loop...")
    max_episodes = 100  # More episodes for off-policy

    training_loop = TrainingLoop(
        env=env,
        agents=agents,
        max_episodes=max_episodes,
        max_steps_per_episode=500,
        update_frequency=1,  # Update every step for off-policy
        log_callback=lambda ep, stats: logger.log(ep, stats.get('agent_0', {})),
        checkpoint_callback=lambda ep, agents: (
            save_checkpoint(agents, ep, 'checkpoints/td3_simple', prefix='td3')
            if ep % 20 == 0 and ep > 0 else None
        )
    )
    print(f"✓ Training loop created ({max_episodes} episodes)")

    # 5. Train!
    print("\n[5/5] Starting training...")
    print("="*60)
    print("Note: TD3 needs warmup period to fill replay buffer")
    print(f"Warmup steps: {agent_config['warmup_steps']}")
    print("="*60)

    try:
        history = training_loop.run()

        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)

        # Print summary
        summary = logger.get_summary()
        print("\nTraining Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value:.4f}")

        print(f"\n✓ Logs saved to: logs/td3_simple/")
        print(f"✓ Checkpoints saved to: checkpoints/td3_simple/")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Saving final checkpoint...")
        save_checkpoint(agents, training_loop.max_episodes, 'checkpoints/td3_simple', prefix='td3_interrupted')
        print("✓ Final checkpoint saved")


if __name__ == '__main__':
    main()
