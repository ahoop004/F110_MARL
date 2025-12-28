"""Simple PPO training example using v2 architecture.

This demonstrates the complete v2 system:
- Factory-based setup from config
- Protocol-compliant agents
- Simple training loop
- Checkpoint saving
- Metrics logging

Usage:
    python v2/examples/train_ppo_simple.py
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
    """Run simple PPO training."""
    print("="*60)
    print("v2 Simple PPO Training Example")
    print("="*60)

    # Set random seeds for reproducibility
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
    print(f"✓ Created environment with {env_config['num_agents']} agent(s)")

    # 2. Create Agent
    print("\n[2/5] Creating PPO agent...")
    agent_config = {
        'obs_dim': 370,  # F110 observation dimension
        'act_dim': 2,     # [steering, velocity]
        'device': 'cpu',
        'lr': 3e-4,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_ratio': 0.2,
        'n_epochs': 10,
        'batch_size': 64,
        'max_buffer_size': 2048,
        'squash_tanh': True,
    }
    agent = AgentFactory.create('ppo', agent_config)
    agents = {'agent_0': agent}
    print(f"✓ Created PPO agent")

    # 3. Create Logger
    print("\n[3/5] Setting up logging...")
    logger = SimpleLogger(log_dir='logs/ppo_simple', verbose=True)
    print("✓ Logger initialized")

    # 4. Create Training Loop
    print("\n[4/5] Creating training loop...")
    max_episodes = 50  # Short training for demo

    training_loop = TrainingLoop(
        env=env,
        agents=agents,
        max_episodes=max_episodes,
        max_steps_per_episode=500,
        update_frequency=1,  # Update every episode for on-policy
        log_callback=lambda ep, stats: logger.log(ep, stats.get('agent_0', {})),
        checkpoint_callback=lambda ep, agents: (
            save_checkpoint(agents, ep, 'checkpoints/ppo_simple', prefix='ppo')
            if ep % 10 == 0 and ep > 0 else None
        )
    )
    print(f"✓ Training loop created ({max_episodes} episodes)")

    # 5. Train!
    print("\n[5/5] Starting training...")
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

        print(f"\n✓ Logs saved to: logs/ppo_simple/")
        print(f"✓ Checkpoints saved to: checkpoints/ppo_simple/")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Saving final checkpoint...")
        save_checkpoint(agents, training_loop.max_episodes, 'checkpoints/ppo_simple', prefix='ppo_interrupted')
        print("✓ Final checkpoint saved")


if __name__ == '__main__':
    main()
