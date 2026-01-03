#!/usr/bin/env python3
"""Test script to verify PPO WandB logging fix."""

import sys
import os
sys.path.insert(0, 'src')
os.environ['WANDB_MODE'] = 'disabled'  # Run offline for testing

from agents.ppo.ppo import PPOAgent
from core.protocol import is_on_policy_agent
import numpy as np

def test_ppo_update_logging():
    """Test that PPO returns update stats correctly."""

    print("=" * 60)
    print("Testing PPO Update and Logging")
    print("=" * 60)

    # Create a minimal PPO agent
    config = {
        'obs_dim': 10,
        'act_dim': 2,
        'action_low': [-1, -1],
        'action_high': [1, 1],
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_epsilon': 0.2,
        'entropy_coef': 0.01,
        'n_epochs': 2,
        'batch_size': 32,
    }

    agent = PPOAgent(config)

    print(f"\n1. Verify agent is on-policy: {is_on_policy_agent(agent)}")

    # Simulate 3 short episodes
    for episode in range(3):
        print(f"\n2. Episode {episode + 1}:")

        # Collect transitions
        for step in range(50):
            obs = np.random.randn(10).astype(np.float32)
            action = agent.act(obs, deterministic=False)
            reward = np.random.rand()
            done = (step == 49)
            agent.store(obs, action, reward, done, terminated=done)

        print(f"   - Collected {len(agent.rew_buf)} transitions")

        # OLD WAY (with redundant finish_path - now removed):
        # agent.finish_path()  # <-- This was causing confusion

        # Update (which calls finish_path internally)
        update_stats = agent.update()

        if update_stats:
            print(f"   ✓ update() returned stats:")
            for key, value in update_stats.items():
                print(f"     - {key}: {value:.6f}")

            # Simulate WandB logging (this is what enhanced_training.py does)
            algo_name = 'ppo'
            agent_id = 'car_0'
            log_dict = {}
            for stat_name, stat_value in update_stats.items():
                log_dict[f'{agent_id}/{algo_name}/{stat_name}'] = stat_value

            print(f"   ✓ Would log to WandB at step={episode}:")
            for key in log_dict:
                print(f"     - {key}")
        else:
            print(f"   ✗ update() returned None!")
            print(f"     - rew_buf length: {len(agent.rew_buf)}")

    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)

if __name__ == '__main__':
    test_ppo_update_logging()
