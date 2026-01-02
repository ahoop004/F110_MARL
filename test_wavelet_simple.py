#!/usr/bin/env python3
"""Minimal test: 5 episodes with WaveletEpisodicAgent, no wandb."""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from env import F110ParallelEnv
from agents.episodic import WaveletEpisodicAgent
from loggers import ConsoleLogger
from core.enhanced_training import EnhancedTrainingLoop


def main():
    print("Testing WaveletEpisodicAgent - 5 episodes, no wandb\n")

    # 1. Create environment
    spawn_poses = np.array([[0.0, 0.0, 0.0]])

    env = F110ParallelEnv(
        map="maps/line2/line2",
        num_agents=1,
        timestep=0.01,
        integrator="rk4",
        model="st",
        observation_config="original",
        start_poses=spawn_poses,  # Add start poses to config
    )

    # 2. Create agent with minimal config
    agent_config = {
        'obs_dim': 107,  # F110 obs dimension
        'act_dim': 2,
        'chunk_size': 25,
        'chunk_stride': 12,
        'episodic_buffer': {'capacity': 100, 'selection_mode': 'uniform'},
        'wavelet': {'type': 'haar'},  # Try 'identity' to disable wavelets
        'encoder': {
            'latent_dim': 64,
            'cnn': {'channels': [16, 32, 64]},
            'rnn': {'use': True, 'type': 'lstm', 'hidden_size': 128, 'num_layers': 1},
        },
        'warmup_chunks': 10,
        'update_freq': 4,
        'batch_size': 16,
        'device': 'cpu',  # Use 'cuda' if you have GPU
    }

    agent = WaveletEpisodicAgent(agent_config)

    agents = {'car_0': agent}

    # 3. Create console logger (no wandb!)
    console_logger = ConsoleLogger(
        log_level='INFO',
        metrics_every_n_episodes=1,
    )

    # 4. Create training loop
    loop = EnhancedTrainingLoop(
        env=env,
        agents=agents,
        wandb_logger=None,  # ← No wandb!
        console_logger=console_logger,
        csv_logger=None,
        rich_console=None,
        max_steps_per_episode=500,
    )

    # 5. Run 5 episodes
    print("Running 5 test episodes...\n")
    loop.run(episodes=5)

    print("\n✓ Test complete! Agent is working.")
    print(f"  - Chronological buffer: {len(agent.chronological_buffer)} transitions")
    print(f"  - Episodic buffer: {len(agent.episodic_buffer)} chunks")
    print(f"  - Updates performed: {agent.update_count}")


if __name__ == "__main__":
    main()
