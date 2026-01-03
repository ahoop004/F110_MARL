#!/usr/bin/env python3
"""Train SB3 baseline agents for F110 gaplock task.

This script provides baselines using Stable-Baselines3 (SB3) implementations
of various RL algorithms. These are well-tested, proven algorithms that should
converge reliably.

Supported algorithms:
    - Off-policy (continuous): sac, td3, ddpg, tqc
    - Off-policy (discrete): dqn, qrdqn
    - On-policy: ppo, a2c

Usage:
    # Train SAC baseline
    python run_sb3_baseline.py --algo sac --scenario scenarios/v2/gaplock_sb3_sac.yaml

    # Train TD3 baseline
    python run_sb3_baseline.py --algo td3 --scenario scenarios/v2/gaplock_sb3_td3.yaml

    # Train PPO baseline
    python run_sb3_baseline.py --algo ppo --scenario scenarios/v2/gaplock_sb3_ppo.yaml

    # Train A2C baseline
    python run_sb3_baseline.py --algo a2c --scenario scenarios/v2/gaplock_sb3_a2c.yaml

    # With wandb logging
    python run_sb3_baseline.py --algo sac --scenario scenarios/v2/gaplock_sb3_sac.yaml --wandb
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC, TD3, PPO, DDPG, A2C, DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

# Optional sb3-contrib algorithms
try:
    from sb3_contrib import TQC, QRDQN
    CONTRIB_AVAILABLE = True
except ImportError:
    CONTRIB_AVAILABLE = False
    TQC = None
    QRDQN = None

# Optional wandb integration
try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")

from src.core.scenario import load_and_expand_scenario
from src.core.setup import create_training_setup
from src.baselines.sb3_wrapper import SB3SingleAgentWrapper


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train SB3 baseline for F110 gaplock'
    )
    parser.add_argument(
        '--algo',
        type=str,
        required=True,
        choices=['sac', 'td3', 'ppo', 'a2c', 'ddpg', 'tqc', 'dqn', 'qrdqn'],
        help='RL algorithm to use'
    )
    parser.add_argument(
        '--scenario',
        type=str,
        required=True,
        help='Path to scenario YAML file'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=2500,
        help='Number of episodes to train (default: 2500)'
    )
    parser.add_argument(
        '--wandb',
        action='store_true',
        help='Enable wandb logging'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./sb3_models',
        help='Directory to save models (default: ./sb3_models)'
    )
    return parser.parse_args()


def create_sb3_agent(algo_name: str, env, seed: int = 42):
    """Create SB3 agent with good default hyperparameters.

    Args:
        algo_name: Algorithm name ('sac', 'td3', 'ppo', 'a2c', 'ddpg', 'tqc', 'dqn', 'qrdqn')
        env: Gym environment
        seed: Random seed

    Returns:
        SB3 model
    """
    common_kwargs = {
        'policy': 'MlpPolicy',
        'env': env,
        'verbose': 1,
        'seed': seed,
    }

    if algo_name == 'sac':
        return SAC(
            **common_kwargs,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            batch_size=256,
            tau=0.005,
            gamma=0.995,
            ent_coef='auto',  # Automatic entropy tuning
            target_entropy='auto',
            policy_kwargs=dict(net_arch=[256, 256]),
        )
    elif algo_name == 'td3':
        return TD3(
            **common_kwargs,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            batch_size=256,
            tau=0.005,
            gamma=0.995,
            policy_delay=2,
            target_policy_noise=0.2,
            target_noise_clip=0.5,
            policy_kwargs=dict(net_arch=[256, 256]),
        )
    elif algo_name == 'ppo':
        return PPO(
            **common_kwargs,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.02,
            policy_kwargs=dict(net_arch=[256, 256]),
        )
    elif algo_name == 'a2c':
        return A2C(
            **common_kwargs,
            learning_rate=3e-4,
            n_steps=5,
            gamma=0.995,
            gae_lambda=1.0,
            ent_coef=0.0,
            vf_coef=0.5,
            policy_kwargs=dict(net_arch=[256, 256]),
        )
    elif algo_name == 'ddpg':
        from stable_baselines3.common.noise import NormalActionNoise
        n_actions = env.action_space.shape[0]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.1 * np.ones(n_actions)
        )
        return DDPG(
            **common_kwargs,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            batch_size=256,
            tau=0.005,
            gamma=0.995,
            action_noise=action_noise,
            policy_kwargs=dict(net_arch=[256, 256]),
        )
    elif algo_name == 'tqc':
        if not CONTRIB_AVAILABLE or TQC is None:
            raise ImportError("TQC requires sb3-contrib. Install with: pip install sb3-contrib")
        return TQC(
            **common_kwargs,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            batch_size=256,
            tau=0.005,
            gamma=0.995,
            ent_coef='auto',
            target_entropy='auto',
            top_quantiles_to_drop_per_net=2,
            policy_kwargs=dict(net_arch=[256, 256]),
        )
    elif algo_name == 'dqn':
        return DQN(
            **common_kwargs,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            batch_size=256,
            tau=0.005,
            gamma=0.995,
            exploration_fraction=0.1,
            exploration_final_eps=0.05,
            exploration_initial_eps=1.0,
            policy_kwargs=dict(net_arch=[256, 256]),
        )
    elif algo_name == 'qrdqn':
        if not CONTRIB_AVAILABLE or QRDQN is None:
            raise ImportError("QR-DQN requires sb3-contrib. Install with: pip install sb3-contrib")
        return QRDQN(
            **common_kwargs,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            batch_size=256,
            tau=0.005,
            gamma=0.995,
            exploration_fraction=0.1,
            exploration_final_eps=0.05,
            exploration_initial_eps=1.0,
            policy_kwargs=dict(net_arch=[256, 256]),
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")


def main():
    args = parse_args()

    # Load scenario
    print(f"Loading scenario: {args.scenario}")
    scenario = load_and_expand_scenario(args.scenario)

    # Override episodes if specified
    if args.episodes:
        scenario['experiment']['episodes'] = args.episodes

    # Create environment and agents (including FTG defender)
    print("Creating environment...")
    env, agents, reward_strategies = create_training_setup(scenario)

    # Wrap environment for SB3
    print("Wrapping environment for SB3...")
    wrapped_env = SB3SingleAgentWrapper(
        env,
        agent_id='car_0',  # Attacker
        obs_dim=126,  # Gaplock observation dimension
    )

    # Set other agents (FTG defender) so they can act
    wrapped_env.set_other_agents(agents)

    # Wrap with Monitor for logging
    wrapped_env = Monitor(wrapped_env)

    # Initialize wandb if requested
    wandb_run = None
    if args.wandb:
        if not WANDB_AVAILABLE:
            print("Warning: wandb requested but not available. Skipping.")
        else:
            wandb_run = wandb.init(
                project=scenario.get('wandb', {}).get('project', 'marl-f110'),
                entity=scenario.get('wandb', {}).get('entity'),
                tags=['sb3', args.algo, 'baseline'],
                config={
                    'algorithm': args.algo,
                    'scenario': args.scenario,
                    'episodes': args.episodes,
                    'seed': args.seed,
                },
                sync_tensorboard=True,
            )

    # Create output directory
    output_dir = Path(args.output_dir) / args.algo / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create callbacks
    callbacks = []

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=str(output_dir / 'checkpoints'),
        name_prefix=f'{args.algo}_model',
        save_replay_buffer=True,
    )
    callbacks.append(checkpoint_callback)

    # Wandb callback
    if wandb_run is not None:
        wandb_callback = WandbCallback(
            model_save_freq=10000,
            model_save_path=str(output_dir / 'models'),
            verbose=2,
        )
        callbacks.append(wandb_callback)

    # Create SB3 agent
    print(f"Creating {args.algo.upper()} agent...")
    model = create_sb3_agent(args.algo, wrapped_env, seed=args.seed)

    # Calculate total timesteps
    # Approximate: episodes * avg_steps_per_episode
    max_steps = scenario['environment'].get('max_steps', 2500)
    total_timesteps = args.episodes * max_steps

    print(f"\nStarting training:")
    print(f"  Algorithm: {args.algo.upper()}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Output directory: {output_dir}")
    print()

    # Train
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=10,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    finally:
        # Save final model
        final_model_path = output_dir / f'{args.algo}_final.zip'
        model.save(final_model_path)
        print(f"\nSaved final model to: {final_model_path}")

        # Close wandb
        if wandb_run is not None:
            wandb.finish()

        # Close environment
        wrapped_env.close()

    print("\nTraining complete!")


if __name__ == '__main__':
    main()
