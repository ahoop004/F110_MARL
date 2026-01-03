#!/usr/bin/env python3
"""Wandb sweep runner for SB3 baseline experiments.

This script bridges wandb sweep parameter injection with the SB3 baseline
training script. It loads a base scenario YAML, overrides parameters
from wandb.config, and runs SB3 training.

Usage:
    wandb sweep sweeps/gaplock_sb3_sac_sweep.yaml
    wandb agent <sweep-id>
"""

import argparse
import sys
from pathlib import Path
import wandb

from src.core.scenario import load_and_expand_scenario


def set_nested_value(config: dict, path: str, value):
    """Set a nested dictionary value using dot notation path.

    Args:
        config: Configuration dictionary to modify
        path: Dot-separated path (e.g., 'agents.car_0.params.learning_rate')
        value: Value to set
    """
    keys = path.split('.')
    current = config

    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]

    current[keys[-1]] = value


def apply_sweep_params(scenario: dict, sweep_params: dict) -> dict:
    """Apply sweep parameters to scenario configuration.

    Args:
        scenario: Base scenario configuration
        sweep_params: Wandb sweep parameters (flat dict with dot-notation keys)

    Returns:
        Modified scenario with sweep parameters applied
    """
    for param_path, value in sweep_params.items():
        if param_path == 'episodes':
            scenario['experiment']['episodes'] = value
        elif param_path == 'seed':
            scenario['experiment']['seed'] = value
        else:
            set_nested_value(scenario, param_path, value)

    return scenario


def main():
    """Main entry point for SB3 sweep runner."""
    parser = argparse.ArgumentParser(description='Wandb sweep runner for SB3 baselines')
    parser.add_argument(
        '--scenario',
        type=str,
        required=True,
        help='Path to base scenario YAML file'
    )
    parser.add_argument(
        '--algo',
        type=str,
        required=True,
        choices=['sac', 'td3', 'ppo', 'a2c', 'ddpg', 'tqc', 'dqn', 'qrdqn'],
        help='RL algorithm to use'
    )
    args = parser.parse_args()

    # Initialize wandb (sweep parameters will be in wandb.config)
    run = wandb.init()

    # Load base scenario
    try:
        scenario = load_and_expand_scenario(args.scenario)
    except Exception as e:
        print(f"Error loading scenario: {e}", file=sys.stderr)
        sys.exit(1)

    # Apply sweep parameters from wandb.config
    sweep_params = dict(wandb.config)
    scenario = apply_sweep_params(scenario, sweep_params)

    # Force wandb enabled
    if 'wandb' not in scenario:
        scenario['wandb'] = {}
    scenario['wandb']['enabled'] = True

    # Import run_sb3_baseline main and modify it for sweep
    from run_sb3_baseline import (
        create_training_setup,
        resolve_sb3_agent_config,
        resolve_observation_preset,
        SB3SingleAgentWrapper,
        create_sb3_agent,
        Monitor,
        CheckpointCallback,
        WandbCallback,
        WANDB_AVAILABLE,
    )
    from src.core.obs_flatten import flatten_observation
    from gymnasium import spaces
    import numpy as np

    # Get seed from scenario (may have been overridden by sweep)
    seed = scenario['experiment'].get('seed', 42)
    episodes = scenario['experiment'].get('episodes', 2500)

    # Create environment and agents
    print("Creating environment...")
    env, agents, reward_strategies = create_training_setup(scenario)

    sb3_agent_id, sb3_agent_cfg = resolve_sb3_agent_config(scenario, args.algo)
    observation_preset = resolve_observation_preset(sb3_agent_cfg)
    target_id = sb3_agent_cfg.get('target_id')

    obs_space = env.observation_spaces.get(sb3_agent_id)
    if obs_space is None:
        raise ValueError(f"Agent '{sb3_agent_id}' not found in observation spaces")

    # Determine observation dimension
    def get_space_dim(space) -> int:
        if isinstance(space, spaces.Dict):
            return sum(get_space_dim(s) for s in space.spaces.values())
        if isinstance(space, spaces.Box):
            return int(np.prod(space.shape))
        if isinstance(space, spaces.Discrete):
            return 1
        if isinstance(space, spaces.MultiDiscrete):
            return len(space.nvec)
        return 1

    if observation_preset:
        dummy_obs = obs_space.sample()
        if target_id:
            dummy_obs = dict(dummy_obs)
            dummy_obs['central_state'] = obs_space.sample()
        flat_dummy = flatten_observation(dummy_obs, preset=observation_preset, target_id=target_id)
        obs_dim = int(flat_dummy.shape[0])
    else:
        obs_dim = get_space_dim(obs_space)

    action_space = env.action_spaces.get(sb3_agent_id)
    if action_space is None:
        raise ValueError(f"Agent '{sb3_agent_id}' not found in action spaces")
    action_low = None
    action_high = None
    if isinstance(action_space, spaces.Box):
        action_low = action_space.low
        action_high = action_space.high

    # Wrap environment for SB3
    print("Wrapping environment for SB3...")
    wrapped_env = SB3SingleAgentWrapper(
        env,
        agent_id=sb3_agent_id,
        obs_dim=obs_dim,
        action_low=action_low if action_low is not None else np.array([-0.46, -1.0]),
        action_high=action_high if action_high is not None else np.array([0.46, 1.0]),
        observation_preset=observation_preset,
        target_id=target_id,
    )

    # Set other agents
    wrapped_env.set_other_agents(agents)

    # Wrap with Monitor for logging
    wrapped_env = Monitor(wrapped_env)

    # Create output directory
    output_dir = Path('./sb3_sweeps') / args.algo / f"run_{wandb.run.id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create tensorboard log directory (required for wandb sync)
    tensorboard_dir = output_dir / 'tensorboard'
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

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
    if WANDB_AVAILABLE:
        wandb_callback = WandbCallback(
            model_save_freq=10000,
            model_save_path=str(output_dir / 'models'),
            verbose=2,
        )
        callbacks.append(wandb_callback)

    # Extract hyperparameters from scenario
    agent_params = sb3_agent_cfg.get('params', {})

    # Create SB3 agent
    print(f"Creating {args.algo.upper()} agent...")
    model = create_sb3_agent(
        args.algo,
        wrapped_env,
        params=agent_params,
        seed=seed,
        tensorboard_log=str(tensorboard_dir)
    )

    # Calculate total timesteps
    max_steps = scenario['environment'].get('max_steps', 2500)
    total_timesteps = episodes * max_steps

    print(f"\nStarting training:")
    print(f"  Algorithm: {args.algo.upper()}")
    print(f"  Episodes: {episodes}")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Output directory: {output_dir}")
    print(f"  WandB run: {wandb.run.name}")
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
        wandb.finish()

        # Close environment
        wrapped_env.close()

    print("\nTraining complete!")


if __name__ == '__main__':
    main()
