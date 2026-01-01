#!/usr/bin/env python3
"""Wandb sweep runner for F110 MARL experiments.

This script bridges wandb sweep parameter injection with the scenario-based
configuration system. It loads a base scenario YAML, overrides parameters
from wandb.config, and runs training.

Usage:
    wandb sweep sweeps/gaplock_td3_v2_sweep.yaml
    wandb agent <sweep-id>
"""

import argparse
import sys
from pathlib import Path
import yaml
import wandb

from src.core.scenario import load_and_expand_scenario
from src.loggers import WandbLogger, ConsoleLogger, CSVLogger, RichConsole


def set_nested_value(config: dict, path: str, value):
    """Set a nested dictionary value using dot notation path.

    Args:
        config: Configuration dictionary to modify
        path: Dot-separated path (e.g., 'agents.car_0.params.lr_actor')
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
    """Main entry point for sweep runner."""
    parser = argparse.ArgumentParser(description='Wandb sweep runner for F110 MARL')
    parser.add_argument(
        '--scenario',
        type=str,
        required=True,
        help='Path to base scenario YAML file'
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

    # Import and run training
    from src.core.enhanced_training import EnhancedTrainingLoop
    from src.core.setup import create_training_setup
    from src.core.run_id import resolve_run_id, set_run_id_env, get_checkpoint_dir, get_output_dir
    from src.core.run_metadata import RunMetadata
    from src.core.checkpoint_manager import CheckpointManager
    from src.core.best_model_tracker import BestModelTracker

    console_logger = ConsoleLogger(verbose=False)

    # Get algorithm for run ID
    algorithm = 'unknown'
    for agent_id, agent_config in scenario['agents'].items():
        algo = agent_config.get('algorithm', '').lower()
        if algo not in ['ftg', 'pp', 'pure_pursuit']:
            algorithm = algo
            break

    # Use wandb run ID for alignment
    run_id = wandb.run.id
    set_run_id_env(run_id)

    # Create run name with algorithm prefix
    base_name = scenario['wandb'].get('name', scenario['experiment']['name'])
    run_name = f"{algorithm}_{wandb.run.name}" if algorithm != 'unknown' else wandb.run.name

    # Create wandb logger (already initialized)
    wandb_logger = WandbLogger(
        project=scenario['wandb'].get('project', 'marl-f110'),
        name=run_name,
        config=scenario,
        tags=scenario['wandb'].get('tags', []),
        group=scenario['wandb'].get('group', None),
        entity=scenario['wandb'].get('entity', None),
        notes=scenario['wandb'].get('notes', None),
        mode='online',
        run_id=run_id,
    )

    # Create training setup
    try:
        env, agents, reward_strategies = create_training_setup(scenario)
    except Exception as e:
        console_logger.print_error(f"Failed to create training setup: {e}")
        wandb_logger.finish()
        sys.exit(1)

    # Extract observation presets and target_ids
    observation_presets = {}
    target_ids = {}
    ftg_schedules = {}

    for agent_id, agent_config in scenario['agents'].items():
        if 'observation' in agent_config:
            obs_config = agent_config['observation']
            if 'preset' in obs_config:
                observation_presets[agent_id] = obs_config['preset']
            elif isinstance(obs_config, dict) and len(obs_config) > 0:
                observation_presets[agent_id] = 'gaplock'

        if 'target_id' in agent_config:
            target_ids[agent_id] = agent_config['target_id']

        if agent_config.get('algorithm', '').lower() == 'ftg':
            schedule = agent_config.get('ftg_schedule')
            if isinstance(schedule, dict):
                ftg_schedules[agent_id] = schedule

    # Infer target_ids for 2-agent scenarios
    if len(scenario['agents']) == 2:
        attacker_id = None
        defender_id = None
        for agent_id, agent_config in scenario['agents'].items():
            role = agent_config.get('role', '').lower()
            if role == 'attacker':
                attacker_id = agent_id
            elif role == 'defender':
                defender_id = agent_id

        if attacker_id and defender_id and attacker_id not in target_ids:
            target_ids[attacker_id] = defender_id

    # Create spawn curriculum
    spawn_curriculum = None
    spawn_config = scenario['environment'].get('spawn_curriculum', {})
    if spawn_config.get('enabled', False):
        from src.core.spawn_curriculum import SpawnCurriculumManager
        spawn_configs = spawn_config.get('spawn_configs', {})

        if spawn_configs:
            try:
                spawn_curriculum = SpawnCurriculumManager(
                    config=spawn_config,
                    available_spawn_points=spawn_configs
                )
            except Exception as e:
                console_logger.print_warning(f"Failed to create spawn curriculum: {e}")

    # Initialize checkpoint system (disabled for sweeps)
    checkpoint_manager = None
    best_model_tracker = None
    csv_logger = None

    # Initialize Rich console
    rich_console = RichConsole(refresh_rate=4.0, enabled=True)

    # Extract agent algorithms
    agent_algorithms = {}
    for agent_id, agent_config in scenario['agents'].items():
        algo = agent_config.get('algorithm', '').lower()
        if algo:
            agent_algorithms[agent_id] = algo

    # Create training loop
    training_loop = EnhancedTrainingLoop(
        env=env,
        agents=agents,
        agent_rewards=reward_strategies,
        observation_presets=observation_presets,
        target_ids=target_ids,
        agent_algorithms=agent_algorithms,
        spawn_curriculum=spawn_curriculum,
        ftg_schedules=ftg_schedules,
        wandb_logger=wandb_logger,
        console_logger=console_logger,
        csv_logger=csv_logger,
        rich_console=rich_console,
        checkpoint_manager=checkpoint_manager,
        best_model_tracker=best_model_tracker,
        max_steps_per_episode=scenario['environment'].get('max_steps', 5000),
        save_every_n_episodes=None,
        normalize_observations=bool(
            scenario.get('experiment', {}).get('normalize_observations', True)
        ),
        obs_clip=float(scenario.get('experiment', {}).get('obs_clip', 10.0)),
    )

    # Run training
    try:
        episodes = scenario['experiment'].get('episodes', 1500)
        training_loop.run(episodes=episodes, start_episode=0)
    except KeyboardInterrupt:
        console_logger.print_warning("\nTraining interrupted")
    except Exception as e:
        console_logger.print_error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        wandb_logger.finish()
        sys.exit(1)

    # Cleanup
    env.close()
    wandb_logger.finish()


if __name__ == '__main__':
    main()
