#!/usr/bin/env python3
"""Main entry point for v2 training pipeline.

Usage:
    python v2/run.py --scenario scenarios/v2/gaplock_ppo.yaml
    python v2/run.py --scenario scenarios/v2/gaplock_ppo.yaml --wandb
    python v2/run.py --scenario scenarios/v2/gaplock_ppo.yaml --no-render

Example scenario file:
    experiment:
      name: gaplock_ppo
      episodes: 1500
      seed: 42

    environment:
      map: maps/line2.yaml
      num_agents: 2
      max_steps: 5000

    agents:
      car_0:
        role: attacker
        algorithm: ppo
        observation:
          preset: gaplock
        reward:
          preset: gaplock_full

      car_1:
        role: defender
        algorithm: ftg

    wandb:
      enabled: true
      project: f110-gaplock
      tags: [ppo, baseline]
"""

import argparse
import sys
from pathlib import Path

from v2.core.scenario import load_and_expand_scenario, ScenarioError
from v2.loggers import WandbLogger, ConsoleLogger


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='F110 Multi-Agent RL Training (v2)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        '--scenario',
        type=str,
        required=True,
        help='Path to scenario YAML file',
    )

    parser.add_argument(
        '--wandb',
        action='store_true',
        help='Enable Weights & Biases logging (overrides scenario config)',
    )

    parser.add_argument(
        '--no-wandb',
        action='store_true',
        help='Disable Weights & Biases logging (overrides scenario config)',
    )

    parser.add_argument(
        '--render',
        action='store_true',
        help='Enable rendering (default: False)',
    )

    parser.add_argument(
        '--no-render',
        action='store_true',
        help='Disable rendering',
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose console output (default: True)',
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Disable console output',
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed (overrides scenario config)',
    )

    parser.add_argument(
        '--episodes',
        type=int,
        default=None,
        help='Number of episodes (overrides scenario config)',
    )

    return parser.parse_args()


def resolve_cli_overrides(scenario: dict, args) -> dict:
    """Apply CLI argument overrides to scenario.

    Args:
        scenario: Loaded scenario configuration
        args: Parsed command-line arguments

    Returns:
        Scenario with CLI overrides applied
    """
    # Override seed
    if args.seed is not None:
        scenario['experiment']['seed'] = args.seed

    # Override episodes
    if args.episodes is not None:
        scenario['experiment']['episodes'] = args.episodes

    # Override W&B setting
    if args.wandb:
        if 'wandb' not in scenario:
            scenario['wandb'] = {}
        scenario['wandb']['enabled'] = True
    elif args.no_wandb:
        if 'wandb' not in scenario:
            scenario['wandb'] = {}
        scenario['wandb']['enabled'] = False

    # Override rendering
    if args.render:
        if 'environment' not in scenario:
            scenario['environment'] = {}
        scenario['environment']['render'] = True
    elif args.no_render:
        if 'environment' not in scenario:
            scenario['environment'] = {}
        scenario['environment']['render'] = False

    return scenario


def initialize_loggers(scenario: dict, args) -> tuple:
    """Initialize W&B and console loggers.

    Args:
        scenario: Scenario configuration
        args: Command-line arguments

    Returns:
        Tuple of (wandb_logger, console_logger)
    """
    # Initialize W&B logger
    wandb_config = scenario.get('wandb', {})
    wandb_enabled = wandb_config.get('enabled', False)

    if wandb_enabled:
        console_logger = ConsoleLogger(verbose=not args.quiet)
        console_logger.print_info("Initializing Weights & Biases...")

        wandb_logger = WandbLogger(
            project=wandb_config.get('project', 'f110-marl'),
            name=wandb_config.get('name', scenario['experiment']['name']),
            config=scenario,
            tags=wandb_config.get('tags', []),
            group=wandb_config.get('group', None),
            notes=wandb_config.get('notes', None),
            mode=wandb_config.get('mode', 'online'),
        )
    else:
        wandb_logger = None

    # Initialize console logger
    console_logger = ConsoleLogger(verbose=not args.quiet)

    return wandb_logger, console_logger


def print_scenario_summary(scenario: dict, console: ConsoleLogger):
    """Print scenario summary to console.

    Args:
        scenario: Scenario configuration
        console: Console logger
    """
    experiment = scenario['experiment']
    environment = scenario['environment']
    agents = scenario['agents']

    console.print_header(
        f"Training: {experiment['name']}",
        f"Episodes: {experiment.get('episodes', 'N/A')} | "
        f"Map: {Path(environment['map']).name}"
    )

    # Print agent summary
    console.console.print("\n[bold cyan]Agents:[/bold cyan]")
    for agent_id, config in agents.items():
        role = config.get('role', 'N/A')
        algorithm = config.get('algorithm', 'N/A')
        console.console.print(f"  • {agent_id}: {algorithm} ({role})")

    console.console.print()


def main():
    """Main entry point."""
    args = parse_args()

    # Load and expand scenario
    try:
        scenario = load_and_expand_scenario(args.scenario)
    except ScenarioError as e:
        print(f"Error loading scenario: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"Scenario file not found: {args.scenario}", file=sys.stderr)
        sys.exit(1)

    # Apply CLI overrides
    scenario = resolve_cli_overrides(scenario, args)

    # Initialize loggers
    wandb_logger, console_logger = initialize_loggers(scenario, args)

    # Print scenario summary
    print_scenario_summary(scenario, console_logger)

    # Import training components
    try:
        from v2.core.enhanced_training import EnhancedTrainingLoop
        from v2.core.setup import create_training_setup
        from v2.render import MinimalHUD, RewardRingExtension
    except ImportError as e:
        console_logger.print_error(f"Failed to import training components: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Create training setup (environment, agents, reward strategies)
    try:
        console_logger.print_info("Creating training setup...")
        env, agents, reward_strategies = create_training_setup(scenario)
        console_logger.print_success(f"Created {len(agents)} agents")
    except Exception as e:
        console_logger.print_error(f"Failed to create training setup: {e}")
        import traceback
        traceback.print_exc()
        if wandb_logger:
            wandb_logger.finish()
        sys.exit(1)

    # Add rendering extensions if rendering is enabled
    if env.renderer is not None:
        console_logger.print_info("Adding rendering extensions...")

        # Add minimal HUD
        hud = MinimalHUD(env.renderer)
        env.renderer.add_extension(hud)
        hud.configure(enabled=True)

        # Add reward ring if there's a defender (attacker vs defender scenario)
        defender_id = None
        for agent_id, config in scenario['agents'].items():
            if config.get('role') == 'defender':
                defender_id = agent_id
                break

        if defender_id:
            ring = RewardRingExtension(env.renderer)
            env.renderer.add_extension(ring)
            ring.configure(
                enabled=True,
                target_agent=defender_id,
                inner_radius=1.0,
                outer_radius=3.0,
                preferred_radius=2.0,
            )
            console_logger.print_info(f"Reward ring tracking: {defender_id}")

    # Create and run training loop
    try:
        console_logger.print_info("Starting training...")

        # Extract observation presets from scenario
        observation_presets = {}
        target_ids = {}

        # First pass: collect observation presets and explicit target_ids
        for agent_id, agent_config in scenario['agents'].items():
            # Get observation preset if specified
            if 'observation' in agent_config:
                obs_config = agent_config['observation']
                if 'preset' in obs_config:
                    # Explicit preset specified
                    observation_presets[agent_id] = obs_config['preset']
                elif isinstance(obs_config, dict) and len(obs_config) > 0:
                    # Observation config exists but preset was expanded
                    # Default to 'gaplock' for now (only preset we support)
                    observation_presets[agent_id] = 'gaplock'
                    console_logger.print_info(f"Inferred observation preset 'gaplock' for {agent_id}")

            # Get target_id if specified (for adversarial tasks)
            if 'target_id' in agent_config:
                target_ids[agent_id] = agent_config['target_id']

        # Second pass: infer target_ids for adversarial tasks based on roles
        # For attacker-defender scenarios, attacker targets defender
        if len(scenario['agents']) == 2:
            attacker_id = None
            defender_id = None
            for agent_id, agent_config in scenario['agents'].items():
                role = agent_config.get('role', '').lower()
                if role == 'attacker':
                    attacker_id = agent_id
                elif role == 'defender':
                    defender_id = agent_id

            # If attacker doesn't have explicit target_id, set it to defender
            if attacker_id and defender_id and attacker_id not in target_ids:
                target_ids[attacker_id] = defender_id

        training_loop = EnhancedTrainingLoop(
            env=env,
            agents=agents,
            agent_rewards=reward_strategies,
            observation_presets=observation_presets,
            target_ids=target_ids,
            wandb_logger=wandb_logger,
            console_logger=console_logger,
            max_steps_per_episode=scenario['environment'].get('max_steps', 5000),
        )

        episodes = scenario['experiment'].get('episodes', 100)
        training_loop.run(episodes=episodes)

    except KeyboardInterrupt:
        console_logger.print_warning("\nTraining interrupted by user")
    except Exception as e:
        console_logger.print_error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        if wandb_logger:
            wandb_logger.finish()
        sys.exit(1)

    # Cleanup
    console_logger.print_info("Cleaning up...")
    env.close()

    # Finish W&B logging
    if wandb_logger:
        console_logger.print_info("Finishing W&B run...")
        wandb_logger.finish()

    console_logger.print_success("Training complete!")


if __name__ == '__main__':
    main()
