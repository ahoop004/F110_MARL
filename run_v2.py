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

from src.core.scenario import load_and_expand_scenario, ScenarioError
from src.loggers import WandbLogger, ConsoleLogger, CSVLogger, RichConsole


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

    # Checkpoint arguments
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Resume training from checkpoint (path or "latest")',
    )

    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default=None,
        help='Custom checkpoint directory (default: outputs/checkpoints/{scenario}/{run_id})',
    )

    parser.add_argument(
        '--save-every',
        type=int,
        default=100,
        help='Save checkpoint every N episodes (default: 100, 0 to disable periodic saves)',
    )

    parser.add_argument(
        '--no-checkpoints',
        action='store_true',
        help='Disable all checkpointing',
    )

    parser.add_argument(
        '--run-id',
        type=str,
        default=None,
        help='Custom run ID for checkpoint/logging alignment',
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


def initialize_loggers(scenario: dict, args, run_id: str = None) -> tuple:
    """Initialize W&B and console loggers.

    Args:
        scenario: Scenario configuration
        args: Command-line arguments
        run_id: Run identifier for checkpoint alignment (optional)

    Returns:
        Tuple of (wandb_logger, console_logger)
    """
    # Initialize console logger first
    console_logger = ConsoleLogger(verbose=not args.quiet)

    # Initialize W&B logger
    wandb_config = scenario.get('wandb', {})
    wandb_enabled = wandb_config.get('enabled', False)

    if wandb_enabled:
        console_logger.print_info("Initializing Weights & Biases...")

        wandb_logger = WandbLogger(
            project=wandb_config.get('project', 'f110-marl'),
            name=wandb_config.get('name', scenario['experiment']['name']),
            config=scenario,
            tags=wandb_config.get('tags', []),
            group=wandb_config.get('group', None),
            entity=wandb_config.get('entity', None),
            notes=wandb_config.get('notes', None),
            mode=wandb_config.get('mode', 'online'),
            run_id=run_id,  # Pass run_id for alignment
        )
    else:
        wandb_logger = None

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

    # Resolve run ID early (for W&B and checkpoint alignment)
    from src.core.run_id import resolve_run_id, set_run_id_env, get_checkpoint_dir
    from src.core.run_metadata import RunMetadata
    from src.core.checkpoint_manager import CheckpointManager
    from src.core.best_model_tracker import BestModelTracker

    # Get algorithm name for run ID
    algorithm = 'unknown'
    for agent_id, agent_config in scenario['agents'].items():
        algo = agent_config.get('algorithm', '').lower()
        if algo not in ['ftg', 'pp', 'pure_pursuit']:
            algorithm = algo
            break

    # Resolve run ID
    if args.run_id:
        run_id = args.run_id
    else:
        run_id = resolve_run_id(
            scenario_name=scenario['experiment']['name'],
            algorithm=algorithm,
            seed=scenario['experiment'].get('seed')
        )

    # Set run ID in environment for child processes
    set_run_id_env(run_id)

    # Initialize loggers with run ID
    wandb_logger, console_logger = initialize_loggers(scenario, args, run_id=run_id)

    # Print scenario summary
    print_scenario_summary(scenario, console_logger)

    # Print run ID
    console_logger.print_info(f"Run ID: {run_id}")

    # Import training components
    try:
        from src.core.enhanced_training import EnhancedTrainingLoop
        from src.core.setup import create_training_setup
        # Rendering extensions imported lazily when needed (to avoid pyglet on HPC)
        # from src.render import TelemetryHUD, RewardRingExtension, RewardHeatmap
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
    # Use render callbacks since renderer is created lazily on first render() call
    if scenario['environment'].get('render', False):
        console_logger.print_info("Configuring rendering extensions...")

        # Find defender and attacker agents
        defender_id = None
        attacker_id = None
        for agent_id, config in scenario['agents'].items():
            role = config.get('role', '').lower()
            if role == 'defender':
                defender_id = agent_id
            elif role == 'attacker':
                attacker_id = agent_id

        # Extract reward parameters from attacker agent config
        reward_params = {}
        if attacker_id and 'reward' in scenario['agents'][attacker_id]:
            from src.rewards.presets import load_preset, merge_config

            reward_config = scenario['agents'][attacker_id]['reward']

            # Load preset if specified
            if 'preset' in reward_config:
                preset_name = reward_config['preset']
                try:
                    base_config = load_preset(preset_name)
                    # Merge any overrides
                    overrides = {k: v for k, v in reward_config.items() if k != 'preset'}
                    if overrides:
                        full_config = merge_config(base_config, overrides)
                    else:
                        full_config = base_config

                    # Extract distance reward parameters
                    if 'distance' in full_config:
                        dist = full_config['distance']
                        reward_params = {
                            'near_distance': dist.get('near_distance', 1.0),
                            'far_distance': dist.get('far_distance', 2.5),
                            'reward_near': dist.get('reward_near', 0.12),
                            'penalty_far': dist.get('penalty_far', 0.08),
                        }
                        console_logger.print_info(f"Extracted reward params: near={reward_params['near_distance']:.2f}m, far={reward_params['far_distance']:.2f}m")
                except Exception as e:
                    console_logger.print_warning(f"Could not load reward preset: {e}")

        # Get visualization config from environment
        viz_config = scenario['environment'].get('visualization', {})

        # Track if extensions have been added (to avoid duplicates)
        extensions_added = [False]

        def setup_extensions(renderer):
            """Callback to add extensions when renderer is created."""
            if extensions_added[0]:
                return  # Already added
            extensions_added[0] = True

            # Lazy import rendering extensions (only when rendering is enabled)
            from src.render import TelemetryHUD, RewardRingExtension, RewardHeatmap

            # Add telemetry HUD
            telemetry = TelemetryHUD(renderer)
            telemetry.configure(enabled=True, mode=TelemetryHUD.MODE_BASIC)
            renderer.add_extension(telemetry)
            print("✓ Added TelemetryHUD (press T to cycle modes)")

            # Add reward ring if there's a defender
            if defender_id:
                ring_config = viz_config.get('reward_ring', {})
                ring = RewardRingExtension(renderer)
                ring.configure(
                    enabled=ring_config.get('enabled', True),
                    target_agent=defender_id,
                    inner_radius=reward_params.get('near_distance', ring_config.get('inner_radius', 1.0)),
                    outer_radius=reward_params.get('far_distance', ring_config.get('outer_radius', 2.5)),
                    preferred_radius=ring_config.get('preferred_radius', 1.5),
                )
                renderer.add_extension(ring)
                print(f"✓ Added RewardRing (target: {defender_id}, press R to toggle)")

            # Add reward heatmap (disabled by default, toggle with H)
            if defender_id and attacker_id:
                heatmap_config = viz_config.get('heatmap', {})
                heatmap = RewardHeatmap(renderer)

                # Get reward strategy for attacker to query actual rewards
                attacker_reward_strategy = reward_strategies.get(attacker_id)

                heatmap.configure(
                    enabled=heatmap_config.get('enabled', False),  # Start disabled
                    target_agent=defender_id,
                    attacker_agent=attacker_id,
                    reward_strategy=attacker_reward_strategy,
                    extent_m=heatmap_config.get('extent_m', 6.0),
                    cell_size_m=heatmap_config.get('cell_size_m', 0.25),
                    alpha=heatmap_config.get('alpha', 0.22),
                    update_frequency=heatmap_config.get('update_frequency', 5),
                )
                renderer.add_extension(heatmap)
                print(f"✓ Added RewardHeatmap (disabled, press H to enable)")
                if attacker_reward_strategy:
                    print(f"  Using actual reward strategy for spatial visualization")
                else:
                    print(f"  WARNING: No reward strategy found for {attacker_id}")

            print()
            print("Keyboard controls: T=telemetry | R=ring | H=heatmap | F=follow | 1-2=focus agent")
            print()

        # Register callback to add extensions when renderer is created
        env.add_render_callback(setup_extensions)
        console_logger.print_success("Rendering extensions configured (will activate on first render)")

    # Create and run training loop
    try:
        console_logger.print_info("Starting training...")

        # Extract observation presets from scenario
        observation_presets = {}
        target_ids = {}
        ftg_schedules = {}

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

            # Collect FTG schedules (optional)
            if agent_config.get('algorithm', '').lower() == 'ftg':
                schedule = agent_config.get('ftg_schedule')
                if isinstance(schedule, dict):
                    ftg_schedules[agent_id] = schedule

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

        # Create spawn curriculum if enabled
        spawn_curriculum = None
        spawn_config = scenario['environment'].get('spawn_curriculum', {})
        if spawn_config.get('enabled', False):
            from src.core.spawn_curriculum import SpawnCurriculumManager

            # Get spawn point configurations from environment
            spawn_configs = spawn_config.get('spawn_configs', {})

            if spawn_configs:
                console_logger.print_info("Creating spawn curriculum...")
                try:
                    spawn_curriculum = SpawnCurriculumManager(
                        config=spawn_config,
                        available_spawn_points=spawn_configs
                    )
                    console_logger.print_success(
                        f"Spawn curriculum: {len(spawn_curriculum.stages)} stages, "
                        f"starting at '{spawn_curriculum.current_stage.name}'"
                    )
                except Exception as e:
                    console_logger.print_warning(f"Failed to create spawn curriculum: {e}")
                    spawn_curriculum = None
            else:
                console_logger.print_warning("Spawn curriculum enabled but no spawn_configs provided")

        # Initialize checkpoint system
        checkpoint_manager = None
        best_model_tracker = None
        start_episode = 0

        if not args.no_checkpoints:
            console_logger.print_info("Initializing checkpoint system...")

            # Determine checkpoint directory
            if args.checkpoint_dir:
                checkpoint_dir = args.checkpoint_dir
            else:
                checkpoint_dir = get_checkpoint_dir(
                    run_id=run_id,
                    scenario_name=scenario['experiment']['name']
                )

            # Create or load run metadata
            metadata_path = Path(checkpoint_dir) / "run_metadata.json"
            if metadata_path.exists() and args.resume:
                console_logger.print_info("Loading existing run metadata...")
                run_metadata = RunMetadata.load(str(metadata_path))
            else:
                console_logger.print_info("Creating new run metadata...")
                wandb_config = scenario.get('wandb', {}) if wandb_logger else None
                run_metadata = RunMetadata.from_scenario(
                    run_id=run_id,
                    scenario=scenario,
                    checkpoint_dir=checkpoint_dir,
                    wandb_config=wandb_config
                )

            # Update metadata with W&B info if available
            if wandb_logger:
                wandb_info = wandb_logger.get_wandb_info()
                run_metadata.update_wandb_info(**wandb_info)

            # Create checkpoint manager
            checkpoint_manager = CheckpointManager(
                checkpoint_dir=checkpoint_dir,
                run_metadata=run_metadata,
                keep_best_n=3,
                keep_latest_n=2,
                keep_every_n_episodes=None,  # Don't keep all periodic checkpoints
            )

            # Create best model tracker
            best_model_tracker = BestModelTracker(
                window_size=50,  # Smooth over 50 episodes
                metric_name="success_rate",
                higher_is_better=True,
                min_improvement=0.01,  # Require 1% improvement
                patience=5,  # Wait 5 episodes before declaring new best
            )

            console_logger.print_success(f"Checkpoint dir: {checkpoint_dir}")

            # Handle resume if requested
            if args.resume:
                resume_info = checkpoint_manager.get_resume_info()
                if resume_info:
                    console_logger.print_info(f"Resuming from episode {resume_info['episode']}...")
                    # We'll load the checkpoint after creating the training loop
                    start_episode = resume_info['episode'] + 1
                else:
                    console_logger.print_warning("No checkpoint found to resume from, starting fresh")
        else:
            console_logger.print_info("Checkpointing disabled")

        # Initialize CSV logger (uses same output directory as checkpoints)
        csv_logger = None
        if not args.no_checkpoints:
            from src.core.run_id import get_output_dir
            output_dir = get_output_dir(
                run_id=run_id,
                scenario_name=scenario['experiment']['name']
            )
            csv_logger = CSVLogger(
                output_dir=output_dir,
                scenario_config=scenario,
                enabled=True,
            )
            console_logger.print_info(f"CSV output dir: {output_dir}")

        # Initialize Rich console dashboard (only updates at end of each episode)
        rich_console = RichConsole(
            enabled=True,
        )

        # Extract algorithm names for each agent
        agent_algorithms = {}
        for agent_id, agent_config in scenario['agents'].items():
            algo = agent_config.get('algorithm', '').lower()
            if algo:
                agent_algorithms[agent_id] = algo

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
            save_every_n_episodes=args.save_every if not args.no_checkpoints else None,
            normalize_observations=bool(
                scenario.get('experiment', {}).get('normalize_observations', True)
            ),
            obs_clip=float(scenario.get('experiment', {}).get('obs_clip', 10.0)),
        )

        # Load checkpoint if resuming
        if args.resume and checkpoint_manager:
            resume_info = checkpoint_manager.get_resume_info()
            if resume_info:
                console_logger.print_info(f"Loading checkpoint from {resume_info['checkpoint_path']}...")
                try:
                    training_state = training_loop.load_checkpoint(resume_info['checkpoint_path'])
                    console_logger.print_success(f"Checkpoint loaded! Resuming from episode {start_episode}")

                    # Restore curriculum stage if available
                    if spawn_curriculum and 'curriculum_stage' in training_state:
                        saved_stage = training_state['curriculum_stage']
                        console_logger.print_info(f"Restored curriculum stage: {saved_stage}")

                except Exception as e:
                    console_logger.print_error(f"Failed to load checkpoint: {e}")
                    import traceback
                    traceback.print_exc()
                    console_logger.print_warning("Starting training from scratch")
                    start_episode = 0

        episodes = scenario['experiment'].get('episodes', 100)
        training_loop.run(episodes=episodes, start_episode=start_episode)

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
