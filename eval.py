#!/usr/bin/env python3
"""Evaluate trained F110 MARL agents.

Standalone evaluation script for testing trained agents with deterministic,
fixed scenarios for consistent performance measurement.

Usage:
    # Evaluate by checkpoint path
    python eval.py --checkpoint outputs/checkpoints/sac/run_abc/sac_fa4f_best_ep000500.pt

    # Evaluate by run ID (auto-finds best checkpoint)
    python eval.py --run-id sac_sb3_sac_s42_1767463361_fa4f

    # Evaluate latest run for an algorithm
    python eval.py --latest --algo sac

    # Custom evaluation parameters
    python eval.py --checkpoint path/to/ckpt.pt --num-episodes 20 --render

Example output:
    Evaluation Results (10 episodes)
      Success Rate: 65.0% (6/10)
      Avg Episode Length: 425.3 ± 87.2 steps
      Avg Reward: 156.8 ± 42.1

    Outcome Distribution:
      TARGET_CRASH: 6 (60.0%)
      SELF_CRASH: 2 (20.0%)
      TIMEOUT: 2 (20.0%)
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch

# Allow running from repo root without installing the package.
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if SRC_DIR.is_dir() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.scenario import load_and_expand_scenario
from core.setup import create_training_setup
from core.checkpoint_manager import CheckpointManager
from core.run_metadata import RunMetadata
from core.evaluator import Evaluator, EvaluationConfig
from core.obs_flatten import flatten_observation
from wrappers.normalize import ObservationNormalizer


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate trained F110 MARL agents',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Checkpoint identification
    ckpt_group = parser.add_mutually_exclusive_group(required=True)
    ckpt_group.add_argument(
        '--checkpoint',
        type=str,
        help='Path to checkpoint file (.pt)'
    )
    ckpt_group.add_argument(
        '--run-id',
        type=str,
        help='Run ID to load best checkpoint from'
    )
    ckpt_group.add_argument(
        '--latest',
        action='store_true',
        help='Load latest run for specified algorithm'
    )

    # Required for --latest
    parser.add_argument(
        '--algo',
        type=str,
        help='Algorithm name (required with --latest)'
    )

    # Evaluation parameters
    parser.add_argument(
        '--num-episodes',
        type=int,
        default=10,
        help='Number of evaluation episodes (default: 10)'
    )
    parser.add_argument(
        '--spawn-points',
        type=str,
        nargs='+',
        default=['spawn_pinch_left', 'spawn_pinch_right'],
        help='Spawn points to use sequentially (default: spawn_pinch_left spawn_pinch_right)'
    )
    parser.add_argument(
        '--spawn-speeds',
        type=float,
        nargs='+',
        default=[0.44, 0.44],
        help='Initial speeds for spawn points (default: 0.44 0.44)'
    )
    parser.add_argument(
        '--no-speed-lock',
        action='store_true',
        help='Disable speed locking (default: no lock)'
    )
    parser.add_argument(
        '--render',
        action='store_true',
        default=False,
        help='Enable rendering during evaluation (default: False)'
    )

    # Output options
    parser.add_argument(
        '--save-results',
        type=str,
        help='Save evaluation results to JSON file'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed progress during evaluation'
    )

    return parser.parse_args()


def find_checkpoint_by_run_id(run_id: str) -> str:
    """Find best checkpoint for a run ID.

    Args:
        run_id: Run identifier

    Returns:
        Path to best checkpoint

    Raises:
        FileNotFoundError: If run metadata not found
    """
    # Search in outputs/checkpoints for run_metadata.json
    checkpoints_dir = Path('outputs/checkpoints')

    # Try to find run_metadata.json for this run_id
    for algo_dir in checkpoints_dir.glob('*'):
        if not algo_dir.is_dir():
            continue

        for run_dir in algo_dir.glob('*'):
            if not run_dir.is_dir():
                continue

            metadata_file = run_dir / 'run_metadata.json'
            if metadata_file.exists():
                try:
                    metadata = RunMetadata.load(str(metadata_file))
                    if metadata.run_id == run_id:
                        # Found it! Return best checkpoint
                        if metadata.best_checkpoint:
                            return metadata.best_checkpoint
                        elif metadata.latest_checkpoint:
                            print(f"Warning: No best checkpoint found, using latest")
                            return metadata.latest_checkpoint
                        else:
                            raise FileNotFoundError(f"No checkpoints found for run {run_id}")
                except Exception as e:
                    continue

    raise FileNotFoundError(f"Run ID '{run_id}' not found in outputs/checkpoints")


def find_latest_checkpoint(algo: str) -> str:
    """Find latest checkpoint for an algorithm.

    Args:
        algo: Algorithm name

    Returns:
        Path to latest best checkpoint

    Raises:
        FileNotFoundError: If no runs found
    """
    checkpoints_dir = Path('outputs/checkpoints') / algo

    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"No checkpoints found for algorithm '{algo}'")

    # Find all run_metadata.json files
    metadata_files = []
    for run_dir in checkpoints_dir.glob('*'):
        if not run_dir.is_dir():
            continue

        metadata_file = run_dir / 'run_metadata.json'
        if metadata_file.exists():
            try:
                metadata = RunMetadata.load(str(metadata_file))
                metadata_files.append((metadata_file, metadata))
            except Exception:
                continue

    if not metadata_files:
        raise FileNotFoundError(f"No run metadata found for algorithm '{algo}'")

    # Sort by creation time (most recent first)
    metadata_files.sort(key=lambda x: x[1].created_at, reverse=True)

    # Get latest
    latest_metadata = metadata_files[0][1]

    print(f"Latest run: {latest_metadata.run_id}")
    print(f"Created: {latest_metadata.created_at}")
    print(f"Episodes: {latest_metadata.episodes_completed}/{latest_metadata.total_episodes}")

    if latest_metadata.best_checkpoint:
        return latest_metadata.best_checkpoint
    elif latest_metadata.latest_checkpoint:
        print(f"Warning: No best checkpoint found, using latest")
        return latest_metadata.latest_checkpoint
    else:
        raise FileNotFoundError(f"No checkpoints found for run {latest_metadata.run_id}")


def load_scenario_from_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """Load scenario configuration from checkpoint's run metadata.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Scenario configuration dict

    Raises:
        FileNotFoundError: If run metadata not found
    """
    # Look for run_metadata.json in same directory
    checkpoint_dir = Path(checkpoint_path).parent
    metadata_file = checkpoint_dir / 'run_metadata.json'

    if metadata_file.exists():
        metadata = RunMetadata.load(str(metadata_file))
        if metadata.config_snapshot:
            return metadata.config_snapshot

    raise FileNotFoundError(
        f"Could not find scenario config for checkpoint: {checkpoint_path}\n"
        f"Expected run_metadata.json at: {metadata_file}"
    )


def main():
    args = parse_args()

    # Validate arguments
    if args.latest and not args.algo:
        print("Error: --algo required when using --latest")
        sys.exit(1)

    # Find checkpoint
    print("=" * 60)
    print("F110 MARL Evaluation")
    print("=" * 60)
    print()

    if args.checkpoint:
        checkpoint_path = args.checkpoint
        print(f"Checkpoint: {checkpoint_path}")
    elif args.run_id:
        print(f"Finding checkpoint for run: {args.run_id}")
        checkpoint_path = find_checkpoint_by_run_id(args.run_id)
        print(f"Found: {checkpoint_path}")
    else:  # args.latest
        print(f"Finding latest checkpoint for algorithm: {args.algo}")
        checkpoint_path = find_latest_checkpoint(args.algo)
        print(f"Found: {checkpoint_path}")

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    print()

    # Load scenario configuration
    print("Loading scenario configuration...")
    scenario = load_scenario_from_checkpoint(str(checkpoint_path))

    # Override render setting
    if args.render:
        scenario.setdefault('environment', {})['render'] = True
    else:
        scenario.setdefault('environment', {})['render'] = False

    # Create environment and agents
    print("Creating environment and agents...")
    env, agents, reward_strategies = create_training_setup(scenario)

    # Extract observation presets and target IDs
    observation_presets = {}
    target_ids = {}
    for agent_id, agent_config in scenario['agents'].items():
        # Observation preset
        obs_cfg = agent_config.get('observation', {})
        if isinstance(obs_cfg, dict) and 'preset' in obs_cfg:
            observation_presets[agent_id] = obs_cfg['preset']

        # Target ID
        if 'target_id' in agent_config:
            target_ids[agent_id] = agent_config['target_id']

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path.name}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Restore agent states
    agent_states = checkpoint.get('agent_states', {})
    for agent_id, state in agent_states.items():
        if agent_id in agents and hasattr(agents[agent_id], 'load_state'):
            agents[agent_id].load_state(state)
            print(f"  Loaded state for agent: {agent_id}")

    episode_num = checkpoint.get('episode', 'unknown')
    metric_value = checkpoint.get('metric_value', None)
    print(f"  Episode: {episode_num}")
    if metric_value is not None:
        print(f"  Metric value: {metric_value:.2%}")

    print()

    # Get observation scales from environment
    obs_scales = {}
    lidar_range = getattr(env, 'lidar_range', None)
    if lidar_range:
        obs_scales['lidar_range'] = float(lidar_range)

    params = getattr(env, 'params', None)
    if isinstance(params, dict):
        v_max = params.get('v_max')
        if v_max:
            obs_scales['speed'] = float(abs(v_max))

    # Create evaluation configuration
    eval_config = EvaluationConfig(
        num_episodes=args.num_episodes,
        deterministic=True,
        spawn_points=args.spawn_points,
        spawn_speeds=args.spawn_speeds,
        lock_speed_steps=0,  # No speed locking for eval
        ftg_override={
            'max_speed': 1.0,
            'bubble_radius': 3.0,
            'steering_gain': 0.35,
        },
        max_steps=scenario['environment'].get('max_steps', 2500),
    )

    # Get spawn configs from scenario (can be at environment level or in spawn_curriculum)
    env_config = scenario.get('environment', {})
    spawn_configs = env_config.get('spawn_configs', {})
    if not spawn_configs and 'spawn_curriculum' in env_config:
        spawn_configs = env_config['spawn_curriculum'].get('spawn_configs', {})

    # Create evaluator
    evaluator = Evaluator(
        env=env,
        agents=agents,
        config=eval_config,
        observation_presets=observation_presets,
        target_ids=target_ids,
        obs_normalizer=None,  # Don't use normalization for eval
        obs_scales=obs_scales,
        spawn_configs=spawn_configs,
    )

    # Run evaluation
    print("Running evaluation...")
    print(f"  Episodes: {args.num_episodes}")
    print(f"  Spawn points: {args.spawn_points}")
    print(f"  Spawn speeds: {args.spawn_speeds}")
    print(f"  Deterministic: True")
    print(f"  FTG: Full strength (max_speed=1.0, bubble_radius=3.0)")
    print()

    result = evaluator.evaluate(verbose=args.verbose)

    # Print results
    print("=" * 60)
    print(result.summary_str())
    print("=" * 60)

    # Save results if requested
    if args.save_results:
        save_path = Path(args.save_results)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Add checkpoint info to results
        results_dict = result.to_dict()
        results_dict['checkpoint_path'] = str(checkpoint_path)
        results_dict['checkpoint_episode'] = episode_num
        results_dict['checkpoint_metric'] = metric_value

        with open(save_path, 'w') as f:
            json.dump(results_dict, f, indent=2)

        print()
        print(f"Results saved to: {save_path}")

    # Return success rate as exit code (for scripting)
    success_rate_pct = int(result.success_rate * 100)
    return min(success_rate_pct, 99)  # Cap at 99 for valid exit codes


if __name__ == '__main__':
    sys.exit(main())
