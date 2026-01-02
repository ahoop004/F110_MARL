#!/usr/bin/env python3
"""Run WaveletEpisodicAgent using scenario configuration."""

import sys
from pathlib import Path
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.scenario import load_scenario, create_training_loop


def main():
    parser = argparse.ArgumentParser(description="Train WaveletEpisodicAgent")
    parser.add_argument(
        "--scenario",
        type=str,
        default="scenarios/v2/test_wavelet.yaml",
        help="Path to scenario config (default: test_wavelet.yaml)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Override number of episodes"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable wandb logging"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable rendering"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (cpu/cuda)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("WAVELET EPISODIC AGENT TRAINING")
    print("=" * 60)
    print(f"Scenario: {args.scenario}")

    # Load scenario
    scenario = load_scenario(args.scenario)

    # Apply overrides
    if args.episodes is not None:
        scenario['experiment']['episodes'] = args.episodes
        print(f"Episodes: {args.episodes} (override)")
    else:
        print(f"Episodes: {scenario['experiment']['episodes']}")

    if args.wandb:
        scenario['wandb']['enabled'] = True
        print("Wandb: enabled (override)")
    else:
        wandb_enabled = scenario.get('wandb', {}).get('enabled', False)
        print(f"Wandb: {'enabled' if wandb_enabled else 'disabled'}")

    if args.render:
        scenario['environment']['render'] = True
        print("Rendering: enabled (override)")

    if args.device is not None:
        # Override device for wavelet agent
        for agent_id, agent_config in scenario['agents'].items():
            if agent_config.get('algorithm') == 'wavelet_episodic':
                agent_config['params']['device'] = args.device
                print(f"Device: {args.device} (override)")

    print("=" * 60)
    print()

    # Create training loop
    loop = create_training_loop(scenario)

    # Run training
    episodes = scenario['experiment']['episodes']
    print(f"Starting training for {episodes} episodes...")
    print()

    try:
        loop.run(episodes=episodes)
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE!")
        print("=" * 60)
    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("Training interrupted by user")
        print("=" * 60)
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"Training failed with error: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
