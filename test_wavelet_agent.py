#!/usr/bin/env python3
"""Simple test script for WaveletEpisodicAgent without wandb."""

import sys
from pathlib import Path
import numpy as np
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from env import F110ParallelEnv
from agents.episodic import WaveletEpisodicAgent
from core.enhanced_training import EnhancedTrainingLoop
from loggers import ConsoleLogger


def create_simple_env():
    """Create a minimal F110 environment for testing."""
    # Simple spawn poses
    spawn_poses = np.array([[0.0, 0.0, 0.0]])  # [x, y, theta]

    env_config = {
        "map": "maps/line2/line2",
        "num_agents": 1,
        "timestep": 0.01,
        "integrator": "rk4",
        "model": "st",
        "observation_config": "original",
        "params": {"mu": 1.0},
        "start_poses": spawn_poses,  # Add start poses to config
    }

    return F110ParallelEnv(**env_config), spawn_poses


def test_wavelet_agent_basic():
    """Test basic agent initialization and forward pass."""
    print("=" * 60)
    print("TEST 1: Basic Agent Initialization")
    print("=" * 60)

    # Minimal config
    config = {
        'obs_dim': 107,  # F110 observation dimension
        'act_dim': 2,    # steering + velocity
        'action_low': [-0.4, 0.0],
        'action_high': [0.4, 8.0],
        'chunk_size': 25,
        'chunk_stride': 12,
        'grid_shape': [5, 5],
        'n_channels': 5,
        'chronological_buffer': {'max_capacity': 1000},
        'episodic_buffer': {'capacity': 100, 'selection_mode': 'uniform'},
        'wavelet': {'type': 'haar', 'mode': 'symmetric'},
        'encoder': {
            'latent_dim': 64,  # Smaller for testing
            'cnn': {
                'channels': [16, 32, 64],
                'kernel_size': 3,
                'pooling': [2, 2, None],
            },
            'rnn': {
                'use': True,
                'type': 'lstm',
                'hidden_size': 128,
                'num_layers': 1,
                'dropout': 0.0,
            }
        },
        'heads': {
            'policy_head': {'hidden_dims': [128, 128]},
            'value_head': {'hidden_dims': [128, 64]},
            'reconstruction_head': {'hidden_dims': [128, 256]},
            'forward_head': {'hidden_dims': [128, 256]},
        },
        'loss_weights': {
            'policy': 1.0,
            'value': 0.5,
            'reconstruction': 0.1,
            'forward': 0.1,
        },
        'learning_rate': 3e-4,
        'batch_size': 16,
        'warmup_chunks': 10,
        'update_freq': 4,
        'device': 'cpu',  # Use CPU for testing
    }

    try:
        agent = WaveletEpisodicAgent(config)
        print("✓ Agent initialized successfully!")
        print(f"  - Chronological buffer: {agent.chronological_buffer}")
        print(f"  - Episodic buffer: {agent.episodic_buffer}")
        print(f"  - Wavelet: {agent.wavelet}")
        print(f"  - Encoder: {agent.encoder.__class__.__name__}")
        print(f"  - Device: {agent.device}")
        return True
    except Exception as e:
        print(f"✗ Agent initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_wavelet_agent_episode():
    """Test running a short episode with the agent."""
    print("\n" + "=" * 60)
    print("TEST 2: Run Short Episode (50 steps)")
    print("=" * 60)

    # Create environment
    try:
        env, spawn_poses = create_simple_env()
        print("✓ Environment created")
    except Exception as e:
        print(f"✗ Environment creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Create agent
    config = {
        'obs_dim': 107,
        'act_dim': 2,
        'action_low': [-0.4, 0.0],
        'action_high': [0.4, 8.0],
        'chunk_size': 25,
        'chunk_stride': 12,
        'grid_shape': [5, 5],
        'chronological_buffer': {'max_capacity': 1000},
        'episodic_buffer': {'capacity': 50, 'selection_mode': 'uniform'},
        'wavelet': {'type': 'haar'},
        'encoder': {
            'latent_dim': 32,
            'cnn': {'channels': [8, 16, 32], 'pooling': [2, 2, None]},
            'rnn': {'use': False},  # Disable RNN for speed
        },
        'warmup_chunks': 5,
        'update_freq': 10,
        'batch_size': 4,
        'device': 'cpu',
    }

    try:
        agent = WaveletEpisodicAgent(config)
        print("✓ Agent initialized")
    except Exception as e:
        print(f"✗ Agent initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Run episode
    try:
        obs_dict, info = env.reset(options={"poses": spawn_poses})
        print("✓ Environment reset")

        agent_id = list(obs_dict.keys())[0]
        obs = obs_dict[agent_id]

        print(f"\nRunning 50 steps with agent...")

        for step in range(50):
            # Get action
            action = agent.act(obs)

            # Step environment
            next_obs_dict, rewards, dones, truncated, infos = env.step({agent_id: action})
            next_obs = next_obs_dict[agent_id]
            reward = rewards[agent_id]
            done = dones[agent_id]

            # Store transition
            agent.store_transition(obs, action, reward, next_obs, done)

            # Update agent
            update_stats = agent.update()

            if step % 10 == 0:
                chrono_size = len(agent.chronological_buffer)
                episodic_size = len(agent.episodic_buffer)
                print(f"  Step {step:2d}: chrono_buffer={chrono_size}, episodic_buffer={episodic_size}", end="")
                if update_stats:
                    print(f", loss={update_stats['total_loss']:.4f}")
                else:
                    print(" (warmup)")

            obs = next_obs

            if done:
                break

        print(f"\n✓ Episode completed!")
        print(f"  - Chronological buffer size: {len(agent.chronological_buffer)}")
        print(f"  - Episodic buffer size: {len(agent.episodic_buffer)}")
        print(f"  - Update count: {agent.update_count}")

        return True

    except Exception as e:
        print(f"✗ Episode execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_save_load():
    """Test saving and loading agent."""
    print("\n" + "=" * 60)
    print("TEST 3: Save/Load Agent")
    print("=" * 60)

    config = {
        'obs_dim': 107,
        'act_dim': 2,
        'chunk_size': 25,
        'episodic_buffer': {'capacity': 50},
        'encoder': {'latent_dim': 32, 'cnn': {'channels': [8, 16]}, 'rnn': {'use': False}},
        'device': 'cpu',
    }

    try:
        # Create agent
        agent = WaveletEpisodicAgent(config)
        agent.step_count = 42  # Set some state

        # Save
        save_path = "/tmp/test_wavelet_agent"
        agent.save(save_path)
        print(f"✓ Saved agent to {save_path}")

        # Load
        agent2 = WaveletEpisodicAgent(config)
        agent2.load(save_path)
        print(f"✓ Loaded agent from {save_path}")

        # Verify
        assert agent2.step_count == 42, "Step count mismatch!"
        print("✓ State verified (step_count matches)")

        return True

    except Exception as e:
        print(f"✗ Save/load failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("WAVELET EPISODIC AGENT TEST SUITE")
    print("=" * 60)
    print("Testing without wandb (local mode)\n")

    results = []

    # Test 1: Basic initialization
    results.append(("Initialization", test_wavelet_agent_basic()))

    # Test 2: Episode execution
    results.append(("Episode Execution", test_wavelet_agent_episode()))

    # Test 3: Save/Load
    results.append(("Save/Load", test_save_load()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name:20s}: {status}")

    all_passed = all(result[1] for result in results)

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED! 🎉")
        print("The WaveletEpisodicAgent is working correctly.")
    else:
        print("SOME TESTS FAILED")
        print("Check the errors above for details.")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
