"""Pytest configuration and shared fixtures."""
import os
import sys
from pathlib import Path

import pytest
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def test_data_dir(project_root):
    """Return the test data directory."""
    return project_root / "tests" / "fixtures"


@pytest.fixture(scope="function")
def seed_rng():
    """Seed random number generators for reproducibility."""
    np.random.seed(42)
    try:
        import torch
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
    except ImportError:
        pass


@pytest.fixture(scope="function")
def simple_env_config():
    """Minimal environment config for testing."""
    return {
        "map": "maps/line.yaml",
        "map_dir": "maps",
        "n_agents": 2,
        "num_beams": 360,
        "timestep": 0.01,
        "integrator": "rk4",
        "max_steps": 100,
    }


@pytest.fixture(scope="function")
def ppo_agent_config():
    """Minimal PPO agent config for testing."""
    return {
        "obs_dim": 370,  # 360 lidar + 10 other features
        "act_dim": 2,    # steering + acceleration
        "action_low": [-0.4, -1.0],
        "action_high": [0.4, 1.0],
        "gamma": 0.99,
        "lam": 0.95,
        "clip_eps": 0.2,
        "update_epochs": 3,
        "minibatch_size": 32,
        "actor_lr": 3e-4,
        "critic_lr": 1e-3,
        "ent_coef": 0.0,
        "device": "cpu",
    }


@pytest.fixture(scope="function")
def td3_agent_config():
    """Minimal TD3 agent config for testing."""
    return {
        "obs_dim": 370,
        "act_dim": 2,
        "action_low": [-0.4, -1.0],
        "action_high": [0.4, 1.0],
        "gamma": 0.99,
        "tau": 0.005,
        "policy_delay": 2,
        "batch_size": 64,
        "buffer_size": 10000,
        "warmup_steps": 100,
        "exploration_noise": 0.1,
        "actor_lr": 1e-3,
        "critic_lr": 1e-3,
        "device": "cpu",
    }


@pytest.fixture(scope="function")
def dqn_agent_config():
    """Minimal DQN agent config for testing."""
    # Define discrete action set
    action_set = np.array([
        [-0.4, 1.0],   # Hard left
        [-0.2, 1.0],   # Soft left
        [0.0, 1.0],    # Straight
        [0.2, 1.0],    # Soft right
        [0.4, 1.0],    # Hard right
    ], dtype=np.float32)

    return {
        "obs_dim": 370,
        "action_set": action_set,
        "gamma": 0.99,
        "batch_size": 32,
        "buffer_size": 10000,
        "target_update_interval": 100,
        "warmup_steps": 100,
        "epsilon_start": 0.9,
        "epsilon_end": 0.05,
        "epsilon_decay": 1000,
        "lr": 5e-4,
        "device": "cpu",
    }
