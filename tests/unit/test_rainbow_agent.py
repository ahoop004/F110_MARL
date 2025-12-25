"""Unit tests for Rainbow DQN agent."""
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from v2.agents.rainbow.r_dqn import RainbowDQNAgent


class TestRainbowAgent:
    """Test suite for Rainbow DQN agent implementation."""

    @pytest.fixture
    def rainbow_config(self):
        """Create a minimal Rainbow config."""
        return {
            "obs_dim": 10,
            "action_set": np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]),  # 3 discrete actions
            "hidden_dims": [64, 64],
            "lr": 5e-4,
            "gamma": 0.99,
            "batch_size": 32,
            "buffer_size": 10000,
            "learning_starts": 100,  # Rainbow uses learning_starts
            "target_update_interval": 100,  # Rainbow uses target_update_interval
            "noisy_layers": True,
            "noisy_sigma0": 0.5,
            "atoms": 51,
            "v_min": -10.0,
            "v_max": 10.0,
            "n_step": 3,
            "use_per": True,
            "epsilon_enabled": False,  # Noisy layers instead
            "device": "cpu",
        }

    @pytest.fixture
    def agent(self, rainbow_config):
        """Create Rainbow agent instance."""
        return RainbowDQNAgent(rainbow_config)

    def test_agent_creation(self, agent, rainbow_config):
        """Test that agent is created with correct architecture."""
        assert agent.obs_dim == rainbow_config["obs_dim"]
        assert agent.n_actions == len(rainbow_config["action_set"])
        assert agent.gamma == rainbow_config["gamma"]
        assert agent.batch_size == rainbow_config["batch_size"]
        # learning_starts is derived from config
        assert agent.learning_starts >= agent.batch_size

    def test_categorical_dqn_support(self, agent, rainbow_config):
        """Test that categorical DQN (distributional RL) is configured."""
        assert agent.atoms == rainbow_config["atoms"]
        assert agent.v_min == rainbow_config["v_min"]
        assert agent.v_max == rainbow_config["v_max"]
        assert agent.delta_z == (agent.v_max - agent.v_min) / (agent.atoms - 1)

    def test_noisy_layers_enabled(self, agent, rainbow_config):
        """Test that noisy layers are enabled for exploration."""
        if rainbow_config["noisy_layers"]:
            assert agent.use_noisy is True
            assert agent.use_epsilon is False  # Should use noisy instead of epsilon
            assert agent.noisy_sigma0 == rainbow_config["noisy_sigma0"]

    def test_n_step_learning(self, agent, rainbow_config):
        """Test that n-step learning is configured."""
        assert agent.n_step == rainbow_config["n_step"]
        assert agent.n_step >= 1

    def test_prioritized_replay_enabled(self, agent, rainbow_config):
        """Test that prioritized experience replay is enabled."""
        if rainbow_config.get("use_per", True):
            assert agent._use_per is True  # Private attribute in base class

    def test_target_network_exists(self, agent):
        """Test that target network exists."""
        assert hasattr(agent, "q_net")
        assert hasattr(agent, "target_q_net")
        assert id(agent.q_net) != id(agent.target_q_net)

    def test_act_returns_valid_action_index(self, agent, rainbow_config):
        """Test that act() returns valid discrete action index."""
        obs = np.random.randn(rainbow_config["obs_dim"])

        # Deterministic action
        action_idx = agent.act(obs, deterministic=True)
        assert isinstance(action_idx, (int, np.integer))
        assert 0 <= action_idx < agent.n_actions

        # Stochastic action (noisy layers provide exploration)
        action_idx = agent.act(obs, deterministic=False)
        assert isinstance(action_idx, (int, np.integer))
        assert 0 <= action_idx < agent.n_actions

    def test_noisy_exploration_vs_epsilon(self):
        """Test that noisy layers and epsilon-greedy are mutually exclusive."""
        config = {
            "obs_dim": 10,
            "action_set": np.array([[1.0], [0.0], [-1.0]]),
            "noisy_layers": True,
            "epsilon_enabled": True,  # This should raise error
            "device": "cpu",
        }

        with pytest.raises(ValueError, match="cannot enable epsilon-greedy when noisy"):
            RainbowDQNAgent(config)

    def test_replay_buffer_storage(self, agent, rainbow_config):
        """Test that transitions are stored in replay buffer."""
        obs = np.random.randn(rainbow_config["obs_dim"])
        action_idx = 0
        reward = 1.0
        next_obs = np.random.randn(rainbow_config["obs_dim"])
        done = False

        # Store n_step + 1 transitions (Rainbow uses n-step buffer)
        for _ in range(agent.n_step + 1):
            agent.store_transition(obs, action_idx, reward, next_obs, done)

        # Buffer should have at least one complete n-step sequence
        assert len(agent.buffer) >= 1

    def test_update_before_warmup(self, agent):
        """Test that update returns None before warmup period."""
        obs = np.random.randn(agent.obs_dim)

        # Don't add enough samples for learning_starts
        for _ in range(agent.learning_starts - agent.n_step - 1):
            agent.store_transition(obs, 0, 0.0, obs, False)

        stats = agent.update()
        assert stats is None  # Not enough data yet

    def test_update_after_warmup(self, agent):
        """Test that update runs after warmup period."""
        obs = np.random.randn(agent.obs_dim)

        # Fill buffer past learning_starts
        for _ in range(agent.learning_starts + agent.batch_size):
            action_idx = np.random.randint(0, agent.n_actions)
            agent.store_transition(obs, action_idx, 1.0, obs, False)

        stats = agent.update()
        assert stats is not None
        assert isinstance(stats, dict)
        assert "loss" in stats

    def test_target_network_updates(self, agent):
        """Test that target network is updated at specified frequency."""
        obs = np.random.randn(agent.obs_dim)

        # Fill buffer
        for _ in range(agent.learning_starts + agent.batch_size):
            action_idx = np.random.randint(0, agent.n_actions)
            agent.store_transition(obs, action_idx, 1.0, obs, False)

        # Get initial target parameters
        target_params_before = [p.clone() for p in agent.target_q_net.parameters()]

        # Update until target should be updated (target_update_interval)
        for _ in range(agent.target_update_interval + 1):
            agent.update()

        # Target network should have changed
        target_params_after = list(agent.target_q_net.parameters())
        params_changed = any(
            not torch.allclose(p_before, p_after)
            for p_before, p_after in zip(target_params_before, target_params_after)
        )
        assert params_changed

    def test_save_and_load(self, agent, rainbow_config):
        """Test that agent can save and load checkpoints."""
        # Fill buffer with some data
        obs = np.random.randn(rainbow_config["obs_dim"])

        for _ in range(100):
            action_idx = np.random.randint(0, agent.n_actions)
            agent.store_transition(obs, action_idx, 1.0, obs, False)

        # Get parameters before saving
        q_params_before = {k: v.clone() for k, v in agent.q_net.state_dict().items()}

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "rainbow_checkpoint.pt"
            agent.save(str(save_path))
            assert save_path.exists()

            # Create new agent and load
            new_agent = RainbowDQNAgent(rainbow_config)
            new_agent.load(str(save_path))

            # Verify parameters match
            q_params_after = new_agent.q_net.state_dict()
            for key in q_params_before:
                assert torch.allclose(q_params_before[key], q_params_after[key])

    def test_rainbow_with_epsilon_greedy(self):
        """Test that Rainbow can work with epsilon-greedy (when noisy layers disabled)."""
        config = {
            "obs_dim": 10,
            "action_set": np.array([[1.0], [0.0], [-1.0]]),
            "noisy_layers": False,
            "epsilon_enabled": True,
            "epsilon_start": 0.9,
            "epsilon_end": 0.05,
            "epsilon_decay": 10000,
            "device": "cpu",
            "learning_starts": 100,
            "batch_size": 32,
        }

        agent = RainbowDQNAgent(config)
        assert agent.use_noisy is False
        assert agent.use_epsilon is True

        # Should still be able to act
        obs = np.random.randn(config["obs_dim"])
        action_idx = agent.act(obs)
        assert 0 <= action_idx < agent.n_actions

    def test_atoms_validation(self):
        """Test that invalid atom counts are rejected."""
        config = {
            "obs_dim": 10,
            "action_set": np.array([[1.0], [0.0]]),
            "atoms": 1,  # Too few atoms
            "device": "cpu",
        }

        with pytest.raises(ValueError, match="at least two atoms"):
            RainbowDQNAgent(config)

    def test_value_range_validation(self):
        """Test that invalid value ranges are rejected."""
        config = {
            "obs_dim": 10,
            "action_set": np.array([[1.0], [0.0]]),
            "v_min": 10.0,
            "v_max": -10.0,  # v_max < v_min
            "device": "cpu",
        }

        with pytest.raises(ValueError, match="v_max > v_min"):
            RainbowDQNAgent(config)
