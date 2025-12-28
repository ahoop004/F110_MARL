"""Unit tests for SAC agent."""
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from agents.sac.sac import SACAgent


class TestSACAgent:
    """Test suite for SAC agent implementation."""

    @pytest.fixture
    def sac_config(self):
        """Create a minimal SAC config."""
        return {
            "obs_dim": 10,
            "act_dim": 2,
            "action_low": np.array([-1.0, -1.0]),
            "action_high": np.array([1.0, 1.0]),
            "hidden_dims": [64, 64],
            "actor_lr": 3e-4,
            "critic_lr": 3e-4,
            "gamma": 0.99,
            "tau": 0.005,
            "batch_size": 32,
            "buffer_size": 10000,
            "warmup_steps": 100,
            "auto_alpha": True,
            "alpha": 0.2,
            "device": "cpu",
        }

    @pytest.fixture
    def agent(self, sac_config):
        """Create SAC agent instance."""
        return SACAgent(sac_config)

    def test_agent_creation(self, agent, sac_config):
        """Test that agent is created with correct architecture."""
        assert agent.obs_dim == sac_config["obs_dim"]
        assert agent.act_dim == sac_config["act_dim"]
        assert agent.gamma == sac_config["gamma"]
        assert agent.tau == sac_config["tau"]
        assert agent.batch_size == sac_config["batch_size"]
        assert agent.warmup_steps == sac_config["warmup_steps"]

    def test_twin_q_networks_exist(self, agent):
        """Test that SAC has twin Q-networks and targets."""
        assert hasattr(agent, "q1")
        assert hasattr(agent, "q2")
        assert hasattr(agent, "q1_target")
        assert hasattr(agent, "q2_target")

        # Q-networks should not share parameters
        assert id(agent.q1) != id(agent.q2)
        assert id(agent.q1_target) != id(agent.q2_target)

    def test_gaussian_policy_exists(self, agent):
        """Test that SAC has Gaussian policy network."""
        assert hasattr(agent, "actor")
        assert agent.actor is not None

    def test_automatic_entropy_tuning(self, agent, sac_config):
        """Test that automatic entropy tuning is configured correctly."""
        if sac_config["auto_alpha"]:
            assert agent.auto_alpha is True
            assert agent.log_alpha is not None
            assert agent.log_alpha.requires_grad is True
            assert agent.alpha_opt is not None
            assert agent.target_entropy == -float(sac_config["act_dim"])
        else:
            assert agent.auto_alpha is False
            assert agent.log_alpha is None
            assert agent.alpha_opt is None

    def test_act_returns_valid_action(self, agent, sac_config):
        """Test that act() returns valid continuous action."""
        obs = np.random.randn(sac_config["obs_dim"])

        # Stochastic action
        action = agent.act(obs, deterministic=False)
        assert isinstance(action, np.ndarray)
        assert action.shape == (sac_config["act_dim"],)
        assert np.all(action >= sac_config["action_low"])
        assert np.all(action <= sac_config["action_high"])

        # Deterministic action
        action_det = agent.act(obs, deterministic=True)
        assert isinstance(action_det, np.ndarray)
        assert action_det.shape == (sac_config["act_dim"],)
        assert np.all(action_det >= sac_config["action_low"])
        assert np.all(action_det <= sac_config["action_high"])

    def test_action_bounds_respected(self, agent, sac_config):
        """Test that actions are properly bounded."""
        obs = np.random.randn(sac_config["obs_dim"])

        # Test multiple samples to ensure bounds are always respected
        for _ in range(10):
            action = agent.act(obs, deterministic=False)
            assert np.all(action >= sac_config["action_low"] - 1e-5)
            assert np.all(action <= sac_config["action_high"] + 1e-5)

    def test_replay_buffer_storage(self, agent, sac_config):
        """Test that transitions are stored in replay buffer."""
        initial_size = len(agent.buffer)

        obs = np.random.randn(sac_config["obs_dim"])
        action = np.random.randn(sac_config["act_dim"])
        reward = 1.0
        next_obs = np.random.randn(sac_config["obs_dim"])
        done = False

        agent.store_transition(obs, action, reward, next_obs, done)

        assert len(agent.buffer) == initial_size + 1

    def test_update_before_warmup(self, agent):
        """Test that update returns None before warmup period."""
        # Don't add enough samples for warmup
        obs = np.random.randn(agent.obs_dim)
        action = np.random.randn(agent.act_dim)

        for _ in range(agent.warmup_steps - 1):
            agent.store_transition(obs, action, 0.0, obs, False)

        stats = agent.update()
        assert stats is None  # Not enough data yet

    def test_update_after_warmup(self, agent):
        """Test that update runs after warmup period."""
        obs = np.random.randn(agent.obs_dim)
        action = np.random.randn(agent.act_dim)

        # Fill buffer past warmup
        for _ in range(agent.warmup_steps + agent.batch_size):
            agent.store_transition(obs, action, 1.0, obs, False)

        stats = agent.update()
        assert stats is not None
        assert isinstance(stats, dict)

        # Check that stats contain expected keys
        assert "actor_loss" in stats
        assert "critic_loss" in stats  # SAC combines Q1 and Q2 losses

        if agent.auto_alpha:
            assert "alpha_loss" in stats
            assert "alpha" in stats

    def test_target_network_soft_update(self, agent):
        """Test that target networks are updated with soft update."""
        # Store initial target parameters
        q1_target_params_before = [p.clone() for p in agent.q1_target.parameters()]
        q2_target_params_before = [p.clone() for p in agent.q2_target.parameters()]

        # Fill buffer and update
        obs = np.random.randn(agent.obs_dim)
        action = np.random.randn(agent.act_dim)
        for _ in range(agent.warmup_steps + agent.batch_size):
            agent.store_transition(obs, action, 1.0, obs, False)

        agent.update()

        # Check that target parameters changed slightly (soft update)
        q1_target_params_after = list(agent.q1_target.parameters())
        q2_target_params_after = list(agent.q2_target.parameters())

        for p_before, p_after in zip(q1_target_params_before, q1_target_params_after):
            # Parameters should change but not be identical to online network
            assert not torch.allclose(p_before, p_after)

        for p_before, p_after in zip(q2_target_params_before, q2_target_params_after):
            assert not torch.allclose(p_before, p_after)

    def test_save_and_load(self, agent, sac_config):
        """Test that agent can save and load checkpoints."""
        # Fill buffer with some data
        obs = np.random.randn(sac_config["obs_dim"])
        action = np.random.randn(sac_config["act_dim"])

        for _ in range(100):
            agent.store_transition(obs, action, 1.0, obs, False)

        # Get parameters before saving
        actor_params_before = {k: v.clone() for k, v in agent.actor.state_dict().items()}

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "sac_checkpoint.pt"
            agent.save(str(save_path))
            assert save_path.exists()

            # Create new agent and load
            new_agent = SACAgent(sac_config)
            new_agent.load(str(save_path))

            # Verify parameters match
            actor_params_after = new_agent.actor.state_dict()
            for key in actor_params_before:
                assert torch.allclose(actor_params_before[key], actor_params_after[key])

    def test_entropy_coefficient_updates(self, agent):
        """Test that entropy coefficient is updated when auto_alpha=True."""
        if not agent.auto_alpha:
            pytest.skip("Auto alpha not enabled")

        # Store initial alpha
        alpha_before = agent.log_alpha.item()

        # Fill buffer and update
        obs = np.random.randn(agent.obs_dim)
        action = np.random.randn(agent.act_dim)
        for _ in range(agent.warmup_steps + agent.batch_size):
            agent.store_transition(obs, action, 1.0, obs, False)

        # Run multiple updates
        for _ in range(10):
            agent.update()

        # Alpha should have changed
        alpha_after = agent.log_alpha.item()
        # Note: alpha might increase or decrease depending on entropy
        assert alpha_before != alpha_after

    def test_fixed_alpha_mode(self):
        """Test that SAC works with fixed alpha (no automatic tuning)."""
        config = {
            "obs_dim": 10,
            "act_dim": 2,
            "action_low": np.array([-1.0, -1.0]),
            "action_high": np.array([1.0, 1.0]),
            "auto_alpha": False,
            "alpha": 0.1,
            "device": "cpu",
            "warmup_steps": 100,
            "batch_size": 32,
        }

        agent = SACAgent(config)
        assert agent.auto_alpha is False
        assert agent.log_alpha is None
        assert agent._alpha_constant == 0.1

        # Should still be able to act and update
        obs = np.random.randn(config["obs_dim"])
        action = agent.act(obs)
        assert action.shape == (config["act_dim"],)
