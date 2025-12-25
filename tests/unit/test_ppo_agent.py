"""Unit tests for PPO agent."""
import pytest
import numpy as np
import torch

from f110x.policies.ppo.ppo import PPOAgent


@pytest.mark.unit
class TestPPOAgent:
    """Test PPO agent core functionality."""

    def test_agent_creation(self, ppo_agent_config, seed_rng):
        """Test PPO agent can be instantiated."""
        agent = PPOAgent(ppo_agent_config)
        assert agent is not None
        assert agent.obs_dim == ppo_agent_config["obs_dim"]
        assert agent.act_dim == ppo_agent_config["act_dim"]

    def test_act_returns_valid_action(self, ppo_agent_config, seed_rng):
        """Test that act() returns action in valid range."""
        agent = PPOAgent(ppo_agent_config)
        obs = np.random.randn(ppo_agent_config["obs_dim"]).astype(np.float32)

        action = agent.act(obs)

        assert action.shape == (ppo_agent_config["act_dim"],)
        assert np.all(action >= ppo_agent_config["action_low"])
        assert np.all(action <= ppo_agent_config["action_high"])

    def test_act_deterministic(self, ppo_agent_config, seed_rng):
        """Test deterministic action selection."""
        agent = PPOAgent(ppo_agent_config)
        obs = np.random.randn(ppo_agent_config["obs_dim"]).astype(np.float32)

        action1 = agent.act_deterministic(obs)
        action2 = agent.act_deterministic(obs)

        np.testing.assert_array_almost_equal(action1, action2)

    def test_store_transition(self, ppo_agent_config, seed_rng):
        """Test storing transitions."""
        agent = PPOAgent(ppo_agent_config)
        obs = np.random.randn(ppo_agent_config["obs_dim"]).astype(np.float32)

        # Act to populate buffers
        action = agent.act(obs)

        # Store transition
        agent.store(obs, action, 1.0, False, terminated=False)

        assert len(agent.rew_buf) == 1
        assert len(agent.obs_buf) == 1

    def test_update_with_data(self, ppo_agent_config, seed_rng):
        """Test that update runs with collected data."""
        agent = PPOAgent(ppo_agent_config)

        # Collect some transitions
        for _ in range(50):
            obs = np.random.randn(ppo_agent_config["obs_dim"]).astype(np.float32)
            action = agent.act(obs)
            agent.store(obs, action, np.random.randn(), False, terminated=False)

        # Final transition (done)
        obs = np.random.randn(ppo_agent_config["obs_dim"]).astype(np.float32)
        action = agent.act(obs)
        agent.store(obs, action, 1.0, True, terminated=True)

        # Update should return stats
        stats = agent.update()

        assert stats is not None
        assert "policy_loss" in stats
        assert "value_loss" in stats
        assert "entropy" in stats
        assert "approx_kl" in stats

    def test_save_and_load(self, ppo_agent_config, seed_rng, tmp_path):
        """Test checkpoint save and load."""
        agent = PPOAgent(ppo_agent_config)

        # Get initial action
        obs = np.random.randn(ppo_agent_config["obs_dim"]).astype(np.float32)
        action_before = agent.act_deterministic(obs)

        # Save
        checkpoint_path = tmp_path / "ppo_checkpoint.pt"
        agent.save(str(checkpoint_path))
        assert checkpoint_path.exists()

        # Create new agent and load
        agent2 = PPOAgent(ppo_agent_config)
        agent2.load(str(checkpoint_path))

        # Should produce same action
        action_after = agent2.act_deterministic(obs)
        np.testing.assert_array_almost_equal(action_before, action_after, decimal=5)

    def test_gae_computation(self, ppo_agent_config, seed_rng):
        """Test GAE (Generalized Advantage Estimation) computation."""
        agent = PPOAgent(ppo_agent_config)

        # Collect episode
        rewards = [1.0, 1.0, 1.0, 5.0]  # Increasing reward
        for i, r in enumerate(rewards):
            obs = np.random.randn(ppo_agent_config["obs_dim"]).astype(np.float32)
            action = agent.act(obs)
            done = (i == len(rewards) - 1)
            agent.store(obs, action, r, done, terminated=done)

        # Finish path computes advantages
        agent.finish_path(normalize_advantage=True)

        assert len(agent.adv_buf) == len(rewards)
        assert len(agent.ret_buf) == len(rewards)
        # Normalized advantages should have mean ≈ 0
        assert abs(np.mean(agent.adv_buf)) < 0.1
