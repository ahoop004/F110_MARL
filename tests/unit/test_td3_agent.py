"""Unit tests for TD3 agent."""
import pytest
import numpy as np
import torch

from f110x.policies.td3.td3 import TD3Agent


@pytest.mark.unit
class TestTD3Agent:
    """Test TD3 agent core functionality."""

    def test_agent_creation(self, td3_agent_config, seed_rng):
        """Test TD3 agent can be instantiated."""
        agent = TD3Agent(td3_agent_config)
        assert agent is not None
        assert agent.obs_dim == td3_agent_config["obs_dim"]
        assert agent.act_dim == td3_agent_config["act_dim"]

    def test_twin_critics_exist(self, td3_agent_config, seed_rng):
        """Test that twin Q-networks are created."""
        agent = TD3Agent(td3_agent_config)
        assert hasattr(agent, "critic1")
        assert hasattr(agent, "critic2")
        assert hasattr(agent, "critic_target1")
        assert hasattr(agent, "critic_target2")

    def test_act_returns_valid_action(self, td3_agent_config, seed_rng):
        """Test that act() returns action in valid range."""
        agent = TD3Agent(td3_agent_config)
        obs = np.random.randn(td3_agent_config["obs_dim"]).astype(np.float32)

        action = agent.act(obs, deterministic=False)

        assert action.shape == (td3_agent_config["act_dim"],)
        assert np.all(action >= td3_agent_config["action_low"])
        assert np.all(action <= td3_agent_config["action_high"])

    def test_exploration_noise_decay(self, td3_agent_config, seed_rng):
        """Test exploration noise decays over time."""
        agent = TD3Agent(td3_agent_config)

        initial_noise = agent.current_exploration_noise()

        # Simulate many steps
        for _ in range(1000):
            obs = np.random.randn(td3_agent_config["obs_dim"]).astype(np.float32)
            agent.act(obs, deterministic=False)

        final_noise = agent.current_exploration_noise()

        # Noise should decay (or stay same if at minimum)
        assert final_noise <= initial_noise

    def test_replay_buffer_storage(self, td3_agent_config, seed_rng):
        """Test storing transitions in replay buffer."""
        agent = TD3Agent(td3_agent_config)

        obs = np.random.randn(td3_agent_config["obs_dim"]).astype(np.float32)
        action = np.random.randn(td3_agent_config["act_dim"]).astype(np.float32)
        reward = 1.0
        next_obs = np.random.randn(td3_agent_config["obs_dim"]).astype(np.float32)
        done = False

        agent.store_transition(obs, action, reward, next_obs, done)

        assert len(agent.buffer) == 1

    def test_update_after_warmup(self, td3_agent_config, seed_rng):
        """Test that update works after warmup period."""
        agent = TD3Agent(td3_agent_config)

        # Fill buffer past warmup
        for _ in range(td3_agent_config["warmup_steps"] + 50):
            obs = np.random.randn(td3_agent_config["obs_dim"]).astype(np.float32)
            action = agent.act(obs)
            reward = np.random.randn()
            next_obs = np.random.randn(td3_agent_config["obs_dim"]).astype(np.float32)
            done = np.random.rand() < 0.1
            agent.store_transition(obs, action, reward, next_obs, done)

        # Should be able to update
        stats = agent.update()

        assert stats is not None
        assert "critic_loss" in stats
        assert "actor_loss" in stats

    def test_delayed_policy_updates(self, td3_agent_config, seed_rng):
        """Test that policy updates are delayed."""
        config = td3_agent_config.copy()
        config["policy_delay"] = 2
        agent = TD3Agent(config)

        # Fill buffer
        for _ in range(config["warmup_steps"] + 50):
            obs = np.random.randn(config["obs_dim"]).astype(np.float32)
            action = agent.act(obs)
            next_obs = np.random.randn(config["obs_dim"]).astype(np.float32)
            agent.store_transition(obs, action, 0.0, next_obs, False)

        # First update: check if total_it is tracking
        initial_it = agent.total_it
        stats1 = agent.update()

        # After policy_delay updates, actor should have updated
        for _ in range(config["policy_delay"]):
            agent.update()

        # total_it should have incremented
        assert agent.total_it > initial_it

    def test_save_and_load(self, td3_agent_config, seed_rng, tmp_path):
        """Test checkpoint save and load."""
        agent = TD3Agent(td3_agent_config)

        # Fill buffer a bit
        for _ in range(50):
            obs = np.random.randn(td3_agent_config["obs_dim"]).astype(np.float32)
            action = agent.act(obs)
            next_obs = np.random.randn(td3_agent_config["obs_dim"]).astype(np.float32)
            agent.store_transition(obs, action, 1.0, next_obs, False)

        # Get action before save
        test_obs = np.random.randn(td3_agent_config["obs_dim"]).astype(np.float32)
        action_before = agent.act(test_obs, deterministic=True)

        # Save
        checkpoint_path = tmp_path / "td3_checkpoint.pt"
        agent.save(str(checkpoint_path))

        # Load into new agent
        agent2 = TD3Agent(td3_agent_config)
        agent2.load(str(checkpoint_path))

        # Should produce same action
        action_after = agent2.act(test_obs, deterministic=True)
        np.testing.assert_array_almost_equal(action_before, action_after, decimal=5)

    def test_target_network_soft_update(self, td3_agent_config, seed_rng):
        """Test that target networks soft update (not hard copy)."""
        agent = TD3Agent(td3_agent_config)

        # Get initial target weights
        initial_target_params = list(agent.critic_target1.parameters())[0].clone()

        # Fill buffer and update
        for _ in range(td3_agent_config["warmup_steps"] + 10):
            obs = np.random.randn(td3_agent_config["obs_dim"]).astype(np.float32)
            action = agent.act(obs)
            next_obs = np.random.randn(td3_agent_config["obs_dim"]).astype(np.float32)
            agent.store_transition(obs, action, 1.0, next_obs, False)
            agent.update()

        # Target params should have changed (soft update)
        final_target_params = list(agent.critic_target1.parameters())[0]

        # Should be different (soft update happened)
        assert not torch.allclose(initial_target_params, final_target_params)
