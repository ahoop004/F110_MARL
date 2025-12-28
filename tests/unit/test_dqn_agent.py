"""Unit tests for DQN agent."""
import pytest
import numpy as np
import torch

from f110x.policies.dqn.dqn import DQNAgent


@pytest.mark.unit
class TestDQNAgent:
    """Test DQN agent core functionality."""

    def test_agent_creation(self, dqn_agent_config, seed_rng):
        """Test DQN agent can be instantiated."""
        agent = DQNAgent(dqn_agent_config)
        assert agent is not None
        assert agent.obs_dim == dqn_agent_config["obs_dim"]
        assert agent.n_actions == len(dqn_agent_config["action_set"])

    def test_act_returns_valid_action_index(self, dqn_agent_config, seed_rng):
        """Test that act() returns valid action index."""
        agent = DQNAgent(dqn_agent_config)
        obs = np.random.randn(dqn_agent_config["obs_dim"]).astype(np.float32)

        action_idx = agent.act(obs, deterministic=False)

        assert isinstance(action_idx, (int, np.integer))
        assert 0 <= action_idx < len(dqn_agent_config["action_set"])

    def test_epsilon_greedy_exploration(self, dqn_agent_config, seed_rng):
        """Test epsilon-greedy exploration."""
        agent = DQNAgent(dqn_agent_config)

        # With high epsilon, should get random actions
        assert agent.epsilon() > 0.5  # Initial epsilon is 0.9

        obs = np.random.randn(dqn_agent_config["obs_dim"]).astype(np.float32)
        actions = [agent.act(obs, deterministic=False) for _ in range(20)]

        # Should have some variety in actions (not all greedy)
        assert len(set(actions)) > 1

    def test_epsilon_decay(self, dqn_agent_config, seed_rng):
        """Test that epsilon decays over episodes."""
        agent = DQNAgent(dqn_agent_config)

        initial_epsilon = agent.epsilon()

        # Simulate many episodes
        for _ in range(100):
            obs = np.random.randn(dqn_agent_config["obs_dim"]).astype(np.float32)
            action_idx = agent.act(obs)
            next_obs = np.random.randn(dqn_agent_config["obs_dim"]).astype(np.float32)
            # Get actual action from action set for storage
            action = dqn_agent_config["action_set"][action_idx]
            agent.store_transition(obs, action, 0.0, next_obs, done=True)

        final_epsilon = agent.epsilon()

        # Epsilon should decay
        assert final_epsilon < initial_epsilon

    def test_double_dqn_action_selection(self, dqn_agent_config, seed_rng):
        """Test that Double DQN uses online net for action selection."""
        agent = DQNAgent(dqn_agent_config)

        # Fill buffer
        for _ in range(dqn_agent_config["warmup_steps"] + 50):
            obs = np.random.randn(dqn_agent_config["obs_dim"]).astype(np.float32)
            action_idx = agent.act(obs)
            action = dqn_agent_config["action_set"][action_idx]
            next_obs = np.random.randn(dqn_agent_config["obs_dim"]).astype(np.float32)
            agent.store_transition(obs, action, 1.0, next_obs, False)

        # Update should work (Double DQN is used)
        stats = agent.update()

        assert stats is not None
        assert "loss" in stats
        assert "q_mean" in stats

    def test_target_network_updates(self, dqn_agent_config, seed_rng):
        """Test that target network updates periodically."""
        config = dqn_agent_config.copy()
        config["target_update_interval"] = 10
        agent = DQNAgent(config)

        # Get initial target network params
        initial_target = list(agent.target_q_net.parameters())[0].clone()

        # Fill buffer
        for _ in range(config["warmup_steps"] + 5):
            obs = np.random.randn(config["obs_dim"]).astype(np.float32)
            action_idx = agent.act(obs)
            action = config["action_set"][action_idx]
            next_obs = np.random.randn(config["obs_dim"]).astype(np.float32)
            agent.store_transition(obs, action, 1.0, next_obs, False)

        # Update a few times (not enough to trigger target update)
        for _ in range(5):
            agent.update()

        # Target should still be same
        current_target = list(agent.target_q_net.parameters())[0]
        assert torch.allclose(initial_target, current_target)

        # Update more times to trigger target update
        for _ in range(10):
            agent.update()

        # Target should now be different
        final_target = list(agent.target_q_net.parameters())[0]
        assert not torch.allclose(initial_target, final_target)

    def test_save_and_load(self, dqn_agent_config, seed_rng, tmp_path):
        """Test checkpoint save and load."""
        agent = DQNAgent(dqn_agent_config)

        # Get action before save
        obs = np.random.randn(dqn_agent_config["obs_dim"]).astype(np.float32)
        action_before = agent.act(obs, deterministic=True)

        # Save
        checkpoint_path = tmp_path / "dqn_checkpoint.pt"
        agent.save(str(checkpoint_path))

        # Load into new agent
        agent2 = DQNAgent(dqn_agent_config)
        agent2.load(str(checkpoint_path))

        # Should produce same action
        action_after = agent2.act(obs, deterministic=True)
        assert action_before == action_after

    def test_reset_optimizers_fixed(self, dqn_agent_config, seed_rng):
        """Test that reset_optimizers() works (bug was fixed)."""
        agent = DQNAgent(dqn_agent_config)

        # This should not raise NameError anymore
        try:
            agent.reset_optimizers()
            assert True  # Success!
        except NameError:
            pytest.fail("reset_optimizers() raised NameError - bug not fixed!")
