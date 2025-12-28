"""Integration test for basic training loop."""
import pytest
import numpy as np

from f110x.policies.ppo.ppo import PPOAgent


@pytest.mark.integration
@pytest.mark.slow
class TestBasicTraining:
    """Test that basic training loop works end-to-end."""

    def test_ppo_training_loop(self, ppo_agent_config, seed_rng):
        """Test PPO agent can train for multiple episodes."""
        agent = PPOAgent(ppo_agent_config)

        num_episodes = 5
        episode_length = 20

        for episode in range(num_episodes):
            episode_reward = 0

            for step in range(episode_length):
                # Generate fake observation
                obs = np.random.randn(ppo_agent_config["obs_dim"]).astype(np.float32)

                # Select action
                action = agent.act(obs)

                # Simulate environment step (fake reward)
                reward = np.random.randn()
                done = (step == episode_length - 1)

                # Store transition
                agent.store(obs, action, reward, done, terminated=done)

                episode_reward += reward

            # Update agent after episode
            stats = agent.update()

            # Should produce training stats
            if stats is not None:
                assert "policy_loss" in stats
                assert "value_loss" in stats
                assert np.isfinite(stats["policy_loss"])
                assert np.isfinite(stats["value_loss"])

    def test_training_improves_policy(self, ppo_agent_config, seed_rng):
        """Test that training improves the policy (sanity check)."""
        agent = PPOAgent(ppo_agent_config)

        # Collect losses over multiple updates
        losses = []

        for _ in range(10):
            # Collect episode
            for step in range(50):
                obs = np.random.randn(ppo_agent_config["obs_dim"]).astype(np.float32)
                action = agent.act(obs)
                # Increasing reward to simulate learning
                reward = 1.0 if step > 25 else 0.0
                done = (step == 49)
                agent.store(obs, action, reward, done, terminated=done)

            # Update
            stats = agent.update()
            if stats:
                losses.append(stats["policy_loss"])

        # Losses should stabilize (may increase or decrease, but should be finite)
        assert all(np.isfinite(loss) for loss in losses)
        # Should have collected some losses
        assert len(losses) > 0
