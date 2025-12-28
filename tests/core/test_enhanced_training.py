"""Tests for enhanced training loop."""

import pytest
from unittest.mock import Mock, MagicMock, patch
import numpy as np

from src.core.enhanced_training import EnhancedTrainingLoop
from src.metrics import EpisodeOutcome


@pytest.fixture
def mock_env():
    """Create mock PettingZoo environment."""
    env = Mock()
    env.reset.return_value = (
        {
            'car_0': np.zeros(738),
            'car_1': np.zeros(738),
        },
        {}
    )
    env.step.return_value = (
        {  # next_obs
            'car_0': np.zeros(738),
            'car_1': np.zeros(738),
        },
        {  # rewards
            'car_0': 1.0,
            'car_1': 0.5,
        },
        {  # terminations
            'car_0': True,
            'car_1': True,
        },
        {  # truncations
            'car_0': False,
            'car_1': False,
        },
        {  # info
            'collision': False,
            'target_collision': True,
        }
    )
    return env


@pytest.fixture
def mock_agents():
    """Create mock agents."""
    agent0 = Mock()
    agent0.act.return_value = 0
    agent0.store_experience = Mock()

    agent1 = Mock()
    agent1.act.return_value = 1
    agent1.store_experience = Mock()

    return {
        'car_0': agent0,
        'car_1': agent1,
    }


@pytest.fixture
def mock_reward_strategy():
    """Create mock reward strategy."""
    strategy = Mock()
    strategy.reset = Mock()
    strategy.compute.return_value = (100.0, {'terminal/success': 60.0, 'pressure/bonus': 0.12})
    return strategy


class TestEnhancedTrainingLoop:
    """Test EnhancedTrainingLoop class."""

    def test_initialization(self, mock_env, mock_agents):
        """Test loop initialization."""
        loop = EnhancedTrainingLoop(
            env=mock_env,
            agents=mock_agents,
        )

        assert loop.env == mock_env
        assert loop.agents == mock_agents
        assert len(loop.metrics_trackers) == 2

    def test_initialization_with_loggers(self, mock_env, mock_agents):
        """Test initialization with loggers."""
        wandb_logger = Mock()
        console_logger = Mock()

        loop = EnhancedTrainingLoop(
            env=mock_env,
            agents=mock_agents,
            wandb_logger=wandb_logger,
            console_logger=console_logger,
        )

        assert loop.wandb_logger == wandb_logger
        assert loop.console_logger == console_logger

    def test_initialization_with_custom_rewards(self, mock_env, mock_agents, mock_reward_strategy):
        """Test initialization with custom reward strategies."""
        agent_rewards = {
            'car_0': mock_reward_strategy,
        }

        loop = EnhancedTrainingLoop(
            env=mock_env,
            agents=mock_agents,
            agent_rewards=agent_rewards,
        )

        assert 'car_0' in loop.agent_rewards
        assert loop.agent_rewards['car_0'] == mock_reward_strategy

    def test_run_single_episode(self, mock_env, mock_agents):
        """Test running a single episode."""
        loop = EnhancedTrainingLoop(
            env=mock_env,
            agents=mock_agents,
        )

        stats = loop.run(episodes=1)

        # Should have stats for both agents
        assert 'car_0' in stats
        assert 'car_1' in stats

        # Each agent should have 1 episode recorded
        assert len(loop.metrics_trackers['car_0'].episodes) == 1
        assert len(loop.metrics_trackers['car_1'].episodes) == 1

    def test_run_multiple_episodes(self, mock_env, mock_agents):
        """Test running multiple episodes."""
        loop = EnhancedTrainingLoop(
            env=mock_env,
            agents=mock_agents,
        )

        loop.run(episodes=5)

        # Should have 5 episodes recorded
        assert len(loop.metrics_trackers['car_0'].episodes) == 5

    def test_custom_rewards_used(self, mock_env, mock_agents, mock_reward_strategy):
        """Test that custom rewards are used when provided."""
        agent_rewards = {
            'car_0': mock_reward_strategy,
        }

        loop = EnhancedTrainingLoop(
            env=mock_env,
            agents=mock_agents,
            agent_rewards=agent_rewards,
        )

        loop.run(episodes=1)

        # Reward strategy should have been reset
        mock_reward_strategy.reset.assert_called()

        # Reward strategy should have been used to compute rewards
        mock_reward_strategy.compute.assert_called()

    def test_environment_rewards_used_as_fallback(self, mock_env, mock_agents):
        """Test that environment rewards are used when no custom rewards."""
        loop = EnhancedTrainingLoop(
            env=mock_env,
            agents=mock_agents,
        )

        loop.run(episodes=1)

        # Should use environment rewards (1.0 for car_0)
        tracker = loop.metrics_trackers['car_0']
        episode = tracker.episodes[0]
        assert episode.total_reward == 1.0

    def test_metrics_tracking(self, mock_env, mock_agents):
        """Test that metrics are tracked correctly."""
        loop = EnhancedTrainingLoop(
            env=mock_env,
            agents=mock_agents,
        )

        loop.run(episodes=3)

        tracker = loop.metrics_trackers['car_0']

        # Should have 3 episodes
        assert len(tracker.episodes) == 3

        # Each episode should have outcome
        for episode in tracker.episodes:
            assert isinstance(episode.outcome, EpisodeOutcome)

    @patch('src.core.enhanced_training.determine_outcome')
    def test_outcome_determination(self, mock_determine, mock_env, mock_agents):
        """Test that outcome is determined correctly."""
        mock_determine.return_value = EpisodeOutcome.TARGET_CRASH

        loop = EnhancedTrainingLoop(
            env=mock_env,
            agents=mock_agents,
        )

        loop.run(episodes=1)

        # determine_outcome should have been called
        mock_determine.assert_called()

        # Outcome should be recorded
        episode = loop.metrics_trackers['car_0'].episodes[0]
        assert episode.outcome == EpisodeOutcome.TARGET_CRASH

    def test_wandb_logging(self, mock_env, mock_agents):
        """Test W&B logging integration."""
        wandb_logger = Mock()

        loop = EnhancedTrainingLoop(
            env=mock_env,
            agents=mock_agents,
            wandb_logger=wandb_logger,
        )

        loop.run(episodes=2)

        # Should have logged episodes to W&B
        assert wandb_logger.log_episode.called
        # 2 episodes * 2 agents = 4 logs
        assert wandb_logger.log_episode.call_count == 4

    def test_console_logging(self, mock_env, mock_agents):
        """Test console logging integration."""
        console_logger = Mock()
        console_logger.create_progress.return_value.__enter__ = Mock()
        console_logger.create_progress.return_value.__exit__ = Mock()
        console_logger.create_progress.return_value.add_task = Mock(return_value=0)
        console_logger.create_progress.return_value.update = Mock()

        loop = EnhancedTrainingLoop(
            env=mock_env,
            agents=mock_agents,
            console_logger=console_logger,
        )

        loop.run(episodes=2)

        # Should have created progress bar
        console_logger.create_progress.assert_called()

        # Should have logged episodes
        assert console_logger.log_episode.called

    def test_rolling_stats_computed(self, mock_env, mock_agents):
        """Test that rolling stats are computed."""
        loop = EnhancedTrainingLoop(
            env=mock_env,
            agents=mock_agents,
            rolling_window=10,
        )

        loop.run(episodes=15)

        tracker = loop.metrics_trackers['car_0']

        # Get rolling stats with window
        stats = tracker.get_rolling_stats(window=10)

        # Should have stats for last 10 episodes
        assert stats['total_episodes'] == 10
        assert 'success_rate' in stats
        assert 'avg_reward' in stats

    def test_agent_experience_storage(self, mock_env, mock_agents):
        """Test that agent experiences are stored."""
        loop = EnhancedTrainingLoop(
            env=mock_env,
            agents=mock_agents,
        )

        loop.run(episodes=1)

        # Agents should have stored experiences
        mock_agents['car_0'].store_experience.assert_called()
        mock_agents['car_1'].store_experience.assert_called()

    def test_max_steps_per_episode(self, mock_env, mock_agents):
        """Test max steps per episode limit."""
        # Make environment never terminate naturally
        mock_env.step.return_value = (
            {'car_0': np.zeros(738), 'car_1': np.zeros(738)},
            {'car_0': 1.0, 'car_1': 0.5},
            {'car_0': False, 'car_1': False},  # Never done
            {'car_0': False, 'car_1': False},
            {}
        )

        loop = EnhancedTrainingLoop(
            env=mock_env,
            agents=mock_agents,
            max_steps_per_episode=10,  # Low limit
        )

        loop.run(episodes=1)

        # Episode should have been truncated at 10 steps
        episode = loop.metrics_trackers['car_0'].episodes[0]
        assert episode.steps == 10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
