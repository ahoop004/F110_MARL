"""Tests for W&B logger."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.loggers import WandbLogger
from src.metrics import EpisodeMetrics, EpisodeOutcome


class TestWandbLogger:
    """Test WandbLogger class."""

    @patch('src.loggers.wandb_logger.wandb')
    def test_initialization_online(self, mock_wandb):
        """Test logger initialization in online mode."""
        logger = WandbLogger(
            project="test-project",
            config={"algorithm": "ppo"},
            tags=["test"],
        )

        assert logger.enabled is True
        mock_wandb.init.assert_called_once()
        call_kwargs = mock_wandb.init.call_args[1]
        assert call_kwargs['project'] == "test-project"
        assert call_kwargs['tags'] == ["test"]

    @patch('src.loggers.wandb_logger.wandb')
    def test_initialization_disabled(self, mock_wandb):
        """Test logger initialization in disabled mode."""
        logger = WandbLogger(
            project="test-project",
            mode="disabled",
        )

        assert logger.enabled is False
        mock_wandb.init.assert_not_called()

    @patch('src.loggers.wandb_logger.wandb')
    def test_flatten_config_simple(self, mock_wandb):
        """Test config flattening with simple dict."""
        config = {
            'algorithm': 'ppo',
            'lr': 0.0005,
            'gamma': 0.995,
        }

        flat = WandbLogger._flatten_config(config)

        assert flat == {
            'algorithm': 'ppo',
            'lr': 0.0005,
            'gamma': 0.995,
        }

    @patch('src.loggers.wandb_logger.wandb')
    def test_flatten_config_nested(self, mock_wandb):
        """Test config flattening with nested dict."""
        config = {
            'agent': {
                'lr': 0.0005,
                'gamma': 0.995,
            },
            'reward': {
                'terminal': {
                    'target_crash': 60.0,
                    'self_crash': -90.0,
                },
                'pressure': {
                    'enabled': True,
                },
            },
        }

        flat = WandbLogger._flatten_config(config)

        assert flat == {
            'agent/lr': 0.0005,
            'agent/gamma': 0.995,
            'reward/terminal/target_crash': 60.0,
            'reward/terminal/self_crash': -90.0,
            'reward/pressure/enabled': True,
        }

    @patch('src.loggers.wandb_logger.wandb')
    def test_log_episode_enabled(self, mock_wandb):
        """Test episode logging when enabled."""
        logger = WandbLogger(project="test", mode="online")

        metrics = EpisodeMetrics(
            episode=0,
            outcome=EpisodeOutcome.TARGET_CRASH,
            total_reward=125.5,
            steps=450,
            reward_components={'terminal/success': 60.0},
        )

        rolling_stats = {
            'success_rate': 0.75,
            'avg_reward': 100.0,
            'outcome_counts': {'target_crash': 3, 'self_crash': 1},
        }

        logger.log_episode(0, metrics, rolling_stats)

        # Should have called wandb.log
        mock_wandb.log.assert_called_once()
        logged_data = mock_wandb.log.call_args[0][0]

        # Check episode metrics
        assert logged_data['episode'] == 0
        assert logged_data['total_reward'] == 125.5
        assert logged_data['steps'] == 450
        assert logged_data['reward/terminal/success'] == 60.0

        # Check rolling stats
        assert logged_data['rolling/success_rate'] == 0.75
        assert logged_data['rolling/avg_reward'] == 100.0

        # Check outcome counts
        assert logged_data['rolling/outcomes/target_crash'] == 3
        assert logged_data['rolling/outcomes/self_crash'] == 1

    @patch('src.loggers.wandb_logger.wandb')
    def test_log_episode_with_agent_id(self, mock_wandb):
        """Test episode logging with agent_id namespacing."""
        logger = WandbLogger(project="test", mode="online")

        metrics = EpisodeMetrics(
            episode=0,
            outcome=EpisodeOutcome.TARGET_CRASH,
            total_reward=125.5,
            steps=450,
            reward_components={'terminal/success': 60.0},
        )

        rolling_stats = {
            'success_rate': 0.75,
            'avg_reward': 100.0,
            'outcome_counts': {'target_crash': 3, 'self_crash': 1},
            'outcome_rates': {'target_crash': 0.75, 'self_crash': 0.25},
        }

        logger.log_episode(0, metrics, rolling_stats, agent_id='car_0')

        # Should have called wandb.log
        mock_wandb.log.assert_called_once()
        logged_data = mock_wandb.log.call_args[0][0]

        # Episode should NOT be namespaced (global counter)
        assert logged_data['episode'] == 0

        # Episode metrics should be namespaced
        assert logged_data['car_0/total_reward'] == 125.5
        assert logged_data['car_0/steps'] == 450
        assert logged_data['car_0/outcome'] == 'target_crash'
        assert logged_data['car_0/reward/terminal/success'] == 60.0

        # Rolling stats should be namespaced
        assert logged_data['car_0/rolling/success_rate'] == 0.75
        assert logged_data['car_0/rolling/avg_reward'] == 100.0

        # Outcome counts/rates should be namespaced
        assert logged_data['car_0/rolling/outcomes/target_crash'] == 3
        assert logged_data['car_0/rolling/outcomes/self_crash'] == 1
        assert logged_data['car_0/rolling/outcome_rates/target_crash'] == 0.75
        assert logged_data['car_0/rolling/outcome_rates/self_crash'] == 0.25

    @patch('src.loggers.wandb_logger.wandb')
    def test_log_episode_disabled(self, mock_wandb):
        """Test that logging does nothing when disabled."""
        logger = WandbLogger(project="test", mode="disabled")

        metrics = EpisodeMetrics(
            episode=0,
            outcome=EpisodeOutcome.TARGET_CRASH,
            total_reward=125.5,
            steps=450,
        )

        logger.log_episode(0, metrics)

        # Should not call wandb.log
        mock_wandb.log.assert_not_called()

    @patch('src.loggers.wandb_logger.wandb')
    def test_log_metrics(self, mock_wandb):
        """Test arbitrary metrics logging."""
        logger = WandbLogger(project="test", mode="online")

        metrics = {'custom_metric': 42.0, 'another_metric': 'test'}
        logger.log_metrics(metrics, step=100)

        mock_wandb.log.assert_called_once_with(metrics, step=100)

    @patch('src.loggers.wandb_logger.wandb')
    def test_log_component_stats(self, mock_wandb):
        """Test component statistics logging."""
        logger = WandbLogger(project="test", mode="online")

        component_stats = {
            'terminal/success': {
                'mean': 60.0,
                'std': 0.0,
                'count': 10,
            },
            'pressure/bonus': {
                'mean': 0.12,
                'std': 0.02,
                'count': 8,
            },
        }

        logger.log_component_stats(component_stats, step=100)

        mock_wandb.log.assert_called_once()
        logged_data = mock_wandb.log.call_args[0][0]

        assert logged_data['components/terminal/success/mean'] == 60.0
        assert logged_data['components/terminal/success/std'] == 0.0
        assert logged_data['components/pressure/bonus/mean'] == 0.12

    @patch('src.loggers.wandb_logger.wandb')
    def test_finish(self, mock_wandb):
        """Test finishing the run."""
        logger = WandbLogger(project="test", mode="online")
        logger.finish()

        mock_wandb.finish.assert_called_once()

    @patch('src.loggers.wandb_logger.wandb')
    def test_finish_disabled(self, mock_wandb):
        """Test that finish does nothing when disabled."""
        logger = WandbLogger(project="test", mode="disabled")
        logger.finish()

        mock_wandb.finish.assert_not_called()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
