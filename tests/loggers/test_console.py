"""Tests for console logger."""

import pytest
from unittest.mock import Mock, patch
from src.loggers import ConsoleLogger


class TestConsoleLogger:
    """Test ConsoleLogger class."""

    def test_initialization(self):
        """Test logger initialization."""
        logger = ConsoleLogger(verbose=True)
        assert logger.verbose is True
        assert logger.console is not None

    def test_initialization_not_verbose(self):
        """Test logger initialization with verbose=False."""
        logger = ConsoleLogger(verbose=False)
        assert logger.verbose is False

    @patch('src.loggers.console.Console')
    def test_print_header(self, mock_console_class):
        """Test printing header."""
        mock_console = Mock()
        mock_console_class.return_value = mock_console

        logger = ConsoleLogger()
        logger.print_header("Test Title", "Test Subtitle")

        # Should call rule and print
        assert mock_console.rule.called
        assert mock_console.print.called

    @patch('src.loggers.console.Console')
    def test_print_config(self, mock_console_class):
        """Test printing config."""
        mock_console = Mock()
        mock_console_class.return_value = mock_console

        logger = ConsoleLogger()
        config = {'algorithm': 'ppo', 'lr': 0.0005}
        logger.print_config(config)

        # Should create and print table
        assert mock_console.print.called

    @patch('src.loggers.console.Console')
    def test_log_episode_verbose(self, mock_console_class):
        """Test episode logging in verbose mode."""
        mock_console = Mock()
        mock_console_class.return_value = mock_console

        logger = ConsoleLogger(verbose=True)
        logger.log_episode(
            episode=0,
            outcome="target_crash",
            reward=125.5,
            steps=450,
            success_rate=0.75,
            avg_reward=100.0,
        )

        # Should print the episode
        assert mock_console.print.called

    @patch('src.loggers.console.Console')
    def test_log_episode_not_verbose(self, mock_console_class):
        """Test episode logging with verbose=False."""
        mock_console = Mock()
        mock_console_class.return_value = mock_console

        logger = ConsoleLogger(verbose=False)
        logger.log_episode(
            episode=0,
            outcome="target_crash",
            reward=125.5,
            steps=450,
        )

        # Should not print anything
        assert not mock_console.print.called

    @patch('src.loggers.console.Console')
    def test_print_summary(self, mock_console_class):
        """Test printing summary."""
        mock_console = Mock()
        mock_console_class.return_value = mock_console

        logger = ConsoleLogger()
        stats = {
            'total_episodes': 1500,
            'success_rate': 0.68,
            'avg_reward': 82.4,
        }
        logger.print_summary(stats)

        # Should print table
        assert mock_console.print.called

    @patch('src.loggers.console.Console')
    def test_print_outcome_distribution(self, mock_console_class):
        """Test printing outcome distribution."""
        mock_console = Mock()
        mock_console_class.return_value = mock_console

        logger = ConsoleLogger()
        outcome_counts = {
            'target_crash': 1020,
            'self_crash': 350,
            'timeout': 130,
        }
        logger.print_outcome_distribution(outcome_counts)

        # Should print table
        assert mock_console.print.called

    @patch('src.loggers.console.Console')
    def test_print_outcome_distribution_empty(self, mock_console_class):
        """Test printing empty outcome distribution."""
        mock_console = Mock()
        mock_console_class.return_value = mock_console

        logger = ConsoleLogger()
        outcome_counts = {}
        logger.print_outcome_distribution(outcome_counts)

        # Should not print anything for empty counts
        assert not mock_console.print.called

    @patch('src.loggers.console.Console')
    def test_create_progress(self, mock_console_class):
        """Test progress bar creation."""
        mock_console = Mock()
        mock_console_class.return_value = mock_console

        logger = ConsoleLogger()
        progress = logger.create_progress(1500, "Training")

        # Should return a Progress instance
        assert progress is not None

    @patch('src.loggers.console.Console')
    def test_print_success(self, mock_console_class):
        """Test success message."""
        mock_console = Mock()
        mock_console_class.return_value = mock_console

        logger = ConsoleLogger()
        logger.print_success("Test success")

        assert mock_console.print.called

    @patch('src.loggers.console.Console')
    def test_print_warning(self, mock_console_class):
        """Test warning message."""
        mock_console = Mock()
        mock_console_class.return_value = mock_console

        logger = ConsoleLogger()
        logger.print_warning("Test warning")

        assert mock_console.print.called

    @patch('src.loggers.console.Console')
    def test_print_error(self, mock_console_class):
        """Test error message."""
        mock_console = Mock()
        mock_console_class.return_value = mock_console

        logger = ConsoleLogger()
        logger.print_error("Test error")

        assert mock_console.print.called

    @patch('src.loggers.console.Console')
    def test_print_info(self, mock_console_class):
        """Test info message."""
        mock_console = Mock()
        mock_console_class.return_value = mock_console

        logger = ConsoleLogger()
        logger.print_info("Test info")

        assert mock_console.print.called


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
