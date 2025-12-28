"""Tests for CLI entry point."""

import pytest
import tempfile
import yaml
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Import functions from run.py
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from run import (
    parse_args,
    resolve_cli_overrides,
    initialize_loggers,
    print_scenario_summary,
)


@pytest.fixture
def valid_scenario():
    """Valid scenario for testing."""
    return {
        'experiment': {
            'name': 'test_experiment',
            'episodes': 100,
            'seed': 42,
        },
        'environment': {
            'map': 'maps/test.yaml',
            'num_agents': 2,
        },
        'agents': {
            'car_0': {
                'role': 'attacker',
                'algorithm': 'ppo',
            },
            'car_1': {
                'role': 'defender',
                'algorithm': 'ftg',
            },
        },
        'wandb': {
            'enabled': False,
            'project': 'test-project',
        },
    }


class TestParseArgs:
    """Test command-line argument parsing."""

    def test_parse_minimal_args(self):
        """Test parsing minimal required arguments."""
        with patch('sys.argv', ['run.py', '--scenario', 'test.yaml']):
            args = parse_args()
            assert args.scenario == 'test.yaml'
            assert args.wandb is False
            assert args.no_wandb is False

    def test_parse_with_wandb(self):
        """Test parsing with --wandb flag."""
        with patch('sys.argv', ['run.py', '--scenario', 'test.yaml', '--wandb']):
            args = parse_args()
            assert args.wandb is True

    def test_parse_with_no_wandb(self):
        """Test parsing with --no-wandb flag."""
        with patch('sys.argv', ['run.py', '--scenario', 'test.yaml', '--no-wandb']):
            args = parse_args()
            assert args.no_wandb is True

    def test_parse_with_seed(self):
        """Test parsing with custom seed."""
        with patch('sys.argv', ['run.py', '--scenario', 'test.yaml', '--seed', '123']):
            args = parse_args()
            assert args.seed == 123

    def test_parse_with_episodes(self):
        """Test parsing with custom episodes."""
        with patch('sys.argv', ['run.py', '--scenario', 'test.yaml', '--episodes', '500']):
            args = parse_args()
            assert args.episodes == 500

    def test_parse_with_render(self):
        """Test parsing with --render flag."""
        with patch('sys.argv', ['run.py', '--scenario', 'test.yaml', '--render']):
            args = parse_args()
            assert args.render is True

    def test_parse_with_quiet(self):
        """Test parsing with --quiet flag."""
        with patch('sys.argv', ['run.py', '--scenario', 'test.yaml', '--quiet']):
            args = parse_args()
            assert args.quiet is True


class TestResolveCLIOverrides:
    """Test CLI argument override resolution."""

    def test_override_seed(self, valid_scenario):
        """Test overriding seed from CLI."""
        args = Mock(
            seed=999,
            episodes=None,
            wandb=False,
            no_wandb=False,
            render=False,
            no_render=False,
        )

        scenario = resolve_cli_overrides(valid_scenario.copy(), args)
        assert scenario['experiment']['seed'] == 999

    def test_override_episodes(self, valid_scenario):
        """Test overriding episodes from CLI."""
        args = Mock(
            seed=None,
            episodes=500,
            wandb=False,
            no_wandb=False,
            render=False,
            no_render=False,
        )

        scenario = resolve_cli_overrides(valid_scenario.copy(), args)
        assert scenario['experiment']['episodes'] == 500

    def test_override_enable_wandb(self, valid_scenario):
        """Test enabling W&B from CLI."""
        args = Mock(
            seed=None,
            episodes=None,
            wandb=True,
            no_wandb=False,
            render=False,
            no_render=False,
        )

        scenario = resolve_cli_overrides(valid_scenario.copy(), args)
        assert scenario['wandb']['enabled'] is True

    def test_override_disable_wandb(self, valid_scenario):
        """Test disabling W&B from CLI."""
        args = Mock(
            seed=None,
            episodes=None,
            wandb=False,
            no_wandb=True,
            render=False,
            no_render=False,
        )

        scenario = resolve_cli_overrides(valid_scenario.copy(), args)
        assert scenario['wandb']['enabled'] is False

    def test_override_enable_render(self, valid_scenario):
        """Test enabling render from CLI."""
        args = Mock(
            seed=None,
            episodes=None,
            wandb=False,
            no_wandb=False,
            render=True,
            no_render=False,
        )

        scenario = resolve_cli_overrides(valid_scenario.copy(), args)
        assert scenario['environment']['render'] is True

    def test_override_multiple(self, valid_scenario):
        """Test overriding multiple values."""
        args = Mock(
            seed=999,
            episodes=500,
            wandb=True,
            no_wandb=False,
            render=True,
            no_render=False,
        )

        scenario = resolve_cli_overrides(valid_scenario.copy(), args)
        assert scenario['experiment']['seed'] == 999
        assert scenario['experiment']['episodes'] == 500
        assert scenario['wandb']['enabled'] is True
        assert scenario['environment']['render'] is True


class TestInitializeLoggers:
    """Test logger initialization."""

    @patch('v2.run.WandbLogger')
    @patch('v2.run.ConsoleLogger')
    def test_initialize_with_wandb_enabled(self, mock_console, mock_wandb, valid_scenario):
        """Test initializing loggers with W&B enabled."""
        valid_scenario['wandb']['enabled'] = True
        args = Mock(quiet=False)

        wandb_logger, console_logger = initialize_loggers(valid_scenario, args)

        # Should create W&B logger
        mock_wandb.assert_called_once()
        assert wandb_logger is not None

        # Should create console logger
        mock_console.assert_called()
        assert console_logger is not None

    @patch('v2.run.ConsoleLogger')
    def test_initialize_with_wandb_disabled(self, mock_console, valid_scenario):
        """Test initializing loggers with W&B disabled."""
        valid_scenario['wandb']['enabled'] = False
        args = Mock(quiet=False)

        wandb_logger, console_logger = initialize_loggers(valid_scenario, args)

        # Should not create W&B logger
        assert wandb_logger is None

        # Should create console logger
        mock_console.assert_called_once()
        assert console_logger is not None

    @patch('v2.run.ConsoleLogger')
    def test_initialize_with_quiet(self, mock_console, valid_scenario):
        """Test initializing loggers with quiet mode."""
        args = Mock(quiet=True)

        initialize_loggers(valid_scenario, args)

        # Console logger should be initialized with verbose=False
        mock_console.assert_called_with(verbose=False)


class TestPrintScenarioSummary:
    """Test scenario summary printing."""

    @patch('v2.run.ConsoleLogger')
    def test_print_summary(self, mock_console_class, valid_scenario):
        """Test printing scenario summary."""
        mock_console = Mock()
        mock_console_class.return_value = mock_console

        console = mock_console_class()
        print_scenario_summary(valid_scenario, console)

        # Should call print_header
        assert console.print_header.called


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
