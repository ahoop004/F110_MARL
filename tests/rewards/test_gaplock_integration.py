"""Integration tests for complete Gaplock reward system."""

import pytest
import numpy as np
from src.rewards import load_preset
from src.rewards.gaplock import GaplockReward


class TestGaplockIntegration:
    """Test complete gaplock reward integration."""

    @pytest.fixture
    def reward_simple(self):
        """Create gaplock reward with simple preset."""
        config = load_preset('gaplock_simple')
        return GaplockReward(config)

    @pytest.fixture
    def reward_full(self):
        """Create gaplock reward with full preset."""
        config = load_preset('gaplock_full')
        return GaplockReward(config)

    def test_initialization_simple(self, reward_simple):
        """Test that simple preset initializes correctly."""
        assert reward_simple is not None
        assert reward_simple.composer is not None
        assert len(reward_simple.composer.components) > 0

    def test_initialization_full(self, reward_full):
        """Test that full preset initializes correctly."""
        assert reward_full is not None
        assert reward_full.composer is not None
        # Full should have more components than simple (includes forcing)
        assert len(reward_full.composer.components) >= 7

    def test_reset(self, reward_full):
        """Test that reset propagates to all components."""
        reward_full.reset()
        # Should not raise any errors

    def test_not_done_no_terminal(self, reward_full):
        """Test that terminal rewards only fire when done."""
        step_info = {
            'obs': {
                'pose': [0, 0, 0],
                'velocity': [1, 0],
            },
            'target_obs': {
                'pose': [1, 0, 0],
                'velocity': [1, 0],
                'scans': np.full(720, 5.0),
            },
            'done': False,
            'truncated': False,
            'info': {},
            'timestep': 0.01,
        }

        reward_full.reset()
        total, components = reward_full.compute(step_info)

        # Should have some rewards but no terminal
        assert 'terminal/success' not in components
        assert 'terminal/self_crash' not in components
        assert 'terminal/timeout' not in components

    def test_success_episode(self, reward_full):
        """Test successful episode (target crashes)."""
        step_info = {
            'obs': {
                'pose': [0.5, 0, 0],
                'velocity': [0.8, 0],
                'scans': np.full(720, 5.0),
            },
            'target_obs': {
                'pose': [1, 0, 0],
                'velocity': [0.5, 0],
                'scans': np.full(720, 5.0),
            },
            'done': True,
            'truncated': False,
            'info': {
                'collision': False,
                'target_collision': True,
            },
            'timestep': 0.01,
        }

        reward_full.reset()
        total, components = reward_full.compute(step_info)

        # Should have terminal success reward
        assert 'terminal/success' in components
        assert components['terminal/success'] == 60.0
        assert total >= 60.0  # May have additional rewards

    def test_failure_episode(self, reward_full):
        """Test failed episode (attacker crashes)."""
        step_info = {
            'obs': {
                'pose': [0, 0, 0],
                'velocity': [1, 0],
            },
            'target_obs': {
                'pose': [5, 0, 0],
                'velocity': [1, 0],
            },
            'done': True,
            'truncated': False,
            'info': {
                'collision': True,
                'target_collision': False,
            },
            'timestep': 0.01,
        }

        reward_full.reset()
        total, components = reward_full.compute(step_info)

        # Should have terminal failure penalty
        assert 'terminal/self_crash' in components
        assert components['terminal/self_crash'] == -90.0

    def test_multi_step_episode(self, reward_full):
        """Test reward accumulation over multiple steps."""
        reward_full.reset()

        # Step 1: Approaching target
        step_info_1 = {
            'obs': {'pose': [0, 0, 0], 'velocity': [0.8, 0]},
            'target_obs': {'pose': [2, 0, 0], 'velocity': [0.5, 0], 'scans': np.full(720, 5.0)},
            'done': False,
            'info': {},
            'timestep': 0.01,
        }
        total_1, components_1 = reward_full.compute(step_info_1)

        # Should get distance and speed rewards
        assert total_1 > 0  # Getting closer should be rewarded

        # Step 2: Close to target (in pressure)
        step_info_2 = {
            'obs': {'pose': [0.8, 0, 0], 'velocity': [0.8, 0]},
            'target_obs': {'pose': [1.5, 0, 0], 'velocity': [0.5, 0], 'scans': np.full(720, 5.0)},
            'done': False,
            'info': {},
            'timestep': 0.01,
        }
        total_2, components_2 = reward_full.compute(step_info_2)

        # Should potentially get pressure rewards (if within threshold)
        assert total_2 > 0

        # Step 3: Target crashes (success!)
        step_info_3 = {
            'obs': {'pose': [1.0, 0, 0], 'velocity': [0.8, 0]},
            'target_obs': {'pose': [1.5, 0, 0], 'velocity': [0, 0], 'scans': np.full(720, 0.1)},
            'done': True,
            'info': {'collision': False, 'target_collision': True},
            'timestep': 0.01,
        }
        total_3, components_3 = reward_full.compute(step_info_3)

        # Should get terminal success
        assert 'terminal/success' in components_3
        assert total_3 >= 60.0

    def test_component_breakdown(self, reward_full):
        """Test that component breakdown is provided."""
        step_info = {
            'obs': {
                'pose': [0, 0, 0],
                'velocity': [0.8, 0],
            },
            'target_obs': {
                'pose': [0.8, 0, 0],  # Close
                'velocity': [0.5, 0],
                'scans': np.full(720, 5.0),
            },
            'done': False,
            'info': {},
            'timestep': 0.01,
        }

        reward_full.reset()

        # Run a few steps to accumulate pressure
        for _ in range(10):
            total, components = reward_full.compute(step_info)

        # Should have multiple component types
        assert isinstance(components, dict)
        assert len(components) > 0

        # Components should sum to total
        assert abs(total - sum(components.values())) < 1e-6

    def test_simple_vs_full(self, reward_simple, reward_full):
        """Test that simple has fewer active components than full."""
        step_info = {
            'obs': {'pose': [0, 0, 0], 'velocity': [0.8, 0]},
            'target_obs': {'pose': [1, 0, 0], 'velocity': [0.5, 0], 'scans': np.full(720, 5.0)},
            'done': False,
            'info': {},
            'timestep': 0.01,
        }

        reward_simple.reset()
        reward_full.reset()

        _, components_simple = reward_simple.compute(step_info)
        _, components_full = reward_full.compute(step_info)

        # Simple should not have forcing components
        forcing_keys = [k for k in components_full if k.startswith('forcing/')]
        for key in forcing_keys:
            assert key not in components_simple

    def test_preset_modification(self):
        """Test that preset can be modified with overrides."""
        from rewards import merge_config

        base = load_preset('gaplock_simple')
        overrides = {
            'terminal': {
                'target_crash': 100.0,  # Increase success reward
            }
        }

        config = merge_config(base, overrides)
        reward = GaplockReward(config)

        step_info = {
            'obs': {'pose': [0, 0, 0], 'velocity': [1, 0]},
            'target_obs': {'pose': [1, 0, 0], 'velocity': [1, 0]},
            'done': True,
            'info': {'collision': False, 'target_collision': True},
            'timestep': 0.01,
        }

        reward.reset()
        total, components = reward.compute(step_info)

        # Should have modified success reward
        assert components['terminal/success'] == 100.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
