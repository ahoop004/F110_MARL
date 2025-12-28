"""Tests for terminal rewards."""

import pytest
from rewards.gaplock.terminal import TerminalReward


class TestTerminalReward:
    """Test terminal reward component."""

    @pytest.fixture
    def reward(self):
        """Create terminal reward with default config."""
        config = {
            'target_crash': 60.0,
            'self_crash': -90.0,
            'collision': -90.0,
            'timeout': -20.0,
            'idle_stop': -5.0,
            'target_finish': -20.0,
        }
        return TerminalReward(config)

    def test_not_done(self, reward):
        """Test that no reward is given when episode not done."""
        step_info = {
            'done': False,
            'info': {},
        }

        result = reward.compute(step_info)

        assert result == {}

    def test_target_crash_success(self, reward):
        """Test reward when target crashes (success)."""
        step_info = {
            'done': True,
            'info': {
                'collision': False,
                'target_collision': True,
            },
        }

        result = reward.compute(step_info)

        assert result == {'terminal/success': 60.0}

    def test_self_crash(self, reward):
        """Test penalty when attacker crashes alone."""
        step_info = {
            'done': True,
            'info': {
                'collision': True,
                'target_collision': False,
            },
        }

        result = reward.compute(step_info)

        assert result == {'terminal/self_crash': -90.0}

    def test_mutual_collision(self, reward):
        """Test penalty when both crash."""
        step_info = {
            'done': True,
            'info': {
                'collision': True,
                'target_collision': True,
            },
        }

        result = reward.compute(step_info)

        assert result == {'terminal/collision': -90.0}

    def test_timeout(self, reward):
        """Test penalty for timeout."""
        step_info = {
            'done': True,
            'truncated': True,
            'info': {
                'collision': False,
                'target_collision': False,
            },
        }

        result = reward.compute(step_info)

        assert result == {'terminal/timeout': -20.0}

    def test_idle_stop(self, reward):
        """Test penalty for idle truncation."""
        step_info = {
            'done': True,
            'info': {
                'collision': False,
                'target_collision': False,
                'idle_triggered': True,
            },
        }

        result = reward.compute(step_info)

        assert result == {'terminal/idle_stop': -5.0}

    def test_target_finish(self, reward):
        """Test penalty when target crosses finish line."""
        step_info = {
            'done': True,
            'info': {
                'target_finished': True,
                'collision': False,
                'target_collision': False,
            },
        }

        result = reward.compute(step_info)

        assert result == {'terminal/target_finish': -20.0}

    def test_alternative_key_format(self, reward):
        """Test with car_0/car_1 key format."""
        step_info = {
            'done': True,
            'info': {
                'car_0/collision': False,
                'car_1/collision': True,
            },
        }

        result = reward.compute(step_info)

        assert result == {'terminal/success': 60.0}

    def test_priority_finish_over_crash(self, reward):
        """Test that finish has priority over crashes."""
        step_info = {
            'done': True,
            'info': {
                'target_finished': True,
                'collision': True,  # This should be ignored
                'target_collision': True,
            },
        }

        result = reward.compute(step_info)

        # Finish should take priority
        assert result == {'terminal/target_finish': -20.0}

    def test_custom_values(self):
        """Test with custom reward values."""
        config = {
            'target_crash': 100.0,
            'self_crash': -50.0,
        }
        reward = TerminalReward(config)

        step_info = {
            'done': True,
            'info': {'collision': False, 'target_collision': True},
        }

        result = reward.compute(step_info)

        assert result == {'terminal/success': 100.0}


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
