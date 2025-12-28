"""Tests for reward system base infrastructure."""

import pytest
from rewards.base import RewardComponent, RewardStrategy
from rewards.composer import ComposedReward
from rewards.presets import load_preset, merge_config


# Mock reward component for testing
class MockRewardComponent:
    """Simple mock component that returns fixed rewards."""

    def __init__(self, rewards: dict):
        self.rewards = rewards
        self.reset_called = False

    def reset(self):
        self.reset_called = True

    def compute(self, step_info: dict) -> dict:
        return self.rewards.copy()


class TestProtocols:
    """Test that protocols are correctly defined."""

    def test_reward_component_protocol(self):
        """Test RewardComponent protocol compliance."""
        mock = MockRewardComponent({'test/reward': 1.0})

        # Should be recognized as RewardComponent
        assert isinstance(mock, RewardComponent)

        # Should have compute method
        result = mock.compute({})
        assert result == {'test/reward': 1.0}

    def test_reward_strategy_protocol(self):
        """Test RewardStrategy protocol compliance."""
        components = [MockRewardComponent({'test/reward': 1.0})]
        strategy = ComposedReward(components)

        # Should be recognized as RewardStrategy
        assert isinstance(strategy, RewardStrategy)

        # Should have reset and compute methods
        strategy.reset()
        total, breakdown = strategy.compute({})
        assert isinstance(total, float)
        assert isinstance(breakdown, dict)


class TestComposedReward:
    """Test reward composition."""

    def test_single_component(self):
        """Test with single component."""
        component = MockRewardComponent({'reward/value': 5.0})
        composed = ComposedReward([component])

        total, breakdown = composed.compute({})

        assert total == 5.0
        assert breakdown == {'reward/value': 5.0}

    def test_multiple_components(self):
        """Test with multiple components."""
        components = [
            MockRewardComponent({'comp1/value': 2.0}),
            MockRewardComponent({'comp2/value': 3.0}),
            MockRewardComponent({'comp3/value': -1.0}),
        ]
        composed = ComposedReward(components)

        total, breakdown = composed.compute({})

        assert total == 4.0  # 2.0 + 3.0 - 1.0
        assert breakdown == {
            'comp1/value': 2.0,
            'comp2/value': 3.0,
            'comp3/value': -1.0,
        }

    def test_empty_components(self):
        """Test with no components."""
        composed = ComposedReward([])

        total, breakdown = composed.compute({})

        assert total == 0.0
        assert breakdown == {}

    def test_reset_propagation(self):
        """Test that reset is called on all components."""
        components = [
            MockRewardComponent({}),
            MockRewardComponent({}),
        ]
        composed = ComposedReward(components)

        composed.reset()

        # All components should have reset called
        for comp in components:
            assert comp.reset_called


class TestPresets:
    """Test preset loading and merging."""

    def test_load_gaplock_full(self):
        """Test loading gaplock_full preset."""
        config = load_preset('gaplock_full')

        assert 'terminal' in config
        assert 'pressure' in config
        assert 'forcing' in config
        assert config['terminal']['target_crash'] == 60.0

    def test_load_gaplock_simple(self):
        """Test loading gaplock_simple preset."""
        config = load_preset('gaplock_simple')

        assert 'terminal' in config
        assert 'pressure' in config
        assert config['forcing']['enabled'] is False

    def test_load_unknown_preset(self):
        """Test loading unknown preset raises error."""
        with pytest.raises(ValueError, match="Unknown preset"):
            load_preset('nonexistent_preset')

    def test_merge_config_simple(self):
        """Test simple config merging."""
        base = {'a': 1, 'b': 2}
        overrides = {'b': 3, 'c': 4}

        result = merge_config(base, overrides)

        assert result == {'a': 1, 'b': 3, 'c': 4}

    def test_merge_config_nested(self):
        """Test nested config merging."""
        base = {
            'terminal': {'target_crash': 60.0, 'self_crash': -90.0},
            'pressure': {'enabled': True},
        }
        overrides = {
            'terminal': {'target_crash': 100.0},  # Override just this value
        }

        result = merge_config(base, overrides)

        assert result['terminal']['target_crash'] == 100.0  # Overridden
        assert result['terminal']['self_crash'] == -90.0     # Preserved
        assert result['pressure']['enabled'] is True          # Preserved

    def test_preset_independence(self):
        """Test that loading preset doesn't modify original."""
        config1 = load_preset('gaplock_full')
        config2 = load_preset('gaplock_full')

        # Modify config1
        config1['terminal']['target_crash'] = 999.0

        # config2 should be unchanged
        assert config2['terminal']['target_crash'] == 60.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
