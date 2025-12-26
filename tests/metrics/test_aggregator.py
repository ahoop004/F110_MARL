"""Tests for metrics aggregation."""

import pytest
import numpy as np
from v2.metrics import EpisodeOutcome, EpisodeMetrics, MetricsAggregator


class TestMetricsAggregator:
    """Test MetricsAggregator class."""

    @pytest.fixture
    def sample_episodes(self):
        """Create sample episodes for testing."""
        episodes = []
        for i in range(10):
            outcome = EpisodeOutcome.TARGET_CRASH if i % 3 == 0 else EpisodeOutcome.SELF_CRASH
            reward = 100.0 + i * 10 if outcome.is_success() else -50.0 - i * 5
            steps = 400 + i * 10

            episodes.append(EpisodeMetrics(
                episode=i,
                outcome=outcome,
                total_reward=reward,
                steps=steps,
                reward_components={'terminal/success': 60.0} if outcome.is_success() else {},
            ))

        return episodes

    def test_compute_stats_empty(self):
        """Test stats computation with no episodes."""
        stats = MetricsAggregator.compute_stats([])
        assert stats == {}

    def test_compute_stats_basic(self, sample_episodes):
        """Test basic stats computation."""
        stats = MetricsAggregator.compute_stats(sample_episodes)

        assert 'reward_mean' in stats
        assert 'reward_std' in stats
        assert 'reward_min' in stats
        assert 'reward_max' in stats
        assert 'steps_mean' in stats
        assert 'success_rate' in stats

        # Verify types
        assert isinstance(stats['reward_mean'], float)
        assert isinstance(stats['success_rate'], float)

    def test_compute_stats_percentiles(self, sample_episodes):
        """Test percentile calculations."""
        stats = MetricsAggregator.compute_stats(sample_episodes)

        assert 'reward_p25' in stats
        assert 'reward_p50' in stats
        assert 'reward_p75' in stats
        assert 'reward_p95' in stats

        # Percentiles should be ordered
        assert stats['reward_min'] <= stats['reward_p25']
        assert stats['reward_p25'] <= stats['reward_p50']
        assert stats['reward_p50'] <= stats['reward_p75']
        assert stats['reward_p75'] <= stats['reward_p95']
        assert stats['reward_p95'] <= stats['reward_max']

    def test_compute_stats_window(self, sample_episodes):
        """Test stats with window."""
        # Get stats for last 5 episodes
        stats = MetricsAggregator.compute_stats(sample_episodes, window=5)

        # Should only consider last 5 episodes
        expected_rewards = [ep.total_reward for ep in sample_episodes[-5:]]
        assert stats['reward_mean'] == np.mean(expected_rewards)

    def test_compute_ema_empty(self):
        """Test EMA with empty list."""
        ema = MetricsAggregator.compute_ema([])
        assert ema == []

    def test_compute_ema_single(self):
        """Test EMA with single value."""
        ema = MetricsAggregator.compute_ema([10.0])
        assert ema == [10.0]

    def test_compute_ema_multiple(self):
        """Test EMA with multiple values."""
        values = [10.0, 20.0, 15.0, 25.0]
        ema = MetricsAggregator.compute_ema(values, alpha=0.5)

        # First value should be unchanged
        assert ema[0] == 10.0

        # Length should match input
        assert len(ema) == len(values)

        # EMA should smooth the values
        # With alpha=0.5, ema[1] = 0.5 * 20.0 + 0.5 * 10.0 = 15.0
        assert ema[1] == 15.0

    def test_compute_ema_smoothing(self):
        """Test that EMA smooths fluctuations."""
        # Create fluctuating values
        values = [10.0, 50.0, 10.0, 50.0, 10.0, 50.0]
        ema = MetricsAggregator.compute_ema(values, alpha=0.1)

        # EMA should be smoother than original
        # Range of EMA should be smaller than range of values
        ema_range = max(ema) - min(ema)
        values_range = max(values) - min(values)
        assert ema_range < values_range

    def test_compute_moving_average_empty(self):
        """Test moving average with empty list."""
        ma = MetricsAggregator.compute_moving_average([], window=3)
        assert ma == []

    def test_compute_moving_average_window(self):
        """Test moving average with different windows."""
        values = [10.0, 20.0, 30.0, 40.0, 50.0]

        # Window of 3
        ma = MetricsAggregator.compute_moving_average(values, window=3)
        assert len(ma) == len(values)

        # First value should be just itself
        assert ma[0] == 10.0

        # Second value should be average of first two
        assert ma[1] == (10.0 + 20.0) / 2

        # Third and onwards should be average of window
        assert ma[2] == (10.0 + 20.0 + 30.0) / 3
        assert ma[3] == (20.0 + 30.0 + 40.0) / 3
        assert ma[4] == (30.0 + 40.0 + 50.0) / 3

    def test_aggregate_components_empty(self):
        """Test component aggregation with no episodes."""
        stats = MetricsAggregator.aggregate_components([])
        assert stats == {}

    def test_aggregate_components_basic(self):
        """Test basic component aggregation."""
        episodes = [
            EpisodeMetrics(
                episode=0,
                outcome=EpisodeOutcome.TARGET_CRASH,
                total_reward=100.0,
                steps=400,
                reward_components={
                    'terminal/success': 60.0,
                    'pressure/bonus': 0.12,
                },
            ),
            EpisodeMetrics(
                episode=1,
                outcome=EpisodeOutcome.TARGET_CRASH,
                total_reward=110.0,
                steps=420,
                reward_components={
                    'terminal/success': 60.0,
                    'pressure/bonus': 0.15,
                },
            ),
        ]

        stats = MetricsAggregator.aggregate_components(episodes)

        assert 'terminal/success' in stats
        assert 'pressure/bonus' in stats

        # Terminal success should be constant 60.0
        assert stats['terminal/success']['mean'] == 60.0
        assert stats['terminal/success']['std'] == 0.0
        assert stats['terminal/success']['count'] == 2

        # Pressure bonus should average 0.12 and 0.15
        assert stats['pressure/bonus']['mean'] == (0.12 + 0.15) / 2
        assert stats['pressure/bonus']['count'] == 2

    def test_aggregate_components_sparse(self):
        """Test component aggregation with sparse components."""
        episodes = [
            EpisodeMetrics(
                episode=0,
                outcome=EpisodeOutcome.TARGET_CRASH,
                total_reward=100.0,
                steps=400,
                reward_components={'terminal/success': 60.0},
            ),
            EpisodeMetrics(
                episode=1,
                outcome=EpisodeOutcome.SELF_CRASH,
                total_reward=-50.0,
                steps=200,
                reward_components={'terminal/self_crash': -90.0},
            ),
        ]

        stats = MetricsAggregator.aggregate_components(episodes)

        # Each component should appear only once
        assert stats['terminal/success']['count'] == 1
        assert stats['terminal/self_crash']['count'] == 1

    def test_detect_improvement_insufficient_data(self):
        """Test improvement detection with insufficient data."""
        episodes = [
            EpisodeMetrics(0, EpisodeOutcome.TARGET_CRASH, 100.0, 400),
            EpisodeMetrics(1, EpisodeOutcome.SELF_CRASH, -50.0, 200),
        ]

        # Not enough data for windows of 50
        improvement = MetricsAggregator.detect_improvement(episodes, window1=50, window2=50)

        assert improvement['reward_delta'] == 0.0
        assert improvement['success_delta'] == 0.0
        assert improvement['improved'] == False

    def test_detect_improvement_positive(self):
        """Test detection of positive improvement."""
        episodes = []

        # First 10 episodes: mostly failures
        for i in range(10):
            outcome = EpisodeOutcome.SELF_CRASH
            reward = -50.0
            episodes.append(EpisodeMetrics(i, outcome, reward, 200))

        # Next 10 episodes: mostly successes
        for i in range(10, 20):
            outcome = EpisodeOutcome.TARGET_CRASH
            reward = 100.0
            episodes.append(EpisodeMetrics(i, outcome, reward, 400))

        improvement = MetricsAggregator.detect_improvement(
            episodes, window1=10, window2=10
        )

        assert improvement['reward_delta'] > 0
        assert improvement['success_delta'] > 0
        assert improvement['improved'] == True

    def test_detect_improvement_negative(self):
        """Test detection of negative improvement."""
        episodes = []

        # First 10 episodes: mostly successes
        for i in range(10):
            outcome = EpisodeOutcome.TARGET_CRASH
            reward = 100.0
            episodes.append(EpisodeMetrics(i, outcome, reward, 400))

        # Next 10 episodes: mostly failures
        for i in range(10, 20):
            outcome = EpisodeOutcome.SELF_CRASH
            reward = -50.0
            episodes.append(EpisodeMetrics(i, outcome, reward, 200))

        improvement = MetricsAggregator.detect_improvement(
            episodes, window1=10, window2=10
        )

        assert improvement['reward_delta'] < 0
        assert improvement['success_delta'] < 0
        assert improvement['improved'] == False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
