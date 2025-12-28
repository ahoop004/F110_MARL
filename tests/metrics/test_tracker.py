"""Tests for episode metrics tracking."""

import pytest
from src.metrics import EpisodeOutcome, EpisodeMetrics, MetricsTracker


class TestEpisodeMetrics:
    """Test EpisodeMetrics dataclass."""

    def test_creation(self):
        """Test creating episode metrics."""
        metrics = EpisodeMetrics(
            episode=0,
            outcome=EpisodeOutcome.TARGET_CRASH,
            total_reward=125.5,
            steps=450,
            reward_components={'terminal/success': 60.0},
        )
        assert metrics.episode == 0
        assert metrics.outcome == EpisodeOutcome.TARGET_CRASH
        assert metrics.total_reward == 125.5
        assert metrics.steps == 450
        assert metrics.reward_components == {'terminal/success': 60.0}
        assert metrics.success is True

    def test_success_flag_from_outcome(self):
        """Test that success flag is set based on outcome."""
        # Success outcome
        metrics = EpisodeMetrics(
            episode=0,
            outcome=EpisodeOutcome.TARGET_CRASH,
            total_reward=100.0,
            steps=400,
        )
        assert metrics.success is True

        # Failure outcome
        metrics = EpisodeMetrics(
            episode=1,
            outcome=EpisodeOutcome.SELF_CRASH,
            total_reward=-50.0,
            steps=200,
        )
        assert metrics.success is False

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = EpisodeMetrics(
            episode=5,
            outcome=EpisodeOutcome.TARGET_CRASH,
            total_reward=125.5,
            steps=450,
            reward_components={
                'terminal/success': 60.0,
                'pressure/bonus': 0.12,
            },
        )
        d = metrics.to_dict()

        assert d['episode'] == 5
        assert d['outcome'] == 'target_crash'
        assert d['total_reward'] == 125.5
        assert d['steps'] == 450
        assert d['success'] is True
        assert d['reward/terminal/success'] == 60.0
        assert d['reward/pressure/bonus'] == 0.12


class TestMetricsTracker:
    """Test MetricsTracker class."""

    def test_initialization(self):
        """Test tracker initialization."""
        tracker = MetricsTracker()
        assert len(tracker.episodes) == 0
        assert all(count == 0 for count in tracker._outcome_counts.values())

    def test_add_episode(self):
        """Test adding episodes."""
        tracker = MetricsTracker()

        metrics = tracker.add_episode(
            episode=0,
            outcome=EpisodeOutcome.TARGET_CRASH,
            total_reward=125.5,
            steps=450,
            reward_components={'terminal/success': 60.0},
        )

        assert len(tracker.episodes) == 1
        assert metrics.episode == 0
        assert metrics.total_reward == 125.5
        assert tracker._outcome_counts[EpisodeOutcome.TARGET_CRASH] == 1

    def test_add_multiple_episodes(self):
        """Test adding multiple episodes."""
        tracker = MetricsTracker()

        # Add success
        tracker.add_episode(
            episode=0,
            outcome=EpisodeOutcome.TARGET_CRASH,
            total_reward=100.0,
            steps=400,
        )

        # Add failure
        tracker.add_episode(
            episode=1,
            outcome=EpisodeOutcome.SELF_CRASH,
            total_reward=-50.0,
            steps=200,
        )

        # Add another success
        tracker.add_episode(
            episode=2,
            outcome=EpisodeOutcome.TARGET_CRASH,
            total_reward=120.0,
            steps=450,
        )

        assert len(tracker.episodes) == 3
        assert tracker._outcome_counts[EpisodeOutcome.TARGET_CRASH] == 2
        assert tracker._outcome_counts[EpisodeOutcome.SELF_CRASH] == 1

    def test_get_latest(self):
        """Test getting latest episodes."""
        tracker = MetricsTracker()

        for i in range(5):
            tracker.add_episode(
                episode=i,
                outcome=EpisodeOutcome.TARGET_CRASH,
                total_reward=100.0,
                steps=400,
            )

        latest = tracker.get_latest(n=2)
        assert len(latest) == 2
        assert latest[0].episode == 4  # Most recent first
        assert latest[1].episode == 3

    def test_get_outcome_counts_all(self):
        """Test getting outcome counts over all episodes."""
        tracker = MetricsTracker()

        tracker.add_episode(0, EpisodeOutcome.TARGET_CRASH, 100.0, 400)
        tracker.add_episode(1, EpisodeOutcome.TARGET_CRASH, 110.0, 420)
        tracker.add_episode(2, EpisodeOutcome.SELF_CRASH, -50.0, 200)
        tracker.add_episode(3, EpisodeOutcome.TIMEOUT, -20.0, 500)

        counts = tracker.get_outcome_counts()
        assert counts['target_crash'] == 2
        assert counts['self_crash'] == 1
        assert counts['timeout'] == 1
        assert counts['collision'] == 0
        assert counts['idle_stop'] == 0
        assert counts['target_finish'] == 0

    def test_get_outcome_counts_windowed(self):
        """Test getting outcome counts with window."""
        tracker = MetricsTracker()

        # Add 5 episodes
        tracker.add_episode(0, EpisodeOutcome.TARGET_CRASH, 100.0, 400)
        tracker.add_episode(1, EpisodeOutcome.SELF_CRASH, -50.0, 200)
        tracker.add_episode(2, EpisodeOutcome.TARGET_CRASH, 110.0, 420)
        tracker.add_episode(3, EpisodeOutcome.TARGET_CRASH, 120.0, 430)
        tracker.add_episode(4, EpisodeOutcome.TIMEOUT, -20.0, 500)

        # Get counts for last 3 episodes
        counts = tracker.get_outcome_counts(window=3)
        assert counts['target_crash'] == 2  # Episodes 2 and 3
        assert counts['self_crash'] == 0
        assert counts['timeout'] == 1

    def test_get_rolling_stats_empty(self):
        """Test rolling stats with no episodes."""
        tracker = MetricsTracker()
        stats = tracker.get_rolling_stats()

        assert stats['success_rate'] == 0.0
        assert stats['avg_reward'] == 0.0
        assert stats['avg_steps'] == 0.0
        assert stats['total_episodes'] == 0

    def test_get_rolling_stats_all(self):
        """Test rolling stats over all episodes."""
        tracker = MetricsTracker()

        tracker.add_episode(0, EpisodeOutcome.TARGET_CRASH, 100.0, 400)
        tracker.add_episode(1, EpisodeOutcome.SELF_CRASH, -50.0, 200)
        tracker.add_episode(2, EpisodeOutcome.TARGET_CRASH, 120.0, 450)

        stats = tracker.get_rolling_stats()

        assert stats['success_rate'] == 2 / 3  # 2 successes out of 3
        assert stats['avg_reward'] == (100.0 - 50.0 + 120.0) / 3
        assert stats['avg_steps'] == (400 + 200 + 450) / 3
        assert stats['total_episodes'] == 3
        assert stats['outcome_counts']['target_crash'] == 2
        assert stats['outcome_counts']['self_crash'] == 1
        assert stats['outcome_rates']['target_crash'] == 2 / 3

    def test_get_rolling_stats_windowed(self):
        """Test rolling stats with window."""
        tracker = MetricsTracker()

        # Add 5 episodes
        tracker.add_episode(0, EpisodeOutcome.SELF_CRASH, -50.0, 200)
        tracker.add_episode(1, EpisodeOutcome.SELF_CRASH, -40.0, 180)
        tracker.add_episode(2, EpisodeOutcome.TARGET_CRASH, 100.0, 400)
        tracker.add_episode(3, EpisodeOutcome.TARGET_CRASH, 110.0, 420)
        tracker.add_episode(4, EpisodeOutcome.TARGET_CRASH, 120.0, 430)

        # Get stats for last 3 episodes (all successes)
        stats = tracker.get_rolling_stats(window=3)

        assert stats['success_rate'] == 1.0  # All 3 are successes
        assert stats['avg_reward'] == (100.0 + 110.0 + 120.0) / 3
        assert stats['total_episodes'] == 3

    def test_clear(self):
        """Test clearing tracker."""
        tracker = MetricsTracker()

        tracker.add_episode(0, EpisodeOutcome.TARGET_CRASH, 100.0, 400)
        tracker.add_episode(1, EpisodeOutcome.SELF_CRASH, -50.0, 200)

        assert len(tracker.episodes) == 2

        tracker.clear()

        assert len(tracker.episodes) == 0
        assert all(count == 0 for count in tracker._outcome_counts.values())


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
