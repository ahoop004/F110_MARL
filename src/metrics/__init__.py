"""Metrics tracking system for F110 training.

This module provides comprehensive metrics tracking capabilities:
- Episode outcome classification (6 types)
- Per-episode metrics tracking
- Rolling statistics and aggregation
- Advanced statistical analysis

Example usage:
    >>> from metrics import MetricsTracker, determine_outcome, EpisodeOutcome
    >>>
    >>> # Create tracker
    >>> tracker = MetricsTracker()
    >>>
    >>> # After episode ends
    >>> outcome = determine_outcome(info, truncated)
    >>> tracker.add_episode(
    ...     episode=0,
    ...     outcome=outcome,
    ...     total_reward=125.5,
    ...     steps=450,
    ...     reward_components=components
    ... )
    >>>
    >>> # Get rolling statistics
    >>> stats = tracker.get_rolling_stats(window=100)
    >>> print(f"Success rate: {stats['success_rate']:.2%}")
"""

from .outcomes import EpisodeOutcome, determine_outcome
from .tracker import EpisodeMetrics, MetricsTracker
from .aggregator import MetricsAggregator

__all__ = [
    'EpisodeOutcome',
    'determine_outcome',
    'EpisodeMetrics',
    'MetricsTracker',
    'MetricsAggregator',
]
