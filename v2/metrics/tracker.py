"""Episode metrics tracking for training sessions.

Tracks per-episode metrics and provides aggregation capabilities.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from .outcomes import EpisodeOutcome


@dataclass
class EpisodeMetrics:
    """Metrics for a single episode.

    Attributes:
        episode: Episode number (0-indexed)
        outcome: Episode outcome type
        total_reward: Total accumulated reward
        steps: Number of steps taken
        reward_components: Dict of individual reward component values
        success: Whether episode was successful (target crashed)

    Example:
        >>> metrics = EpisodeMetrics(
        ...     episode=0,
        ...     outcome=EpisodeOutcome.TARGET_CRASH,
        ...     total_reward=125.5,
        ...     steps=450,
        ...     reward_components={'terminal/success': 60.0, 'pressure/bonus': 0.12},
        ...     success=True
        ... )
    """
    episode: int
    outcome: EpisodeOutcome
    total_reward: float
    steps: int
    reward_components: Dict[str, float] = field(default_factory=dict)
    success: bool = field(init=False)

    def __post_init__(self):
        """Set success flag based on outcome."""
        self.success = self.outcome.is_success()

    def to_dict(self) -> Dict:
        """Convert to dictionary for logging."""
        return {
            'episode': self.episode,
            'outcome': self.outcome.value,
            'total_reward': self.total_reward,
            'steps': self.steps,
            'success': self.success,
            **{f'reward/{k}': v for k, v in self.reward_components.items()},
        }


class MetricsTracker:
    """Tracks metrics across multiple episodes.

    Maintains a history of all episodes and provides aggregation methods
    for computing rolling statistics and outcome distributions.

    Example:
        >>> tracker = MetricsTracker()
        >>> tracker.add_episode(
        ...     episode=0,
        ...     outcome=EpisodeOutcome.TARGET_CRASH,
        ...     total_reward=125.5,
        ...     steps=450,
        ...     reward_components={'terminal/success': 60.0}
        ... )
        >>> stats = tracker.get_rolling_stats(window=100)
        >>> stats['success_rate']
        1.0
    """

    def __init__(self):
        """Initialize empty metrics tracker."""
        self.episodes: List[EpisodeMetrics] = []
        self._outcome_counts: Dict[EpisodeOutcome, int] = {
            outcome: 0 for outcome in EpisodeOutcome
        }

    def add_episode(
        self,
        episode: int,
        outcome: EpisodeOutcome,
        total_reward: float,
        steps: int,
        reward_components: Optional[Dict[str, float]] = None,
    ) -> EpisodeMetrics:
        """Add a completed episode to the tracker.

        Args:
            episode: Episode number
            outcome: Episode outcome type
            total_reward: Total accumulated reward
            steps: Number of steps taken
            reward_components: Optional dict of reward component values

        Returns:
            EpisodeMetrics instance that was added
        """
        metrics = EpisodeMetrics(
            episode=episode,
            outcome=outcome,
            total_reward=total_reward,
            steps=steps,
            reward_components=reward_components or {},
        )
        self.episodes.append(metrics)
        self._outcome_counts[outcome] += 1
        return metrics

    def get_latest(self, n: int = 1) -> List[EpisodeMetrics]:
        """Get the n most recent episodes.

        Args:
            n: Number of recent episodes to return

        Returns:
            List of EpisodeMetrics (most recent first)
        """
        return self.episodes[-n:][::-1]

    def get_outcome_counts(self, window: Optional[int] = None) -> Dict[str, int]:
        """Get counts of each outcome type.

        Args:
            window: If specified, only count last N episodes

        Returns:
            Dict mapping outcome names to counts
        """
        episodes = self.episodes[-window:] if window else self.episodes
        counts = {outcome.value: 0 for outcome in EpisodeOutcome}
        for ep in episodes:
            counts[ep.outcome.value] += 1
        return counts

    def get_rolling_stats(self, window: Optional[int] = None) -> Dict[str, float]:
        """Compute rolling statistics over recent episodes.

        Args:
            window: Number of recent episodes to include (None = all)

        Returns:
            Dict containing:
            - success_rate: Fraction of successful episodes
            - avg_reward: Average total reward
            - avg_steps: Average episode length
            - total_episodes: Number of episodes in window
            - outcome_counts: Dict of outcome counts
            - outcome_rates: Dict of outcome rates (fraction)

        Example:
            >>> tracker = MetricsTracker()
            >>> # ... add episodes ...
            >>> stats = tracker.get_rolling_stats(window=100)
            >>> print(f"Success rate: {stats['success_rate']:.2%}")
        """
        episodes = self.episodes[-window:] if window else self.episodes

        if not episodes:
            return {
                'success_rate': 0.0,
                'avg_reward': 0.0,
                'avg_steps': 0.0,
                'total_episodes': 0,
                'outcome_counts': {outcome.value: 0 for outcome in EpisodeOutcome},
                'outcome_rates': {outcome.value: 0.0 for outcome in EpisodeOutcome},
            }

        # Compute basic stats
        total = len(episodes)
        successes = sum(1 for ep in episodes if ep.success)
        total_reward = sum(ep.total_reward for ep in episodes)
        total_steps = sum(ep.steps for ep in episodes)

        # Compute outcome distribution
        outcome_counts = self.get_outcome_counts(window)
        outcome_rates = {k: v / total for k, v in outcome_counts.items()}

        return {
            'success_rate': successes / total,
            'avg_reward': total_reward / total,
            'avg_steps': total_steps / total,
            'total_episodes': total,
            'outcome_counts': outcome_counts,
            'outcome_rates': outcome_rates,
        }

    def clear(self):
        """Clear all tracked episodes."""
        self.episodes.clear()
        self._outcome_counts = {outcome: 0 for outcome in EpisodeOutcome}


__all__ = ['EpisodeMetrics', 'MetricsTracker']
