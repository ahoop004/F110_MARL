"""Advanced metrics aggregation and rolling statistics.

Provides utilities for computing various statistical measures over
episode metrics, including percentiles, moving averages, and trends.
"""

from typing import Dict, List, Optional
import numpy as np
from .tracker import EpisodeMetrics


class MetricsAggregator:
    """Advanced aggregation and statistical analysis of episode metrics.

    Extends basic MetricsTracker with additional statistical capabilities:
    - Percentile calculations (p50, p95, etc.)
    - Moving averages (exponential and simple)
    - Trend detection
    - Component-level statistics

    Example:
        >>> aggregator = MetricsAggregator()
        >>> stats = aggregator.compute_stats(episodes, window=100)
        >>> print(f"Median reward: {stats['reward_p50']:.1f}")
        >>> print(f"95th percentile: {stats['reward_p95']:.1f}")
    """

    @staticmethod
    def compute_stats(
        episodes: List[EpisodeMetrics],
        window: Optional[int] = None,
    ) -> Dict[str, float]:
        """Compute comprehensive statistics over episodes.

        Args:
            episodes: List of episode metrics
            window: Number of recent episodes to include (None = all)

        Returns:
            Dict containing:
            - reward_mean: Mean reward
            - reward_std: Standard deviation of reward
            - reward_min: Minimum reward
            - reward_max: Maximum reward
            - reward_p25: 25th percentile
            - reward_p50: Median reward
            - reward_p75: 75th percentile
            - reward_p95: 95th percentile
            - steps_mean: Mean episode length
            - steps_std: Standard deviation of steps
            - success_rate: Fraction of successful episodes
        """
        if not episodes:
            return {}

        # Apply window
        eps = episodes[-window:] if window else episodes

        # Extract rewards and steps
        rewards = np.array([ep.total_reward for ep in eps])
        steps = np.array([ep.steps for ep in eps])
        successes = sum(1 for ep in eps if ep.success)

        stats = {
            # Reward statistics
            'reward_mean': float(np.mean(rewards)),
            'reward_std': float(np.std(rewards)),
            'reward_min': float(np.min(rewards)),
            'reward_max': float(np.max(rewards)),
            'reward_p25': float(np.percentile(rewards, 25)),
            'reward_p50': float(np.percentile(rewards, 50)),
            'reward_p75': float(np.percentile(rewards, 75)),
            'reward_p95': float(np.percentile(rewards, 95)),
            # Step statistics
            'steps_mean': float(np.mean(steps)),
            'steps_std': float(np.std(steps)),
            # Success rate
            'success_rate': successes / len(eps),
        }

        return stats

    @staticmethod
    def compute_ema(
        values: List[float],
        alpha: float = 0.1,
    ) -> List[float]:
        """Compute exponential moving average.

        Args:
            values: List of values to smooth
            alpha: Smoothing factor (0 < alpha <= 1)
                - Higher alpha = more weight on recent values
                - Lower alpha = smoother curve

        Returns:
            List of EMA values (same length as input)

        Example:
            >>> rewards = [10.0, 15.0, 12.0, 18.0, 20.0]
            >>> ema = MetricsAggregator.compute_ema(rewards, alpha=0.3)
        """
        if not values:
            return []

        ema = [values[0]]
        for value in values[1:]:
            ema.append(alpha * value + (1 - alpha) * ema[-1])

        return ema

    @staticmethod
    def compute_moving_average(
        values: List[float],
        window: int,
    ) -> List[float]:
        """Compute simple moving average.

        Args:
            values: List of values to smooth
            window: Window size for averaging

        Returns:
            List of moving averages (same length as input)
            First window-1 values are padded with cumulative average

        Example:
            >>> rewards = [10.0, 15.0, 12.0, 18.0, 20.0]
            >>> ma = MetricsAggregator.compute_moving_average(rewards, window=3)
        """
        if not values:
            return []

        result = []
        for i in range(len(values)):
            start = max(0, i - window + 1)
            result.append(np.mean(values[start:i + 1]))

        return result

    @staticmethod
    def aggregate_components(
        episodes: List[EpisodeMetrics],
        window: Optional[int] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Aggregate reward component statistics.

        Args:
            episodes: List of episode metrics
            window: Number of recent episodes to include

        Returns:
            Dict mapping component names to their statistics:
            - mean: Average value
            - std: Standard deviation
            - min: Minimum value
            - max: Maximum value
            - count: Number of episodes where component was active

        Example:
            >>> stats = MetricsAggregator.aggregate_components(episodes)
            >>> print(stats['terminal/success']['mean'])
            60.0
        """
        if not episodes:
            return {}

        eps = episodes[-window:] if window else episodes

        # Collect all component names
        all_components = set()
        for ep in eps:
            all_components.update(ep.reward_components.keys())

        # Aggregate each component
        component_stats = {}
        for component in all_components:
            values = [
                ep.reward_components[component]
                for ep in eps
                if component in ep.reward_components
            ]

            if values:
                component_stats[component] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'count': len(values),
                }

        return component_stats

    @staticmethod
    def detect_improvement(
        episodes: List[EpisodeMetrics],
        window1: int = 50,
        window2: int = 50,
    ) -> Dict[str, float]:
        """Detect improvement by comparing two consecutive windows.

        Args:
            episodes: List of episode metrics
            window1: Size of earlier window
            window2: Size of later window

        Returns:
            Dict containing:
            - reward_delta: Change in mean reward
            - success_delta: Change in success rate
            - steps_delta: Change in mean steps
            - improved: Whether metrics improved overall

        Example:
            >>> improvement = MetricsAggregator.detect_improvement(episodes)
            >>> if improvement['improved']:
            ...     print("Training is improving!")
        """
        if len(episodes) < window1 + window2:
            return {
                'reward_delta': 0.0,
                'success_delta': 0.0,
                'steps_delta': 0.0,
                'improved': False,
            }

        # Split into two windows
        earlier = episodes[-(window1 + window2):-window2]
        later = episodes[-window2:]

        # Compute stats for each window
        earlier_reward = np.mean([ep.total_reward for ep in earlier])
        later_reward = np.mean([ep.total_reward for ep in later])

        earlier_success = sum(1 for ep in earlier if ep.success) / len(earlier)
        later_success = sum(1 for ep in later if ep.success) / len(later)

        earlier_steps = np.mean([ep.steps for ep in earlier])
        later_steps = np.mean([ep.steps for ep in later])

        reward_delta = later_reward - earlier_reward
        success_delta = later_success - earlier_success
        steps_delta = later_steps - earlier_steps

        # Consider improved if reward or success rate increased
        improved = reward_delta > 0 or success_delta > 0

        return {
            'reward_delta': float(reward_delta),
            'success_delta': float(success_delta),
            'steps_delta': float(steps_delta),
            'improved': improved,
        }


__all__ = ['MetricsAggregator']
