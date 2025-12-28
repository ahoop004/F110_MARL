"""Best model tracking with rolling window smoothing.

Tracks best performing models based on smoothed metrics to avoid
saving checkpoints based on noisy single-episode spikes.
"""

from collections import deque
from typing import Optional, List, Tuple, Deque
import numpy as np


class BestModelTracker:
    """Tracks best model performance with rolling window smoothing.

    Uses a rolling window to smooth noisy metrics and determine when
    a new best model should be saved. Prevents spurious best model
    updates from random spikes.

    Example:
        >>> tracker = BestModelTracker(window_size=50, metric_name="success_rate")
        >>> for episode in range(1000):
        ...     success_rate = compute_success_rate()
        ...     if tracker.is_new_best(success_rate, episode):
        ...         save_checkpoint(episode, "best")
        ...         tracker.update_best(success_rate, episode, checkpoint_path)
    """

    def __init__(
        self,
        window_size: int = 50,
        metric_name: str = "avg_reward",
        higher_is_better: bool = True,
        min_improvement: float = 0.0,
        patience: int = 0,
    ):
        """Initialize best model tracker.

        Args:
            window_size: Rolling window size for smoothing (default: 50)
            metric_name: Name of metric being tracked (for logging)
            higher_is_better: True if higher metric values are better (default: True)
            min_improvement: Minimum improvement required to update best (default: 0.0)
            patience: Number of episodes to wait before declaring new best (default: 0)
        """
        self.window_size = window_size
        self.metric_name = metric_name
        self.higher_is_better = higher_is_better
        self.min_improvement = min_improvement
        self.patience = patience

        # Rolling window of metric values
        self.metric_window: Deque[float] = deque(maxlen=window_size)

        # Best model tracking
        self.best_smoothed_value: Optional[float] = None
        self.best_episode: Optional[int] = None
        self.best_checkpoint_path: Optional[str] = None

        # Patience tracking
        self.patience_counter = 0
        self.pending_best_value: Optional[float] = None
        self.pending_best_episode: Optional[int] = None

        # History of best models (for cleanup)
        self.best_history: List[Tuple[str, float]] = []

    def add_value(self, value: float) -> float:
        """Add a new metric value to the rolling window.

        Args:
            value: New metric value

        Returns:
            Current smoothed value (mean of window)
        """
        self.metric_window.append(value)
        return self.get_smoothed_value()

    def get_smoothed_value(self) -> Optional[float]:
        """Get current smoothed metric value.

        Returns:
            Mean of rolling window, or None if window is empty
        """
        if not self.metric_window:
            return None
        return float(np.mean(self.metric_window))

    def is_new_best(self, value: float, episode: int) -> bool:
        """Check if current value represents a new best model.

        Args:
            value: New metric value
            episode: Current episode number

        Returns:
            True if this is a new best model, False otherwise
        """
        # Add value to window
        smoothed_value = self.add_value(value)

        if smoothed_value is None:
            return False

        # Not enough samples yet
        if len(self.metric_window) < self.window_size:
            return False

        # First valid smoothed value
        if self.best_smoothed_value is None:
            if self.patience == 0:
                return True
            else:
                # Start patience counter
                self.pending_best_value = smoothed_value
                self.pending_best_episode = episode
                self.patience_counter = 1
                return False

        # Check if smoothed value is better than current best
        is_better = self._is_better(smoothed_value, self.best_smoothed_value)

        if is_better:
            if self.patience == 0:
                # No patience required
                return True
            else:
                # Check if this is a new pending best
                if (self.pending_best_value is None or
                    self._is_better(smoothed_value, self.pending_best_value)):
                    self.pending_best_value = smoothed_value
                    self.pending_best_episode = episode
                    self.patience_counter = 1
                else:
                    # Continuing same pending best
                    self.patience_counter += 1

                # Check if patience threshold met
                if self.patience_counter >= self.patience:
                    return True
                else:
                    return False
        else:
            # Not better - reset patience
            self.pending_best_value = None
            self.pending_best_episode = None
            self.patience_counter = 0
            return False

    def update_best(
        self,
        value: float,
        episode: int,
        checkpoint_path: str
    ) -> None:
        """Update best model information.

        Args:
            value: Metric value (raw, not smoothed)
            episode: Episode number
            checkpoint_path: Path to checkpoint file
        """
        smoothed_value = self.get_smoothed_value()

        if smoothed_value is None:
            return

        self.best_smoothed_value = smoothed_value
        self.best_episode = episode
        self.best_checkpoint_path = checkpoint_path

        # Add to history for cleanup
        self.best_history.append((checkpoint_path, smoothed_value))

        # Reset patience
        self.pending_best_value = None
        self.pending_best_episode = None
        self.patience_counter = 0

    def _is_better(self, new_value: float, old_value: float) -> bool:
        """Check if new value is better than old value.

        Args:
            new_value: New metric value
            old_value: Old metric value

        Returns:
            True if new value is better
        """
        if self.higher_is_better:
            return new_value >= (old_value + self.min_improvement)
        else:
            return new_value <= (old_value - self.min_improvement)

    def get_best_info(self) -> Optional[Tuple[float, int, str]]:
        """Get information about current best model.

        Returns:
            Tuple of (smoothed_value, episode, checkpoint_path), or None if no best yet
        """
        if self.best_smoothed_value is None:
            return None

        return (
            self.best_smoothed_value,
            self.best_episode,
            self.best_checkpoint_path
        )

    def get_best_checkpoints(self, n: int = 3) -> List[Tuple[str, float]]:
        """Get top N best checkpoints by metric value.

        Args:
            n: Number of best checkpoints to return

        Returns:
            List of (checkpoint_path, metric_value) tuples, sorted by metric value
        """
        if not self.best_history:
            return []

        # Sort by metric value
        sorted_history = sorted(
            self.best_history,
            key=lambda x: x[1],
            reverse=self.higher_is_better
        )

        return sorted_history[:n]

    def reset(self) -> None:
        """Reset tracker to initial state."""
        self.metric_window.clear()
        self.best_smoothed_value = None
        self.best_episode = None
        self.best_checkpoint_path = None
        self.pending_best_value = None
        self.pending_best_episode = None
        self.patience_counter = 0
        self.best_history.clear()

    def __str__(self) -> str:
        """String representation of tracker state."""
        if self.best_smoothed_value is None:
            return f"BestModelTracker({self.metric_name}): No best model yet"

        return (
            f"BestModelTracker({self.metric_name}): "
            f"best={self.best_smoothed_value:.4f} @ episode {self.best_episode}"
        )

    def get_status(self) -> str:
        """Get detailed status string for logging.

        Returns:
            Multi-line status string
        """
        smoothed = self.get_smoothed_value()
        best_info = self.get_best_info()

        lines = [
            f"Metric: {self.metric_name}",
            f"Window: {len(self.metric_window)}/{self.window_size}",
        ]

        if smoothed is not None:
            lines.append(f"Current (smoothed): {smoothed:.4f}")

        if best_info:
            best_val, best_ep, _ = best_info
            lines.append(f"Best: {best_val:.4f} @ episode {best_ep}")

        if self.patience > 0 and self.pending_best_value is not None:
            lines.append(
                f"Patience: {self.patience_counter}/{self.patience} "
                f"(pending: {self.pending_best_value:.4f})"
            )

        return "\n".join(lines)
