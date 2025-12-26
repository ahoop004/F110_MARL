"""Weights & Biases logging integration for F110 training.

Provides automatic W&B initialization, configuration tracking,
and per-episode/rolling metrics logging.
"""

from typing import Dict, Any, Optional
import wandb


class WandbLogger:
    """Logger for Weights & Biases integration.

    Handles W&B initialization, configuration tracking, and metrics logging
    for training runs. Supports both per-episode and rolling statistics.

    Example:
        >>> from v2.loggers import WandbLogger
        >>> from v2.metrics import MetricsTracker, determine_outcome
        >>>
        >>> # Initialize logger
        >>> logger = WandbLogger(
        ...     project="f110-gaplock",
        ...     config={"algorithm": "ppo", "lr": 0.0005},
        ...     tags=["baseline", "gaplock"],
        ... )
        >>>
        >>> # After episode
        >>> logger.log_episode(
        ...     episode=0,
        ...     metrics=tracker.get_latest(1)[0],
        ...     rolling_stats=tracker.get_rolling_stats(window=100),
        ... )
        >>>
        >>> # Finish run
        >>> logger.finish()
    """

    def __init__(
        self,
        project: str,
        config: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        tags: Optional[list] = None,
        group: Optional[str] = None,
        notes: Optional[str] = None,
        mode: str = "online",
        **kwargs,
    ):
        """Initialize W&B logger.

        Args:
            project: W&B project name
            config: Configuration dict (can be nested)
            name: Run name (optional, W&B will auto-generate if not provided)
            tags: List of tags for this run
            group: Group name for organizing runs
            notes: Notes about this run
            mode: W&B mode ("online", "offline", or "disabled")
            **kwargs: Additional arguments passed to wandb.init()

        Example:
            >>> logger = WandbLogger(
            ...     project="f110-gaplock",
            ...     config={
            ...         "algorithm": "ppo",
            ...         "agent": {"lr": 0.0005, "gamma": 0.995},
            ...         "reward": {"terminal": {"target_crash": 60.0}},
            ...     },
            ...     tags=["baseline"],
            ... )
        """
        self.project = project
        self.enabled = mode != "disabled"

        if self.enabled:
            # Flatten nested config for W&B
            flat_config = self._flatten_config(config) if config else {}

            # Initialize W&B
            self.run = wandb.init(
                project=project,
                config=flat_config,
                name=name,
                tags=tags,
                group=group,
                notes=notes,
                mode=mode,
                **kwargs,
            )
        else:
            self.run = None

    def log_episode(
        self,
        episode: int,
        metrics: Any,  # EpisodeMetrics
        rolling_stats: Optional[Dict[str, float]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ):
        """Log metrics for a single episode.

        Args:
            episode: Episode number
            metrics: EpisodeMetrics instance
            rolling_stats: Optional dict of rolling statistics
            extra: Optional extra metrics to log

        Example:
            >>> logger.log_episode(
            ...     episode=0,
            ...     metrics=episode_metrics,
            ...     rolling_stats={'success_rate': 0.75, 'avg_reward': 85.2},
            ... )
        """
        if not self.enabled:
            return

        # Convert metrics to dict
        log_dict = metrics.to_dict()

        # Add rolling stats with 'rolling/' prefix
        if rolling_stats:
            for key, value in rolling_stats.items():
                if key not in ['outcome_counts', 'outcome_rates', 'total_episodes']:
                    log_dict[f'rolling/{key}'] = value

            # Log outcome distribution
            if 'outcome_counts' in rolling_stats:
                for outcome, count in rolling_stats['outcome_counts'].items():
                    log_dict[f'rolling/outcomes/{outcome}'] = count

            if 'outcome_rates' in rolling_stats:
                for outcome, rate in rolling_stats['outcome_rates'].items():
                    log_dict[f'rolling/outcome_rates/{outcome}'] = rate

        # Add extra metrics
        if extra:
            log_dict.update(extra)

        # Log to W&B
        wandb.log(log_dict, step=episode)

    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
    ):
        """Log arbitrary metrics dict.

        Args:
            metrics: Dict of metrics to log
            step: Optional step number

        Example:
            >>> logger.log_metrics({'custom_metric': 42.0}, step=100)
        """
        if not self.enabled:
            return

        wandb.log(metrics, step=step)

    def log_component_stats(
        self,
        component_stats: Dict[str, Dict[str, float]],
        step: Optional[int] = None,
    ):
        """Log reward component statistics.

        Args:
            component_stats: Dict mapping component names to their stats
                (from MetricsAggregator.aggregate_components)
            step: Optional step number

        Example:
            >>> stats = aggregator.aggregate_components(episodes, window=100)
            >>> logger.log_component_stats(stats, step=500)
        """
        if not self.enabled:
            return

        log_dict = {}
        for component, stats in component_stats.items():
            for stat_name, value in stats.items():
                # components/terminal/success/mean
                log_dict[f'components/{component}/{stat_name}'] = value

        wandb.log(log_dict, step=step)

    def finish(self):
        """Finish the W&B run."""
        if self.enabled and self.run is not None:
            wandb.finish()

    @staticmethod
    def _flatten_config(config: Dict[str, Any], parent_key: str = '', sep: str = '/') -> Dict[str, Any]:
        """Flatten nested config dict for W&B.

        Args:
            config: Nested configuration dict
            parent_key: Parent key for recursion
            sep: Separator for keys

        Returns:
            Flattened dict with keys like 'agent/lr', 'reward/terminal/target_crash'

        Example:
            >>> config = {
            ...     'agent': {'lr': 0.0005, 'gamma': 0.995},
            ...     'reward': {'terminal': {'target_crash': 60.0}},
            ... }
            >>> WandbLogger._flatten_config(config)
            {
                'agent/lr': 0.0005,
                'agent/gamma': 0.995,
                'reward/terminal/target_crash': 60.0,
            }
        """
        items = []
        for key, value in config.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key

            if isinstance(value, dict):
                # Recursively flatten nested dicts
                items.extend(WandbLogger._flatten_config(value, new_key, sep=sep).items())
            else:
                # Add leaf values
                items.append((new_key, value))

        return dict(items)


__all__ = ['WandbLogger']
