"""Weights & Biases logging integration for F110 training.

Provides automatic W&B initialization, configuration tracking,
and per-episode/rolling metrics logging.
"""

import os
from typing import Dict, Any, Optional
import wandb


class WandbLogger:
    """Logger for Weights & Biases integration.

    Handles W&B initialization, configuration tracking, and metrics logging
    for training runs. Supports both per-episode and rolling statistics.

    Example:
        >>> from loggers import WandbLogger
        >>> from metrics import MetricsTracker, determine_outcome
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
        job_type: Optional[str] = None,
        entity: Optional[str] = None,
        notes: Optional[str] = None,
        mode: str = "online",
        run_id: Optional[str] = None,
        logging_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Initialize W&B logger.

        Args:
            project: W&B project name
            config: Configuration dict (can be nested)
            name: Run name (optional, W&B will auto-generate if not provided)
            tags: List of tags for this run
            group: Group name for organizing runs
            job_type: Job type for organizing runs
            entity: W&B entity (username or team name)
            notes: Notes about this run
            mode: W&B mode ("online", "offline", or "disabled")
            run_id: Custom run ID for checkpoint alignment (optional)
            logging_config: Optional logging toggles (e.g., groups/metrics maps)
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
            ...     run_id="gaplock_ppo_s42_1234567890_abcd",
            ... )
        """
        self.project = project
        self.enabled = mode != "disabled"
        self.logging_config = logging_config if isinstance(logging_config, dict) else None

        # Store run ID for alignment with checkpoints
        self.custom_run_id = run_id

        # W&B run information (captured after init)
        self.wandb_run_id: Optional[str] = None
        self.wandb_run_name: Optional[str] = None
        self.wandb_url: Optional[str] = None

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
                job_type=job_type,
                entity=entity,
                notes=notes,
                mode=mode,
                **kwargs,
            )

            # Capture W&B run information
            if self.run is not None:
                self.wandb_run_id = self.run.id
                self.wandb_run_name = self.run.name
                self.wandb_url = self.run.get_url()
                if self.logging_config:
                    try:
                        logging_payload = {"wandb_logging": self.logging_config}
                        flat_logging = self._flatten_config(logging_payload)
                        wandb.config.update(flat_logging, allow_val_change=True)
                    except Exception:
                        pass
                if self.should_log("define_metrics"):
                    try:
                        wandb.define_metric("train/episode")
                        wandb.define_metric("train/*", step_metric="train/episode")
                        wandb.define_metric("target/*", step_metric="train/episode")
                        wandb.define_metric("curriculum/*", step_metric="train/episode")
                        wandb.define_metric("eval/episode")
                        wandb.define_metric("eval/episode_*", step_metric="eval/episode")
                        wandb.define_metric("eval/training_episode", step_metric="eval/episode")
                        wandb.define_metric("eval/spawn_point", step_metric="eval/episode")
                        wandb.define_metric("eval_agg/*", step_metric="train/episode")
                    except Exception:
                        pass
        else:
            self.run = None

    def should_log(self, key: str) -> bool:
        """Check if a logging group is enabled."""
        if not self.enabled:
            return False
        group_config = self._get_group_config()
        if group_config is None:
            return True
        return bool(group_config.get(key, False))

    def _get_group_config(self) -> Optional[Dict[str, Any]]:
        if not isinstance(self.logging_config, dict):
            return None
        if "groups" in self.logging_config:
            groups = self.logging_config.get("groups")
            return groups if isinstance(groups, dict) else {}
        return self.logging_config

    def _get_metrics_config(self) -> Optional[Dict[str, Any]]:
        if not isinstance(self.logging_config, dict):
            return None
        metrics = self.logging_config.get("metrics")
        if metrics is None:
            return None
        if isinstance(metrics, dict):
            return metrics
        if isinstance(metrics, (list, tuple, set)):
            return {name: True for name in metrics}
        return None

    def _filter_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        metrics_config = self._get_metrics_config()
        if metrics_config is None:
            return metrics
        return {key: value for key, value in metrics.items() if metrics_config.get(key, False)}

    def log_episode(
        self,
        episode: int,
        metrics: Any,  # EpisodeMetrics
        rolling_stats: Optional[Dict[str, float]] = None,
        extra: Optional[Dict[str, Any]] = None,
        agent_id: Optional[str] = None,
    ):
        """Log metrics for a single episode.

        Args:
            episode: Episode number
            metrics: EpisodeMetrics instance
            rolling_stats: Optional dict of rolling statistics
            extra: Optional extra metrics to log
            agent_id: Optional agent ID for namespacing metrics

        Example:
            >>> logger.log_episode(
            ...     episode=0,
            ...     metrics=episode_metrics,
            ...     rolling_stats={'success_rate': 0.75, 'avg_reward': 85.2},
            ...     agent_id='car_0',
            ... )
        """
        if not self.enabled:
            return

        # Convert metrics to dict
        metrics_dict = metrics.to_dict()

        log_dict = {}

        # Namespace episode metrics by agent_id if provided
        if agent_id:
            # Remove 'episode' from namespacing as it's a global counter
            if 'episode' in metrics_dict:
                log_dict['episode'] = metrics_dict.pop('episode')

            # Namespace all other episode metrics
            for key, value in metrics_dict.items():
                log_dict[f'{agent_id}/{key}'] = value
        else:
            # No namespacing if agent_id not provided
            log_dict = metrics_dict

        # Add rolling stats with agent namespace
        if rolling_stats:
            for key, value in rolling_stats.items():
                if key not in ['outcome_counts', 'outcome_rates', 'total_episodes']:
                    if agent_id:
                        log_dict[f'{agent_id}/rolling/{key}'] = value
                    else:
                        log_dict[f'rolling/{key}'] = value

            # Log outcome distribution
            if 'outcome_counts' in rolling_stats:
                for outcome, count in rolling_stats['outcome_counts'].items():
                    if agent_id:
                        log_dict[f'{agent_id}/rolling/outcomes/{outcome}'] = count
                    else:
                        log_dict[f'rolling/outcomes/{outcome}'] = count

            if 'outcome_rates' in rolling_stats:
                for outcome, rate in rolling_stats['outcome_rates'].items():
                    if agent_id:
                        log_dict[f'{agent_id}/rolling/outcome_rates/{outcome}'] = rate
                    else:
                        log_dict[f'rolling/outcome_rates/{outcome}'] = rate

        # Add extra metrics (not namespaced - assume they're already properly named)
        if extra:
            log_dict.update(extra)

        # Log to W&B
        log_dict = self._filter_metrics(log_dict)
        if not log_dict:
            return
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

        metrics = self._filter_metrics(metrics)
        if not metrics:
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

    def get_wandb_info(self) -> Dict[str, Optional[str]]:
        """Get W&B run information for metadata tracking.

        Returns:
            Dict with keys: run_id, run_name, url
                Values are None if W&B is disabled or not initialized

        Example:
            >>> info = logger.get_wandb_info()
            >>> print(info['url'])
            'https://wandb.ai/username/project/runs/abc123'
        """
        return {
            'run_id': self.wandb_run_id,
            'run_name': self.wandb_run_name,
            'url': self.wandb_url,
        }

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
