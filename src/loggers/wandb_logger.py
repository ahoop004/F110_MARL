"""Weights & Biases logging integration for F110 training.

Provides automatic W&B initialization, configuration tracking,
and per-episode/rolling metrics logging.
"""

from typing import Dict, Any, Optional
import wandb

from loggers.metric_policy import AXES, MetricPolicy


class WandbLogger:
    """Logger for Weights & Biases integration.

    Handles W&B initialization, configuration tracking, and metrics logging
    for training runs. Supports both per-episode and rolling statistics.

    Training hooks send scalar dictionaries through ``log_metrics``.
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
        self.logging_config = (logging_config if logging_config is not None else
                               (config or {}).get("wandb", {}).get("logging", {}))
        self.policy = MetricPolicy(self.logging_config, config)

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
                        for namespace, axis in AXES.items():
                            wandb.define_metric(axis)
                            wandb.define_metric(f"{namespace}/*", step_metric=axis)
                    except Exception:
                        pass
        else:
            self.run = None

    def should_log(self, key: str) -> bool:
        """Check an effective logging group before preparing optional metrics."""
        return self.enabled and self.policy.group_enabled(key)

    def _filter_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        return self.policy.filter(metrics)

    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
    ):
        """Log metrics selected by the configured profile, groups, and allowlist.

        Args:
            metrics: Dict of metrics to log
            step: Optional step number

        Example:
            >>> logger.log_metrics({'train/policy_loss': 0.2, 'train/environment_steps': 100})
        """
        if not self.enabled:
            return

        metrics = self._filter_metrics(metrics)
        if not metrics:
            return
        wandb.log(metrics, step=step)

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
