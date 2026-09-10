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
                        # Keep episode and optimizer-update charts on explicit,
                        # independent x-axes. W&B's internal step remains
                        # monotonic because callers do not provide a global step.
                        wandb.define_metric("episode/number")
                        wandb.define_metric("episode/*", step_metric="episode/number")
                        wandb.define_metric("train/update")
                        wandb.define_metric("train/*", step_metric="train/update")
                        wandb.define_metric("train/episode")
                        wandb.define_metric("target/*", step_metric="train/episode")
                        wandb.define_metric("curriculum/*", step_metric="train/episode")
                        wandb.define_metric("eval/episode")
                        wandb.define_metric("eval/episode_*", step_metric="eval/episode")
                        wandb.define_metric("eval/rolling_*", step_metric="eval/episode")
                        wandb.define_metric("eval/training_episode", step_metric="eval/episode")
                        wandb.define_metric("eval/spawn_point", step_metric="eval/episode")
                        wandb.define_metric("eval/run")
                        wandb.define_metric("eval_agg/episode_*", step_metric="eval/episode")
                        wandb.define_metric("eval_agg/*", step_metric="eval/run")
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
