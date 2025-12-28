"""Logging system for F110 training.

Provides W&B integration, rich console output, and file-based logging for training runs.

Example usage:
    >>> from loggers import WandbLogger, ConsoleLogger, CSVLogger
    >>>
    >>> # Initialize loggers
    >>> wandb_logger = WandbLogger(
    ...     project="f110-gaplock",
    ...     config={"algorithm": "ppo"},
    ...     tags=["baseline"],
    ... )
    >>> console_logger = ConsoleLogger()
    >>> csv_logger = CSVLogger(output_dir="outputs/run_001", scenario_config=scenario)
    >>>
    >>> # Log training
    >>> console_logger.print_header("Training PPO")
    >>> for episode in range(1500):
    ...     # Train episode
    ...     wandb_logger.log_episode(episode, metrics, rolling_stats)
    ...     console_logger.log_episode(episode, outcome, reward, steps)
    ...     csv_logger.log_episode(episode, metrics, agent_metrics, rolling_stats)
    >>>
    >>> wandb_logger.finish()
    >>> csv_logger.save_summary(final_stats)
    >>> csv_logger.close()
"""

from .wandb_logger import WandbLogger
from .console import ConsoleLogger
from .csv_logger import CSVLogger
from .rich_console import RichConsole

__all__ = ['WandbLogger', 'ConsoleLogger', 'CSVLogger', 'RichConsole']
