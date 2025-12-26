"""Logging system for F110 training.

Provides W&B integration and rich console output for training runs.

Example usage:
    >>> from v2.loggers import WandbLogger, ConsoleLogger
    >>>
    >>> # Initialize loggers
    >>> wandb_logger = WandbLogger(
    ...     project="f110-gaplock",
    ...     config={"algorithm": "ppo"},
    ...     tags=["baseline"],
    ... )
    >>> console_logger = ConsoleLogger()
    >>>
    >>> # Log training
    >>> console_logger.print_header("Training PPO")
    >>> for episode in range(1500):
    ...     # Train episode
    ...     wandb_logger.log_episode(episode, metrics, rolling_stats)
    ...     console_logger.log_episode(episode, outcome, reward, steps)
    >>>
    >>> wandb_logger.finish()
"""

from .wandb_logger import WandbLogger
from .console import ConsoleLogger

__all__ = ['WandbLogger', 'ConsoleLogger']
