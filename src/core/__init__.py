"""Core infrastructure for the F110 training pipeline."""

from src.core.config import (
    AgentFactory,
    register_builtin_agents,
)
from src.core.setup import create_training_setup

__all__ = [
    # Factory
    "AgentFactory",
    "register_builtin_agents",
    # Setup
    "create_training_setup",
]
