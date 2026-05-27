"""Core infrastructure for the F110 training pipeline."""

from src.core.protocol import (
    Agent,
    OnPolicyAgent,
    OffPolicyAgent,
    HeuristicPolicy,
    is_on_policy_agent,
    is_off_policy_agent,
    is_heuristic_policy,
)
from src.core.config import (
    AgentFactory,
    register_builtin_agents,
)
from src.core.setup import create_training_setup, get_experiment_config

__all__ = [
    # Protocols
    "Agent",
    "OnPolicyAgent",
    "OffPolicyAgent",
    "HeuristicPolicy",
    "is_on_policy_agent",
    "is_off_policy_agent",
    "is_heuristic_policy",
    # Factory
    "AgentFactory",
    "register_builtin_agents",
    # Setup
    "create_training_setup",
    "get_experiment_config",
]
