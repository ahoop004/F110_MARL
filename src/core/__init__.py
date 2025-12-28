"""Core infrastructure for v2 training pipeline.

This module provides:
- Agent protocol (clean interface all agents must implement)
- Training loops (simple, no wrapper layers)
- Configuration system (simple YAML loading and agent factory)
"""

from core.protocol import (
    Agent,
    OnPolicyAgent,
    OffPolicyAgent,
    is_on_policy_agent,
    is_off_policy_agent,
)
from core.training import TrainingLoop, EvaluationLoop
from core.enhanced_training import EnhancedTrainingLoop
from core.config import (
    load_yaml,
    resolve_paths,
    AgentFactory,
    EnvironmentFactory,
    WrapperFactory,
)
from core.setup import create_training_setup, get_experiment_config
from core.utils import (
    save_checkpoint,
    load_checkpoint,
    SimpleLogger,
    set_random_seeds,
    compute_episode_metrics,
)
from core.observations import (
    compute_obs_dim,
    load_observation_preset,
    merge_observation_config,
    get_observation_config,
    OBSERVATION_PRESETS,
)

__all__ = [
    # Protocol
    "Agent",
    "OnPolicyAgent",
    "OffPolicyAgent",
    "is_on_policy_agent",
    "is_off_policy_agent",
    # Training
    "TrainingLoop",
    "EvaluationLoop",
    "EnhancedTrainingLoop",
    # Config & Factories
    "load_yaml",
    "resolve_paths",
    "AgentFactory",
    "EnvironmentFactory",
    "WrapperFactory",
    "create_training_setup",
    # Utils
    "save_checkpoint",
    "load_checkpoint",
    "SimpleLogger",
    "set_random_seeds",
    "compute_episode_metrics",
    # Observations
    "compute_obs_dim",
    "load_observation_preset",
    "merge_observation_config",
    "get_observation_config",
    "OBSERVATION_PRESETS",
]
