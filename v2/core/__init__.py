"""Core infrastructure for v2 training pipeline.

This module provides:
- Agent protocol (clean interface all agents must implement)
- Training loops (simple, no wrapper layers)
- Configuration system (simple YAML loading and agent factory)
"""

from v2.core.protocol import (
    Agent,
    OnPolicyAgent,
    OffPolicyAgent,
    is_on_policy_agent,
    is_off_policy_agent,
)
from v2.core.training import TrainingLoop, EvaluationLoop
from v2.core.config import load_yaml, resolve_paths, AgentFactory

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
    # Config
    "load_yaml",
    "resolve_paths",
    "AgentFactory",
]
