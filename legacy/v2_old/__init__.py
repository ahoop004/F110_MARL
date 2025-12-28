"""v2 - Refactored F110 MARL Training Pipeline

A clean, simplified architecture for multi-agent reinforcement learning in F1TENTH racing.

Key improvements over v1:
- 4 layers instead of 7 (removed trainer wrappers and redundant factories)
- ~3,500 fewer lines of code
- Protocol-based agent interface (no wrapper classes needed)
- Simple training loops (replaces 2,011-line train_runner.py)
- Clean configuration system (replaces complex builders.py)

Architecture:
    1. Core (protocol, training loops, config)
    2. Agents (RL algorithms: PPO, TD3, SAC, DQN, Rainbow)
    3. Environment (F110ParallelEnv, wrappers)
    4. Physics & Tasks (vehicle dynamics, rewards)
"""

__version__ = "2.0.0-alpha"

# Core exports
from v2.core import (
    Agent,
    OnPolicyAgent,
    OffPolicyAgent,
    TrainingLoop,
    EvaluationLoop,
    AgentFactory,
    load_yaml,
)

# Environment
from v2.env.f110ParallelEnv import F110ParallelEnv

__all__ = [
    # Core
    "Agent",
    "OnPolicyAgent",
    "OffPolicyAgent",
    "TrainingLoop",
    "EvaluationLoop",
    "AgentFactory",
    "load_yaml",
    # Environment
    "F110ParallelEnv",
]
