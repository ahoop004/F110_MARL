"""Core infrastructure for the F110 training pipeline.

Load public exports on demand so configuration helpers do not initialize the
simulator or fixed-policy registry merely by importing the core package.
"""

__all__ = ["AgentFactory", "register_builtin_agents", "create_training_setup"]


def __getattr__(name: str):
    if name in {"AgentFactory", "register_builtin_agents"}:
        from core.config import AgentFactory, register_builtin_agents
        return {"AgentFactory": AgentFactory,
                "register_builtin_agents": register_builtin_agents}[name]
    if name == "create_training_setup":
        from core.setup import create_training_setup
        return create_training_setup
    raise AttributeError(f"module 'core' has no attribute {name!r}")
