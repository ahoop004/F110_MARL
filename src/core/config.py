"""AgentFactory registry for fixed-policy agents.

This module is intentionally small: only the ``AgentFactory`` registry and
``register_builtin_agents`` live here.  All other config helpers have moved
to dedicated modules:

- YAML loading / scenario expansion → ``src/core/scenario.py``
- Path resolution → ``src/core/map_selection.py``
- Env kwargs construction → ``src/core/env_builder.py``
- Agent role split / builder helpers → ``src/core/agent_builder.py``
"""
from __future__ import annotations

from typing import Any, Dict


class AgentFactory:
    """Registry for fixed-policy agent constructors.

    Pure PyTorch RL agents are instantiated directly in ``run.py`` because
    their constructors need dimensions and bounds resolved from the composed
    scenario.  Heuristic agents are registered here and instantiated by
    :func:`~src.core.agent_builder.build_fixed_policy_agents`.
    """

    _registry: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str, agent_class: type) -> None:
        """Register *agent_class* under *name* (case-insensitive)."""
        cls._registry[name.lower()] = agent_class

    @classmethod
    def create(cls, agent_type: str, config: Dict[str, Any]) -> Any:
        """Instantiate the agent registered under *agent_type*.

        Raises
        ------
        ValueError
            When *agent_type* is not in the registry.
        """
        key = agent_type.lower()
        if key not in cls._registry:
            raise ValueError(
                f"Unknown agent type: {agent_type!r}. "
                f"Available: {sorted(cls._registry)}"
            )
        return cls._registry[key](config)

    @classmethod
    def available_agents(cls) -> list:
        """Return sorted list of registered agent-type names."""
        return sorted(cls._registry)


def register_builtin_agents() -> None:
    """Register built-in fixed-policy agents into :class:`AgentFactory`."""
    try:
        from agents.ftg import FTGAgent
        AgentFactory.register("ftg", FTGAgent)
        AgentFactory.register("follow_gap", FTGAgent)
        AgentFactory.register("gap_follow", FTGAgent)
        AgentFactory.register("followthegap", FTGAgent)
    except ImportError:
        pass

    try:
        from agents.waypoint import PurePursuitAgent, StanleyAgent, HybridPPFTGAgent
        AgentFactory.register("pure_pursuit", PurePursuitAgent)
        AgentFactory.register("stanley", StanleyAgent)
        AgentFactory.register("hybrid_pp_ftg", HybridPPFTGAgent)
    except ImportError:
        pass


# Register on import so callers don't have to remember to call this explicitly.
register_builtin_agents()
