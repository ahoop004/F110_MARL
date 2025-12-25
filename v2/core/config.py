"""Simple configuration system - replaces complex config_models.py and builders.py."""
from typing import Any, Dict
import yaml
from pathlib import Path


def load_yaml(path: str) -> Dict[str, Any]:
    """Load YAML configuration file.

    Args:
        path: Path to YAML file

    Returns:
        config: Configuration dictionary
    """
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def resolve_paths(config: Dict[str, Any], base_dir: str = ".") -> Dict[str, Any]:
    """Resolve relative paths in config.

    Args:
        config: Configuration dictionary
        base_dir: Base directory for relative paths

    Returns:
        config: Configuration with resolved paths
    """
    base_path = Path(base_dir)

    # Common path fields to resolve
    path_fields = ['map', 'checkpoint_path', 'log_dir', 'scenario_path']

    def resolve_recursive(d: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively resolve paths in nested dict."""
        for key, value in d.items():
            if isinstance(value, dict):
                d[key] = resolve_recursive(value)
            elif isinstance(value, str) and key in path_fields:
                # Resolve relative path
                if not Path(value).is_absolute():
                    d[key] = str(base_path / value)
        return d

    return resolve_recursive(config)


class AgentFactory:
    """Simple agent factory - replaces builders.py (1,586 lines -> ~100 lines)."""

    # Agent class registry
    _registry = {}

    @classmethod
    def register(cls, name: str, agent_class: type):
        """Register an agent class.

        Args:
            name: Agent name (e.g., "ppo", "td3", "dqn")
            agent_class: Agent class to register
        """
        cls._registry[name.lower()] = agent_class

    @classmethod
    def create(cls, agent_type: str, config: Dict[str, Any]) -> Any:
        """Create an agent from config.

        Args:
            agent_type: Agent type (e.g., "ppo", "td3")
            config: Agent configuration dictionary

        Returns:
            agent: Instantiated agent

        Raises:
            ValueError: If agent_type not registered
        """
        agent_type_lower = agent_type.lower()
        if agent_type_lower not in cls._registry:
            raise ValueError(
                f"Unknown agent type: {agent_type}. "
                f"Available types: {list(cls._registry.keys())}"
            )

        agent_class = cls._registry[agent_type_lower]
        return agent_class(config)

    @classmethod
    def available_agents(cls) -> list:
        """Get list of registered agent types."""
        return list(cls._registry.keys())


# Auto-register agents
def register_builtin_agents():
    """Register all built-in agents."""
    try:
        from v2.agents.ppo.ppo import PPOAgent
        AgentFactory.register("ppo", PPOAgent)
    except ImportError:
        pass

    try:
        from v2.agents.ppo.rec_ppo import RecurrentPPOAgent
        AgentFactory.register("rec_ppo", RecurrentPPOAgent)
        AgentFactory.register("recurrent_ppo", RecurrentPPOAgent)
    except ImportError:
        pass

    try:
        from v2.agents.td3.td3 import TD3Agent
        AgentFactory.register("td3", TD3Agent)
    except ImportError:
        pass

    try:
        from v2.agents.sac.sac import SACAgent
        AgentFactory.register("sac", SACAgent)
    except ImportError:
        pass

    try:
        from v2.agents.dqn.dqn import DQNAgent
        AgentFactory.register("dqn", DQNAgent)
    except ImportError:
        pass

    try:
        from v2.agents.rainbow.r_dqn import RainbowDQNAgent
        AgentFactory.register("rainbow", RainbowDQNAgent)
        AgentFactory.register("rainbow_dqn", RainbowDQNAgent)
    except ImportError:
        pass


# Register agents on import
register_builtin_agents()
