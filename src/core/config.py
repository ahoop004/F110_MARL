"""Small config helpers and heuristic-agent factory for the active training path."""
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
    """Factory for fixed-policy agents used by ``core.setup``.

    Pure PyTorch RL agents are instantiated directly in ``run.py`` because their
    constructors need dimensions and bounds resolved from the composed scenario.
    """

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
    """Register built-in fixed-policy agents."""
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


# Register agents on import
register_builtin_agents()


class EnvironmentFactory:
    """Factory for creating F110 environments."""

    @staticmethod
    def create(config: Dict[str, Any]) -> Any:
        """Create F110 parallel environment from config.

        Args:
            config: Environment configuration dictionary

        Returns:
            env: F110ParallelEnv instance
        """
        from env.f110ParallelEnv import F110ParallelEnv

        # Extract environment parameters
        env_config = {
            'map': config.get('map', 'maps/example_map.yaml'),
            'n_agents': config.get('num_agents', config.get('n_agents', 1)),  # Support both num_agents and n_agents
            'timestep': config.get('timestep', 0.01),
            'ego_idx': config.get('ego_idx', 0),
            'integrator': config.get('integrator', 'rk4'),
            'render_mode': config.get('render_mode', None),
        }

        # Pass through additional config keys
        for key in ['control_mode', 'observation_config', 'reset_config', 'start_poses', 'random_spawn']:
            if key in config:
                env_config[key] = config[key]

        return F110ParallelEnv(**env_config)
