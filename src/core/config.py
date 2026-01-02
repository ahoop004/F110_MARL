"""Simple configuration system - replaces complex config_models.py and builders.py."""
from typing import Any, Dict, Optional, Tuple
import yaml
from pathlib import Path
import numpy as np


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
        from agents.ppo.ppo import PPOAgent
        AgentFactory.register("ppo", PPOAgent)
    except ImportError:
        pass

    try:
        from agents.ppo.rec_ppo import RecurrentPPOAgent
        AgentFactory.register("rec_ppo", RecurrentPPOAgent)
        AgentFactory.register("recurrent_ppo", RecurrentPPOAgent)
    except ImportError:
        pass

    try:
        from agents.td3.td3 import TD3Agent
        AgentFactory.register("td3", TD3Agent)
    except ImportError:
        pass

    try:
        from agents.sac.sac import SACAgent
        AgentFactory.register("sac", SACAgent)
    except ImportError:
        pass

    try:
        from agents.dqn.dqn import DQNAgent
        AgentFactory.register("dqn", DQNAgent)
    except ImportError:
        pass

    try:
        from agents.rainbow.r_dqn import RainbowDQNAgent
        AgentFactory.register("rainbow", RainbowDQNAgent)
        AgentFactory.register("rainbow_dqn", RainbowDQNAgent)
    except ImportError:
        pass

    try:
        from agents.ftg import FTGAgent
        AgentFactory.register("ftg", FTGAgent)
        AgentFactory.register("follow_gap", FTGAgent)
        AgentFactory.register("gap_follow", FTGAgent)
        AgentFactory.register("followthegap", FTGAgent)
    except ImportError:
        pass

    try:
        from agents.episodic import WaveletEpisodicAgent
        AgentFactory.register("wavelet_episodic", WaveletEpisodicAgent)
        AgentFactory.register("wavelet", WaveletEpisodicAgent)
        AgentFactory.register("episodic", WaveletEpisodicAgent)
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


class WrapperFactory:
    """Factory for applying observation/action/reward wrappers."""

    @staticmethod
    def wrap_observation(env: Any, config: Dict[str, Any]) -> Any:
        """Apply observation wrappers to environment.

        Args:
            env: Environment to wrap
            config: Observation wrapper configuration

        Returns:
            wrapped_env: Environment with observation wrappers applied
        """
        if not config or not config.get('enabled', False):
            return env

        from wrappers.observation import ObsWrapper

        obs_config = config.get('config', {})
        return ObsWrapper(env, **obs_config)

    @staticmethod
    def wrap_action(env: Any, config: Dict[str, Any]) -> Any:
        """Apply action wrappers to environment.

        Args:
            env: Environment to wrap
            config: Action wrapper configuration

        Returns:
            wrapped_env: Environment with action wrappers applied
        """
        if not config or not config.get('enabled', False):
            return env

        # Action wrappers would go here
        # Currently F110 doesn't have a standard action wrapper
        return env

    # REMOVED: wrap_reward() method
    # The old task-based reward system has been removed.
    # Use src/rewards/ with build_reward_strategy() instead.
    # See docs/REWARD_SYSTEM_REMOVAL.md for migration instructions.

    @staticmethod
    def wrap_all(env: Any, wrapper_configs: Dict[str, Any]) -> Any:
        """Apply all configured wrappers to environment.

        Args:
            env: Environment to wrap
            wrapper_configs: Dictionary of wrapper configurations
                {
                    'observation': {...},
                    'action': {...},
                    'reward': {...}
                }

        Returns:
            wrapped_env: Fully wrapped environment
        """
        # Apply wrappers in order: observation -> action
        if 'observation' in wrapper_configs:
            env = WrapperFactory.wrap_observation(env, wrapper_configs['observation'])

        if 'action' in wrapper_configs:
            env = WrapperFactory.wrap_action(env, wrapper_configs['action'])

        if 'reward' in wrapper_configs:
            import warnings
            warnings.warn(
                "Reward wrapper in config is deprecated and has been removed. "
                "Use the new reward system in src/rewards/ instead. "
                "See docs/REWARD_SYSTEM_REMOVAL.md for migration instructions.",
                DeprecationWarning,
                stacklevel=2
            )

        return env


def create_training_setup(config_path: str) -> Dict[str, Any]:
    """Create complete training setup from YAML config.

    This is the main entry point that ties everything together.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        setup: Dictionary containing:
            - env: F110ParallelEnv
            - agents: Dict[str, Agent]
            - config: Parsed configuration
    """
    # Load and resolve config
    config = load_yaml(config_path)
    config = resolve_paths(config, base_dir=str(Path(config_path).parent))

    # Create environment
    env_config = config.get('environment', {})
    env = EnvironmentFactory.create(env_config)

    # Apply wrappers if configured
    wrapper_configs = config.get('wrappers', {})
    if wrapper_configs:
        env = WrapperFactory.wrap_all(env, wrapper_configs)

    # Create agents
    agents = {}
    agents_config = config.get('agents', {})

    for agent_id, agent_cfg in agents_config.items():
        agent_type = agent_cfg.get('type', 'ppo')
        agent_params = agent_cfg.get('params', {})

        # Create agent
        agent = AgentFactory.create(agent_type, agent_params)
        agents[agent_id] = agent

    return {
        'env': env,
        'agents': agents,
        'config': config,
    }
