"""Training setup builder - creates environment and agents from scenario config."""
from typing import Any, Dict, Optional, Tuple, List
from pathlib import Path
import yaml
import numpy as np

from src.env import F110ParallelEnv
from src.core.config import AgentFactory, register_builtin_agents
from src.rewards import RewardStrategy, build_reward_strategy


def load_spawn_points_from_map(map_path: str, spawn_names: List[str]) -> np.ndarray:
    """Load spawn point poses from map YAML file.

    Args:
        map_path: Path to map YAML file (e.g., 'maps/line2/line2.yaml')
        spawn_names: List of spawn point names (e.g., ['spawn_2', 'spawn_1'])

    Returns:
        numpy array of poses with shape (N, 3) where N = len(spawn_names)
        Each pose is [x, y, theta]

    Raises:
        FileNotFoundError: If map file doesn't exist
        ValueError: If spawn point not found in map
    """
    map_yaml_path = Path(map_path)

    if not map_yaml_path.exists():
        raise FileNotFoundError(f"Map YAML not found: {map_path}")

    # Load map YAML
    with open(map_yaml_path, 'r') as f:
        map_data = yaml.safe_load(f)

    # Extract spawn points from annotations
    spawn_points = map_data.get('annotations', {}).get('spawn_points', [])

    # Build lookup dict
    spawn_dict = {sp['name']: sp['pose'] for sp in spawn_points}

    # Extract poses in order
    poses = []
    for name in spawn_names:
        if name not in spawn_dict:
            available = list(spawn_dict.keys())
            raise ValueError(
                f"Spawn point '{name}' not found in map. "
                f"Available: {available}"
            )
        poses.append(spawn_dict[name])

    return np.array(poses, dtype=np.float64)


def create_training_setup(scenario: Dict[str, Any]) -> Tuple[F110ParallelEnv, Dict[str, Any], Dict[str, RewardStrategy]]:
    """Create training setup from scenario configuration.

    Args:
        scenario: Expanded scenario configuration with:
            - experiment: {name, episodes, seed}
            - environment: {map, num_agents, max_steps, ...}
            - agents: {agent_id: {algorithm, params, observation, reward, ...}}

    Returns:
        Tuple of (env, agents, reward_strategies):
            - env: F110ParallelEnv instance
            - agents: Dict mapping agent_id -> agent instance
            - reward_strategies: Dict mapping agent_id -> RewardStrategy (for trainable agents)
    """
    # Register built-in agents
    register_builtin_agents()

    # Extract configuration sections
    experiment_config = scenario['experiment']
    env_config = scenario['environment']
    agent_configs = scenario['agents']

    # Set random seed if specified
    seed = experiment_config.get('seed')
    if seed is not None:
        import numpy as np
        import random
        np.random.seed(seed)
        random.seed(seed)

    # Build environment configuration
    num_agents = env_config.get('num_agents', env_config.get('n_agents', 1))
    env_kwargs = {
        'map': env_config['map'],
        'n_agents': num_agents,
        'timestep': env_config.get('timestep', 0.01),
        'max_steps': env_config.get('max_steps', 5000),
    }

    # Add optional environment parameters
    if 'lidar_beams' in env_config:
        env_kwargs['lidar_beams'] = env_config['lidar_beams']
    if 'lidar_range' in env_config:
        env_kwargs['lidar_range'] = env_config['lidar_range']
    if 'render' in env_config:
        env_kwargs['render_mode'] = 'human' if env_config['render'] else None
    if 'vehicle_params' in env_config:
        env_kwargs['vehicle_params'] = env_config['vehicle_params']

    # Load spawn points from map YAML if specified
    if 'spawn_points' in env_config:
        spawn_names = env_config['spawn_points']
        map_path = env_config['map']
        start_poses = load_spawn_points_from_map(map_path, spawn_names)
        env_kwargs['start_poses'] = start_poses

    # Create environment
    env = F110ParallelEnv(**env_kwargs)

    # Create agents
    agents = {}
    reward_strategies = {}
    target_mapping = {}  # Track agent -> target relationships

    for agent_id, agent_config in agent_configs.items():
        algorithm = agent_config['algorithm']

        # Build agent-specific configuration
        agent_kwargs = {
            'agent_id': agent_id,
        }

        # Add algorithm hyperparameters
        if 'params' in agent_config:
            agent_kwargs.update(agent_config['params'])

        # Add observation and action space info
        # Access observation_spaces directly as dict (PettingZoo compatibility)
        obs_space = env.observation_spaces.get(agent_id)
        action_space = env.action_spaces.get(agent_id)

        if obs_space is None or action_space is None:
            # Debug: print available spaces
            print(f"DEBUG: observation_spaces = {env.observation_spaces}")
            print(f"DEBUG: action_spaces = {env.action_spaces}")
            print(f"DEBUG: possible_agents = {env.possible_agents}")
            print(f"DEBUG: agent_id = {agent_id}")
            raise ValueError(
                f"Agent '{agent_id}' not found in environment spaces. "
                f"Available agents: {list(env.possible_agents)}"
            )

        agent_kwargs['observation_space'] = obs_space
        agent_kwargs['action_space'] = action_space

        # Extract dimensions for agents that need them
        # For Dict spaces, compute total dimension
        from gymnasium import spaces
        import numpy as np

        def get_space_dim(space):
            """Get total dimension of a gym space."""
            if isinstance(space, spaces.Dict):
                return sum(get_space_dim(s) for s in space.spaces.values())
            elif isinstance(space, spaces.Box):
                return int(np.prod(space.shape))
            elif isinstance(space, spaces.Discrete):
                return 1
            elif isinstance(space, spaces.MultiDiscrete):
                return len(space.nvec)
            else:
                return 1

        # Calculate observation dimension
        # If observation preset specified, use flattened dimension
        if 'observation' in agent_config and isinstance(agent_config['observation'], dict):
            obs_config = agent_config['observation']
            if 'preset' in obs_config:
                preset_name = obs_config['preset']
            elif len(obs_config) > 0:
                # Expanded preset - default to gaplock
                preset_name = 'gaplock'
            else:
                preset_name = None

            if preset_name:
                # Create a dummy observation and flatten it to get dimension
                from core.obs_flatten import flatten_observation
                dummy_obs = obs_space.sample()
                # Get target_id if specified
                target_id = agent_config.get('target_id')
                # Add dummy central_state if target exists
                if target_id:
                    # Create dummy central state (same structure as ego obs)
                    dummy_obs['central_state'] = obs_space.sample()
                flat_dummy = flatten_observation(dummy_obs, preset=preset_name, target_id=target_id)
                obs_dim = flat_dummy.shape[0]
            else:
                obs_dim = get_space_dim(obs_space)
        else:
            obs_dim = get_space_dim(obs_space)

        action_dim = get_space_dim(action_space)

        # Add dimension parameters (support both naming conventions)
        agent_kwargs['obs_dim'] = obs_dim
        agent_kwargs['action_dim'] = action_dim
        agent_kwargs['act_dim'] = action_dim  # Alias for PPO

        # Extract action bounds for continuous action spaces (SAC, TD3, etc.)
        if isinstance(action_space, spaces.Box):
            agent_kwargs['action_low'] = action_space.low
            agent_kwargs['action_high'] = action_space.high

        # Create agent
        agent = AgentFactory.create(algorithm, agent_kwargs)
        agents[agent_id] = agent

        # Build reward strategy if configured
        # Only trainable agents will have reward configs; FTG and other
        # non-trainable agents won't have reward configs in scenarios
        if 'reward' in agent_config:
            reward_config = agent_config['reward']

            # Get target agent ID if specified
            target_id = agent_config.get('target_id')

            # Store target mapping for environment configuration
            if target_id is not None:
                target_mapping[agent_id] = target_id

            # Build reward strategy
            reward_strategy = build_reward_strategy(
                reward_config,
                agent_id=agent_id,
                target_id=target_id,
            )
            reward_strategies[agent_id] = reward_strategy

    # Configure environment with agent target relationships
    if target_mapping:
        env.configure_agent_targets(target_mapping)

    return env, agents, reward_strategies


def get_experiment_config(scenario: Dict[str, Any]) -> Dict[str, Any]:
    """Extract experiment configuration from scenario.

    Args:
        scenario: Scenario configuration

    Returns:
        Experiment configuration dict
    """
    return scenario.get('experiment', {})
