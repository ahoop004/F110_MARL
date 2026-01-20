"""Training setup builder - creates environment and agents from scenario config."""
from typing import Any, Dict, Optional, Tuple, List
import math
from pathlib import Path
import yaml
import numpy as np

from src.env import F110ParallelEnv
from src.core.config import AgentFactory, register_builtin_agents
from src.rewards import RewardStrategy, build_reward_strategy
from src.utils.map_loader import MapLoader


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


def _coerce_bundle_list(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            raise ValueError("environment.map_bundles cannot be empty")
        return [value]
    if isinstance(value, (list, tuple)):
        bundles = [str(item).strip() for item in value]
        bundles = [item for item in bundles if item]
        if not bundles:
            raise ValueError("environment.map_bundles cannot be empty")
        return bundles
    raise TypeError("environment.map_bundles must be a string or list of strings")


def _resolve_bundle_yaml(map_dir: Path, bundle: str) -> Path:
    bundle_str = str(bundle).strip()
    if not bundle_str:
        raise ValueError("map bundle identifier cannot be empty")

    candidate_path = Path(bundle_str)
    if candidate_path.is_absolute():
        resolved = candidate_path
        if resolved.is_file():
            return resolved
        if resolved.with_suffix(".yaml").is_file():
            return resolved.with_suffix(".yaml")
        raise FileNotFoundError(f"Map YAML not found for bundle '{bundle_str}': {resolved}")

    if candidate_path.suffix:
        resolved = (map_dir / candidate_path).resolve()
        if resolved.is_file():
            return resolved

    resolved = (map_dir / candidate_path).resolve()
    if resolved.is_file():
        return resolved

    yaml_with_suffix = resolved.with_suffix(".yaml")
    if yaml_with_suffix.is_file():
        return yaml_with_suffix

    if resolved.is_dir():
        yaml_files = sorted(resolved.glob("*.yaml"))
        if yaml_files:
            return yaml_files[0].resolve()

    search_name = candidate_path.name
    matches = sorted(map_dir.rglob(f"{search_name}.yaml"))
    if matches:
        return matches[0].resolve()

    raise FileNotFoundError(f"Map YAML not found for bundle '{bundle_str}' within {map_dir}")


def _relative_yaml_name(map_dir: Path, yaml_path: Path) -> str:
    try:
        return yaml_path.relative_to(map_dir).as_posix()
    except ValueError:
        return str(yaml_path)


def _apply_map_bundle(env_config: Dict[str, Any], bundle: str) -> Dict[str, Any]:
    map_root = env_config.get("map_dir") or env_config.get("map_root") or "maps"
    map_dir = Path(str(map_root)).expanduser()
    if not map_dir.is_absolute():
        map_dir = (Path.cwd() / map_dir).resolve()

    yaml_path = _resolve_bundle_yaml(map_dir, bundle)
    env_config["map_dir"] = str(map_dir)
    env_config["map_yaml"] = _relative_yaml_name(map_dir, yaml_path)
    env_config["map"] = env_config["map_yaml"]
    env_config["map_bundle"] = str(bundle)
    return env_config


def _apply_map_split(
    env_config: Dict[str, Any],
    experiment_config: Dict[str, Any],
    mode: str,
) -> Dict[str, Any]:
    map_bundles = _coerce_bundle_list(env_config.get("map_bundles"))
    if not map_bundles:
        return env_config

    split_cfg = env_config.get("map_split") or {}
    if not isinstance(split_cfg, dict):
        raise TypeError("environment.map_split must be a mapping when provided")

    train_ratio = split_cfg.get("train_ratio", 0.8)
    try:
        train_ratio = float(train_ratio)
    except (TypeError, ValueError):
        train_ratio = 0.8
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("map_split.train_ratio must be between 0 and 1")

    seed = split_cfg.get("seed")
    if seed is None:
        seed = experiment_config.get("seed", env_config.get("seed", 0))
    try:
        seed = int(seed)
    except (TypeError, ValueError):
        seed = 0

    shuffle = split_cfg.get("shuffle", True)
    rng = np.random.default_rng(seed)
    bundles = list(map_bundles)
    if shuffle:
        rng.shuffle(bundles)

    total = len(bundles)
    if total == 1:
        train_bundles = bundles
        eval_bundles: List[str] = []
    else:
        train_count = int(math.floor(train_ratio * total))
        train_count = max(1, min(total - 1, train_count))
        train_bundles = bundles[:train_count]
        eval_bundles = bundles[train_count:]

    is_eval = str(mode).lower() in {"eval", "evaluation", "test"}
    active_bundles = eval_bundles if is_eval else train_bundles
    if not active_bundles:
        active_bundles = train_bundles

    pick_key = "eval_pick" if is_eval else "train_pick"
    pick_strategy = split_cfg.get(pick_key, split_cfg.get("pick", "first"))
    if str(env_config.get("map_cycle", "")).lower() == "per_episode":
        pick_strategy = env_config.get("map_pick", pick_strategy)
    if pick_strategy not in {"first", "random"}:
        pick_strategy = "first"
    if pick_strategy == "random":
        chosen = active_bundles[int(rng.integers(0, len(active_bundles)))]
    else:
        chosen = active_bundles[0]

    env_config = dict(env_config)
    env_config["map_bundles_train"] = list(train_bundles)
    env_config["map_bundles_eval"] = list(eval_bundles)
    env_config["map_bundle_active"] = chosen
    env_config["map_split_mode"] = "eval" if is_eval else "train"
    return _apply_map_bundle(env_config, chosen)


def create_training_setup(
    scenario: Dict[str, Any],
    *,
    mode: str = "train",
) -> Tuple[F110ParallelEnv, Dict[str, Any], Dict[str, RewardStrategy]]:
    """Create training setup from scenario configuration.

    Args:
        scenario: Expanded scenario configuration with:
            - experiment: {name, episodes, seed}
            - environment: {map, num_agents, max_steps, ...}
            - agents: {agent_id: {algorithm, params, observation, reward, ...}}
        mode: "train" or "eval" (used for map bundle splits)

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
    env_config = dict(scenario['environment'])
    env_config = _apply_map_split(env_config, experiment_config, mode)
    agent_configs = scenario['agents']

    # Set random seed if specified
    seed = experiment_config.get('seed')
    if seed is not None:
        import numpy as np
        import random
        np.random.seed(seed)
        random.seed(seed)
        try:
            import torch
        except ImportError:
            torch = None
        if torch is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    # Build environment configuration
    num_agents = env_config.get('num_agents', env_config.get('n_agents', 1))
    env_seed = env_config.get('seed', seed)
    env_kwargs = {
        'map': env_config['map'],
        'n_agents': num_agents,
        'timestep': env_config.get('timestep', 0.01),
        'max_steps': env_config.get('max_steps', 5000),
    }
    if 'map_dir' in env_config:
        env_kwargs['map_dir'] = env_config['map_dir']
    if 'map_yaml' in env_config:
        env_kwargs['map_yaml'] = env_config['map_yaml']
    if 'map_ext' in env_config:
        env_kwargs['map_ext'] = env_config['map_ext']
    if env_seed is not None:
        env_kwargs['seed'] = env_seed

    # Add optional environment parameters
    if 'lidar_beams' in env_config:
        env_kwargs['lidar_beams'] = env_config['lidar_beams']
    if 'lidar_range' in env_config:
        env_kwargs['lidar_range'] = env_config['lidar_range']
    if 'render' in env_config:
        env_kwargs['render_mode'] = 'human' if env_config['render'] else None
    if 'vehicle_params' in env_config:
        env_kwargs['vehicle_params'] = env_config['vehicle_params']
    passthrough_keys = [
        "map_root",
        "map_bundle",
        "map_bundle_active",
        "map_bundles",
        "map_bundles_train",
        "map_bundles_eval",
        "map_split_mode",
        "map_cycle",
        "map_pick",
        "epoch_shuffle",
        "centerline_autoload",
        "centerline_csv",
        "centerline_render",
        "centerline_features",
        "walls_autoload",
        "walls_csv",
        "track_threshold",
        "track_inverted",
        "spawn_policy",
        "spawn_centerline",
        "spawn_offsets",
        "spawn_target",
        "spawn_ego",
    ]
    for key in passthrough_keys:
        if key in env_config and key not in env_kwargs:
            env_kwargs[key] = env_config[key]

    map_data = None
    centerline_requested = bool(
        env_config.get('centerline_autoload')
        or env_config.get('centerline_csv')
        or env_config.get('centerline_render')
        or env_config.get('centerline_features')
    )
    if centerline_requested:
        map_loader_cfg = dict(env_config)
        map_loader_cfg['centerline_autoload'] = bool(
            env_config.get('centerline_autoload', False)
            or env_config.get('centerline_csv')
            or env_config.get('centerline_render')
            or env_config.get('centerline_features')
        )
        map_value = map_loader_cfg.get('map')
        if isinstance(map_value, str):
            map_path = Path(map_value)
            if map_path.parent != Path(".") and not map_loader_cfg.get('map_dir'):
                map_file = map_path if map_path.suffix else map_path.with_suffix(".yaml")
                map_loader_cfg['map_dir'] = str(map_file.parent)
                if not map_loader_cfg.get('map_yaml'):
                    map_loader_cfg['map_yaml'] = map_file.name
                map_loader_cfg['map'] = map_file.name
        try:
            map_loader = MapLoader(base_dir=Path.cwd())
            map_data = map_loader.load(map_loader_cfg)
        except Exception as exc:
            print(f"Warning: failed to load centerline data: {exc}")
            map_data = None

    if map_data is not None:
        env_kwargs['map_data'] = map_data
        env_kwargs['map'] = map_data.yaml_path.name

    # Load spawn points from map YAML if specified
    if 'spawn_points' in env_config:
        spawn_names = env_config['spawn_points']
        map_path = env_config['map']
        start_poses = load_spawn_points_from_map(map_path, spawn_names)
        env_kwargs['start_poses'] = start_poses

    # Create environment
    env = F110ParallelEnv(**env_kwargs)
    if map_data is not None and map_data.centerline is not None:
        env.set_centerline(map_data.centerline, path=map_data.centerline_path)
        env.register_centerline_usage(
            require_render=bool(env_config.get('centerline_render')),
            require_features=bool(env_config.get('centerline_features')),
        )

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

        frame_stack = agent_config.get('frame_stack', 1)
        if frame_stack is None:
            frame_stack = 1
        try:
            frame_stack = int(frame_stack)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"frame_stack must be an integer >= 1 (agent {agent_id})") from exc
        if frame_stack < 1:
            raise ValueError(f"frame_stack must be >= 1 (agent {agent_id})")

        if frame_stack > 1:
            obs_dim *= frame_stack

        # Add dimension parameters (support both naming conventions)
        agent_kwargs['obs_dim'] = obs_dim
        agent_kwargs['action_dim'] = action_dim
        agent_kwargs['act_dim'] = action_dim  # Alias for PPO
        agent_kwargs['frame_stack'] = frame_stack

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
