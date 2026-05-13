"""Training setup builder - creates environment and agents from scenario config."""
from typing import Any, Dict, Optional, Tuple, List
import math
from pathlib import Path
import yaml
import numpy as np

from src.env import F110ParallelEnv
from src.core.config import AgentFactory, register_builtin_agents
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


def _discover_map_bundles(env_config: Dict[str, Any]) -> List[str]:
    map_root = env_config.get("map_dir") or env_config.get("map_root") or "maps"
    map_dir = Path(str(map_root)).expanduser()
    if not map_dir.is_absolute():
        map_dir = (Path.cwd() / map_dir).resolve()

    bundles: List[str] = []
    for entry in sorted(map_dir.iterdir()):
        if not entry.is_dir():
            continue
        yaml_files = sorted(entry.glob("*.yaml"))
        if not yaml_files:
            continue
        yaml_path = yaml_files[0]
        try:
            metadata = yaml.safe_load(yaml_path.read_text())
        except Exception:
            continue
        if not isinstance(metadata, dict):
            continue
        image_field = metadata.get("image")
        if image_field:
            image_path = (yaml_path.parent / image_field).expanduser().resolve()
        else:
            image_path = None
            for ext in (".png", ".pgm", ".jpg", ".jpeg"):
                candidate = yaml_path.with_suffix(ext)
                if candidate.exists():
                    image_path = candidate
                    break
        if image_path is None or not image_path.exists():
            continue
        stem = yaml_path.stem
        centerline_path = yaml_path.with_name(f"{stem}_centerline.csv")
        walls_path = yaml_path.with_name(f"{stem}_walls.csv")
        if not centerline_path.exists() or not walls_path.exists():
            continue
        bundles.append(entry.name)

    return bundles


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


def _normalize_maps_key(env_config: Dict[str, Any]) -> Dict[str, Any]:
    """Translate the new `maps:` key into the legacy map/map_bundles representation."""
    maps_raw = env_config.get("maps")
    if maps_raw is None:
        return env_config

    env_config = dict(env_config)
    env_config.pop("maps")

    if isinstance(maps_raw, str) and maps_raw.strip().lower() in {"auto", "all"}:
        env_config["map_bundles"] = True
        return env_config

    maps_list = [maps_raw] if isinstance(maps_raw, str) else list(maps_raw)

    if len(maps_list) == 1:
        bundle = str(maps_list[0]).strip()
        map_dir = Path(str(env_config.get("map_dir", "maps"))).expanduser()
        if not map_dir.is_absolute():
            map_dir = (Path.cwd() / map_dir).resolve()
        try:
            yaml_path = _resolve_bundle_yaml(map_dir, bundle)
            env_config["map"] = str(yaml_path)
            env_config.setdefault("map_dir", str(map_dir))
        except FileNotFoundError:
            env_config["map"] = bundle
    else:
        env_config["map_bundles"] = maps_list

    return env_config


def _apply_map_split(
    env_config: Dict[str, Any],
    experiment_config: Dict[str, Any],
    mode: str,
) -> Dict[str, Any]:
    env_config = _normalize_maps_key(env_config)
    map_bundles_raw = env_config.get("map_bundles")
    if map_bundles_raw is None:
        map_bundles = _coerce_bundle_list(map_bundles_raw)
    elif (
        map_bundles_raw is True
        or (isinstance(map_bundles_raw, str) and map_bundles_raw.strip().lower() in {"auto", "all"})
    ):
        map_bundles = _discover_map_bundles(env_config)
        env_config = dict(env_config)
        env_config["map_bundles"] = list(map_bundles)
    else:
        map_bundles = _coerce_bundle_list(map_bundles_raw)
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
) -> Tuple[F110ParallelEnv, Dict[str, Any], Dict]:
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

    # Build environment configuration.
    # n_agents is always derived from the agents dict (max car index + 1) so
    # that commenting out agents in the scenario YAML reduces the car count.
    # The env requires contiguous IDs car_0..car_{n-1}, so car_3 and car_4
    # in a scenario mean n_agents=5 (car_0..car_4 all exist; gaps sit still).
    # To get exactly N cars, use car_0..car_{N-1} in the scenario agents block.
    import re as _re
    _indices = []
    for _aid in (agent_configs or {}):
        _m = _re.search(r'(\d+)$', str(_aid))
        if _m:
            _indices.append(int(_m.group(1)))
    num_agents = max(_indices) + 1 if _indices else int(
        env_config.get('num_agents', env_config.get('n_agents', 1))
    )
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
        "random_spawn",
        "random_spawn_allow_reuse",
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
        map_dir_value = env_kwargs.get("map_dir")
        if map_dir_value:
            env_kwargs['map'] = _relative_yaml_name(Path(map_dir_value), map_data.yaml_path)
            env_kwargs['map_yaml'] = env_kwargs['map']
        else:
            env_kwargs['map'] = str(map_data.yaml_path)

    # Load spawn points from map YAML if specified
    if 'spawn_points' in env_config:
        spawn_names = env_config['spawn_points']
        map_path = env_config['map']
        start_poses = load_spawn_points_from_map(map_path, spawn_names)
        env_kwargs['start_poses'] = start_poses
    elif 'start_poses' in env_config and 'start_poses' not in env_kwargs:
        env_kwargs['start_poses'] = np.array(env_config['start_poses'], dtype=np.float64)

    # Create environment
    env = F110ParallelEnv(**env_kwargs)
    if map_data is not None and map_data.centerline is not None:
        env.set_centerline(map_data.centerline, path=map_data.centerline_path)
        env.register_centerline_usage(
            require_render=bool(env_config.get('centerline_render')),
            require_features=bool(env_config.get('centerline_features')),
        )

    # Pure PyTorch RL algorithms are instantiated by run.py, not here.
    # setup.py only creates heuristic / fixed-policy agents.
    _PYTORCH_RL_ALGOS = {"ppo", "a2c", "sac", "td3", "ddpg", "dqn", "qrdqn", "mappo"}

    # Create agents (heuristic/fixed-policy only — RL agents are created by run.py)
    _HEURISTIC_ALGOS = {"ftg", "follow_gap", "gap_follow", "followthegap",
                        "pure_pursuit", "stanley", "hybrid_pp_ftg"}
    agents = {}

    for agent_id, agent_config in agent_configs.items():
        algorithm = agent_config['algorithm'].lower()

        if algorithm in _PYTORCH_RL_ALGOS:
            continue

        if algorithm in _HEURISTIC_ALGOS:
            heuristic_kwargs = dict(agent_config.get('params', {}))
            agents[agent_id] = AgentFactory.create(algorithm, heuristic_kwargs)
            continue

    return env, agents, {}


def get_experiment_config(scenario: Dict[str, Any]) -> Dict[str, Any]:
    """Extract experiment configuration from scenario.

    Args:
        scenario: Scenario configuration

    Returns:
        Experiment configuration dict
    """
    return scenario.get('experiment', {})
