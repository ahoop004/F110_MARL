"""Training setup builder - creates environment and agents from scenario config."""
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from src.env import F110ParallelEnv
from src.core.agent_builder import (
    build_fixed_policy_agents,
    get_fixed_agent_ids,
    get_trainable_agent_ids,
)
from src.core.config import register_builtin_agents
from src.core.env_builder import create_environment
from src.core.feature_requirements import derive_environment_feature_requirements
from src.core.map_selection import apply_map_split


def create_training_setup(
    scenario: Dict[str, Any],
    *,
    mode: str = "train",
    scenario_dir: Optional[Path] = None,
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
    env_config = apply_map_split(env_config, experiment_config, mode)
    agent_configs = scenario['agents']
    env_config.setdefault("trainable_agents", get_trainable_agent_ids(agent_configs))
    env_config.setdefault("fixed_policy_agents", get_fixed_agent_ids(agent_configs))
    if scenario_dir is not None:
        requirements = derive_environment_feature_requirements(
            agent_configs,
            scenario_dir=Path(scenario_dir),
            centerline_render=bool(env_config.get("centerline_render", False)),
        )
        env_config["feature_requirements"] = requirements.as_dict()
        geometry_required = bool(
            requirements.requires_centerline_facts
            or requirements.requires_track_preview
            or requirements.requires_frenet_neighbors
            or requirements.centerline_render
        )
        if geometry_required:
            env_config["centerline_autoload"] = True
        if (
            requirements.requires_centerline_facts
            or requirements.requires_track_preview
            or requirements.requires_frenet_neighbors
        ):
            env_config["centerline_features"] = True

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

    env = create_environment(env_config, agent_configs, seed)

    # Wire per-agent target_id (set explicitly or via resolve_target_ids) into
    # the env's agent_target_index, which feeds target_pose/target_state/
    # relative_pose observations and target_collision/target_finished info.
    target_mapping = {
        aid: cfg["target_id"]
        for aid, cfg in agent_configs.items()
        if cfg.get("target_id")
    }
    if target_mapping:
        env.configure_agent_targets(target_mapping)

    agents = build_fixed_policy_agents(agent_configs)
    return env, agents, {}


def get_experiment_config(scenario: Dict[str, Any]) -> Dict[str, Any]:
    """Extract experiment configuration from scenario.

    Args:
        scenario: Scenario configuration

    Returns:
        Experiment configuration dict
    """
    return scenario.get('experiment', {})
