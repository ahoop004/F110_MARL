"""Training setup builder - creates environment and agents from scenario config."""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from env import F110ParallelEnv
from core.agent_builder import (
    build_fixed_policy_agents,
    get_fixed_agent_ids,
    get_trainable_agent_ids,
)
from core.config import register_builtin_agents
from core.env_builder import create_environment
from core.feature_requirements import derive_environment_feature_requirements
from core.map_selection import apply_map_split
from core.provenance import physics_contract
from wrappers.actions.composer import ActionComposer
from wrappers.observations.composer import ObservationComposer
from wrappers.rewards.composer import RewardComposer


def build_obs_composer(
    agent_cfg: Dict, env_config: Dict, scenario_dir: Path, action_dim: int = 2
) -> ObservationComposer:
    """Build one ObservationComposer for a single agent config."""
    obs_ref = agent_cfg.get("observation")
    if isinstance(obs_ref, str):
        obs_path = (scenario_dir / obs_ref).resolve()
        return ObservationComposer.from_file(str(obs_path), env_config, action_dim=action_dim)
    elif isinstance(obs_ref, dict):
        return ObservationComposer.from_config(obs_ref, env_config, action_dim=action_dim)
    raise ValueError("Agent 'observation' must be a file path or inline config dict.")


def build_reward_composer(agent_cfg: Dict, scenario_dir: Path) -> RewardComposer:
    """Build one RewardComposer for a single agent config."""
    reward_ref = agent_cfg.get("reward")
    if isinstance(reward_ref, str):
        reward_path = (scenario_dir / reward_ref).resolve()
        return RewardComposer.from_file(str(reward_path))
    elif isinstance(reward_ref, dict):
        return RewardComposer.from_config(reward_ref)
    raise ValueError("Agent 'reward' must be a file path or inline config dict.")


def build_obs_composers(
    agent_configs: Dict,
    trainable_ids: List[str],
    env_config: Dict,
    scenario_dir: Path,
    action_dim: int = 2,
) -> Dict[str, ObservationComposer]:
    """Build one ObservationComposer per trainable agent.

    Returns a dict keyed by agent_id.  Single-agent trainers index into it
    with their ``rl_agent_id``; MAPPO iterates over all entries.
    """
    return {
        aid: build_obs_composer(agent_configs[aid], env_config, scenario_dir, action_dim)
        for aid in trainable_ids
    }


def build_reward_composers(
    agent_configs: Dict,
    trainable_ids: List[str],
    scenario_dir: Path,
) -> Dict[str, RewardComposer]:
    """Build one RewardComposer per trainable agent.

    Returns a dict keyed by agent_id.
    """
    return {
        aid: build_reward_composer(agent_configs[aid], scenario_dir)
        for aid in trainable_ids
    }


def resolve_training_params(agent_cfg: Dict, scenario: Dict) -> Dict:
    """Merge training_defaults with agent params — agent params win."""
    defaults = scenario.get("training_defaults", {})
    params = agent_cfg.get("params", {})
    environment = scenario.get("environment", {})
    decision_dt = float(environment.get("timestep", 0.01)) * int(environment.get("action_repeat", 1))
    return {**defaults, **params, "_physics_contract": physics_contract(environment),
            "_action_contract": ActionComposer.contract_from_config(
        agent_cfg.get("action_constraints", {}), decision_dt,
    )}


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
    env_config["physics_phase"] = "eval" if mode in {"eval", "evaluation", "test"} else "train"
    evaluation = scenario.get("evaluation", {}) or {}
    if env_config["physics_phase"] == "eval" and "target_laps" in evaluation:
        # Continuous training can disable finishing and timeouts. Evaluation
        # explicitly restores a finite race for PPO and multi-agent MAPPO alike.
        env_config["target_laps"] = int(evaluation["target_laps"])
        env_config["episode_termination"] = {**env_config.get("episode_termination", {}),
                                             "lap_completion": True}
        if "max_steps" in evaluation:
            env_config["max_steps"] = int(evaluation["max_steps"])
    if env_config["physics_phase"] == "eval" and "terminate_on_collision" in evaluation:
        env_config["terminate_on_collision"] = evaluation["terminate_on_collision"]
    if env_config["physics_phase"] == "eval" and env_config.get("track_limits", {}).get("enabled"):
        # Default paper evaluation records excursions; safety evaluation can
        # explicitly retain boundary termination as well as collision checks.
        env_config["track_limits"] = {
            "enabled": True,
            "terminate": bool(evaluation.get("terminate_on_track_limit", False)),
        }
        env_config["episode_termination"] = {**env_config.get("episode_termination", {}),
                                             "lap_completion": True}
        env_config["target_laps"] = int(evaluation.get("target_laps", 20))
        env_config["max_steps"] = int(evaluation.get("max_steps", 16000))
    if env_config["physics_phase"] == "eval" and "lap_completion" in evaluation:
        env_config["episode_termination"] = {**env_config.get("episode_termination", {}),
                                             "lap_completion": evaluation["lap_completion"]}
    if env_config["physics_phase"] == "eval" and "episode_termination_mode" in evaluation:
        env_config["episode_termination"] = {**env_config.get("episode_termination", {}),
                                             "mode": evaluation["episode_termination_mode"]}
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

    agents = build_fixed_policy_agents(agent_configs, vehicle_params=env.params)
    return env, agents, {}
