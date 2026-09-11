"""Scenario configuration system for v2 training pipeline.

Provides shared YAML include loading and scenario validation
for algorithms, rewards, and observations. Scenarios define complete
training setups in a concise, readable format.
"""

from typing import Dict, Any, Optional
import copy
import yaml
from pathlib import Path



class ScenarioError(Exception):
    """Exception raised for scenario configuration errors."""
    pass


MAPPO_DEFAULTS: Dict[str, str] = {
    "reward_mode": "individual",
    "critic_mode": "agent_conditioned",
    "team_reward_reduction": "mean",
}


def resolve_mappo_config(scenario: Dict[str, Any]) -> Dict[str, str]:
    """Return the normalized MAPPO reward/critic experiment contract."""
    raw = scenario.get("mappo", {}) or {}
    if not isinstance(raw, dict):
        raise ScenarioError("'mappo' must be a dictionary when provided.")
    unknown = sorted(set(raw) - set(MAPPO_DEFAULTS))
    if unknown:
        raise ScenarioError(f"Unknown MAPPO config field(s): {unknown}.")
    config = dict(MAPPO_DEFAULTS)
    config.update({key: str(value).strip().lower() for key, value in raw.items()})
    return config


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge two dictionaries (override wins)."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_yaml_file(path_obj: Path) -> Dict[str, Any]:
    """Load a YAML file and ensure it returns a dict."""
    if not path_obj.exists():
        raise FileNotFoundError(f"Config file not found: {path_obj}")

    try:
        with open(path_obj, 'r') as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in config {path_obj}: {e}") from e

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML dictionary: {path_obj}")
    return data


def load_yaml_config(path_obj: Path, visited: Optional[set] = None) -> Dict[str, Any]:
    """Load YAML includes relative to each file; later values override earlier ones."""
    path_obj = Path(path_obj).resolve()
    visited = visited or set()
    if path_obj in visited:
        raise ValueError(f"Include cycle detected at: {path_obj}")
    visited.add(path_obj)

    data = _load_yaml_file(path_obj)
    includes = data.pop('includes', None)

    merged: Dict[str, Any] = {}
    if includes:
        if isinstance(includes, (str, Path)):
            includes = [includes]
        if not isinstance(includes, list):
            raise ValueError("'includes' must be a list of file paths")
        for include_path in includes:
            if not isinstance(include_path, (str, Path)):
                raise ValueError("'includes' entries must be file paths")
            include_obj = (path_obj.parent / include_path).resolve()
            merged = _deep_merge(merged, load_yaml_config(include_obj, visited))

    merged = _deep_merge(merged, data)
    visited.remove(path_obj)
    return merged


def load_scenario(path: str) -> Dict[str, Any]:
    """Load scenario from YAML file.

    Args:
        path: Path to YAML scenario file

    Returns:
        Scenario configuration dict

    Raises:
        ScenarioError: If file not found or invalid YAML

    Example:
        >>> scenario = load_scenario('scenarios/ppo.yaml')
        >>> scenario['experiment']['name']
        'gaplock_ppo'
    """
    path_obj = Path(path)

    try:
        return load_yaml_config(path_obj)
    except (OSError, ValueError) as exc:
        raise ScenarioError(str(exc)) from exc


def resolve_evaluation_protocol(scenario: Dict[str, Any], protocol: str) -> Dict[str, Any]:
    """Resolve fixed selection/final seeds without mutating training config."""
    evaluation = scenario.get("evaluation", {}) or {}
    if not isinstance(evaluation, dict):
        raise ScenarioError("'evaluation' must be a dictionary.")
    if evaluation.get("selection_strategy", "completion_safety") not in {"completion_safety", "completion_progress"}:
        raise ScenarioError("evaluation.selection_strategy must be completion_safety or completion_progress.")
    if protocol not in {"selection", "final"}:
        raise ScenarioError(f"Unknown evaluation protocol: {protocol!r}.")
    selection = {
        "seed": evaluation.get("seed", int(scenario["experiment"].get("seed", 0) or 0) + 10_000),
        "episodes": evaluation.get("episodes", 8),
    }
    final = evaluation.get("final_test")
    if final is not None and (not isinstance(final, dict) or not {"seed", "episodes"} <= final.keys()):
        raise ScenarioError("'evaluation.final_test' requires explicit seed and episodes.")
    if final is not None and set(final) - {"seed", "episodes"}:
        raise ScenarioError("'evaluation.final_test' accepts only seed and episodes; both protocols share evaluation.max_steps.")
    for name, config in (("selection", selection), ("final", final)):
        if config is None:
            continue
        for key, minimum in (("seed", 0), ("episodes", 1)):
            value = config[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ScenarioError(f"Evaluation {name} {key} must be an integer >= {minimum}.")
        if config["seed"] + config["episodes"] > 2**32:
            raise ScenarioError(f"Evaluation {name} seeds exceed the NumPy seed range.")
    if final is not None and max(selection["seed"], final["seed"]) < min(
        selection["seed"] + selection["episodes"], final["seed"] + final["episodes"]
    ):
        raise ScenarioError("Checkpoint-selection and final-test seed ranges must be disjoint.")
    if protocol == "final" and final is None:
        raise ScenarioError("--eval-protocol final requires evaluation.final_test.")
    config = selection if protocol == "selection" else final
    # Both fixed protocols share the selection horizon, inheriting training by default.
    max_steps = evaluation.get("max_steps")
    if max_steps is None:
        max_steps = scenario["environment"].get("max_steps", 5000)
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 0:
        raise ScenarioError("Evaluation max_steps must be a nonnegative integer.")
    return {"name": protocol, "seed": config["seed"], "episodes": config["episodes"], "max_steps": max_steps}


def validate_scenario(scenario: Dict[str, Any]) -> None:
    """Validate scenario configuration before env construction.

    Checks
    ------
    - Required top-level sections: ``experiment``, ``environment``, ``agents``.
    - ``experiment.name`` present.
    - ``environment`` has at least one map field.
    - Each agent config is a dict with an ``algorithm`` field.
    - Each agent's ``algorithm`` is a known RL or heuristic algorithm.
    - Each trainable (RL) agent has ``observation`` and ``reward`` config.

    Raises
    ------
    ScenarioError
        On the first validation failure found.
    """
    from src.core.agent_builder import (
        HEURISTIC_ALGOS,
        PYTORCH_RL_ALGOS,
        is_trainable_agent,
    )

    _ALL_KNOWN_ALGOS = PYTORCH_RL_ALGOS | HEURISTIC_ALGOS

    # --- Required top-level sections ---
    for section in ("experiment", "environment", "agents"):
        if section not in scenario:
            raise ScenarioError(f"Scenario must have a '{section}' section.")

    experiment = scenario["experiment"]
    if "name" not in experiment:
        raise ScenarioError("'experiment' section must have a 'name' field.")
    if "total_steps" in experiment:
        raise ScenarioError("PPO/MAPPO use 'experiment.episodes'; 'total_steps' is unsupported.")

    environment = scenario["environment"]
    if scenario.get("evaluation"):
        resolve_evaluation_protocol(scenario, "selection")
    _MAP_KEYS = {"map", "maps", "map_bundle", "map_bundles"}
    if not _MAP_KEYS.intersection(environment):
        raise ScenarioError(
            "Environment must declare a map via one of: "
            + ", ".join(f"'{k}'" for k in sorted(_MAP_KEYS))
        )
    if "target_laps" in environment:
        target_laps = environment["target_laps"]
        if isinstance(target_laps, bool) or not isinstance(target_laps, int) or target_laps <= 0:
            raise ScenarioError("'environment.target_laps' must be a positive integer.")

    agents = scenario["agents"]
    if not isinstance(agents, dict) or not agents:
        raise ScenarioError("'agents' must be a non-empty dictionary.")

    # --- Per-agent checks ---
    for agent_id, agent_cfg in agents.items():
        if not isinstance(agent_cfg, dict):
            raise ScenarioError(
                f"Agent '{agent_id}' config must be a dictionary, got {type(agent_cfg).__name__}."
            )

        algo = str(agent_cfg.get("algorithm", "")).strip().lower()
        if not algo:
            raise ScenarioError(
                f"Agent '{agent_id}' is missing required 'algorithm' field."
            )

        if algo not in _ALL_KNOWN_ALGOS:
            raise ScenarioError(
                f"Agent '{agent_id}' has unknown algorithm '{algo}'. "
                f"Known RL algorithms: {sorted(PYTORCH_RL_ALGOS)}. "
                f"Known heuristic algorithms: {sorted(HEURISTIC_ALGOS)}."
            )

        explicit = agent_cfg.get("trainable")
        if explicit is not None and not isinstance(explicit, bool):
            raise ScenarioError(f"Agent '{agent_id}' trainable must be a boolean.")
        if explicit is not None and explicit != (algo in PYTORCH_RL_ALGOS):
            raise ScenarioError(
                f"Agent '{agent_id}': algorithm '{algo}' does not support trainable={explicit}. "
                "PPO/MAPPO are trainable; fixed opponents must use a registered controller."
            )

        # Trainable agents need observation and reward configs
        if is_trainable_agent(agent_cfg):
            for required_key in ("observation", "reward"):
                if required_key not in agent_cfg:
                    raise ScenarioError(
                        f"Trainable agent '{agent_id}' (algorithm='{algo}') "
                        f"is missing required '{required_key}' config."
                    )

    trainable_ids = [aid for aid, cfg in agents.items() if is_trainable_agent(cfg)]
    trainable_algos = {
        str(agents[aid]["algorithm"]).strip().lower() for aid in trainable_ids
    }
    if len(trainable_algos) > 1:
        raise ScenarioError("Mixed trainable algorithms are unsupported; use one PPO agent or a MAPPO team.")
    if trainable_algos == {"ppo"} and len(trainable_ids) > 1:
        raise ScenarioError("PPO requires exactly one trainable agent; use MAPPO for a trainable team.")
    num_envs = experiment.get("num_envs", 1)
    for name in ("num_envs", "torch_threads"):
        value = experiment.get(name, 1)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ScenarioError(f"'experiment.{name}' must be a positive integer.")
    if num_envs > 1:
        if trainable_algos != {"ppo"}:
            raise ScenarioError("Parallel environments currently support PPO only; MAPPO uses num_envs=1.")
        if environment.get("render") or scenario.get("curriculum"):
            raise ScenarioError("Parallel PPO requires headless training without curriculum.")
        seed = experiment.get("seed")
        env_seed = environment.get("seed", seed)
        if env_seed is None:
            env_seed = seed
        if any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < 2 ** 32
               for v in (seed, env_seed)):
            raise ScenarioError("Parallel PPO requires explicit integer seeds in [0, 2**32).")
        if int(experiment.get("episodes", 1000)) < num_envs:
            raise ScenarioError("Parallel PPO requires at least num_envs total episodes.")
        params = {**scenario.get("training_defaults", {}), **agents[trainable_ids[0]].get("params", {})}
        n_steps = params.get("n_steps", 2048)
        if (isinstance(n_steps, bool) or not isinstance(n_steps, int)
                or n_steps < num_envs or n_steps % num_envs):
            raise ScenarioError("Parallel PPO n_steps must be a positive multiple of num_envs.")
    trainable_mappo = trainable_ids if trainable_algos == {"mappo"} else []
    if trainable_mappo:
        if scenario.get("curriculum"):
            raise ScenarioError("MAPPO does not yet support scenario curriculum; use PPO curriculum experiments.")
        mappo = resolve_mappo_config(scenario)
        reward_mode = mappo["reward_mode"]
        critic_mode = mappo["critic_mode"]
        reduction = mappo["team_reward_reduction"]
        params = {**scenario.get("training_defaults", {}), **agents[trainable_ids[0]].get("params", {})}
        team_return_mode = params.get("team_return_mode", "per_agent")
        if team_return_mode not in {"per_agent", "joint"}:
            raise ScenarioError("team_return_mode must be per_agent or joint")
        if team_return_mode == "joint" and (
            reward_mode != "team_shared" or critic_mode != "shared_team"
            or int(environment.get("action_repeat", 1)) != 1
            or (environment.get("episode_termination", {}) or {}).get("mode") not in {"all_agents", "all_trainable"}
        ):
            raise ScenarioError("Joint team returns require team_shared/shared_team, action_repeat=1, and all_agents/all_trainable termination")
        if reward_mode not in {"individual", "team_shared"}:
            raise ScenarioError(
                "'mappo.reward_mode' must be 'individual' or 'team_shared'."
            )
        if critic_mode not in {"shared_team", "agent_conditioned"}:
            raise ScenarioError(
                "'mappo.critic_mode' must be 'shared_team' or 'agent_conditioned'."
            )
        if reduction not in {"mean", "sum"}:
            raise ScenarioError(
                "'mappo.team_reward_reduction' must be 'mean' or 'sum'."
            )
        if reward_mode == "individual" and critic_mode == "shared_team":
            raise ScenarioError(
                "MAPPO individual rewards require critic_mode='agent_conditioned'; "
                "a shared team critic cannot represent distinct per-agent returns."
            )

        # One MAPPO object owns one shared actor and optimizer. Per-agent
        # reward configs may differ, but policy inputs, action processing, and
        # optimizer/model parameters must not depend on which agent happened
        # to be selected as the focal agent in run.py.
        reference_id = trainable_mappo[0]
        reference = agents[reference_id]
        shared_fields = ("observation", "params", "action_constraints")
        for agent_id in trainable_mappo[1:]:
            for field in shared_fields:
                if agents[agent_id].get(field, {}) != reference.get(field, {}):
                    raise ScenarioError(
                        "Shared MAPPO agents require identical "
                        f"'{field}' configuration; agents '{reference_id}' and "
                        f"'{agent_id}' differ."
                    )


def resolve_target_ids(scenario: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve target_id for agents based on roles.

    For adversarial tasks, automatically resolves which agent is the target
    for each attacker based on explicit roles.

    Args:
        scenario: Scenario configuration

    Returns:
        Scenario with target_id resolved for each agent

    Example:
        >>> scenario = {
        ...     'agents': {
        ...         'car_0': {'role': 'attacker', 'algorithm': 'ppo'},
        ...         'car_1': {'role': 'defender', 'algorithm': 'ftg'},
        ...     }
        ... }
        >>> resolved = resolve_target_ids(scenario)
        >>> resolved['agents']['car_0']['target_id']
        'car_1'
    """
    scenario = copy.deepcopy(scenario)

    if 'agents' not in scenario:
        return scenario

    agents = scenario['agents']

    # Find agents by role
    attackers = []
    defenders = []

    for agent_id, agent_config in agents.items():
        role = agent_config.get('role', None)
        if role == 'attacker':
            attackers.append(agent_id)
        elif role == 'defender':
            defenders.append(agent_id)

    # For each attacker, set target_id to first defender
    # (Simple 1v1 case, can be extended for multi-agent)
    if attackers and defenders:
        for attacker_id in attackers:
            if 'target_id' not in agents[attacker_id]:
                agents[attacker_id]['target_id'] = defenders[0]

    return scenario


def load_and_expand_scenario(path: str, validate: bool = True) -> Dict[str, Any]:
    """Load and validate a scenario, then resolve agent targets.

    The historical entry-point name is retained for callers.

    Args:
        path: Path to scenario YAML file
        validate: Whether to validate the scenario (default: True)

    Returns:
        Fully expanded and validated scenario

    Raises:
        ScenarioError: If scenario is invalid

    Example:
        >>> scenario = load_and_expand_scenario('scenarios/ppo.yaml')
        >>> # Ready to use for training
    """
    # Load raw scenario
    scenario = load_scenario(path)

    # Validate before resolving targets
    if validate:
        validate_scenario(scenario)

    # Resolve target IDs for adversarial tasks
    scenario = resolve_target_ids(scenario)

    return scenario


__all__ = [
    'ScenarioError',
    'load_scenario',
    'load_yaml_config',
    'validate_scenario',
    'resolve_mappo_config',
    'resolve_target_ids',
    'load_and_expand_scenario',
]
