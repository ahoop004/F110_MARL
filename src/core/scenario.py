"""Scenario configuration system for v2 training pipeline.

Provides YAML-based scenario configuration with preset expansion
for algorithms, rewards, and observations. Scenarios define complete
training setups in a concise, readable format.
"""

from typing import Dict, Any, Optional
import copy
import yaml
from pathlib import Path

from rewards import load_preset as load_reward_preset, merge_config as merge_reward_config
from core.observations import load_observation_preset, merge_observation_config


class ScenarioError(Exception):
    """Exception raised for scenario configuration errors."""
    pass


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
        raise ScenarioError(f"Scenario file not found: {path_obj}")

    try:
        with open(path_obj, 'r') as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ScenarioError(f"Invalid YAML in scenario file: {e}")

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ScenarioError("Scenario must be a YAML dictionary")
    return data


def _load_with_includes(path_obj: Path, visited: Optional[set] = None) -> Dict[str, Any]:
    """Load a scenario file with optional includes."""
    path_obj = path_obj.resolve()
    visited = visited or set()
    if path_obj in visited:
        raise ScenarioError(f"Include cycle detected at: {path_obj}")
    visited.add(path_obj)

    data = _load_yaml_file(path_obj)
    includes = data.pop('includes', None)

    merged: Dict[str, Any] = {}
    if includes:
        if isinstance(includes, (str, Path)):
            includes = [includes]
        if not isinstance(includes, list):
            raise ScenarioError("'includes' must be a list of file paths")
        for include_path in includes:
            if not isinstance(include_path, (str, Path)):
                raise ScenarioError("'includes' entries must be file paths")
            include_obj = (path_obj.parent / include_path).resolve()
            merged = _deep_merge(merged, _load_with_includes(include_obj, visited))

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
        >>> scenario = load_scenario('scenarios/v2/gaplock_ppo.yaml')
        >>> scenario['experiment']['name']
        'gaplock_ppo'
    """
    path_obj = Path(path)

    return _load_with_includes(path_obj)


def expand_reward_preset(config: Dict[str, Any]) -> Dict[str, Any]:
    """Expand reward preset with optional overrides.

    Args:
        config: Reward config with 'preset' and optional 'overrides'

    Returns:
        Expanded reward configuration

    Example:
        >>> config = {'preset': 'gaplock_full', 'overrides': {'terminal': {'target_crash': 100.0}}}
        >>> expanded = expand_reward_preset(config)
        >>> expanded['terminal']['target_crash']
        100.0
    """
    if 'preset' not in config:
        # No preset, return config as-is
        return copy.deepcopy(config)

    # Load preset
    preset_name = config['preset']
    try:
        preset = load_reward_preset(preset_name)
    except ValueError as e:
        raise ScenarioError(f"Reward preset error: {e}")

    # Apply overrides if present
    if 'overrides' in config:
        return merge_reward_config(preset, config['overrides'])

    return preset


def expand_observation_preset(config: Dict[str, Any]) -> Dict[str, Any]:
    """Expand observation preset with optional overrides.

    Args:
        config: Observation config with 'preset' and optional 'overrides'

    Returns:
        Expanded observation configuration

    Example:
        >>> config = {'preset': 'gaplock', 'overrides': {'lidar': {'beams': 360}}}
        >>> expanded = expand_observation_preset(config)
        >>> expanded['lidar']['beams']
        360
    """
    if 'preset' not in config:
        # No preset, return config as-is
        return copy.deepcopy(config)

    # Load preset
    preset_name = config['preset']
    try:
        preset = load_observation_preset(preset_name)
    except ValueError as e:
        raise ScenarioError(f"Observation preset error: {e}")

    # Apply overrides if present
    if 'overrides' in config:
        return merge_observation_config(preset, config['overrides'])

    return preset


def expand_agent_config(agent_config: Dict[str, Any]) -> Dict[str, Any]:
    """Expand agent configuration with preset expansion.

    Expands reward and observation presets for an agent.

    Args:
        agent_config: Agent configuration dict

    Returns:
        Expanded agent configuration

    Example:
        >>> agent_config = {
        ...     'algorithm': 'ppo',
        ...     'observation': {'preset': 'gaplock'},
        ...     'reward': {'preset': 'gaplock_full'},
        ... }
        >>> expanded = expand_agent_config(agent_config)
    """
    config = copy.deepcopy(agent_config)

    # Expand observation preset
    if 'observation' in config:
        config['observation'] = expand_observation_preset(config['observation'])

    # Expand reward preset
    if 'reward' in config:
        config['reward'] = expand_reward_preset(config['reward'])

    return config


def expand_scenario(scenario: Dict[str, Any]) -> Dict[str, Any]:
    """Expand all presets in a scenario.

    Expands reward and observation presets for all agents in the scenario.

    Args:
        scenario: Raw scenario configuration

    Returns:
        Scenario with all presets expanded

    Example:
        >>> scenario = load_scenario('scenarios/v2/gaplock_ppo.yaml')
        >>> expanded = expand_scenario(scenario)
    """
    expanded = copy.deepcopy(scenario)

    # Expand each agent's configuration
    if 'agents' in expanded:
        for agent_id, agent_config in expanded['agents'].items():
            expanded['agents'][agent_id] = expand_agent_config(agent_config)

    return expanded


def validate_scenario(scenario: Dict[str, Any]) -> None:
    """Validate scenario configuration.

    Args:
        scenario: Scenario configuration to validate

    Raises:
        ScenarioError: If scenario is invalid

    Example:
        >>> scenario = load_scenario('scenarios/v2/gaplock_ppo.yaml')
        >>> validate_scenario(scenario)  # Raises if invalid
    """
    # Check required top-level keys
    if 'experiment' not in scenario:
        raise ScenarioError("Scenario must have 'experiment' section")

    if 'environment' not in scenario:
        raise ScenarioError("Scenario must have 'environment' section")

    if 'agents' not in scenario:
        raise ScenarioError("Scenario must have 'agents' section")

    # Validate experiment section
    experiment = scenario['experiment']
    if 'name' not in experiment:
        raise ScenarioError("Experiment must have 'name' field")

    # Validate environment section
    environment = scenario['environment']
    if 'map' not in environment:
        raise ScenarioError("Environment must have 'map' field")

    # Validate agents section
    agents = scenario['agents']
    if not isinstance(agents, dict):
        raise ScenarioError("'agents' must be a dictionary")

    if len(agents) == 0:
        raise ScenarioError("Scenario must have at least one agent")

    # Validate each agent
    for agent_id, agent_config in agents.items():
        if not isinstance(agent_config, dict):
            raise ScenarioError(f"Agent '{agent_id}' config must be a dictionary")

        # Check for algorithm or role
        if 'algorithm' not in agent_config and 'role' not in agent_config:
            raise ScenarioError(
                f"Agent '{agent_id}' must have 'algorithm' field "
                "(or 'role' for non-trainable agents)"
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
    """Load, expand, and validate a scenario file.

    Convenience function that combines loading, expansion, and validation.

    Args:
        path: Path to scenario YAML file
        validate: Whether to validate the scenario (default: True)

    Returns:
        Fully expanded and validated scenario

    Raises:
        ScenarioError: If scenario is invalid

    Example:
        >>> scenario = load_and_expand_scenario('scenarios/v2/gaplock_ppo.yaml')
        >>> # Ready to use for training
    """
    # Load raw scenario
    scenario = load_scenario(path)

    # Validate before expansion
    if validate:
        validate_scenario(scenario)

    # Expand presets
    scenario = expand_scenario(scenario)

    # Resolve target IDs for adversarial tasks
    scenario = resolve_target_ids(scenario)

    return scenario


__all__ = [
    'ScenarioError',
    'load_scenario',
    'expand_reward_preset',
    'expand_observation_preset',
    'expand_agent_config',
    'expand_scenario',
    'validate_scenario',
    'resolve_target_ids',
    'load_and_expand_scenario',
]
