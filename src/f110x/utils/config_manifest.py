"""Load simplified scenario manifests into :class:`ExperimentConfig`."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml

from .config_models import ExperimentConfig, AgentRosterConfig


class ScenarioConfigError(ValueError):
    """Raised when scenario manifests are malformed."""


def load_scenario_manifest(path: Path | str) -> ExperimentConfig:
    """Compose a scenario manifest into an :class:`ExperimentConfig`."""

    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Scenario manifest not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as handle:
        doc = yaml.safe_load(handle) or {}

    if not isinstance(doc, Mapping):
        raise ScenarioConfigError("Scenario manifest must contain a mapping")

    scenario = doc.get("scenario") or doc
    if not isinstance(scenario, Mapping):
        raise ScenarioConfigError("Scenario manifest must contain a 'scenario' mapping")

    cfg = ExperimentConfig()
    raw: Dict[str, Any] = {}

    _apply_section(cfg.env.schema.update_from_dict, scenario.get("env"), raw, "env")
    _apply_section(cfg.reward.schema.update_from_dict, scenario.get("reward"), raw, "reward")
    _apply_section(cfg.main.schema.update_from_dict, scenario.get("main"), raw, "main")

    algorithms = scenario.get("algorithms", {})
    if algorithms and not isinstance(algorithms, Mapping):
        raise ScenarioConfigError("Scenario 'algorithms' must be a mapping")

    for name, overrides in algorithms.items():
        if not isinstance(overrides, Mapping):
            raise ScenarioConfigError(f"Algorithm overrides for '{name}' must be a mapping")
        section = getattr(cfg, name, None)
        if section is None or not hasattr(section, "schema"):
            raise ScenarioConfigError(f"Unknown algorithm section '{name}' in scenario")
        section.schema.update_from_dict(dict(overrides))
        raw[name] = dict(overrides)

    agents_block = scenario.get("agents", {})
    if not isinstance(agents_block, Mapping):
        raise ScenarioConfigError("Scenario 'agents' must be a mapping of agent identifiers")

    roster_payload = []
    for index, (agent_name, agent_cfg) in enumerate(agents_block.items()):
        if not isinstance(agent_cfg, Mapping):
            raise ScenarioConfigError(f"Agent '{agent_name}' configuration must be a mapping")
        spec = _normalise_agent_spec(agent_name, agent_cfg)
        spec.setdefault("agent_id", agent_name)
        roster_payload.append(spec)

    if roster_payload:
        cfg.agents = AgentRosterConfig.from_dict({"roster": roster_payload})
        raw["agents"] = {"roster": deepcopy(roster_payload)}

    for key, value in scenario.items():
        if key in {"env", "reward", "main", "algorithms", "agents"}:
            continue
        raw[key] = value

    cfg.raw = raw
    return cfg


def _apply_section(updater, payload: Optional[Mapping[str, Any]], raw: Dict[str, Any], key: str) -> None:
    if not payload:
        return
    if not isinstance(payload, Mapping):
        raise ScenarioConfigError(f"Scenario '{key}' section must be a mapping")
    updater(dict(payload))
    raw[key] = dict(payload)


def _normalise_agent_spec(agent_name: str, agent_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    spec = dict(agent_cfg)

    reward_cfg = spec.get("reward")
    if reward_cfg is not None and not isinstance(reward_cfg, Mapping):
        raise ScenarioConfigError(f"Agent '{agent_name}' reward section must be a mapping")

    params: Dict[str, Any] = {}
    if "params" in spec:
        params.update(_coerce_mapping(spec["params"], name=f"Agent '{agent_name}' params"))

    algo_section = spec.pop("algorithm", None)
    algo_name = spec.get("algo")
    config_ref = spec.get("config_ref")

    if algo_section is not None:
        if not isinstance(algo_section, Mapping):
            raise ScenarioConfigError(f"Agent '{agent_name}' algorithm section must be a mapping")
        algo_map = dict(algo_section)
        algo_params = _coerce_mapping(algo_map.get("params"), name=f"Agent '{agent_name}' algorithm params")
        extra_params = {
            key: value
            for key, value in algo_map.items()
            if key not in {"name", "algo", "type", "params", "config_ref"}
        }
        if algo_params:
            params.update(algo_params)
        if extra_params:
            params.update(extra_params)
        candidate = algo_map.get("name") or algo_map.get("algo") or algo_map.get("type")
        if candidate:
            algo_name = candidate
        if "config_ref" in algo_map:
            config_ref = algo_map.get("config_ref")

    params = _flatten_params(params)

    architecture = params.pop("architecture", None)
    if not algo_name and architecture:
        algo_name = architecture

    if not algo_name:
        raise ScenarioConfigError(f"Agent '{agent_name}' must declare an 'algo'")

    if params:
        spec["params"] = params
    else:
        spec.pop("params", None)

    if config_ref is not None:
        spec["config_ref"] = config_ref

    spec["algo"] = algo_name

    if reward_cfg is not None:
        spec["reward"] = dict(reward_cfg)

    return spec


def _coerce_mapping(value: Any, *, name: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    raise ScenarioConfigError(f"{name} must be a mapping")


def _flatten_params(params: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(params)
    for key in list(params.keys()):
        value = params[key]
        if isinstance(value, Mapping):
            nested_params = value.get("params")
            if isinstance(nested_params, Mapping):
                for inner_key, inner_value in nested_params.items():
                    result.setdefault(inner_key, inner_value)
                if set(value.keys()) <= {"params"}:
                    result.pop(key, None)
    return result


__all__ = ["load_scenario_manifest", "ScenarioConfigError"]
