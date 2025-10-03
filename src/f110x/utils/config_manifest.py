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
        spec = dict(agent_cfg)
        spec.setdefault("slot", index)
        spec.setdefault("agent_id", agent_name)
        if "algo" not in spec:
            raise ScenarioConfigError(f"Agent '{agent_name}' must declare an 'algo'")
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


__all__ = ["load_scenario_manifest", "ScenarioConfigError"]
