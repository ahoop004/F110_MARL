"""Load simplified scenario manifests into :class:`ExperimentConfig`."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml

from .config_models import ExperimentConfig


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

    payload: Dict[str, Any] = {}
    consumed: set[str] = set()

    def _consume_block(key: str) -> None:
        block = scenario.get(key)
        if block is None:
            return
        if not isinstance(block, Mapping):
            raise ScenarioConfigError(f"Scenario '{key}' section must be a mapping")
        payload[key] = deepcopy(block)
        consumed.add(key)

    _consume_block("env")
    _consume_block("reward")
    _consume_block("main")

    algorithms = scenario.get("algorithms")
    if algorithms is not None:
        if not isinstance(algorithms, Mapping):
            raise ScenarioConfigError("Scenario 'algorithms' must be a mapping")
        for name, overrides in algorithms.items():
            if not isinstance(overrides, Mapping):
                raise ScenarioConfigError(f"Algorithm overrides for '{name}' must be a mapping")
            payload[name] = deepcopy(overrides)
        consumed.add("algorithms")

    agents_block = scenario.get("agents")
    if agents_block is not None:
        if not isinstance(agents_block, Mapping):
            raise ScenarioConfigError("Scenario 'agents' must be a mapping of agent identifiers")
        payload["agents"] = deepcopy(agents_block)
        consumed.add("agents")

    for key, value in scenario.items():
        if key in consumed:
            continue
        payload[key] = deepcopy(value)

    meta = doc.get("meta")
    if meta is not None:
        payload.setdefault("meta", deepcopy(meta))

    cfg = ExperimentConfig.from_dict(payload)
    return cfg


__all__ = ["load_scenario_manifest", "ScenarioConfigError"]
