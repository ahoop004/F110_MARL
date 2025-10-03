"""Scenario manifest loader composing layered configs into ExperimentConfig."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import yaml

from .config_layers import (
    LAYER_NAME_TO_DIR,
    CONFIG_BASE_DIR,
    CONFIG_TASK_DIR,
    CONFIG_POLICY_DIR,
    CONFIG_ALGO_DIR,
    deep_merge,
)
from .config_models import ExperimentConfig


class ScenarioConfigError(ValueError):
    """Raised when a scenario manifest cannot be composed."""


def load_scenario_manifest(path: Path | str) -> ExperimentConfig:
    """Load a scenario manifest and return a composed :class:`ExperimentConfig`."""

    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Scenario manifest not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as handle:
        doc = yaml.safe_load(handle) or {}

    if not isinstance(doc, Mapping):
        raise ScenarioConfigError("Scenario manifest root must be a mapping")

    payload = doc.get("scenario") or doc
    if not isinstance(payload, Mapping):
        raise ScenarioConfigError("Scenario manifest must define a 'scenario' mapping")

    scenario_dir = manifest_path.parent

    base_path = payload.get("base", "base/default.yaml")
    base_doc = _load_yaml(base_path, CONFIG_BASE_DIR, scenario_dir)

    merged: Dict[str, Any] = {}
    deep_merge(merged, deepcopy(base_doc))

    # Scenario-level task (optional)
    task_decl = payload.get("task")
    if task_decl:
        task_doc = _load_yaml(task_decl, CONFIG_TASK_DIR, scenario_dir)
        deep_merge(merged, deepcopy(task_doc))

    # Scenario-level overrides (env, episodes, wandb, reward, main, etc.)
    for key in ("env", "episodes", "wandb", "reward", "main", "ppo_agent_idx"):
        if key in payload:
            deep_merge(merged, {key: payload[key]})

    merged.setdefault("main", {})
    if isinstance(merged["main"], Mapping):
        merged["main"].setdefault("experiment_name", payload.get("name") or manifest_path.stem)

    algorithms: Dict[str, Any] = {}
    roster: list[Dict[str, Any]] = []
    agents_decl = payload.get("agents", {})
    if not isinstance(agents_decl, Mapping):
        raise ScenarioConfigError("Scenario 'agents' must be a mapping of agent names")

    for index, (agent_name, agent_decl) in enumerate(agents_decl.items()):
        if not isinstance(agent_decl, Mapping):
            raise ScenarioConfigError(f"Agent '{agent_name}' config must be a mapping")

        slot = agent_decl.get("slot", index)
        role = agent_decl.get("role", agent_name)

        # Agent task overrides (optional per-agent reward tweaks)
        agent_task = agent_decl.get("task")
        if agent_task:
            task_doc = _load_yaml(agent_task, CONFIG_TASK_DIR, scenario_dir)
            deep_merge(merged, deepcopy(task_doc))

        policy_path = agent_decl.get("policy")
        if not policy_path:
            raise ScenarioConfigError(f"Agent '{agent_name}' is missing a policy reference")
        policy_doc = _load_yaml(policy_path, CONFIG_POLICY_DIR, scenario_dir)
        if "agent" not in policy_doc:
            raise ScenarioConfigError(f"Policy '{policy_path}' must contain an 'agent' mapping")
        agent_config = deepcopy(policy_doc["agent"])

        algo_name = str(agent_config.get("algo", "")).strip()
        if not algo_name:
            raise ScenarioConfigError(f"Policy '{policy_path}' for agent '{agent_name}' missing 'algo'")

        config_entry = agent_config.pop("config", None)
        if config_entry:
            algo_doc = _load_yaml(config_entry, CONFIG_ALGO_DIR, scenario_dir)
            deep_merge(algorithms.setdefault(algo_name, {}), deepcopy(algo_doc))
            agent_config.setdefault("config_ref", algo_name)

        agent_config.setdefault("slot", slot)
        agent_config.setdefault("role", role)
        agent_config.setdefault("agent_id", agent_name)

        overrides = agent_decl.get("overrides")
        if isinstance(overrides, Mapping):
            deep_merge(agent_config, overrides)

        roster.append(agent_config)

    if roster:
        merged.setdefault("agents", {})
        merged["agents"]["roster"] = roster

    for algo, payload_doc in algorithms.items():
        deep_merge(merged.setdefault(algo, {}), payload_doc)

    return ExperimentConfig.from_dict(merged)


def _load_yaml(path_str: str, default_dir: Optional[Path], scenario_dir: Path) -> Dict[str, Any]:
    resolved = _resolve_layer_path(path_str, default_dir, scenario_dir)
    with resolved.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, Mapping):
        raise ScenarioConfigError(f"Layer file must be a mapping: {resolved}")
    return dict(data)


def _resolve_layer_path(path_str: str, default_dir: Optional[Path], scenario_dir: Path) -> Path:
    candidate = Path(path_str)
    search_paths: Iterable[Path]

    config_root = Path("configs")

    if candidate.is_absolute():
        search_paths = [candidate]
    else:
        search_paths = [
            scenario_dir / candidate,
            config_root / candidate,
        ]
        if default_dir and len(candidate.parts) == 1:
            search_paths.append(default_dir / candidate)
        search_paths.append(candidate)

    for raw in search_paths:
        if raw is None:
            continue
        if raw.exists():
            return raw.resolve()
        if raw.suffix == "" and raw.with_suffix(".yaml").exists():
            return raw.with_suffix(".yaml").resolve()

    raise FileNotFoundError(f"Unable to resolve config layer path '{path_str}'")


__all__ = ["load_scenario_manifest", "ScenarioConfigError"]
