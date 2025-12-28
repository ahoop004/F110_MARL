"""Shared helpers for loading experiment configuration files."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import yaml

from f110x.utils.config_manifest import load_scenario_manifest, ScenarioConfigError
from f110x.utils.config_models import ExperimentConfig


DEFAULT_ENV_CONFIG_KEY = "F110_CONFIG"
DEFAULT_ENV_EXPERIMENT_KEY = "F110_EXPERIMENT"


def resolve_config_path(
    cfg_path: Path | str | None,
    *,
    default_path: Path | str,
    env_key: str = DEFAULT_ENV_CONFIG_KEY,
) -> Path:
    """Resolve the configuration path, honouring env overrides and defaults."""

    explicit_path: Optional[Path]
    if cfg_path is not None:
        explicit_path = Path(cfg_path)
    else:
        env_value = os.environ.get(env_key)
        explicit_path = Path(env_value) if env_value else None

    if explicit_path is None:
        explicit_path = Path(default_path)

    resolved = explicit_path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Config file not found: {resolved}")
    return resolved


def resolve_experiment_name(
    experiment: Optional[str],
    *,
    env_key: str = DEFAULT_ENV_EXPERIMENT_KEY,
) -> Optional[str]:
    """Resolve experiment name with environment variable fallback."""

    if experiment is not None:
        return experiment

    env_value = os.environ.get(env_key)
    if env_value:
        trimmed = env_value.strip()
        if trimmed:
            return trimmed
    return None


def load_config(
    cfg_path: Path | str | None,
    *,
    default_path: Path | str,
    experiment: Optional[str] = None,
    env_config_key: str = DEFAULT_ENV_CONFIG_KEY,
    env_experiment_key: str = DEFAULT_ENV_EXPERIMENT_KEY,
) -> Tuple[ExperimentConfig, Path, Optional[str]]:
    """Load an :class:`ExperimentConfig` from YAML or scenario manifest.

    Parameters
    ----------
    cfg_path:
        Explicit path provided by the caller (may be ``None``).
    default_path:
        Fallback path used when neither ``cfg_path`` nor the environment override is
        provided.
    experiment:
        Optional experiment name inside the config file. When omitted, an
        environment override is honoured.
    env_config_key / env_experiment_key:
        Environment variable names used for overrides.

    Returns
    -------
    tuple
        A tuple containing the loaded :class:`ExperimentConfig`, the resolved
        configuration path, and the resolved experiment name (if any).
    """

    resolved_path = resolve_config_path(
        cfg_path,
        default_path=default_path,
        env_key=env_config_key,
    )
    resolved_experiment = resolve_experiment_name(
        experiment,
        env_key=env_experiment_key,
    )

    try:
        with resolved_path.open("r", encoding="utf-8") as handle:
            doc = yaml.safe_load(handle) or {}
    except FileNotFoundError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Failed to parse config {resolved_path}: {exc}") from exc

    if isinstance(doc, dict) and "scenario" in doc:
        try:
            cfg = load_scenario_manifest(resolved_path)
        except (ScenarioConfigError, FileNotFoundError) as exc:
            raise RuntimeError(f"Failed to compose scenario {resolved_path}: {exc}") from exc
    else:
        cfg = ExperimentConfig.load(resolved_path, experiment=resolved_experiment)

    return cfg, resolved_path, resolved_experiment


def is_scenario_document(doc: Mapping[str, Any]) -> bool:
    """Return True when the provided mapping represents a scenario manifest."""

    if not isinstance(doc, Mapping):
        return False

    scenario_block = doc.get("scenario")
    if isinstance(scenario_block, Mapping):
        return True

    if "algorithms" in doc and "experiments" not in doc:
        return True
    return False


def resolve_active_config_block(doc: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """Return the configuration block that should receive runtime overrides."""

    if not isinstance(doc, dict):
        raise TypeError("Configuration document must be a mutable mapping")

    scenario_block = doc.get("scenario")
    if isinstance(scenario_block, Mapping):
        if not isinstance(scenario_block, dict):
            scenario_block = dict(scenario_block)
            doc["scenario"] = scenario_block
        return scenario_block, True

    return doc, False


__all__ = [
    "DEFAULT_ENV_CONFIG_KEY",
    "DEFAULT_ENV_EXPERIMENT_KEY",
    "is_scenario_document",
    "load_config",
    "resolve_config_path",
    "resolve_active_config_block",
    "resolve_experiment_name",
]
