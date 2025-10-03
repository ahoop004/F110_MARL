"""Utilities for composing layered configuration files."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping


# Conventional directories for layered configs (relative to project root)
CONFIG_BASE_DIR = Path("configs/base")
CONFIG_TASK_DIR = Path("configs/tasks")
CONFIG_POLICY_DIR = Path("configs/policies")
CONFIG_SCENARIO_DIR = Path("configs/scenarios")
CONFIG_ALGO_DIR = Path("configs/algorithms")


LAYER_NAME_TO_DIR = {
    "base": CONFIG_BASE_DIR,
    "task": CONFIG_TASK_DIR,
    "policy": CONFIG_POLICY_DIR,
    "scenario": CONFIG_SCENARIO_DIR,
    "algorithm": CONFIG_ALGO_DIR,
}


def deep_merge(base: MutableMapping[str, Any], overlay: Mapping[str, Any]) -> MutableMapping[str, Any]:
    """Deep merge ``overlay`` into ``base`` (mutates and returns ``base``).

    Dict values are merged recursively by default. If an overlay mapping contains a
    truthy ``__replace__`` flag, it replaces the corresponding base value entirely.
    """

    for key, value in overlay.items():
        if key == "__replace__":
            # Flag handled by caller when encountered alongside real keys
            continue

        if isinstance(value, Mapping):
            replace = bool(value.get("__replace__")) if isinstance(value, dict) else False
            if replace:
                new_val = {k: v for k, v in value.items() if k != "__replace__"}
                base[key] = new_val
                continue

            existing = base.get(key)
            if isinstance(existing, MutableMapping):
                deep_merge(existing, value)
            else:
                base[key] = _strip_replace(value)
        else:
            base[key] = value

    return base


def _strip_replace(mapping: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a shallow copy of ``mapping`` without ``__replace__`` markers."""

    if not isinstance(mapping, Mapping):
        return dict(mapping)  # type: ignore[arg-type]
    if "__replace__" not in mapping:
        return dict(mapping)
    return {k: v for k, v in mapping.items() if k != "__replace__"}


def ensure_layer_dirs(dirs: Iterable[Path] | None = None) -> None:
    """Ensure the standard layer directories exist.

    Primarily helpful in tests or setup scripts so CI can materialise the expected layout.
    """

    targets = list(dirs) if dirs is not None else list(LAYER_NAME_TO_DIR.values())
    for directory in targets:
        directory.mkdir(parents=True, exist_ok=True)


__all__ = [
    "CONFIG_BASE_DIR",
    "CONFIG_TASK_DIR",
    "CONFIG_POLICY_DIR",
    "CONFIG_SCENARIO_DIR",
    "CONFIG_ALGO_DIR",
    "LAYER_NAME_TO_DIR",
    "deep_merge",
    "ensure_layer_dirs",
]
