"""Registry for task-aware reward strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from .base import RewardRuntimeContext, RewardStrategy
from .presets import resolve_reward_presets


RewardTaskConfig = Dict[str, Any]
RewardTaskFactory = Callable[[RewardRuntimeContext, RewardTaskConfig, "RewardTaskRegistry"], RewardStrategy]


@dataclass(frozen=True)
class RewardTaskSpec:
    """Metadata describing how to construct a specific reward task."""

    name: str
    factory: RewardTaskFactory
    aliases: Tuple[str, ...] = ()
    legacy_sections: Tuple[str, ...] = ()
    param_keys: Tuple[str, ...] = ()
    description: str = ""


class RewardTaskRegistry:
    """Runtime registry responsible for instantiating reward strategies."""

    def __init__(self) -> None:
        self._specs: Dict[str, RewardTaskSpec] = {}
        self._aliases: Dict[str, str] = {}

    def register(self, spec: RewardTaskSpec) -> None:
        key = spec.name.lower()
        if key in self._specs:
            raise ValueError(f"Reward task '{spec.name}' is already registered")

        self._specs[key] = spec
        for alias in spec.aliases:
            alias_key = alias.lower()
            if alias_key in self._aliases and self._aliases[alias_key] != key:
                raise ValueError(
                    f"Alias '{alias}' already registered for reward task '{self._aliases[alias_key]}'"
                )
            self._aliases[alias_key] = key

    def normalize(self, name: Optional[str], *, default: Optional[str] = None) -> str:
        if not name:
            if default is not None:
                name = default
            else:
                raise KeyError("Reward task identifier is required")

        lookup = name.lower()
        if lookup in self._specs:
            return lookup
        if lookup in self._aliases:
            return self._aliases[lookup]
        raise KeyError(f"Unknown reward task '{name}'")

    def get(self, name: str) -> RewardTaskSpec:
        key = self.normalize(name)
        return self._specs[key]

    def create(
        self,
        name: str,
        context: RewardRuntimeContext,
        config: RewardTaskConfig,
    ) -> RewardStrategy:
        spec = self.get(name)
        return spec.factory(context, config, self)

    def items(self) -> Iterable[Tuple[str, RewardTaskSpec]]:
        return self._specs.items()

    def available_tasks(self) -> Tuple[str, ...]:
        return tuple(self._specs.keys())


def _merge_params(
    spec: RewardTaskSpec,
    config: Mapping[str, Any],
    *,
    base: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    params: Dict[str, Any] = dict(base or {})

    candidate = config.get("params")
    if isinstance(candidate, Mapping):
        params.update(candidate)

    for section in spec.legacy_sections:
        section_data = config.get(section)
        if isinstance(section_data, Mapping):
            params.update(section_data)

    for key in spec.param_keys:
        if key in config and key not in params:
            params[key] = config[key]

    return params


def _normalise_component_config(
    label: str,
    definition: Any,
    registry: RewardTaskRegistry,
) -> Dict[str, Any]:
    if isinstance(definition, str):
        task_id = definition
        component_config: Mapping[str, Any] = {}
    elif isinstance(definition, Mapping):
        task_id = definition.get("task") or definition.get("mode") or label
        component_config = definition
    else:
        task_id = label
        component_config = {}

    canonical = registry.normalize(task_id)
    spec = registry.get(canonical)

    params = _merge_params(spec, component_config)
    weight = component_config.get("weight", 1.0)
    try:
        weight_value = float(weight)
    except (TypeError, ValueError):
        weight_value = 1.0

    return {
        "label": label,
        "task": canonical,
        "weight": weight_value,
        "params": params,
    }


def migrate_reward_config(
    raw_config: Optional[Mapping[str, Any]],
    registry: RewardTaskRegistry,
    *,
    default_task: str = "gaplock",
) -> Tuple[RewardTaskConfig, List[str]]:
    """Normalise legacy reward configuration dictionaries."""

    notes: List[str] = []
    config: Mapping[str, Any] = raw_config or {}

    task_name = config.get("task") or config.get("mode") or default_task
    try:
        canonical_task = registry.normalize(task_name, default=default_task)
    except KeyError as exc:
        raise KeyError(f"Unable to resolve reward task '{task_name}'") from exc

    spec = registry.get(canonical_task)
    params = _merge_params(spec, config)

    feature_defs = config.get("features")
    if feature_defs is not None:
        if isinstance(feature_defs, (list, tuple, set)):
            feature_iter = feature_defs
        else:
            feature_iter = [feature_defs]
        preset_params, preset_notes = resolve_reward_presets(feature_iter)
        merged_params = dict(preset_params)
        merged_params.update(params)
        params = merged_params
        notes.extend(preset_notes)

    components: Dict[str, Any] = {}
    raw_components = config.get("components")
    if isinstance(raw_components, Mapping):
        for label, definition in raw_components.items():
            component = _normalise_component_config(str(label), definition, registry)
            components[label] = component

    migrated: RewardTaskConfig = {
        "task": canonical_task,
        "params": params,
    }
    if components:
        migrated["components"] = components

    if config.get("mode") and "task" not in config:
        notes.append("Converted legacy reward.mode to reward.task")
    if config.get("params"):
        notes.append("Flattened reward.params into task parameters")

    return migrated, notes


reward_task_registry = RewardTaskRegistry()


def register_reward_task(spec: RewardTaskSpec) -> None:
    reward_task_registry.register(spec)


def resolve_reward_task(
    context: RewardRuntimeContext,
    *,
    config: Optional[Mapping[str, Any]],
) -> Tuple[RewardStrategy, RewardTaskConfig, List[str]]:
    migrated, notes = migrate_reward_config(config, reward_task_registry)
    strategy = reward_task_registry.create(migrated["task"], context, migrated)
    return strategy, migrated, notes


__all__ = [
    "RewardTaskConfig",
    "RewardTaskFactory",
    "RewardTaskRegistry",
    "RewardTaskSpec",
    "migrate_reward_config",
    "register_reward_task",
    "resolve_reward_task",
    "reward_task_registry",
]
