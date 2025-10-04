"""Composite reward strategy built from registered tasks."""

from __future__ import annotations

from typing import Dict, List, Tuple

from .base import RewardRuntimeContext, RewardStep, RewardStrategy
from .registry import RewardTaskConfig, RewardTaskRegistry, RewardTaskSpec, register_reward_task


class CompositeRewardStrategy(RewardStrategy):
    name = "composite"

    def __init__(self, components: List[Tuple[str, RewardStrategy, float]]) -> None:
        if not components:
            raise ValueError("Composite reward strategy requires at least one component")
        self._components = [(label, strategy, float(weight)) for label, strategy, weight in components]

    def reset(self, episode_index: int) -> None:
        for _, strategy, _ in self._components:
            strategy.reset(episode_index)

    def compute(self, step: RewardStep) -> Tuple[float, Dict[str, float]]:
        total_reward = 0.0
        components: Dict[str, float] = {}
        multi_strategy = len(self._components) > 1

        for label, strategy, weight in self._components:
            base_reward, strat_components = strategy.compute(step)
            weighted_reward = base_reward * weight
            total_reward += weighted_reward

            if not strat_components:
                continue

            for name, value in strat_components.items():
                effective_value = value * weight
                key = f"{strategy.name}/{name}" if multi_strategy else name
                components[key] = components.get(key, 0.0) + effective_value

        return total_reward, components


def _build_composite_strategy(
    context: RewardRuntimeContext,
    config: RewardTaskConfig,
    registry: RewardTaskRegistry,
) -> RewardStrategy:
    raw_components = config.get("components") or {}
    if not isinstance(raw_components, dict) or not raw_components:
        raise ValueError("Composite reward task requires a non-empty 'components' mapping")

    components: List[Tuple[str, RewardStrategy, float]] = []
    for label, component_config in raw_components.items():
        weight = component_config.get("weight", 1.0)
        try:
            weight_value = float(weight)
        except (TypeError, ValueError):
            weight_value = 1.0
        if weight_value == 0.0:
            continue

        task_id = component_config.get("task")
        if not task_id:
            raise ValueError(f"Composite component '{label}' is missing a task identifier")

        child_strategy = registry.create(task_id, context, component_config)
        components.append((str(label), child_strategy, weight_value))

    if not components:
        raise ValueError("Composite reward task requires at least one component with non-zero weight")

    return CompositeRewardStrategy(components)


register_reward_task(
    RewardTaskSpec(
        name="composite",
        factory=_build_composite_strategy,
    )
)


__all__ = ["CompositeRewardStrategy"]
