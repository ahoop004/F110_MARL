"""Reward strategy orchestration utilities powered by the task registry."""

from __future__ import annotations

from typing import Any, Dict, Optional

from f110x.tasks.reward import (
    RewardRuntimeContext,
    RewardStep,
    RewardStrategy,
    resolve_reward_task,
)


class RewardWrapper:
    """Constructs and runs task-specific reward strategies."""

    def __init__(
        self,
        *,
        config: Dict[str, Any],
        context: RewardRuntimeContext,
    ) -> None:
        self.context = context
        self.raw_config = dict(config)
        self.ego_collision_penalty = float(config.get("ego_collision_penalty", 0.0))

        strategy, migrated, notes = resolve_reward_task(context, config=config)
        self.strategy: RewardStrategy = strategy
        self.task_config = migrated
        self.task_id = migrated.get("task", "unknown")
        self.mode = self.task_id  # Backward compatible attribute
        self.migration_notes = tuple(notes)

        self._episode_index = 0
        self._last_components: Dict[str, Dict[str, float]] = {}
        self._step_counter = 0

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    def reset(self, episode_index: int = 0) -> None:
        self._episode_index = episode_index
        self._step_counter = 0
        self._last_components.clear()
        self.strategy.reset(episode_index)

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------
    def __call__(
        self,
        obs: Dict[str, Dict[str, Any]],
        agent_id: str,
        reward: float,
        *,
        done: bool,
        info: Optional[Dict[str, Any]],
        all_obs: Optional[Dict[str, Dict[str, Any]]] = None,
        step_index: Optional[int] = None,
        events: Optional[Dict[str, Any]] = None,
    ) -> float:
        agent_obs = obs.get(agent_id)
        if agent_obs is None:
            return float(reward)

        timestep = float(getattr(self.context.env, "timestep", 0.01))
        current_time = float(getattr(self.context.env, "current_time", 0.0))

        effective_step = step_index if step_index is not None else self._step_counter
        self._step_counter = effective_step + 1

        step = RewardStep(
            agent_id=agent_id,
            obs=agent_obs,
            env_reward=float(reward),
            done=bool(done),
            info=info,
            all_obs=all_obs if isinstance(all_obs, dict) else None,
            episode_index=self._episode_index,
            step_index=effective_step,
            current_time=current_time,
            timestep=timestep,
            events=dict(events or {}),
        )

        total_reward, components = self.strategy.compute(step)
        components = dict(components)

        if self.ego_collision_penalty and bool(agent_obs.get("collision", False)):
            total_reward += self.ego_collision_penalty
            components["ego_collision_penalty"] = (
                components.get("ego_collision_penalty", 0.0) + self.ego_collision_penalty
            )

        components["total"] = total_reward
        self._last_components[agent_id] = components
        return total_reward

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def get_last_components(self, agent_id: str) -> Dict[str, float]:
        return dict(self._last_components.get(agent_id, {}))

    @property
    def task(self) -> str:
        return self.task_id


__all__ = ["RewardWrapper", "RewardRuntimeContext"]
