"""Trainer implementations for on-policy algorithms."""
from __future__ import annotations

from typing import Any, Dict, Optional

from f110x.trainer.base import Trainer, Transition

try:  # Lazy policy imports to avoid optional dependency issues during docs/import-time
    from f110x.policies.ppo.ppo import PPOAgent
    from f110x.policies.ppo.rec_ppo import RecurrentPPOAgent
except Exception:  # pragma: no cover - protects import-time when policies unavailable
    PPOAgent = Any  # type: ignore
    RecurrentPPOAgent = Any  # type: ignore


class PPOTrainer(Trainer):
    """Adapter exposing the PPOAgent through the shared Trainer interface."""

    def __init__(self, agent_id: str, agent: PPOAgent) -> None:
        super().__init__(agent_id)
        self._agent = agent

    def select_action(self, obs: Any, *, deterministic: bool = False) -> Any:
        if deterministic and hasattr(self._agent, "act_deterministic"):
            return self._agent.act_deterministic(obs, self.agent_id)
        return self._agent.act(obs, self.agent_id)

    def observe(self, transition: Transition) -> None:
        next_obs = transition.next_obs
        if transition.truncated and not transition.terminated:
            record_value = getattr(self._agent, "record_final_value", None)
            if callable(record_value):
                record_value(next_obs)

        done_flag = transition.terminated or transition.truncated
        self._agent.store(next_obs, transition.action, transition.reward, done_flag)

    def update(self) -> Optional[Dict[str, Any]]:
        stats = self._agent.update()
        if not stats:
            return None
        return {f"{self.agent_id}/{key}": value for key, value in stats.items()}

    def save(self, path: str) -> None:
        self._agent.save(path)

    def load(self, path: str) -> None:
        self._agent.load(path)


class RecurrentPPOTrainer(Trainer):
    """Adapter exposing the recurrent PPO agent through the Trainer interface."""

    def __init__(self, agent_id: str, agent: RecurrentPPOAgent) -> None:
        super().__init__(agent_id)
        self._agent = agent

    def select_action(self, obs: Any, *, deterministic: bool = False) -> Any:
        del deterministic  # Recurrent agent currently ignores deterministic flag
        act_fn = getattr(self._agent, "act", None)
        if not callable(act_fn):
            raise TypeError("Recurrent PPO agent must expose an 'act' method")
        return act_fn(obs, self.agent_id)

    def observe(self, transition: Transition) -> None:
        done_flag = transition.terminated or transition.truncated
        self._agent.store(
            transition.obs,
            transition.action,
            transition.reward,
            transition.next_obs,
            done_flag,
            transition.info or {},
        )

    def update(self) -> Optional[Dict[str, Any]]:
        stats = self._agent.update()
        if not stats:
            return None
        return {f"{self.agent_id}/{key}": value for key, value in stats.items()}

    def save(self, path: str) -> None:
        self._agent.save(path)

    def load(self, path: str) -> None:
        self._agent.load(path)


__all__ = ["PPOTrainer", "RecurrentPPOTrainer"]
