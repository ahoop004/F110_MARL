"""Trainer implementations for off-policy algorithms."""
from __future__ import annotations

from typing import Any, Dict, Optional

from f110x.trainer.base import Trainer, Transition

try:  # Lazy imports keep module available without policy extras installed
    from f110x.policies.dqn.dqn import DQNAgent
    from f110x.policies.td3.td3 import TD3Agent
    from f110x.policies.sac.sac import SACAgent
except Exception:  # pragma: no cover
    DQNAgent = Any  # type: ignore
    TD3Agent = Any  # type: ignore
    SACAgent = Any  # type: ignore


class DQNTrainer(Trainer):
    """Adapter that exposes the DQN agent via the Trainer interface."""

    def __init__(self, agent_id: str, agent: DQNAgent) -> None:
        super().__init__(agent_id)
        self._agent = agent

    def select_action(self, obs: Any, *, deterministic: bool = False) -> Any:
        return self._agent.act(obs, deterministic=deterministic)

    def observe(self, transition: Transition) -> None:
        done = transition.terminated or transition.truncated
        self._agent.store_transition(
            transition.obs,
            transition.action,
            transition.reward,
            transition.next_obs,
            done,
            transition.info,
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

    def epsilon(self) -> float:
        eps_fn = getattr(self._agent, "epsilon", None)
        if callable(eps_fn):
            return float(eps_fn())
        raise AttributeError("Underlying DQN agent does not expose epsilon()")


class TD3Trainer(Trainer):
    """Adapter exposing TD3Agent via the Trainer contract."""

    def __init__(self, agent_id: str, agent: TD3Agent) -> None:
        super().__init__(agent_id)
        self._agent = agent

    def select_action(self, obs: Any, *, deterministic: bool = False) -> Any:
        return self._agent.act(obs, deterministic=deterministic)

    def observe(self, transition: Transition) -> None:
        done = transition.terminated or transition.truncated
        self._agent.store_transition(
            transition.obs,
            transition.action,
            transition.reward,
            transition.next_obs,
            done,
            transition.info,
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


class SACTrainer(Trainer):
    """Adapter exposing SACAgent via the Trainer contract."""

    def __init__(self, agent_id: str, agent: SACAgent) -> None:
        super().__init__(agent_id)
        self._agent = agent

    def select_action(self, obs: Any, *, deterministic: bool = False) -> Any:
        return self._agent.act(obs, deterministic=deterministic)

    def observe(self, transition: Transition) -> None:
        done = transition.terminated or transition.truncated
        self._agent.store_transition(
            transition.obs,
            transition.action,
            transition.reward,
            transition.next_obs,
            done,
            transition.info,
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


__all__ = ["DQNTrainer", "TD3Trainer", "SACTrainer"]
