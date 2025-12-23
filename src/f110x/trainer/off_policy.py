"""Trainer implementations for off-policy algorithms."""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from f110x.trainer.base import Trainer, Transition


class OffPolicyTrainer(Trainer):
    """Generic adapter for off-policy agents sharing a common API."""

    def __init__(
        self,
        agent_id: str,
        agent: Any,
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(agent_id)
        self._agent = agent
        cfg = dict(config or {})
        self._include_truncation = bool(cfg.get("include_truncation", True))

    def select_action(self, obs: Any, *, deterministic: bool = False) -> Any:
        return self._agent.act(obs, deterministic=deterministic)

    def observe(self, transition: Transition) -> None:
        done = transition.terminated or (self._include_truncation and transition.truncated)
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
        raise AttributeError("Underlying agent does not expose epsilon()")

    def reset_noise_schedule(self, *, restart: bool = False) -> None:
        reset_fn = getattr(self._agent, "reset_noise_schedule", None)
        if callable(reset_fn):
            try:
                reset_fn(restart=restart)
            except TypeError:
                reset_fn()

    def exploration_noise(self) -> Optional[float]:
        accessor = getattr(self._agent, "current_exploration_noise", None)
        if callable(accessor):
            try:
                value = accessor()
            except Exception:
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
        return None

    def state_dict(self, *, include_optimizer: bool = True):
        snapshot_fn = getattr(self._agent, "state_dict", None)
        if not callable(snapshot_fn):
            raise AttributeError("Underlying agent does not expose state_dict()")
        try:
            return snapshot_fn(include_optim=include_optimizer)
        except TypeError:
            # Fallback for agents that use include_optimizer naming
            return snapshot_fn(include_optimizer=include_optimizer)

    def load_state_dict(
        self,
        state,
        *,
        strict: bool = False,
        include_optimizer: bool = True,
    ) -> None:
        restore_fn = getattr(self._agent, "load_state_dict", None)
        if not callable(restore_fn):
            raise AttributeError("Underlying agent does not expose load_state_dict()")
        try:
            restore_fn(state, strict=strict, include_optim=include_optimizer)
        except TypeError:
            # Older agents may accept include_optimizer keyword instead.
            restore_fn(state, strict=strict, include_optimizer=include_optimizer)

    def reset_optimizers(self) -> None:
        reset_fn = getattr(self._agent, "reset_optimizers", None)
        if callable(reset_fn):
            reset_fn()


DQNTrainer = OffPolicyTrainer
TD3Trainer = OffPolicyTrainer
SACTrainer = OffPolicyTrainer

__all__ = ["OffPolicyTrainer", "DQNTrainer", "TD3Trainer", "SACTrainer"]
