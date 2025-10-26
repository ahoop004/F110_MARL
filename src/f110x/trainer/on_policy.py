"""Trainer implementations for on-policy algorithms."""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from f110x.trainer.base import Trainer, Transition

try:  # Lazy policy imports to avoid optional dependency issues during docs/import-time
    from f110x.policies.ppo.ppo import PPOAgent
    from f110x.policies.ppo.rec_ppo import RecurrentPPOAgent
except Exception:  # pragma: no cover - protects import-time when policies unavailable
    PPOAgent = Any  # type: ignore
    RecurrentPPOAgent = Any  # type: ignore


class OnPolicyTrainer(Trainer):
    """Adapter covering both feed-forward and recurrent PPO-style agents."""

    def __init__(
        self,
        agent_id: str,
        agent: Any,
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(agent_id)
        self._agent = agent
        cfg = dict(config or {})
        self._deterministic_attr = cfg.get("deterministic_method", "act_deterministic")
        self._record_final_value = bool(cfg.get("record_final_value", True))
        self._recurrent = bool(cfg.get("recurrent", False))

    def select_action(self, obs: Any, *, deterministic: bool = False) -> Any:
        if self._recurrent:
            act_fn = getattr(self._agent, "act", None)
            if not callable(act_fn):
                raise TypeError("Recurrent agent must expose an 'act' method")
            return act_fn(obs, self.agent_id)

        if deterministic:
            det_fn = getattr(self._agent, self._deterministic_attr, None)
            if callable(det_fn):
                return det_fn(obs, self.agent_id)
        return self._agent.act(obs, self.agent_id)

    def observe(self, transition: Transition) -> None:
        if transition.truncated and not transition.terminated and self._record_final_value:
            record_value = getattr(self._agent, "record_final_value", None)
            if callable(record_value):
                record_value(transition.next_obs)
        done_flag = transition.terminated or transition.truncated
        self._agent.store(
            transition.obs,
            transition.action,
            transition.reward,
            done_flag,
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


PPOTrainer = OnPolicyTrainer
RecurrentPPOTrainer = OnPolicyTrainer

__all__ = ["OnPolicyTrainer", "PPOTrainer", "RecurrentPPOTrainer"]
