"""Trainer adapter for the recurrent PPO agent."""

from __future__ import annotations

from typing import Any, Dict, Optional

from f110x.trainers.base import Trainer, Transition
from f110x.policies.ppo.rec_ppo import RecurrentPPOAgent


class RecurrentPPOTrainer(Trainer):
    def __init__(self, agent_id: str, agent: RecurrentPPOAgent) -> None:
        self.agent_id = agent_id
        self._agent = agent

    def select_action(self, obs: Any, *, deterministic: bool = False) -> Any:
        if deterministic and hasattr(self._agent, "act_deterministic"):
            return self._agent.act_deterministic(obs, self.agent_id)
        return self._agent.act(obs, self.agent_id)

    def observe(self, transition: Transition) -> None:
        next_obs = transition.next_obs
        if transition.truncated and not transition.terminated:
            try:
                self._agent.record_final_value(next_obs)
            except Exception:
                pass

        done_flag = transition.terminated or transition.truncated
        self._agent.store(
            next_obs,
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
