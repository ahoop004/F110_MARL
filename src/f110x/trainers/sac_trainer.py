"""SAC trainer adapter implementing the shared Trainer protocol."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from f110x.trainers.base import Trainer, Transition
from f110x.policies.sac.sac import SACAgent


class SACTrainer(Trainer):
    def __init__(self, agent_id: str, agent: SACAgent) -> None:
        self.agent_id = agent_id
        self._agent = agent

    def select_action(self, obs: Any, *, deterministic: bool = False) -> Any:
        obs_np = np.asarray(obs, dtype=np.float32)
        return self._agent.act(obs_np, deterministic=deterministic)

    def observe(self, transition: Transition) -> None:
        done_flag = transition.terminated or transition.truncated
        self._agent.store_transition(
            transition.obs,
            transition.action,
            transition.reward,
            transition.next_obs,
            done_flag,
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
