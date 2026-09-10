"""Small interfaces shared by current agents; trainers own update contracts."""
from typing import Any, Dict, Protocol, Tuple, Union, runtime_checkable

import numpy as np


@runtime_checkable
class Agent(Protocol):
    """PPO/MAPPO inference and checkpoint interface.

    PPO returns (action, log_prob, value); MAPPO returns (action, log_prob).
    Rollout storage and update signatures belong to each agent and trainer.
    """

    def act(
        self, obs: np.ndarray, deterministic: bool = False,
    ) -> Union[Tuple[np.ndarray, float], Tuple[np.ndarray, float, float]]:
        ...

    def save(self, path: str) -> None:
        ...

    def load(self, path: str) -> None:
        ...


@runtime_checkable
class HeuristicPolicy(Protocol):
    """Fixed controllers consume raw observations and return physical actions."""

    def act(self, obs: Dict[str, Any], deterministic: bool = False) -> np.ndarray:
        ...

    def reset(self) -> None:
        """Clear episode-scoped controller state."""
        ...


def is_heuristic_policy(agent: Any) -> bool:
    return isinstance(agent, HeuristicPolicy)
