"""Action wrappers for continuous scaling and discrete templating."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from f110x.wrappers.common import ensure_index, to_numpy


class DiscreteActionWrapper:
    """Map discrete action indices to continuous control primitives."""

    def __init__(self, action_set: Iterable[Iterable[float]]) -> None:
        action_array = to_numpy(action_set)
        if action_array.ndim != 2:
            raise ValueError("action_set must be 2D: (n_actions, action_dim)")
        self._actions = action_array

    @property
    def actions(self) -> List[np.ndarray]:
        return [action.copy() for action in self._actions]

    def transform(self, _agent_id: str, action: Any) -> np.ndarray:
        if np.isscalar(action):
            return self.index_to_action(ensure_index(action))
        action_arr = to_numpy(action)
        if action_arr.ndim == 0:
            return self.index_to_action(ensure_index(action_arr))
        return action_arr

    def index_to_action(self, index: int) -> np.ndarray:
        return self._actions[index].copy()


class DeltaDiscreteActionWrapper:
    """Maintain per-agent action state and apply discrete deltas."""

    def __init__(
        self,
        action_deltas: Iterable[Iterable[float]],
        low: Iterable[float],
        high: Iterable[float],
    ) -> None:
        delta_array = to_numpy(action_deltas)
        if delta_array.ndim != 2:
            raise ValueError("action_deltas must be 2D: (n_actions, action_dim)")
        self._deltas = delta_array
        self.low = to_numpy(low)
        self.high = to_numpy(high)
        if self.low.shape != self.high.shape:
            raise ValueError("low/high must have matching shapes")
        if self.low.shape[0] != self._deltas.shape[1]:
            raise ValueError("delta dimensionality must match action bounds")
        self._state: Dict[str, np.ndarray] = {}

    def reset(self, agent_id: str, initial_action: Optional[Iterable[float]] = None) -> None:
        if initial_action is None:
            initial_action = np.zeros_like(self.low)
        self._state[agent_id] = to_numpy(initial_action)

    def transform(self, agent_id: str, action: Any) -> np.ndarray:
        index = ensure_index(action)
        if agent_id not in self._state:
            self.reset(agent_id)
        baseline = self._state[agent_id]
        delta = self._deltas[index]
        updated = np.clip(baseline + delta, self.low, self.high)
        self._state[agent_id] = updated
        return updated.copy()
