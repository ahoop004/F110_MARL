"""Action wrappers for continuous scaling and discrete templating."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np


class ContinuousActionWrapper:
    """Scale normalized actions in [-1, 1] to environment bounds."""

    def __init__(self, low: Iterable[float], high: Iterable[float]) -> None:
        self.low = np.asarray(low, dtype=np.float32)
        self.high = np.asarray(high, dtype=np.float32)
        if self.low.shape != self.high.shape:
            raise ValueError("low/high must have matching shapes")
        self.range = self.high - self.low
        if np.any(self.range <= 0):
            raise ValueError("action range must be positive")

    def __call__(self, action: Iterable[float]) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        return np.clip(action, -1.0, 1.0) * (self.range / 2.0) + (self.low + self.high) / 2.0


class DiscreteActionWrapper:
    """Map discrete action indices to continuous control primitives."""

    def __init__(self, action_set: Iterable[Iterable[float]]) -> None:
        action_array = np.asarray(action_set, dtype=np.float32)
        if action_array.ndim != 2:
            raise ValueError("action_set must be 2D: (n_actions, action_dim)")
        self._actions = action_array

    @property
    def actions(self) -> List[np.ndarray]:
        return [action.copy() for action in self._actions]

    def __call__(self, action: Any) -> np.ndarray:
        if np.isscalar(action):
            return self.index_to_action(int(action))
        action_arr = np.asarray(action, dtype=np.float32)
        if action_arr.ndim == 0:
            return self.index_to_action(int(action_arr.item()))
        return action_arr

    def index_to_action(self, index: int) -> np.ndarray:
        return self._actions[index].copy()

    def action_to_index(self, action: Iterable[float]) -> int:
        action_arr = np.asarray(action, dtype=np.float32)
        diffs = np.linalg.norm(self._actions - action_arr, axis=1)
        return int(np.argmin(diffs))


class DeltaDiscreteActionWrapper:
    """Maintain per-agent action state and apply discrete deltas."""

    def __init__(
        self,
        action_deltas: Iterable[Iterable[float]],
        low: Iterable[float],
        high: Iterable[float],
    ) -> None:
        delta_array = np.asarray(action_deltas, dtype=np.float32)
        if delta_array.ndim != 2:
            raise ValueError("action_deltas must be 2D: (n_actions, action_dim)")
        self._deltas = delta_array
        self.low = np.asarray(low, dtype=np.float32)
        self.high = np.asarray(high, dtype=np.float32)
        if self.low.shape != self.high.shape:
            raise ValueError("low/high must have matching shapes")
        if self.low.shape[0] != self._deltas.shape[1]:
            raise ValueError("delta dimensionality must match action bounds")
        self._state: Dict[str, np.ndarray] = {}

    def reset(self, agent_id: str, initial_action: Optional[Iterable[float]] = None) -> None:
        if initial_action is None:
            initial_action = np.zeros_like(self.low)
        self._state[agent_id] = np.asarray(initial_action, dtype=np.float32)

    def __call__(self, agent_id: str, index: int) -> np.ndarray:
        if agent_id not in self._state:
            self.reset(agent_id)
        baseline = self._state[agent_id]
        delta = self._deltas[index]
        updated = np.clip(baseline + delta, self.low, self.high)
        self._state[agent_id] = updated
        return updated.copy()
