"""Discrete action component — maps a scalar index to a physical action from a predefined set."""
from __future__ import annotations

import numpy as np

from wrappers.actions.base import ActionComponent


class DiscreteActionComponent(ActionComponent):
    """Looks up a physical action from an action set using an integer index.

    Replaces denormalization for discrete-action agents (DQN).
    The agent outputs a scalar index; this component returns the corresponding
    (steer, speed) action from the predefined action_set.
    """

    def __init__(self, action_set: np.ndarray) -> None:
        self._action_set = np.asarray(action_set, dtype=np.float32)

    @property
    def n_actions(self) -> int:
        return len(self._action_set)

    def process(self, action: np.ndarray) -> np.ndarray:
        idx = int(np.asarray(action).ravel()[0])
        idx = max(0, min(idx, self.n_actions - 1))
        return self._action_set[idx].copy()
