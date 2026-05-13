"""Previous action observation component."""
from __future__ import annotations

from typing import Dict

import numpy as np

from wrappers.observations.base import ObservationComponent


class PrevActionComponent(ObservationComponent):
    """Last action taken by this agent: [steer, speed] — 2 dims.

    The composer stores the previous action and injects it on each call.
    """

    def __init__(self, action_dim: int = 2) -> None:
        self._action_dim = action_dim
        self._prev_action = np.zeros(action_dim, dtype=np.float32)

    @property
    def dim(self) -> int:
        return self._action_dim

    def update(self, action: np.ndarray) -> None:
        """Call after each env.step() to track the last action."""
        self._prev_action = np.asarray(action, dtype=np.float32).ravel()[: self._action_dim]

    def reset(self) -> None:
        self._prev_action = np.zeros(self._action_dim, dtype=np.float32)

    def compute(self, raw_obs: Dict, info: Dict) -> np.ndarray:
        return self._prev_action.copy()
