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
        arr = np.asarray(action, dtype=np.float32).ravel()
        n = min(arr.shape[0], self._action_dim)
        self._prev_action[:n] = arr[:n]
        if n < self._action_dim:
            self._prev_action[n:] = 0.0

    def reset(self) -> None:
        self._prev_action.fill(0.0)

    def compute_into(self, raw_obs: Dict, info: Dict, out: np.ndarray) -> None:
        np.copyto(out, self._prev_action)

    def compute(self, raw_obs: Dict, info: Dict) -> np.ndarray:
        return self._prev_action.copy()
