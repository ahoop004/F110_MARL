"""Denormalize action — maps [-1, 1] normalized output to physical action range."""
from __future__ import annotations

import numpy as np

from wrappers.actions.base import ActionComponent


class DenormalizeComponent(ActionComponent):
    """Linearly maps each action dimension from [-1, 1] to [low, high]."""

    def __init__(self, action_low: np.ndarray, action_high: np.ndarray) -> None:
        self._low = np.asarray(action_low, dtype=np.float32)
        self._high = np.asarray(action_high, dtype=np.float32)
        self._scale = (self._high - self._low) / 2.0
        self._offset = (self._high + self._low) / 2.0

    def process(self, action: np.ndarray) -> np.ndarray:
        return (np.asarray(action, dtype=np.float32) * self._scale + self._offset)
