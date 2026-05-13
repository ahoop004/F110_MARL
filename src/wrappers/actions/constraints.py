"""Action constraint components — clip or override action dimensions."""
from __future__ import annotations

import numpy as np

from wrappers.actions.base import ActionComponent


class PreventReverseComponent(ActionComponent):
    """Clips the speed dimension to >= 0, preventing the car from reversing."""

    def __init__(self, speed_index: int = 1) -> None:
        self._idx = int(speed_index)

    def process(self, action: np.ndarray) -> np.ndarray:
        a = np.asarray(action, dtype=np.float32).copy()
        if len(a) > self._idx:
            a[self._idx] = max(0.0, a[self._idx])
        return a
