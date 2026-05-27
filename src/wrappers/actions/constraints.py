"""Action constraint components — clip or override action dimensions."""
from __future__ import annotations

import numpy as np

from wrappers.actions.base import ActionComponent


class PreventReverseComponent(ActionComponent):
    """Clips the speed dimension to >= 0, preventing the car from reversing.

    Modifies *action* in-place (the composer guarantees that by this point
    the array is a freshly-allocated float32 buffer owned by the pipeline).
    """

    def __init__(self, speed_index: int = 1) -> None:
        self._idx = int(speed_index)

    def process(self, action: np.ndarray) -> np.ndarray:
        if len(action) > self._idx and action[self._idx] < 0.0:
            action[self._idx] = 0.0
        return action
