"""Base class for action transform components."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ActionComponent(ABC):
    """A single transform applied to an action vector in the processing pipeline."""

    @abstractmethod
    def process(self, action: np.ndarray) -> np.ndarray:
        """Apply this transform and return the modified action."""
