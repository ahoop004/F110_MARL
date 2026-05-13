"""Base class for observation components."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional

import numpy as np


class ObservationComponent(ABC):
    """Single named slice of the flat observation vector."""

    @abstractmethod
    def compute(self, raw_obs: Dict, info: Dict) -> np.ndarray:
        """Extract and process this component from the raw env obs dict.

        Args:
            raw_obs: Dict of arrays keyed by obs field name (lidar, pose, velocity, …)
            info:    Step info dict from env.step() or env.reset()

        Returns:
            1-D float32 array of length self.dim
        """

    @property
    @abstractmethod
    def dim(self) -> int:
        """Number of scalar values this component contributes."""

    @property
    def name(self) -> str:
        return self.__class__.__name__
