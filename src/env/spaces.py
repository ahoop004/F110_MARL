"""Lightweight space specifications for the local training path."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class SpaceSpec:
    """Describes a continuous box space: shape, bounds, dtype."""
    shape: tuple
    low: np.ndarray
    high: np.ndarray
    dtype: type = np.float32

    def __post_init__(self) -> None:
        self.low = np.asarray(self.low, dtype=self.dtype)
        self.high = np.asarray(self.high, dtype=self.dtype)
        if self.low.shape != self.shape:
            self.low = np.broadcast_to(self.low, self.shape).copy()
        if self.high.shape != self.shape:
            self.high = np.broadcast_to(self.high, self.shape).copy()

    def sample(self) -> np.ndarray:
        return np.random.uniform(self.low, self.high).astype(self.dtype)

    @property
    def n(self) -> int:
        return int(np.prod(self.shape))


@dataclass
class DictSpaceSpec:
    """Describes a structured observation space as an ordered dict of SpaceSpecs."""
    spaces: Dict[str, SpaceSpec]

    def sample(self) -> Dict[str, np.ndarray]:
        return {k: v.sample() for k, v in self.spaces.items()}

    @property
    def n(self) -> int:
        return sum(s.n for s in self.spaces.values())

    @property
    def shape(self) -> tuple:
        return (self.n,)
