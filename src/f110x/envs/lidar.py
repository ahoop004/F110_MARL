"""LiDAR processing helpers for the parallel F110 environment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class LidarProcessor:
    """Cache-aware LiDAR downsampler."""

    beam_count: int
    indices: Optional[np.ndarray] = None
    source_size: Optional[int] = None

    def update_beam_count(self, beam_count: int) -> None:
        self.beam_count = max(int(beam_count), 0)
        self.indices = None
        self.source_size = None

    def select(self, scan: np.ndarray) -> np.ndarray:
        target = self.beam_count
        if target <= 0 or scan.size == 0:
            return scan.astype(np.float32, copy=False)
        if scan.size == target:
            return scan.astype(np.float32, copy=False)
        if scan.size < target:
            padded = np.zeros((target,), dtype=np.float32)
            view = scan.astype(np.float32, copy=False)
            padded[: view.size] = view
            return padded
        if self.indices is None or self.source_size != scan.size:
            self.indices = np.linspace(0, scan.size - 1, target, dtype=np.int32)
            self.source_size = scan.size
        return scan[self.indices].astype(np.float32, copy=False)

