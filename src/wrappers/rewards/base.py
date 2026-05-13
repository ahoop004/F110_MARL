"""Base classes for reward components and composers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Tuple


class RewardComponent(ABC):
    """Single named contribution to the total reward signal."""

    @abstractmethod
    def compute(self, step_info: dict) -> Dict[str, float]:
        """Compute this component's reward for one step.

        Args:
            step_info: dict containing at minimum:
                obs, next_obs, info, done, truncated, action, timestep

        Returns:
            Dict mapping 'component/subkey' → float value (can be empty dict).
        """

    def reset(self) -> None:
        """Reset any per-episode state. Override if needed."""
