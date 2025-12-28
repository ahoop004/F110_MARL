"""Abstract trainer interfaces and transition containers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

ObservationDict = Mapping[str, Any]


@dataclass(slots=True)
class Transition:
    """Single-agent transition consumed by trainer implementations."""

    agent_id: str
    obs: Any
    action: Any
    reward: float
    next_obs: Any
    terminated: bool
    truncated: bool = False
    info: Optional[Dict[str, Any]] = None
    raw_obs: Optional[ObservationDict] = None


class Trainer(ABC):
    """Abstract base class that all trainers must implement."""

    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id

    @abstractmethod
    def select_action(self, obs: Any, *, deterministic: bool = False) -> Any:
        """Return an action for the provided observation."""

    @abstractmethod
    def observe(self, transition: Transition) -> None:
        """Record a transition produced by the environment."""

    @abstractmethod
    def update(self) -> Optional[Dict[str, Any]]:
        """Perform a learning step, returning optional stats."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist trainer state (model weights, optimiser, etc.)."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Load trainer state from a checkpoint path."""


__all__ = ["ObservationDict", "Transition", "Trainer"]
