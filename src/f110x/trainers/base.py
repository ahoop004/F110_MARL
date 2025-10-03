"""Common trainer interfaces and transition containers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Protocol


ObservationDict = Mapping[str, Any]


@dataclass(slots=True)
class Transition:
    """Single-agent transition used by trainer adapters."""

    agent_id: str
    obs: Any
    action: Any
    reward: float
    next_obs: Any
    terminated: bool
    truncated: bool = False
    info: Optional[Dict[str, Any]] = None
    raw_obs: Optional[ObservationDict] = None


class Trainer(Protocol):
    """Protocol every trainer implementation should follow."""

    agent_id: str

    def select_action(self, obs: Any, *, deterministic: bool = False) -> Any:
        ...

    def observe(self, transition: Transition) -> None:
        ...

    def update(self) -> Optional[Dict[str, Any]]:
        ...

    def save(self, path: str) -> None:
        ...

    def load(self, path: str) -> None:
        ...
