"""Base primitives shared by reward tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple, TYPE_CHECKING


if TYPE_CHECKING:  # pragma: no cover - typing helpers only
    from v2.env import F110ParallelEnv
    from v2.utils.map_loader import MapData


RewardComponents = Dict[str, float]
RewardComputation = Tuple[float, RewardComponents]


@dataclass
class RewardRuntimeContext:
    """Runtime context provided to reward tasks."""

    env: "F110ParallelEnv"
    map_data: "MapData"
    roster: Optional[Any] = None


@dataclass
class RewardStep:
    """Canonical observation bundle passed to reward strategies."""

    agent_id: str
    obs: Dict[str, Any]
    env_reward: float
    done: bool
    info: Optional[Dict[str, Any]]
    all_obs: Optional[Dict[str, Dict[str, Any]]]
    episode_index: int
    step_index: int
    current_time: float
    timestep: float
    events: Dict[str, Any] = field(default_factory=dict)


class RewardStrategy:
    """Base interface for concrete reward strategies."""

    name: str = "base"

    def reset(self, episode_index: int) -> None:  # pragma: no cover - default noop
        return None

    def compute(self, step: RewardStep) -> RewardComputation:
        raise NotImplementedError


class PerAgentStateMixin(RewardStrategy):
    """Mixin that manages per-agent state dictionaries for reward strategies."""

    def __init__(self, state_factory: Callable[[], Any], *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._state_factory: Callable[[], Any] = state_factory
        self._agent_state: Dict[str, Any] = {}

    def reset(self, episode_index: int) -> None:
        self._agent_state.clear()
        super().reset(episode_index)

    def state_for(self, agent_id: str) -> Any:
        return self._agent_state.setdefault(agent_id, self._state_factory())


__all__ = [
    "PerAgentStateMixin",
    "RewardComponents",
    "RewardComputation",
    "RewardRuntimeContext",
    "RewardStep",
    "RewardStrategy",
]
