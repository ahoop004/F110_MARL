"""Base primitives shared by reward tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING


if TYPE_CHECKING:  # pragma: no cover - typing helpers only
    from f110x.envs import F110ParallelEnv
    from f110x.utils.map_loader import MapData


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


class RewardStrategy:
    """Base interface for concrete reward strategies."""

    name: str = "base"

    def reset(self, episode_index: int) -> None:  # pragma: no cover - default noop
        return None

    def compute(self, step: RewardStep) -> RewardComputation:
        raise NotImplementedError


__all__ = [
    "RewardComponents",
    "RewardComputation",
    "RewardRuntimeContext",
    "RewardStep",
    "RewardStrategy",
]
