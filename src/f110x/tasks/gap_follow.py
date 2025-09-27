from __future__ import annotations

from typing import Any, Dict

from .task import Task


class GapFollowTask(Task):
    """Lightweight task stub for scripted gap-follow agents."""

    def __init__(
        self,
        laps: int = 1,
        time_limit: float = 120.0,
        terminate_on_collision: bool = True,
        **policy_kwargs: Any,
    ) -> None:
        self._laps = int(laps)
        self._time_limit = float(time_limit)
        self._terminate_on_collision = bool(terminate_on_collision)
        self._policy_kwargs: Dict[str, Any] = dict(policy_kwargs)

    def reset(self) -> None:
        pass

    def reward(self, agent_id, state, action) -> float:
        return 0.0

    def done(self, agent_id, state) -> bool:
        agent_state = state.get(agent_id, {}) if isinstance(state, dict) else {}
        if not isinstance(agent_state, dict):
            agent_state = {}

        if self._terminate_on_collision:
            collided = bool(agent_state.get("wall_collision"))
            collided = collided or bool(agent_state.get("collision", False))
            opponent_hits = agent_state.get("opponent_collisions")
            if isinstance(opponent_hits, (list, tuple, set)) and opponent_hits:
                collided = True
            if collided:
                return True

        laps = float(agent_state.get("lap", 0.0))
        if laps >= self._laps:
            return True
        time_elapsed = float(agent_state.get("time", 0.0))
        return time_elapsed >= self._time_limit

    def policy_kwargs(self) -> Dict[str, Any]:
        return dict(self._policy_kwargs)
