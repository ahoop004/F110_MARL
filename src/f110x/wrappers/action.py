from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .base import BaseParallelWrapper


class ActionScaleWrapper(BaseParallelWrapper):
    """Placeholder for normalizing actions into [-1, 1] for MARLlib policies."""

    def __init__(self, env) -> None:
        super().__init__(env)
        self._low: Dict[str, np.ndarray] = {}
        self._high: Dict[str, np.ndarray] = {}

    def reset(self, *args: Any, **kwargs: Any):
        obs, info = super().reset(*args, **kwargs)
        self._capture_bounds()
        return obs, info

    def step(self, actions: Dict[str, np.ndarray]):
        denorm = {
            aid: self._denormalize(aid, act)
            for aid, act in actions.items()
        }
        return super().step(denorm)

    def _capture_bounds(self) -> None:
        for aid in getattr(self.env, "possible_agents", []):
            space = self.env.action_space(aid)
            if hasattr(space, "low"):
                self._low[aid] = np.asarray(space.low, dtype=np.float32)
            if hasattr(space, "high"):
                self._high[aid] = np.asarray(space.high, dtype=np.float32)
        # TODO: update bounds when agents join/leave mid-episode.

    def _denormalize(self, aid: str, action: np.ndarray) -> np.ndarray:
        low = self._low.get(aid)
        high = self._high.get(aid)
        if low is None or high is None:
            return action
        return low + (0.5 * (action + 1.0) * (high - low))

