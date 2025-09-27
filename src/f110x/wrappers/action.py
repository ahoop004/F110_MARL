from __future__ import annotations

from typing import Any, Dict

import numpy as np
from gymnasium import spaces

from .base import BaseParallelWrapper


class ActionScaleWrapper(BaseParallelWrapper):
    """Placeholder for normalizing actions into [-1, 1] for MARLlib policies."""

    def __init__(self, env) -> None:
        super().__init__(env)
        self._low: Dict[str, np.ndarray] = {}
        self._high: Dict[str, np.ndarray] = {}
        self._mid: Dict[str, np.ndarray] = {}
        self._scale: Dict[str, np.ndarray] = {}
        self._normalized_spaces: Dict[str, spaces.Box] = {}
        self._bounds_ready = False

    def reset(self, *args: Any, **kwargs: Any):
        self._bounds_ready = False
        obs, info = super().reset(*args, **kwargs)
        self._capture_bounds()
        return obs, info

    def step(self, actions: Dict[str, np.ndarray]):
        self._capture_bounds()
        denorm = {
            aid: self._denormalize(aid, act)
            for aid, act in actions.items()
        }
        return super().step(denorm)

    def _capture_bounds(self) -> None:
        if self._bounds_ready:
            return
        for aid in getattr(self.env, "possible_agents", []):
            space = self.env.action_space(aid)
            if not isinstance(space, spaces.Box):
                raise TypeError("ActionScaleWrapper requires Box action spaces")
            low = np.asarray(space.low, dtype=np.float32)
            high = np.asarray(space.high, dtype=np.float32)
            self._low[aid] = low
            self._high[aid] = high
            self._mid[aid] = (high + low) * 0.5
            self._scale[aid] = (high - low) * 0.5
            ones = np.ones_like(low, dtype=np.float32)
            self._normalized_spaces[aid] = spaces.Box(-ones, ones, dtype=np.float32)
        self._bounds_ready = True

    def _denormalize(self, aid: str, action: np.ndarray) -> np.ndarray:
        low = self._low.get(aid)
        high = self._high.get(aid)
        mid = self._mid.get(aid)
        scale = self._scale.get(aid)
        if low is None or high is None or mid is None or scale is None:
            return np.asarray(action, dtype=np.float32)
        act = np.asarray(action, dtype=np.float32)
        act = np.clip(act, -1.0, 1.0)
        return mid + act * scale

    def action_space(self, agent: str):
        self._capture_bounds()
        space = self._normalized_spaces.get(agent)
        if space is None:
            base = self.env.action_space(agent)
            if not isinstance(base, spaces.Box):
                raise TypeError("ActionScaleWrapper requires Box action spaces")
            ones = np.ones_like(base.low, dtype=np.float32)
            space = spaces.Box(-ones, ones, dtype=np.float32)
            self._normalized_spaces[agent] = space
        return space
