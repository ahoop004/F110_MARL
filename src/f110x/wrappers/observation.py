from __future__ import annotations

from typing import Any, Dict

import numpy as np
from gymnasium import spaces
from gymnasium.spaces import utils as space_utils

from .base import BaseParallelWrapper


class FlattenObservationWrapper(BaseParallelWrapper):
    """Placeholder for MARLlib-style observation flattening."""

    def __init__(self, env, dtype=np.float32) -> None:
        super().__init__(env)
        self._dtype = dtype
        self._flat_cache: Dict[str, spaces.Box] = {}

    def reset(self, *args: Any, **kwargs: Any):
        obs, info = super().reset(*args, **kwargs)
        return self._flatten_all(obs), info

    def step(self, actions: Dict[str, Any]):
        obs, rewards, dones, truncs, infos = super().step(actions)
        obs = self._flatten_all(obs)
        return obs, rewards, dones, truncs, infos

    def observation_space(self, agent: str):
        base = self.env.observation_space(agent)
        flattened = space_utils.flatten_space(base)
        if not isinstance(flattened, spaces.Box):
            raise TypeError("Flattened observation space must be a Box")
        low = np.asarray(flattened.low, dtype=self._dtype)
        high = np.asarray(flattened.high, dtype=self._dtype)
        # Some entries may be +/-inf; Box handles that already.
        flat_space = spaces.Box(low=low, high=high, dtype=self._dtype)
        self._flat_cache[agent] = flat_space
        return flat_space

    def _flatten_all(self, obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        flattened: Dict[str, np.ndarray] = {}
        for aid, value in obs.items():
            base_space = self.env.observation_space(aid)
            flat = space_utils.flatten(base_space, value).astype(self._dtype, copy=False)
            flattened[aid] = flat
        return flattened
