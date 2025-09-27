from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .base import BaseParallelWrapper


class FlattenObservationWrapper(BaseParallelWrapper):
    """Placeholder for MARLlib-style observation flattening."""

    def __init__(self, env, dtype=np.float32) -> None:
        super().__init__(env)
        self._dtype = dtype

    def reset(self, *args: Any, **kwargs: Any):
        obs, info = super().reset(*args, **kwargs)
        return self._flatten_all(obs), info

    def step(self, actions: Dict[str, Any]):
        obs, rewards, dones, truncs, infos = super().step(actions)
        obs = self._flatten_all(obs)
        return obs, rewards, dones, truncs, infos

    def _flatten_all(self, obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        # TODO: flatten nested dict observations into flat arrays per MARLlib expectations.
        return obs

