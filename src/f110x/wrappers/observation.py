from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import numpy as np
from gymnasium import spaces
from gymnasium.spaces import utils as space_utils

from .base import BaseParallelWrapper


class ObservationSanitizerWrapper(BaseParallelWrapper):
    """Down-sample lidar and strip NaNs/Infs from observations."""

    def __init__(
        self,
        env,
        *,
        lidar_keys: Iterable[str] = ("lidar", "scans"),
        target_beams: int | None = None,
        max_range: float | None = None,
        dtype: np.dtype = np.float32,
    ) -> None:
        super().__init__(env)
        self._lidar_keys = tuple(lidar_keys)
        self._target_beams = int(target_beams) if target_beams and target_beams > 0 else None
        self._max_range = float(max_range) if max_range is not None else None
        self._dtype = dtype
        self._downsample_cache: Dict[Tuple[int, int], np.ndarray] = {}

    def reset(self, *args: Any, **kwargs: Any):
        obs, info = super().reset(*args, **kwargs)
        return self._sanitize_all(obs), info

    def step(self, actions: Dict[str, Any]):
        obs, rewards, dones, truncs, infos = super().step(actions)
        obs = self._sanitize_all(obs)
        return obs, rewards, dones, truncs, infos

    # ---------------------------------------------------------------------
    def _sanitize_all(self, obs: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        sanitized: Dict[str, Dict[str, Any]] = {}
        for aid, agent_obs in obs.items():
            sanitized[aid] = self._sanitize_agent(agent_obs)
        return sanitized

    def _sanitize_agent(self, agent_obs: Dict[str, Any]) -> Dict[str, Any]:
        clean: Dict[str, Any] = {}
        for key, value in agent_obs.items():
            if key in self._lidar_keys:
                clean[key] = self._sanitize_lidar(value)
            elif isinstance(value, np.ndarray):
                clean[key] = self._sanitize_array(value)
            elif np.isscalar(value):
                clean[key] = self._sanitize_scalar(value)
            else:
                clean[key] = value
        return clean

    def _sanitize_lidar(self, value: Any) -> np.ndarray:
        arr = np.asarray(value, dtype=self._dtype).reshape(-1)
        arr = np.nan_to_num(arr, nan=0.0, posinf=self._max_range or 0.0, neginf=0.0)
        if self._max_range is not None:
            arr = np.clip(arr, 0.0, self._max_range)
        if self._target_beams is not None and arr.size != self._target_beams:
            arr = self._match_beam_count(arr, self._target_beams)
        return arr.astype(self._dtype, copy=False)

    def _sanitize_array(self, value: np.ndarray) -> np.ndarray:
        arr = np.asarray(value, dtype=self._dtype)
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    def _sanitize_scalar(self, value: Any) -> np.float32:
        return np.float32(value if np.isfinite(value) else 0.0)

    def _match_beam_count(self, arr: np.ndarray, target: int) -> np.ndarray:
        if arr.size == target:
            return arr
        if arr.size < target:
            padded = np.zeros((target,), dtype=arr.dtype)
            padded[: arr.size] = arr
            return padded
        key = (arr.size, target)
        indices = self._downsample_cache.get(key)
        if indices is None:
            indices = np.linspace(0, arr.size - 1, target, dtype=np.int32)
            self._downsample_cache[key] = indices
        return arr[indices]


class FlattenObservationWrapper(BaseParallelWrapper):
    """Flatten nested dict observations into 1-D vectors."""

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
