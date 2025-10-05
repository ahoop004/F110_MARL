"""Action wrappers for continuous scaling and discrete templating."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from f110x.wrappers.common import ensure_index, to_numpy


class DiscreteActionWrapper:
    """Map discrete action indices to continuous control primitives."""

    def __init__(self, action_set: Iterable[Iterable[float]]) -> None:
        action_array = to_numpy(action_set)
        if action_array.ndim != 2:
            raise ValueError("action_set must be 2D: (n_actions, action_dim)")
        self._actions = action_array

    @property
    def actions(self) -> List[np.ndarray]:
        return [action.copy() for action in self._actions]

    def transform(self, _agent_id: str, action: Any) -> np.ndarray:
        if np.isscalar(action):
            return self.index_to_action(ensure_index(action))
        action_arr = to_numpy(action)
        if action_arr.ndim == 0:
            return self.index_to_action(ensure_index(action_arr))
        return action_arr

    def index_to_action(self, index: int) -> np.ndarray:
        return self._actions[index].copy()


class DeltaDiscreteActionWrapper:
    """Maintain per-agent action state and apply discrete deltas."""

    def __init__(
        self,
        action_deltas: Iterable[Iterable[float]],
        low: Iterable[float],
        high: Iterable[float],
        *,
        initial_action: Optional[Iterable[float]] = None,
        prevent_reverse: bool = False,
        stop_threshold: float = 0.0,
        speed_index: int = 1,
    ) -> None:
        delta_array = to_numpy(action_deltas)
        if delta_array.ndim != 2:
            raise ValueError("action_deltas must be 2D: (n_actions, action_dim)")
        self._deltas = delta_array
        self.low = to_numpy(low)
        self.high = to_numpy(high)
        if self.low.shape != self.high.shape:
            raise ValueError("low/high must have matching shapes")
        if self.low.shape[0] != self._deltas.shape[1]:
            raise ValueError("delta dimensionality must match action bounds")
        self._state: Dict[str, np.ndarray] = {}
        self._default_initial = (
            None if initial_action is None else to_numpy(initial_action)
        )
        self._prevent_reverse = bool(prevent_reverse)
        self._stop_threshold = float(stop_threshold)
        self._speed_index = int(speed_index)

    def reset(self, agent_id: str, initial_action: Optional[Iterable[float]] = None) -> None:
        baseline = initial_action if initial_action is not None else self._default_initial
        if baseline is None:
            baseline = np.zeros_like(self.low)
        self._state[agent_id] = to_numpy(baseline)

    def transform(self, agent_id: str, action: Any) -> np.ndarray:
        index = ensure_index(action)
        if agent_id not in self._state:
            self.reset(agent_id)
        baseline = self._state[agent_id]
        delta = self._deltas[index]

        if self._prevent_reverse and 0 <= self._speed_index < delta.shape[0]:
            throttle_delta = float(delta[self._speed_index])
            current_speed = float(baseline[self._speed_index])
            if throttle_delta < 0.0 and current_speed <= self._stop_threshold:
                delta = delta.copy()
                delta[self._speed_index] = 0.0

        updated = np.clip(baseline + delta, self.low, self.high)
        if self._prevent_reverse and 0 <= self._speed_index < updated.shape[0]:
            if updated[self._speed_index] < max(0.0, self._stop_threshold):
                updated = updated.copy()
                updated[self._speed_index] = max(0.0, self._stop_threshold)
        self._state[agent_id] = updated
        return updated.copy()


class ActionRepeatWrapper:
    """Cache transformed actions and re-emit them for multiple env steps."""

    def __init__(
        self,
        inner: Optional[Any],
        repeat: int,
    ) -> None:
        if repeat <= 0:
            raise ValueError("action repeat must be positive")
        self.inner = inner
        self.repeat = int(repeat)
        self._cached: Dict[str, np.ndarray] = {}
        self._remaining: Dict[str, int] = {}

    def transform(self, agent_id: str, action: Any) -> np.ndarray:
        remaining = self._remaining.get(agent_id, 0)
        if remaining <= 0:
            transformed = self._transform_inner(agent_id, action)
            self._cached[agent_id] = transformed
            self._remaining[agent_id] = self.repeat - 1
            return transformed.copy()

        self._remaining[agent_id] = remaining - 1
        cached = self._cached.get(agent_id)
        if cached is None:
            transformed = self._transform_inner(agent_id, action)
            self._cached[agent_id] = transformed
            self._remaining[agent_id] = self.repeat - 1
            return transformed.copy()
        return cached.copy()

    def reset(self, agent_id: str, *args: Any, **kwargs: Any) -> None:
        self._cached.pop(agent_id, None)
        self._remaining.pop(agent_id, None)
        if self.inner is not None:
            reset_fn = getattr(self.inner, "reset", None)
            if callable(reset_fn):
                reset_fn(agent_id, *args, **kwargs)

    def _transform_inner(self, agent_id: str, action: Any) -> np.ndarray:
        if self.inner is None:
            return to_numpy(action).copy()
        transformed = self.inner.transform(agent_id, action)
        return to_numpy(transformed)
