"""Action wrappers for continuous scaling and discrete templating."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

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
        self._last_meta: Dict[str, Optional[Dict[str, Any]]] = {}

    def transform(self, agent_id: str, action: Any) -> np.ndarray:
        transformed, _ = self._apply(agent_id, action, capture_meta=False)
        return transformed

    def transform_with_info(self, agent_id: str, action: Any) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        return self._apply(agent_id, action, capture_meta=True)

    def reset(self, agent_id: str, *args: Any, **kwargs: Any) -> None:
        self._cached.pop(agent_id, None)
        self._remaining.pop(agent_id, None)
        self._last_meta.pop(agent_id, None)
        if self.inner is not None:
            reset_fn = getattr(self.inner, "reset", None)
            if callable(reset_fn):
                reset_fn(agent_id, *args, **kwargs)

    def _transform_inner(self, agent_id: str, action: Any) -> np.ndarray:
        if self.inner is None:
            return to_numpy(action).copy()
        transformed = self.inner.transform(agent_id, action)
        return to_numpy(transformed)

    def _apply(
        self,
        agent_id: str,
        action: Any,
        *,
        capture_meta: bool,
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        remaining = self._remaining.get(agent_id, 0)
        if remaining <= 0:
            transformed, meta = self._invoke_inner(agent_id, action)
            info = self._prepare_meta(meta, action)
            self._cached[agent_id] = transformed
            self._remaining[agent_id] = self.repeat - 1
            self._last_meta[agent_id] = info.copy() if info is not None else None
        else:
            self._remaining[agent_id] = remaining - 1
            cached = self._cached.get(agent_id)
            if cached is None:
                transformed, meta = self._invoke_inner(agent_id, action)
                info = self._prepare_meta(meta, action)
                self._cached[agent_id] = transformed
                self._remaining[agent_id] = self.repeat - 1
                self._last_meta[agent_id] = info.copy() if info is not None else None
            else:
                transformed = cached
                last_meta = self._last_meta.get(agent_id)
                info = last_meta.copy() if last_meta is not None else None

        if capture_meta:
            info_copy = info.copy() if info is not None else None
            return transformed.copy(), info_copy
        return transformed.copy(), None

    def _invoke_inner(self, agent_id: str, action: Any) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        if self.inner is None:
            return to_numpy(action).copy(), None

        transform_with_info = getattr(self.inner, "transform_with_info", None)
        if callable(transform_with_info):
            transformed, meta = transform_with_info(agent_id, action)
            return to_numpy(transformed), meta

        transformed = self.inner.transform(agent_id, action)
        return to_numpy(transformed), None

    def _prepare_meta(self, meta: Optional[Dict[str, Any]], raw_action: Any) -> Optional[Dict[str, Any]]:
        info = dict(meta) if meta is not None else {}
        if "action_index" not in info:
            if np.isscalar(raw_action) or (
                isinstance(raw_action, np.ndarray) and raw_action.ndim == 0
            ):
                try:
                    info["action_index"] = int(np.asarray(raw_action).item())
                except (TypeError, ValueError):
                    pass
        if not info:
            return None
        return info


class PreventReverseContinuousWrapper:
    """Clamp continuous throttle commands to prevent reverse motion."""

    def __init__(
        self,
        low: Iterable[float],
        high: Iterable[float],
        *,
        min_speed: float = 0.0,
        speed_index: int = 1,
        warmup_steps: int = 0,
        warmup_speed: Optional[float] = None,
    ) -> None:
        self.low = to_numpy(low, copy=True)
        self.high = to_numpy(high, copy=True)
        if self.low.shape != self.high.shape:
            raise ValueError("low/high must have matching shapes for PreventReverseContinuousWrapper")
        self.min_speed = float(min_speed)
        self.speed_index = int(speed_index)
        self.warmup_steps = max(int(warmup_steps), 0)
        self.warmup_speed = None
        if warmup_speed is not None:
            try:
                self.warmup_speed = float(warmup_speed)
            except (TypeError, ValueError):
                self.warmup_speed = None
        self._step_counters: Dict[str, int] = {}

    def _active_warmup_lower_bound(self, agent_id: str) -> Optional[float]:
        if self.warmup_steps <= 0:
            return None
        steps = self._step_counters.get(agent_id, 0)
        if steps >= self.warmup_steps:
            return None
        if self.warmup_speed is not None:
            return self.warmup_speed
        return self.min_speed

    def transform(self, agent_id: str, action: Any) -> np.ndarray:
        action_arr = to_numpy(action, copy=True)
        if action_arr.shape != self.low.shape:
            action_arr = action_arr.reshape(self.low.shape)

        clipped = np.clip(action_arr, self.low, self.high)
        idx = self.speed_index
        if 0 <= idx < clipped.shape[0]:
            lower_bound = max(self.min_speed, float(self.low[idx]))
            # ensure early episodes keep positive throttle
            warmup_bound = self._active_warmup_lower_bound(agent_id)
            if warmup_bound is not None:
                lower_bound = max(lower_bound, warmup_bound)
            max_speed = float(self.high[idx])
            min_speed = lower_bound if lower_bound <= max_speed else max_speed
            clipped[idx] = float(np.clip(clipped[idx], min_speed, max_speed))
            if self.warmup_steps > 0:
                self._step_counters[agent_id] = self._step_counters.get(agent_id, 0) + 1
        return clipped

    def reset(self, agent_id: str) -> None:
        if self.warmup_steps > 0:
            self._step_counters[agent_id] = 0
        else:
            self._step_counters.pop(agent_id, None)
