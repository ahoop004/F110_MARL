from __future__ import annotations

from typing import Any, Dict, Tuple

from .base import BaseParallelWrapper
from .action import ActionScaleWrapper
from .observation import FlattenObservationWrapper


class MarlLibParallelWrapper(BaseParallelWrapper):
    """High-level wrapper chaining observation/action transforms for MARLlib."""

    def __init__(
        self,
        env,
        *,
        flatten_obs: bool = True,
        normalize_actions: bool = True,
        extra_wrappers: Tuple[type, ...] = (),
    ) -> None:
        wrapped = env
        if normalize_actions:
            wrapped = ActionScaleWrapper(wrapped)
        if flatten_obs:
            wrapped = FlattenObservationWrapper(wrapped)
        for wrapper_cls in extra_wrappers:
            wrapped = wrapper_cls(wrapped)
        super().__init__(wrapped)
        self._base_env = env

    def reset(self, *args: Any, **kwargs: Any):
        return self.env.reset(*args, **kwargs)

    def step(self, actions: Dict[str, Any]):
        return self.env.step(actions)

    # TODO: expose MARLlib env_config (observation/action spaces, metadata) here.
