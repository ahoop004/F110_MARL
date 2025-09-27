from __future__ import annotations

from typing import Any


try:  # pragma: no cover - optional dependency
    from pettingzoo.utils.wrappers import BaseParallelWrapper
except Exception:  # PettingZoo not available during scaffolding
    class BaseParallelWrapper:  # type: ignore
        """Minimal stand-in when PettingZoo wrappers are unavailable."""

        def __init__(self, env: Any) -> None:
            self.env = env

        def reset(self, *args: Any, **kwargs: Any):
            return self.env.reset(*args, **kwargs)

        def step(self, actions: Any):
            return self.env.step(actions)

        def __getattr__(self, item: str) -> Any:
            return getattr(self.env, item)

