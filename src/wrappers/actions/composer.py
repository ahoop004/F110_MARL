"""ActionComposer — chains ActionComponents into a processing pipeline."""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from wrappers.actions.base import ActionComponent
from wrappers.actions.denormalize import DenormalizeComponent
from wrappers.actions.constraints import PreventReverseComponent


class ActionComposer:
    """Applies a sequence of ActionComponents to transform a normalized action.

    Built from the action_constraints block in the scenario agent config and
    the physical action bounds from the environment.
    """

    def __init__(self, components: List[ActionComponent]) -> None:
        self._components = components

    def process(self, action: np.ndarray) -> np.ndarray:
        result = np.asarray(action, dtype=np.float32).copy()
        for component in self._components:
            result = component.process(result)
        return result

    @classmethod
    def from_config(
        cls,
        action_low: np.ndarray,
        action_high: np.ndarray,
        constraints: Dict,
    ) -> "ActionComposer":
        """Build from physical action bounds and an action_constraints dict.

        Args:
            action_low:   Physical lower bounds (from env action space).
            action_high:  Physical upper bounds (from env action space).
            constraints:  agent_cfg.get("action_constraints", {})
        """
        components: List[ActionComponent] = [
            DenormalizeComponent(action_low, action_high),
        ]
        if constraints.get("prevent_reverse", False):
            speed_index = int(constraints.get("speed_index", 1))
            components.append(PreventReverseComponent(speed_index))
        return cls(components)
