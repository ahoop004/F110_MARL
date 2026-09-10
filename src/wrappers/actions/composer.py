"""Composable action transforms from normalized policy output to physical controls."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np


class ActionComponent(ABC):
    """A single transform applied to an action vector in the processing pipeline."""

    @abstractmethod
    def process(self, action: np.ndarray) -> np.ndarray:
        """Apply this transform and return the modified action."""


class DenormalizeComponent(ActionComponent):
    """Linearly maps each action dimension from [-1, 1] to [low, high]."""

    def __init__(self, action_low: np.ndarray, action_high: np.ndarray) -> None:
        self._low = np.asarray(action_low, dtype=np.float32)
        self._high = np.asarray(action_high, dtype=np.float32)
        self._scale = (self._high - self._low) / 2.0
        self._offset = (self._high + self._low) / 2.0

    def process(self, action: np.ndarray) -> np.ndarray:
        return (np.asarray(action, dtype=np.float32) * self._scale + self._offset)


class PreventReverseComponent(ActionComponent):
    """Clips the speed dimension to >= 0, preventing the car from reversing.

    Modifies *action* in-place (the composer guarantees that by this point
    the array is a freshly-allocated float32 buffer owned by the pipeline).
    """

    def __init__(self, speed_index: int = 1) -> None:
        self._idx = int(speed_index)

    def process(self, action: np.ndarray) -> np.ndarray:
        if len(action) > self._idx and action[self._idx] < 0.0:
            action[self._idx] = 0.0
        return action


class IntegratedSpeedComponent(ActionComponent):
    """Integrate a normalized acceleration into a bounded speed reference.

    Clamp the stored reference itself to prevent windup at either speed bound.
    Integration happens once per policy decision, before action repetition.
    """

    def __init__(self, low: float, high: float, contract: Dict) -> None:
        self._index = contract["speed_index"]
        self._low = max(0.0, low) if contract["prevent_reverse"] else low
        self._high = high
        if not self._low <= 0.0 <= self._high:
            raise ValueError("Integrated speed bounds must contain the reset reference 0 m/s.")
        self._dt = contract["decision_dt"]
        self._acceleration = contract["max_acceleration"]
        self._deceleration = contract["max_deceleration"]
        self.reset()

    def reset(self) -> None:
        self._speed = 0.0

    def process(self, action: np.ndarray) -> np.ndarray:
        command = float(np.clip(action[self._index], -1.0, 1.0))
        rate = command * (self._acceleration if command >= 0.0 else self._deceleration)
        self._speed = float(np.clip(self._speed + rate * self._dt, self._low, self._high))
        action[self._index] = self._speed
        return action


class ActionComposer:
    """Applies a sequence of ActionComponents to transform a normalized action.

    Built from the action_constraints block in the scenario agent config and
    the physical action bounds from the environment. Reverse prevention defaults
    to enabled for both direct and integrated speed commands.

    The first component in the pipeline (e.g. :class:`DenormalizeComponent`)
    always produces a freshly-allocated float32 array, so subsequent components
    can modify it in-place without needing extra copies.
    """

    def __init__(self, components: List[ActionComponent]) -> None:
        self._components = components

    def reset(self) -> None:
        for component in self._components:
            reset = getattr(component, "reset", None)
            if reset is not None:
                reset()

    @staticmethod
    def contract_from_config(constraints: Dict, decision_dt: float | None = None) -> Dict:
        mode = constraints.get("speed_control", "direct")
        if mode == "direct":
            return {"speed_control": "direct"}
        if mode != "acceleration":
            raise ValueError("speed_control must be 'direct' or 'acceleration'.")
        values = {
            "decision_dt": decision_dt,
            "max_acceleration": constraints.get("max_acceleration"),
            "max_deceleration": constraints.get("max_deceleration"),
        }
        for name, value in values.items():
            if value is None or not np.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"Acceleration speed control requires positive finite {name}.")
        return {
            "speed_control": mode,
            **{name: float(value) for name, value in values.items()},
            "speed_index": int(constraints.get("speed_index", 1)),
            "prevent_reverse": bool(constraints.get("prevent_reverse", True)),
        }

    def process(self, action: np.ndarray) -> np.ndarray:
        """Transform *action* through all components and return the result.

        The first component always allocates a new array (so the caller's
        buffer is never mutated).  Subsequent components may modify in-place.
        """
        result = np.asarray(action, dtype=np.float32)
        for component in self._components:
            result = component.process(result)
        return result

    @classmethod
    def from_config(
        cls,
        action_low: np.ndarray,
        action_high: np.ndarray,
        constraints: Dict,
        *,
        decision_dt: float | None = None,
    ) -> "ActionComposer":
        """Build from physical action bounds and an action_constraints dict.

        Args:
            action_low:   Physical lower bounds (from env action space).
            action_high:  Physical upper bounds (from env action space).
            constraints:  agent_cfg.get("action_constraints", {})
        """
        contract = cls.contract_from_config(constraints, decision_dt)
        low = np.asarray(action_low, dtype=np.float32).copy()
        high = np.asarray(action_high, dtype=np.float32).copy()
        integrated = None
        if contract["speed_control"] == "acceleration":
            index = contract["speed_index"]
            if not 0 <= index < len(low):
                raise ValueError("Acceleration speed_index is outside the action vector.")
            integrated = IntegratedSpeedComponent(float(low[index]), float(high[index]), contract)
            # Denormalize steering normally; retain the speed channel in [-1, 1]
            # until it is scaled to acceleration and integrated below.
            low[index], high[index] = -1.0, 1.0
        components: List[ActionComponent] = [DenormalizeComponent(low, high)]
        if integrated is not None:
            components.append(integrated)
        if constraints.get("prevent_reverse", True):
            speed_index = int(constraints.get("speed_index", 1))
            components.append(PreventReverseComponent(speed_index))
        return cls(components)
