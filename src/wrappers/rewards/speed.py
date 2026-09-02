"""Rewards and penalties derived from vehicle longitudinal speed."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from wrappers.rewards.base import RewardComponent


class SpeedRewardComponent(RewardComponent):
    """Small per-step bonus proportional to forward speed."""

    def __init__(self, config: dict) -> None:
        self.weight = float(config.get("weight", 0.01))
        self.speed_index = int(config.get("speed_index", 1))

    def compute(self, step_info: dict) -> Dict[str, float]:
        action = np.asarray(step_info.get("action", [0.0, 0.0]), dtype=np.float32).ravel()
        speed = float(action[self.speed_index]) if len(action) > self.speed_index else 0.0
        return {"speed/bonus": self.weight * max(speed, 0.0)}


class ReverseVelocityPenaltyComponent(RewardComponent):
    """Penalize measured backward vehicle motion, not a reverse command.

    The simulator reports longitudinal velocity in the vehicle body frame, so
    a negative first velocity element means that the chassis is physically
    moving backward regardless of its orientation on the track. Environment
    step facts are authoritative; raw observations are only a compatibility
    fallback for isolated component use.
    """

    def __init__(self, config: dict) -> None:
        self.weight = float(config.get("weight", 1.0))
        self.deadband = max(float(config.get("deadband", 0.05)), 0.0)
        max_reverse_speed = config.get("max_reverse_speed")
        self.max_reverse_speed: Optional[float] = (
            max(float(max_reverse_speed), 0.0)
            if max_reverse_speed is not None
            else None
        )
        if self.weight < 0.0:
            raise ValueError("reverse_velocity_penalty.weight must be >= 0")

    def compute(self, step_info: dict) -> Dict[str, float]:
        longitudinal_speed = self._longitudinal_speed(step_info)
        if longitudinal_speed is None:
            return {}

        reverse_speed = max(-longitudinal_speed - self.deadband, 0.0)
        if reverse_speed <= 0.0:
            return {}
        if self.max_reverse_speed is not None:
            reverse_speed = min(reverse_speed, self.max_reverse_speed)
        return {"reverse_velocity/penalty": -self.weight * reverse_speed}

    @staticmethod
    def _longitudinal_speed(step_info: dict) -> Optional[float]:
        facts = step_info.get("last_step_facts")
        agent_id = step_info.get("agent_id")
        agent_states = getattr(facts, "agent_states", None)
        if agent_id is not None and agent_states is not None:
            state = agent_states.get(str(agent_id))
            velocity = getattr(state, "velocity", None)
            value = _first_float(velocity)
            if value is not None:
                return value

        for key in ("next_obs", "obs"):
            raw_obs = step_info.get(key)
            if isinstance(raw_obs, dict):
                value = _first_float(raw_obs.get("velocity"))
                if value is not None:
                    return value
        return None


def _first_float(value: object) -> Optional[float]:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32).ravel()
    if array.size == 0:
        return None
    result = float(array[0])
    return result if np.isfinite(result) else None
