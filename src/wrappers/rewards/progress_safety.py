"""Safety penalties derived from centerline progress facts."""
from __future__ import annotations

from typing import Dict

from wrappers.rewards.base import RewardComponent


class ProgressSafetyComponent(RewardComponent):
    """Penalize wrong-way driving, reverse progress, and large deviation."""

    def __init__(self, config: dict) -> None:
        self.wrong_way_penalty = float(config.get("wrong_way_penalty", -2.0))
        self.reverse_progress_weight = float(config.get("reverse_progress_weight", 5.0))
        self.max_abs_d = float(config.get("max_abs_d", 1.5))
        self.offtrack_penalty = float(config.get("offtrack_penalty", -1.0))

    def compute(self, step_info: dict) -> Dict[str, float]:
        info = step_info.get("info") or {}
        centerline = info.get("centerline", {}) if isinstance(info, dict) else {}
        if not isinstance(centerline, dict):
            return {}

        rewards: Dict[str, float] = {}
        if bool(centerline.get("wrong_way", False)):
            rewards["progress_safety/wrong_way"] = self.wrong_way_penalty

        progress_delta = _float_or_none(centerline.get("progress_delta"))
        if progress_delta is not None and progress_delta < 0.0:
            rewards["progress_safety/reverse_progress"] = (
                -self.reverse_progress_weight * abs(progress_delta)
            )

        lateral_d = _float_or_none(centerline.get("d"))
        if lateral_d is not None and abs(lateral_d) > self.max_abs_d:
            rewards["progress_safety/offtrack"] = self.offtrack_penalty

        return rewards


def _float_or_none(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
