"""Utilities for decomposing reward shaping logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class ScalingParams:
    horizon: float | None
    clip: float | None


def apply_reward_scaling(
    total: float,
    components: Dict[str, float],
    scaling: ScalingParams,
) -> Tuple[float, Dict[str, float]]:
    """Scale and clip rewards while updating component bookkeeping."""

    adjusted = dict(components)
    if scaling.horizon:
        scale = 1.0 / scaling.horizon
        total *= scale
        for key in list(adjusted.keys()):
            adjusted[key] *= scale
        adjusted["scale_factor"] = scale

    if scaling.clip:
        clipped = float(np.clip(total, -scaling.clip, scaling.clip))
        if clipped != total:
            adjusted["clip_delta"] = clipped - total
            total = clipped

    adjusted["total_return"] = total
    return total, adjusted
