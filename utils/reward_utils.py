"""Utilities for decomposing reward shaping logic."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class ScalingParams:
    horizon: float | None = None
    clip: float | None = None
    weight: float | None = None
    decay: float | None = None
    smoothing: float | None = None


def apply_reward_scaling(
    total: float,
    components: Dict[str, float],
    scaling: ScalingParams,
) -> Tuple[float, Dict[str, float]]:
    """Scale and clip rewards while updating component bookkeeping."""

    adjusted = dict(components)
    if scaling.weight:
        weight = float(scaling.weight)
        total *= weight
        for key in list(adjusted.keys()):
            adjusted[key] *= weight
        adjusted["weight_factor"] = weight

    if scaling.decay:
        decay = float(scaling.decay)
        factor = exp(-decay)
        total *= factor
        for key in list(adjusted.keys()):
            adjusted[key] *= factor
        adjusted["decay_factor"] = factor

    if scaling.smoothing:
        smooth = float(scaling.smoothing)
        if smooth > 0.0:
            factor = 1.0 / (1.0 + smooth)
            total *= factor
            for key in list(adjusted.keys()):
                adjusted[key] *= factor
            adjusted["smoothing_factor"] = factor

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
