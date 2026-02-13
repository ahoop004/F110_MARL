"""Reusable adaptive scalar controller with optional auto-regression.

The controller updates a scalar value using rolling success-rate feedback.
It supports:
- advance updates when success is high
- optional regression updates when success drops
- update windows, patience, and cooldown controls
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Optional

import numpy as np


class AdaptiveScalarController:
    """Adaptive scalar controller driven by rolling success outcomes."""

    def __init__(
        self,
        *,
        initial_value: float,
        value_min: float,
        value_max: float,
        window_size: int,
        update_every: int,
        advance_sr_low: float,
        advance_sr_high: float,
        advance_step_min: float,
        advance_step_max: float,
        regression_enabled: bool = False,
        regression_sr_trigger: Optional[float] = None,
        regression_sr_floor: Optional[float] = None,
        regression_step_min: Optional[float] = None,
        regression_step_max: Optional[float] = None,
        regression_patience_updates: int = 1,
        regression_cooldown_updates: int = 0,
        progress_start: Optional[float] = None,
        progress_end: Optional[float] = None,
    ):
        self.value_min = float(min(value_min, value_max))
        self.value_max = float(max(value_min, value_max))
        self.value = float(np.clip(initial_value, self.value_min, self.value_max))

        self.window_size = max(1, int(window_size))
        self.update_every = max(1, int(update_every))
        self.history: Deque[bool] = deque(maxlen=self.window_size)
        self.completed_events = 0

        self.advance_sr_low = float(advance_sr_low)
        self.advance_sr_high = float(advance_sr_high)
        self.advance_step_min = float(advance_step_min)
        self.advance_step_max = float(advance_step_max)
        if abs(self.advance_step_max) < abs(self.advance_step_min):
            self.advance_step_min, self.advance_step_max = (
                self.advance_step_max,
                self.advance_step_min,
            )

        self.regression_enabled = bool(regression_enabled)
        if regression_sr_trigger is None:
            regression_sr_trigger = max(0.0, self.advance_sr_low - 0.20)
        if regression_sr_floor is None:
            regression_sr_floor = max(0.0, float(regression_sr_trigger) - 0.20)
        self.regression_sr_trigger = float(regression_sr_trigger)
        self.regression_sr_floor = float(regression_sr_floor)
        self.regression_patience_updates = max(1, int(regression_patience_updates))
        self.regression_cooldown_updates = max(0, int(regression_cooldown_updates))
        self.regression_low_streak = 0
        self.regression_cooldown_left = 0

        if regression_step_min is None:
            regression_step_min = abs(self.advance_step_min)
        if regression_step_max is None:
            regression_step_max = abs(self.advance_step_max)
        self.regression_step_min = float(regression_step_min)
        self.regression_step_max = float(regression_step_max)
        if abs(self.regression_step_max) < abs(self.regression_step_min):
            self.regression_step_min, self.regression_step_max = (
                self.regression_step_max,
                self.regression_step_min,
            )

        self.progress_start = progress_start
        self.progress_end = progress_end

        self.last_window_sr: Optional[float] = None
        self.last_delta = 0.0
        self.last_adjustment = "hold"

    @staticmethod
    def _blend(alpha: float, step_min: float, step_max: float) -> float:
        return float(step_min + alpha * (step_max - step_min))

    def progress(self) -> Optional[float]:
        """Return normalized progress in [0, 1] when progress bounds are configured."""
        if self.progress_start is None or self.progress_end is None:
            return None
        start = float(self.progress_start)
        end = float(self.progress_end)
        denom = end - start
        if abs(denom) <= 1e-12:
            return 1.0
        pct = (self.value - start) / denom
        return float(np.clip(pct, 0.0, 1.0))

    def observe(self, success: bool) -> bool:
        """Record one outcome and update controller state when update window is ready.

        Returns True when a windowed update cycle was executed (change or hold), else False.
        """
        self.history.append(bool(success))
        self.completed_events += 1

        if self.completed_events % self.update_every != 0:
            return False
        if len(self.history) < self.window_size:
            return False

        sr = float(sum(self.history) / len(self.history))
        self.last_window_sr = sr

        if self.regression_enabled:
            if self.regression_cooldown_left > 0:
                self.regression_cooldown_left -= 1
            if sr < self.regression_sr_trigger:
                self.regression_low_streak += 1
            else:
                self.regression_low_streak = 0

        update_mode = "hold"
        delta = 0.0

        if sr >= self.advance_sr_low:
            denom = max(self.advance_sr_high - self.advance_sr_low, 1e-9)
            alpha = float(np.clip((sr - self.advance_sr_low) / denom, 0.0, 1.0))
            delta = self._blend(alpha, self.advance_step_min, self.advance_step_max)
            update_mode = "advance"
            if self.regression_enabled:
                self.regression_low_streak = 0
        elif self.regression_enabled:
            ready = (
                sr < self.regression_sr_trigger
                and self.regression_cooldown_left <= 0
                and self.regression_low_streak >= self.regression_patience_updates
            )
            if ready:
                denom = max(self.regression_sr_trigger - self.regression_sr_floor, 1e-9)
                beta = float(np.clip((self.regression_sr_trigger - sr) / denom, 0.0, 1.0))
                delta = self._blend(beta, self.regression_step_min, self.regression_step_max)
                update_mode = "regress"
                self.regression_cooldown_left = self.regression_cooldown_updates

        next_value = float(np.clip(self.value + delta, self.value_min, self.value_max))
        actual_delta = float(next_value - self.value)
        self.value = next_value
        self.last_delta = actual_delta
        if abs(actual_delta) <= 1e-12:
            self.last_adjustment = "hold"
        else:
            self.last_adjustment = update_mode

        return True

