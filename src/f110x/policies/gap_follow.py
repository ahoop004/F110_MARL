from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


class GapFollowPolicy:
    """Scripted gap-follow controller compatible with MARL policy hooks."""

    def __init__(self, **config: Any) -> None:
        self._max_range = float(config.pop("max_range", 30.0))
        self._max_speed = float(config.pop("max_speed", 6.0))
        self._min_speed = float(config.pop("min_speed", 1.0))
        self._safety_radius = float(config.pop("safety_radius", 1.0))
        self._bubble_radius = max(int(config.pop("bubble_radius", 8)), 0)
        self._smooth_window = max(int(config.pop("smooth_window", 5)), 1)
        fov = config.pop("field_of_view", np.deg2rad(270.0))
        self._field_of_view = float(fov)
        self._steering_gain = float(config.pop("steering_gain", 1.0))
        self._extra_config: Dict[str, Any] = dict(config)

    def reset(self) -> None:
        pass

    def compute_action(self, observation: Dict[str, Any], action_space) -> np.ndarray:
        scan_source = observation.get("lidar")
        if scan_source is None:
            scan_source = observation.get("scans")
        scan = np.asarray(scan_source if scan_source is not None else (), dtype=np.float32)
        if scan.size == 0:
            return self._sample_fallback(action_space)

        ranges = np.nan_to_num(scan, nan=self._max_range, posinf=self._max_range, neginf=0.0)
        ranges = np.clip(ranges, 0.0, self._max_range)

        if self._smooth_window > 1 and ranges.size >= self._smooth_window:
            kernel = np.ones(self._smooth_window, dtype=np.float32) / float(self._smooth_window)
            ranges = np.convolve(ranges, kernel, mode="same")

        if ranges.size:
            closest_idx = int(np.argmin(ranges))
            left = max(closest_idx - self._bubble_radius, 0)
            right = min(closest_idx + self._bubble_radius + 1, ranges.size)
            ranges[left:right] = 0.0

        free_mask = ranges > self._safety_radius
        best_idx = int(np.argmax(ranges)) if ranges.size else 0
        best_len = 0
        best_range = 0.0
        run_start: Optional[int] = None

        for idx, is_free in enumerate(free_mask):
            if is_free and run_start is None:
                run_start = idx
            elif not is_free and run_start is not None:
                length = idx - run_start
                if length > 0:
                    segment = ranges[run_start:idx]
                    segment_idx = int(np.argmax(segment)) if segment.size else 0
                    segment_best = float(segment[segment_idx]) if segment.size else 0.0
                    if length > best_len or (length == best_len and segment_best > best_range):
                        best_len = length
                        best_range = segment_best
                        best_idx = run_start + segment_idx
                run_start = None

        if run_start is not None:
            length = free_mask.size - run_start
            if length > 0:
                segment = ranges[run_start:]
                if segment.size:
                    segment_idx = int(np.argmax(segment))
                    segment_best = float(segment[segment_idx])
                    if length > best_len or (length == best_len and segment_best > best_range):
                        best_idx = run_start + segment_idx

        angle = self._index_to_angle(best_idx, scan.size)
        steer = self._compute_steering(angle, action_space)
        speed = self._compute_speed(angle, action_space)

        return np.array([steer, speed], dtype=np.float32)

    def _index_to_angle(self, index: int, scan_size: int) -> float:
        if scan_size <= 1:
            return 0.0
        norm = (index / float(scan_size - 1)) - 0.5
        return norm * self._field_of_view

    def _compute_steering(self, angle: float, action_space) -> float:
        steer = self._steering_gain * angle
        if hasattr(action_space, "low") and hasattr(action_space, "high"):
            steer_low = float(action_space.low[0])
            steer_high = float(action_space.high[0])
            steer = float(np.clip(steer, steer_low, steer_high))
        return steer

    def _compute_speed(self, angle: float, action_space) -> float:
        span = max(self._max_speed - self._min_speed, 1e-3)
        denom = self._field_of_view / 2.0 if self._field_of_view else 1.0
        slow_factor = min(abs(angle) / denom, 1.0)
        target_speed = self._max_speed - slow_factor * span
        speed = target_speed
        if hasattr(action_space, "low") and hasattr(action_space, "high"):
            speed_low = float(action_space.low[1])
            speed_high = float(action_space.high[1])
            speed = float(np.clip(speed, speed_low, speed_high))
        return speed

    def _sample_fallback(self, action_space) -> np.ndarray:
        if hasattr(action_space, "sample"):
            return action_space.sample()
        return np.zeros(2, dtype=np.float32)
