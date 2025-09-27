from __future__ import annotations

from typing import Any, Dict

import numpy as np


class GapFollowPolicy:
    """Simple reactive follow-the-gap controller for slow, broad sweeps."""

    def __init__(self, **config: Any) -> None:
        self._max_range = float(config.pop("max_range", 30.0))
        self._clip_range = float(config.pop("clip_range", 3.0))
        self._smooth_window = max(int(config.pop("smooth_window", 5)), 1)
        self._bubble_radius = max(int(config.pop("bubble_radius", 35)), 0)
        self._speed = float(config.pop("speed", 0.1))
        self._field_of_view = float(config.pop("field_of_view", np.deg2rad(270.0)))
        self._steering_gain = float(config.pop("steering_gain", 1.0))
        self._steer_smooth = float(np.clip(config.pop("steer_smoothing", 0.6), 0.0, 1.0))
        self._speed_smooth = float(np.clip(config.pop("speed_smoothing", 0.9), 0.0, 1.0))
        self._extra_config: Dict[str, Any] = dict(config)

        self._last_action = np.array([0.0, self._speed], dtype=np.float32)

    def reset(self) -> None:
        self._last_action = np.array([0.0, self._speed], dtype=np.float32)

    def compute_action(self, observation: Dict[str, Any], action_space) -> np.ndarray:
        scan_source = observation.get("lidar") or observation.get("scans")
        scan = np.asarray(scan_source if scan_source is not None else (), dtype=np.float32)
        if scan.size == 0:
            return self._sample_fallback(action_space)

        ranges = self._preprocess(scan)

        closest = int(np.argmin(ranges)) if ranges.size else 0
        if ranges.size:
            min_idx = max(closest - self._bubble_radius, 0)
            max_idx = min(closest + self._bubble_radius + 1, ranges.size)
            ranges[min_idx:max_idx] = 0.0

        gap_start, gap_end = self._find_max_gap(ranges)
        best_idx = self._find_best_point(gap_start, gap_end, ranges)

        angle = self._index_to_angle(best_idx, ranges.size)
        steer = self._compute_steering(angle, action_space)
        speed = self._compute_speed(action_space)

        steer = self._smooth(steer, index=0)
        speed = self._smooth(speed, index=1)

        action = np.array([steer, speed], dtype=np.float32)
        self._last_action = action
        return action

    def _preprocess(self, scan: np.ndarray) -> np.ndarray:
        proc = np.nan_to_num(scan, nan=self._max_range, posinf=self._max_range, neginf=0.0)
        proc = np.clip(proc, 0.0, self._clip_range)
        if self._smooth_window > 1 and proc.size >= self._smooth_window:
            kernel = np.ones(self._smooth_window, dtype=np.float32) / float(self._smooth_window)
            proc = np.convolve(proc, kernel, mode="same")
        return proc

    def _find_max_gap(self, ranges: np.ndarray) -> tuple[int, int]:
        if ranges.size == 0:
            return 0, 0
        mask = ranges > 0.05
        if not np.any(mask):
            return 0, ranges.size
        gaps = []
        start = None
        for idx, val in enumerate(mask):
            if val and start is None:
                start = idx
            elif not val and start is not None:
                gaps.append((start, idx))
                start = None
        if start is not None:
            gaps.append((start, mask.size))
        return max(gaps, key=lambda g: g[1] - g[0]) if gaps else (0, ranges.size)

    def _find_best_point(self, start: int, end: int, ranges: np.ndarray) -> int:
        if end <= start:
            return start
        segment = ranges[start:end]
        return int(np.argmax(segment)) + start if segment.size else start

    def _index_to_angle(self, index: int, size: int) -> float:
        if size <= 1:
            return 0.0
        norm = (index / float(size - 1)) - 0.5
        return norm * self._field_of_view

    def _compute_steering(self, angle: float, action_space) -> float:
        steer = self._steering_gain * angle
        if hasattr(action_space, "low") and hasattr(action_space, "high"):
            steer_low = float(action_space.low[0])
            steer_high = float(action_space.high[0])
            steer = float(np.clip(steer, steer_low, steer_high))
        return steer

    def _compute_speed(self, action_space) -> float:
        speed = float(self._speed)
        if hasattr(action_space, "low") and hasattr(action_space, "high"):
            speed_low = float(action_space.low[1])
            speed_high = float(action_space.high[1])
            speed = float(np.clip(speed, speed_low, speed_high))
        return speed

    def _smooth(self, value: float, index: int) -> float:
        alpha = self._steer_smooth if index == 0 else self._speed_smooth
        return float(alpha * value + (1.0 - alpha) * self._last_action[index])

    def _sample_fallback(self, action_space) -> np.ndarray:
        if hasattr(action_space, "sample"):
            return action_space.sample()
        return np.zeros(2, dtype=np.float32)
