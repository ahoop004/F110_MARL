"""Simple centerline-following heuristic for lap completion."""
from __future__ import annotations

from typing import Optional

import numpy as np

from f110x.utils.centerline import project_to_centerline


class CenterlinePursuitPolicy:
    """Follow the track centerline using a lookahead pure-pursuit heuristic."""

    def __init__(
        self,
        *,
        centerline: Optional[np.ndarray],
        lookahead_distance: float = 3.0,
        base_speed: float = 3.5,
        min_speed: float = 0.5,
        max_speed: float = 6.0,
        heading_gain: float = 1.4,
        lateral_gain: float = 0.4,
        turn_slowdown: float = 2.0,
        lookahead_shrink: float = 0.0,
        lateral_speed_gain: float = 0.0,
        obstacle_turn_gain: float = 0.0,
        obstacle_front_fraction: float = 0.3,
        obstacle_speed_gain: float = 0.0,
        obstacle_distance_threshold: float = 0.0,
        lidar_max_range: float = 0.0,
        lidar_normalized: bool = True,
    ) -> None:
        self.centerline = None if centerline is None else np.asarray(centerline, dtype=np.float32)
        if self.centerline is not None and self.centerline.ndim != 2:
            raise ValueError("centerline must be a 2D array")

        self.lookahead_distance = float(lookahead_distance)
        self.base_speed = float(base_speed)
        self.min_speed = float(min_speed)
        self.max_speed = float(max_speed)
        self.heading_gain = float(heading_gain)
        self.lateral_gain = float(lateral_gain)
        self.turn_slowdown = float(turn_slowdown)
        self.lookahead_shrink = max(float(lookahead_shrink), 0.0)
        self.lateral_speed_gain = max(float(lateral_speed_gain), 0.0)
        self.obstacle_turn_gain = max(float(obstacle_turn_gain), 0.0)
        self.obstacle_front_fraction = float(np.clip(obstacle_front_fraction, 0.0, 1.0))
        self.obstacle_speed_gain = max(float(obstacle_speed_gain), 0.0)
        self.obstacle_distance_threshold = max(float(obstacle_distance_threshold), 0.0)
        self.lidar_max_range = float(lidar_max_range)
        self.lidar_normalized = bool(lidar_normalized)

        if self.centerline is not None and self.centerline.shape[0] >= 2:
            diffs = np.diff(self.centerline[:, :2], axis=0)
            seg_lengths = np.linalg.norm(diffs, axis=1)
            self._avg_spacing = float(np.mean(seg_lengths)) if seg_lengths.size else 1.0
            if self._avg_spacing <= 1e-5:
                self._avg_spacing = 1.0
        else:
            self._avg_spacing = 1.0
        self._last_index: Optional[int] = None

    def reset_hidden_state(self) -> None:
        """Clear cached projection state between episodes."""

        self._last_index = None

    def get_action(self, action_space, obs):
        if self.centerline is None or obs is None:
            return np.zeros(getattr(action_space, "shape", (2,)), dtype=np.float32)

        pose = obs.get("pose")
        if pose is None or len(pose) < 3:
            return np.zeros(getattr(action_space, "shape", (2,)), dtype=np.float32)

        position = np.asarray(pose[:2], dtype=np.float32)
        heading = float(pose[2])

        try:
            projection = project_to_centerline(
                self.centerline,
                position,
                heading,
                last_index=self._last_index,
            )
        except ValueError:
            self._last_index = None
            return np.zeros(getattr(action_space, "shape", (2,)), dtype=np.float32)

        self._last_index = projection.index

        spacing = max(self._avg_spacing, 1e-3)
        lookahead_distance = self.lookahead_distance
        if self.lookahead_shrink > 0.0:
            reference_error = abs(getattr(projection, "heading_error", 0.0))
            lookahead_distance = max(
                0.5,
                lookahead_distance / (1.0 + self.lookahead_shrink * reference_error),
            )

        lookahead_steps = max(1, int(round(lookahead_distance / spacing)))
        target_index = (projection.index + lookahead_steps) % self.centerline.shape[0]
        target_point = self.centerline[target_index, :2]

        to_target = target_point - position
        desired_heading = float(np.arctan2(to_target[1], to_target[0]))
        heading_error = float(np.arctan2(np.sin(desired_heading - heading), np.cos(desired_heading - heading)))

        # Follow the target while nudging back toward the centerline; lateral_error is positive to the left
        steer_cmd = self.heading_gain * heading_error - self.lateral_gain * projection.lateral_error

        speed_cmd_penalty = 0.0

        lidar_scan = obs.get("lidar")
        if lidar_scan is None:
            lidar_scan = obs.get("scan")
        if lidar_scan is None:
            lidar_scan = obs.get("scans")

        if self.obstacle_turn_gain > 0.0 and lidar_scan is not None:
            scan = np.asarray(lidar_scan, dtype=np.float32).reshape(-1)
            if scan.size > 0:
                if self.lidar_normalized and self.lidar_max_range > 0.0:
                    scan = scan * self.lidar_max_range

                mid = scan.size // 2
                if self.obstacle_front_fraction > 0.0:
                    half_width = max(1, int(round(scan.size * self.obstacle_front_fraction * 0.5)))
                    start = max(0, mid - half_width)
                    end = min(scan.size, mid + half_width)
                else:
                    start = 0
                    end = scan.size

                front = scan[start:end]
                if front.size == 0:
                    front = scan
                front_mid = front.size // 2
                left = front[:front_mid] if front_mid > 0 else front
                right = front[front_mid:] if front_mid > 0 else front

                left_min = float(np.min(left)) if left.size else float(np.min(front))
                right_min = float(np.min(right)) if right.size else float(np.min(front))
                inv_left = 1.0 / max(left_min, 1e-3)
                inv_right = 1.0 / max(right_min, 1e-3)
                steer_cmd -= self.obstacle_turn_gain * (inv_left - inv_right)

                if self.obstacle_speed_gain > 0.0 and self.obstacle_distance_threshold > 0.0:
                    min_front = float(np.min(front)) if front.size else float(np.min(scan))
                    shortfall = max(0.0, self.obstacle_distance_threshold - min_front)
                    if shortfall > 0.0:
                        speed_cmd_penalty = self.obstacle_speed_gain * shortfall
                    else:
                        speed_cmd_penalty = 0.0
                else:
                    speed_cmd_penalty = 0.0
            else:
                speed_cmd_penalty = 0.0
        else:
            speed_cmd_penalty = 0.0

        if action_space.shape and action_space.shape[0] >= 1:
            steer_low = float(action_space.low[0])
            steer_high = float(action_space.high[0])
            steer_cmd = float(np.clip(steer_cmd, steer_low, steer_high))

        turn_mag = abs(heading_error)
        speed_cmd = self.base_speed - self.turn_slowdown * turn_mag
        if self.lateral_speed_gain > 0.0:
            speed_cmd -= self.lateral_speed_gain * abs(projection.lateral_error)
        if speed_cmd_penalty > 0.0:
            speed_cmd -= speed_cmd_penalty
        speed_cmd = float(np.clip(speed_cmd, self.min_speed, self.max_speed))

        if action_space.shape and action_space.shape[0] >= 2:
            speed_low = float(action_space.low[1])
            speed_high = float(action_space.high[1])
            speed_cmd = float(np.clip(speed_cmd, speed_low, speed_high))

        if bool(obs.get("collision", False)):
            speed_cmd = min(speed_cmd, max(0.0, self.min_speed * 0.5))

        if action_space.shape and action_space.shape[0] >= 2:
            return np.array([steer_cmd, speed_cmd], dtype=np.float32)

        return np.array([steer_cmd], dtype=np.float32)


__all__ = ["CenterlinePursuitPolicy"]
