from typing import Any, Dict, Optional

import numpy as np

class FollowTheGapPolicy:
    CONFIG_DEFAULTS = {
        "max_distance": 30.0,
        "window_size": 4,
        "bubble_radius": 2,
        "max_steer": 0.32,
        "min_speed": 2.0,
        "max_speed": 20.0,
        "steering_gain": 0.6,
        "fov": np.deg2rad(270),
        "normalized": False,
        "steer_smooth": 0.4,
        "mode": "lidar",
        "center_bias_gain": 0.0,
        "steering_speed_scale": 1.0,
        "inside_bias_gain": 0.0,
        "crawl_steer_ratio": 0.6,
    }

    def __init__(self,
                 max_distance=30.0,   # actual sensor range (m)
                 window_size=4,
                 bubble_radius=2,
                 max_steer=0.32,
                 min_speed=2.0,
                 max_speed=20.0,
                 steering_gain=0.6,
                 fov=np.deg2rad(270),
                 normalized=False,
                 steer_smooth=0.4,   # heavier smoothing slows steering corrections
                 mode="lidar",
                 center_bias_gain=0.0,
                 steering_speed_scale: float = 1.0,
                 inside_bias_gain: float = 0.0,
                 crawl_steer_ratio: float = 0.6):
        self.max_distance = max_distance
        self.window_size = window_size
        self.bubble_radius = bubble_radius
        self.max_steer = max_steer
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.steering_gain = steering_gain
        self.fov = fov
        self.normalized = normalized
        self.steer_smooth = steer_smooth
        self.mode = str(mode).strip().lower()
        self.center_bias_gain = float(center_bias_gain)
        self.steering_speed_scale = max(float(steering_speed_scale), 1e-3)
        self.inside_bias_gain = float(inside_bias_gain)
        self.crawl_steer_ratio = float(np.clip(crawl_steer_ratio, 0.1, 1.0))

        # keep track of last steering for smoothing
        self.last_steer = 0.0

    @staticmethod
    def _coerce_like(value, default):
        if value is None:
            return default
        if isinstance(default, bool):
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "on"}
            return bool(value)
        if isinstance(default, int) and not isinstance(default, bool):
            try:
                return int(value)
            except (TypeError, ValueError):
                try:
                    return int(float(value))
                except (TypeError, ValueError):
                    return default
        if isinstance(default, float):
            try:
                return float(value)
            except (TypeError, ValueError):
                return default
        if isinstance(default, (np.ndarray, list, tuple)):
            return type(default)(value)
        return value

    @classmethod
    def from_config(cls, params):
        if not params:
            return cls()

        def _coerce(value, default):
            return cls._coerce_like(value, default)

        kwargs = {}
        for key, default in cls.CONFIG_DEFAULTS.items():
            if key in params:
                kwargs[key] = _coerce(params[key], default)

        if "fov" not in kwargs:
            if "fov_rad" in params:
                kwargs["fov"] = float(params["fov_rad"])
            elif "fov_deg" in params:
                kwargs["fov"] = np.deg2rad(float(params["fov_deg"]))

        unexpected = set(params) - set(kwargs) - {"fov_deg", "fov_rad"}
        if unexpected:
            print(f"[FollowTheGapPolicy] Ignoring unsupported config keys: {sorted(unexpected)}")

        return cls(**kwargs)

    def apply_config(self, params):
        if not params:
            return
        for key, value in params.items():
            if not hasattr(self, key):
                continue
            default = self.CONFIG_DEFAULTS.get(key, getattr(self, key))
            coerced = self._coerce_like(value, default)
            setattr(self, key, coerced)

    def export_config(self, keys=None):
        if keys is None:
            keys = ("max_speed", "min_speed", "steering_gain", "bubble_radius", "steer_smooth")
        snapshot = {}
        for key in keys:
            if hasattr(self, key):
                snapshot[key] = getattr(self, key)
        return snapshot

    def preprocess_lidar(self, ranges, min_scan: Optional[float] = None):
        """Smooth LiDAR with moving average + clip to max_distance."""
        N = len(ranges)
        window = self._adaptive_window_size(min_scan)
        half = window // 2
        proc = []
        for i in range(N):
            start = max(0, i - half)
            end = min(N - 1, i + half)
            avg = np.mean(np.clip(ranges[start:end+1], 0, self.max_distance))
            proc.append(avg)
        return np.array(proc)

    def create_bubble(self, proc, min_scan: Optional[float] = None):
        """Zero out a bubble around the closest obstacle."""
        closest = np.argmin(proc)
        radius = self._adaptive_bubble_radius(min_scan)
        start = max(0, closest - radius)
        end = min(len(proc) - 1, closest + radius)
        proc[start:end+1] = 0
        return proc

    def find_max_gap(self, proc):
        """Find the largest contiguous nonzero gap."""
        gaps, start = [], None
        for i, v in enumerate(proc > 2.5):
            if v and start is None:
                start = i
            elif not v and start is not None:
                gaps.append((start, i-1))
                start = None
        if start is not None:
            gaps.append((start, len(proc)-1))
        if not gaps:
            return 0, len(proc)-1
        return max(gaps, key=lambda g: g[1]-g[0])

    def best_point_midgap(self, gap):
        """Return the midpoint of the widest gap."""
        return (gap[0] + gap[1]) // 2

    def get_action(self, action_space, obs: dict):
        return self._get_action_lidar(action_space, obs)

    def _get_action_lidar(self, action_space, obs: dict):
        scan = np.array(obs["scans"])
        if self.normalized:
            scan = scan * self.max_distance

        N = len(scan)
        center_idx = N // 2

        min_scan = float(np.min(scan))

        # 1. Base gap-following
        proc = self.preprocess_lidar(scan, min_scan=min_scan)
        proc = self.create_bubble(proc, min_scan=min_scan)
        gap = self.find_max_gap(proc)
        best = self.best_point_midgap(gap)
        offset = (best - center_idx) / center_idx
        steering = offset * self.steering_gain * self.max_steer

        steering = self._apply_lookahead_guard(steering, scan, min_scan, center_idx, obs)
        steering = self._apply_kinematic_preview(steering, scan, obs)

        # 2. Centering bias to avoid drifting in symmetric gaps
        left_min = np.min(scan[:center_idx]) if center_idx > 0 else np.inf
        right_min = np.min(scan[center_idx:]) if center_idx < N else np.inf
        if center_idx > 0 and center_idx < N:
            left_mean = float(np.mean(scan[:center_idx]))
            right_mean = float(np.mean(scan[center_idx:]))
            denom = max(self.max_distance, 1e-3)
            if self.center_bias_gain != 0.0 and min_scan > 3.0:
                bias = np.clip((right_mean - left_mean) / denom, -1.0, 1.0)
                steering += self.center_bias_gain * bias * self.max_steer
            if self.inside_bias_gain != 0.0 and min_scan <= 4.0:
                hug_bias = np.clip((left_mean - right_mean) / denom, -1.0, 1.0)
                steering += self.inside_bias_gain * hug_bias * self.max_steer

        # 3. Sector-based danger weighting
        # Gentler panic scaling keeps the policy committed to the current heading longer
        panic_factor = 1.0
        if min_scan < 4.0:
            panic_factor = 1.15
        if min_scan < 2.0:
            panic_factor = 1.4

        steering *= panic_factor

        # Kick away from closest side
        if left_min < right_min:
            steering += 0.18 * self.max_steer
        elif right_min < left_min:
            steering -= 0.18 * self.max_steer

        # Override when extremely close: pure evasive
        if min_scan < 1.0:
            steering = np.clip(steering, -0.5 * self.max_steer, 0.5 * self.max_steer)

        # Clip and smooth steering
        velocity_vec = obs.get("velocity")
        if velocity_vec is None:
            speed = float(obs.get("speed", 0.0))
        else:
            arr = np.asarray(velocity_vec, dtype=np.float32).reshape(-1)
            speed = float(arr[0]) if arr.size else 0.0
        speed_cap = self._steering_cap_from_speed(abs(speed))
        steering = np.clip(steering, -speed_cap, speed_cap)
        steering = self.steer_smooth * self.last_steer + (1 - self.steer_smooth) * steering
        self.last_steer = steering

        # 4. Speed schedule (more conservative)
        speed = self._compute_speed(scan, center_idx, min_scan, steering, obs)
        if abs(steering) > self.crawl_steer_ratio * self.max_steer:
            speed = min(speed, max(self.min_speed, 0.4 * self.max_speed))

        action = np.array([steering, speed], dtype=np.float32)
        if action_space is not None:
            action = np.clip(action, action_space.low, action_space.high)
        return action

    def _steering_cap_from_speed(self, speed: float) -> float:
        if speed <= 0.0:
            return self.max_steer
        limit = self.steering_speed_scale / (speed + 1e-6)
        return float(np.clip(limit, 0.1 * self.max_steer, self.max_steer))

    def _adaptive_window_size(self, min_scan: Optional[float]) -> int:
        base = max(int(round(self.window_size)), 1)
        if min_scan is None or not np.isfinite(min_scan) or self.max_distance <= 0:
            size = base
        else:
            ratio = np.clip(min_scan / self.max_distance, 0.2, 1.0)
            size = max(1, int(round(base * ratio)))
        if size % 2 == 0:
            size = max(1, size - 1)
        return size

    def _adaptive_bubble_radius(self, min_scan: Optional[float]) -> int:
        base = max(int(round(self.bubble_radius)), 0)
        if base == 0 or min_scan is None or not np.isfinite(min_scan) or self.max_distance <= 0:
            return base
        ratio = np.clip(min_scan / self.max_distance, 0.2, 1.0)
        radius = max(1, int(round(base * ratio)))
        return radius

    def _apply_kinematic_preview(self, steering: float, scan: np.ndarray, obs: Dict[str, Any]) -> float:
        preview_cfg = (
            float(getattr(self, "preview_horizon", 0.0)),
            int(getattr(self, "preview_samples", 0)),
        )
        if preview_cfg[0] <= 0.0 or preview_cfg[1] <= 0:
            return steering
        scan_res = scan * self.max_distance if self.normalized else scan
        proposals = np.linspace(-self.max_steer, self.max_steer, preview_cfg[1])
        velocity_vec = obs.get("velocity")
        if velocity_vec is None:
            speed = float(obs.get("speed", 0.0))
        else:
            arr = np.asarray(velocity_vec, dtype=np.float32).reshape(-1)
            speed = float(arr[0]) if arr.size else 0.0
        best = steering
        best_margin = -np.inf
        for candidate in proposals:
            margin = self._simulate_preview(candidate, speed, scan_res, preview_cfg[0])
            if margin > best_margin:
                best_margin = margin
                best = candidate
        return best

    def _simulate_preview(self, steering: float, speed: float, scan: np.ndarray, horizon: float) -> float:
        dt = 0.1
        steps = max(int(horizon / dt), 1)
        pos = np.array([0.0, 0.0], dtype=np.float32)
        heading = 0.0
        min_margin = np.inf
        for _ in range(steps):
            heading += (speed / max(speed, 1e-3)) * steering * dt
            pos[0] += speed * dt * np.cos(heading)
            pos[1] += speed * dt * np.sin(heading)
            beam_idx = int(len(scan) / 2 + heading / self.fov * len(scan))
            beam_idx = int(np.clip(beam_idx, 0, len(scan) - 1))
            min_margin = min(min_margin, float(scan[beam_idx]))
        return min_margin

    def _compute_speed(
        self,
        scan: np.ndarray,
        center_idx: int,
        min_scan: float,
        steering: float,
        obs: Dict[str, Any],
    ) -> float:
        forward = float(scan[center_idx])
        lateral_window = max(center_idx // 4, 1)
        left_band = scan[max(0, center_idx - lateral_window):center_idx]
        right_band = scan[center_idx:center_idx + lateral_window]
        spread = float(np.std(np.concatenate([left_band, right_band]))) if left_band.size + right_band.size > 0 else 0.0

        speed = self.max_speed
        lookahead_idx = int(center_idx + np.sign(steering) * max(abs(steering) / max(self.max_steer, 1e-3), 0.1) * center_idx)
        lookahead_idx = int(np.clip(lookahead_idx, 0, len(scan) - 1))
        lookahead_range = float(scan[lookahead_idx])

        if min_scan < 1.5 or lookahead_range < 2.0:
            speed = max(self.min_speed, 0.5 * self.max_speed)
        elif forward < 3.0 or spread > 2.0 or lookahead_range < 3.0:
            speed = max(self.min_speed, 0.6 * self.max_speed)
        elif forward < 6.0 or spread > 1.0:
            speed = 0.8 * self.max_speed

        return float(np.clip(speed, self.min_speed, self.max_speed))

    def _apply_lookahead_guard(
        self,
        steering: float,
        scan: np.ndarray,
        min_scan: float,
        center_idx: int,
        obs: Dict[str, Any],
    ) -> float:
        if min_scan < 1.5 or not np.isfinite(min_scan):
            return steering
        velocity_vec = obs.get("velocity")
        if velocity_vec is None:
            speed = float(obs.get("speed", 0.0))
        else:
            arr = np.asarray(velocity_vec, dtype=np.float32).reshape(-1)
            speed = float(arr[0]) if arr.size else 0.0
        dt = 0.15
        curvature = steering / max(self.max_steer, 1e-3)
        yaw_change = curvature * dt
        lookahead_idx = int(center_idx + yaw_change * center_idx)
        lookahead_idx = int(np.clip(lookahead_idx, 0, len(scan) - 1))
        lookahead_range = float(scan[lookahead_idx])
        if lookahead_range < max(1.2, 0.4 * min_scan):
            steering *= 0.5
        return steering
