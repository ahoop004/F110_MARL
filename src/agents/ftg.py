from typing import Any, Dict, Optional, Tuple
import json
import os

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
        "preview_horizon": 0.0,
        "preview_samples": 0,
        "dwa_samples": 0,
        "dwa_horizon": 0.5,
        "dwa_heading_weight": 0.1,
        "enhanced_gap_scoring": False,
        "gap_score_width_weight": 1.0,
        "gap_score_clearance_weight": 1.0,
        "gap_score_center_weight": 0.5,
        "gap_score_curvature_weight": 0.2,
        "u_shape_enabled": False,
        "u_shape_threshold": 0.5,
        "u_shape_crawl_speed": 0.3,
        "gap_min_range": 0.65,
        "target_mode": "farthest",          # "midgap" or "farthest"
        "use_disparity_extender": True,
        "disparity_threshold": 0.35,        # meters
        "vehicle_width": 0.225,             # meters (match your env vehicle_params.width)
        "safety_margin": 0.08,              # meters
        "no_cutback_enabled": True,
        "cutback_clearance": 0.9,           # meters
        "cutback_hold_steps": 8, 
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
                 gap_min_range: float = 0.65,
                target_mode: str = "farthest",
                use_disparity_extender: bool = True,
                disparity_threshold: float = 0.35,
                vehicle_width: float = 0.225,
                safety_margin: float = 0.08,
                no_cutback_enabled: bool = True,
                cutback_clearance: float = 0.9,
                cutback_hold_steps: int = 8,
                steering_speed_scale: float = 1.0,
                inside_bias_gain: float = 0.0,
                crawl_steer_ratio: float = 0.6,
                preview_horizon: float = 0.0,
                preview_samples: int = 0,
                dwa_samples: int = 0,
                dwa_horizon: float = 0.5,
                dwa_heading_weight: float = 0.1,
                enhanced_gap_scoring: bool = False,
                gap_score_width_weight: float = 1.0,
                gap_score_clearance_weight: float = 1.0,
                gap_score_center_weight: float = 0.5,
                gap_score_curvature_weight: float = 0.2,
                u_shape_enabled: bool = False,
                u_shape_threshold: float = 0.5,
                u_shape_crawl_speed: float = 0.3):
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
        self.preview_horizon = max(float(preview_horizon), 0.0)
        self.preview_samples = max(int(preview_samples), 0)
        self.dwa_samples = max(int(dwa_samples), 0)
        self.dwa_horizon = max(float(dwa_horizon), 0.0)
        self.dwa_heading_weight = float(dwa_heading_weight)
        self.enhanced_gap_scoring = bool(enhanced_gap_scoring)
        self.gap_score_width_weight = float(gap_score_width_weight)
        self.gap_score_clearance_weight = float(gap_score_clearance_weight)
        self.gap_score_center_weight = float(gap_score_center_weight)
        self.gap_score_curvature_weight = float(gap_score_curvature_weight)
        self.u_shape_enabled = bool(u_shape_enabled)
        self.u_shape_threshold = float(np.clip(u_shape_threshold, 0.1, 1.0))
        self.u_shape_crawl_speed = float(np.clip(u_shape_crawl_speed, 0.1, 1.0))
        self.gap_min_range = float(gap_min_range)
        self.target_mode = str(target_mode).strip().lower()
        self.use_disparity_extender = bool(use_disparity_extender)
        self.disparity_threshold = float(disparity_threshold)
        self.vehicle_width = float(vehicle_width)
        self.safety_margin = float(safety_margin)
        self.no_cutback_enabled = bool(no_cutback_enabled)
        self.cutback_clearance = float(cutback_clearance)
        self.cutback_hold_steps = int(cutback_hold_steps)
        self._cutback_ttl = 0
        self._cutback_side = None

        # Validate parameters
        if self.min_speed > self.max_speed:
            raise ValueError(
                f"min_speed ({self.min_speed}) must be <= max_speed ({self.max_speed})"
            )
        if self.bubble_radius < 0:
            raise ValueError(f"bubble_radius must be >= 0, got {self.bubble_radius}")
        if not 0 < self.fov <= 2 * np.pi:
            raise ValueError(
                f"fov must be in (0, 2π] radians, got {self.fov} "
                f"({np.rad2deg(self.fov)} degrees)"
            )
        if self.max_steer <= 0:
            raise ValueError(f"max_steer must be > 0, got {self.max_steer}")
        if self.max_distance <= 0:
            raise ValueError(f"max_distance must be > 0, got {self.max_distance}")
        if self.window_size <= 0:
            raise ValueError(f"window_size must be > 0, got {self.window_size}")
        if self.target_mode not in ("farthest", "center", "weighted"):
            raise ValueError(
                f"target_mode must be one of ('farthest', 'center', 'weighted'), "
                f"got '{self.target_mode}'"
            )
        if self.mode not in ("lidar", "waypoint", "hybrid"):
            raise ValueError(
                f"mode must be one of ('lidar', 'waypoint', 'hybrid'), "
                f"got '{self.mode}'"
            )
        if not 0.0 <= self.steering_gain <= 1.0:
            raise ValueError(
                f"steering_gain should be in [0, 1], got {self.steering_gain}"
            )
        if not 0.0 <= self.steer_smooth <= 1.0:
            raise ValueError(
                f"steer_smooth should be in [0, 1], got {self.steer_smooth}"
            )

        # keep track of last steering for smoothing
        self.last_steer = 0.0
        self.action_space = None

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

    def preprocess_lidar(self, ranges, min_scan: Optional[float] = None) -> np.ndarray:
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

    def create_bubble(self, proc, min_scan: Optional[float] = None) -> np.ndarray:
        """Zero out a bubble around the closest obstacle."""
        closest = np.argmin(proc)
        radius = self._adaptive_bubble_radius(min_scan)
        start = max(0, closest - radius)
        end = min(len(proc) - 1, closest + radius)
        proc[start:end+1] = 0
        return proc

    def find_max_gap(self, proc) -> Tuple[int, int]:
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

    def _select_gap(self, proc: np.ndarray, raw_scan: np.ndarray) -> tuple:
        gaps, start = [], None
        mask = proc > float(self.gap_min_range)
        for i, v in enumerate(mask):
            if v and start is None:
                start = i
            elif not v and start is not None:
                gaps.append((start, i - 1))
                start = None
        if start is not None:
            gaps.append((start, len(proc) - 1))
        if not gaps:
            return 0, len(proc) - 1
        if not getattr(self, "enhanced_gap_scoring", False):
            return max(gaps, key=lambda g: g[1] - g[0])

        center_idx = len(raw_scan) // 2
        best_gap = gaps[0]
        best_score = -np.inf
        for gap in gaps:
            score = self._score_gap(gap, raw_scan, center_idx)
            if score > best_score:
                best_score = score
                best_gap = gap
        return best_gap

    def _score_gap(self, gap: tuple, scan: np.ndarray, center_idx: int) -> float:
        start, end = gap
        width = end - start + 1
        midpoint = (start + end) / 2.0
        center_offset = abs(midpoint - center_idx) / max(center_idx, 1)
        min_clearance = float(np.min(scan[start:end + 1])) if end >= start else 0.0
        curvature_penalty = 0.0
        if start > 0 and end < len(scan) - 1:
            left_trend = scan[start] - scan[max(0, start - 1)]
            right_trend = scan[min(len(scan) - 1, end + 1)] - scan[end]
            curvature_penalty = abs(left_trend - right_trend)
        width_weight = getattr(self, "gap_score_width_weight", 1.0)
        clearance_weight = getattr(self, "gap_score_clearance_weight", 1.0)
        center_weight = getattr(self, "gap_score_center_weight", 0.5)
        curvature_weight = getattr(self, "gap_score_curvature_weight", 0.2)
        score = (
            width_weight * width
            + clearance_weight * min_clearance
            - center_weight * center_offset
            - curvature_weight * curvature_penalty
        )
        return float(score)

    def best_point_midgap(self, gap):
        """Return the midpoint of the widest gap."""
        return (gap[0] + gap[1]) // 2
    def best_point_farthest(self, gap, scan):
        start, end = gap
        seg = scan[start:end+1]
        if seg.size == 0:
            return (start + end) // 2
        rel = int(np.argmax(seg))
        return start + rel

    def get_action(self, action_space, obs: dict) -> np.ndarray:
        return self._get_action_lidar(action_space, obs)

    def _get_action_lidar(self, action_space, obs: dict) -> np.ndarray:
        scan = np.asarray(obs["scans"], dtype=np.float32)
        if self.normalized:
            scan = scan * self.max_distance

        max_d = float(self.max_distance)
        scan[~np.isfinite(scan)] = max_d
        scan[scan == 0.0] = max_d   # important for “invalid as 0.0”
        scan = np.clip(scan, 0.0, max_d)

        N = len(scan)
        center_idx = N // 2

        min_scan = float(np.min(scan))

        # 1. Base gap-following
        proc = self.preprocess_lidar(scan, min_scan=min_scan)
        proc = self.create_bubble(proc, min_scan=min_scan)
        gap = self._select_gap(proc, scan)
        if getattr(self, "target_mode", "midgap") == "farthest":
            best = self.best_point_farthest(gap, scan)
        else:
            best = self.best_point_midgap(gap)
        offset = (best - center_idx) / center_idx
        steering = offset * self.steering_gain * self.max_steer

        steering = self._apply_lookahead_guard(steering, scan, min_scan, center_idx, obs)
        steering = self._apply_kinematic_preview(steering, scan, obs)
        steering = self._apply_dynamic_window(steering, scan, obs)

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
        speed = self._extract_speed(obs)
        speed_cap = self._steering_cap_from_speed(speed)
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

    def _extract_speed(self, obs: Dict[str, Any]) -> float:
        velocity_vec = obs.get("velocity")
        speed = 0.0
        if velocity_vec is not None:
            arr = np.asarray(velocity_vec, dtype=np.float32).reshape(-1)
            if arr.size >= 2:
                speed = float(np.linalg.norm(arr[:2]))
            elif arr.size == 1:
                speed = float(abs(arr[0]))
        else:
            raw_speed = obs.get("speed")
            if raw_speed is not None:
                try:
                    speed = float(raw_speed)
                except (TypeError, ValueError):
                    arr = np.asarray(raw_speed, dtype=np.float32).reshape(-1)
                    speed = float(arr[0]) if arr.size else 0.0
        if not np.isfinite(speed):
            return 0.0
        return abs(float(speed))

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
        speed = self._extract_speed(obs)
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

    def _apply_dynamic_window(self, steering: float, scan: np.ndarray, obs: Dict[str, Any]) -> float:
        dwa_samples = int(getattr(self, "dwa_samples", 0))
        if dwa_samples <= 0:
            return steering
        horizon = float(getattr(self, "dwa_horizon", 0.5))
        heading_weight = float(getattr(self, "dwa_heading_weight", 0.1))
        speed_state = self._extract_speed(obs)

        steer_candidates = np.linspace(-self.max_steer, self.max_steer, dwa_samples)
        speed_candidates = np.linspace(self.min_speed, max(self.min_speed, speed_state), dwa_samples)
        best_score = -np.inf
        best = steering
        for s in speed_candidates:
            for st in steer_candidates:
                clearance = self._simulate_preview(st, s, scan, horizon)
                heading_penalty = abs(st)
                score = clearance - heading_weight * heading_penalty
                if score > best_score:
                    best_score = score
                    best = st
        return best

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
        side_band = np.concatenate([left_band, right_band]) if left_band.size + right_band.size > 0 else np.array([])
        spread = float(np.std(side_band)) if side_band.size > 0 else 0.0
        forward_window = max(center_idx // 12, 1)
        forward_band = scan[max(0, center_idx - forward_window):min(len(scan), center_idx + forward_window + 1)]
        forward_min = float(np.min(forward_band)) if forward_band.size > 0 else forward

        speed = self.max_speed
        lookahead_idx = int(center_idx + np.sign(steering) * max(abs(steering) / max(self.max_steer, 1e-3), 0.1) * center_idx)
        lookahead_idx = int(np.clip(lookahead_idx, 0, len(scan) - 1))
        lookahead_range = float(scan[lookahead_idx])

        if forward_min < 1.5 or lookahead_range < 2.0:
            speed = max(self.min_speed, 0.5 * self.max_speed)
        elif forward < 3.0 or spread > 2.0 or lookahead_range < 3.0 or forward_min < 2.5:
            speed = max(self.min_speed, 0.6 * self.max_speed)
        elif forward < 6.0 or spread > 1.0:
            speed = 0.8 * self.max_speed

        speed = float(np.clip(speed, self.min_speed, self.max_speed))
        if (
            getattr(self, "u_shape_enabled", False)
            and self._detect_u_shape(scan, center_idx, min_scan)
        ):
            speed = min(speed, max(self.min_speed, self.u_shape_crawl_speed * self.max_speed))

        return speed

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
        speed = self._extract_speed(obs)
        dt = 0.15
        curvature = steering / max(self.max_steer, 1e-3)
        yaw_change = curvature * dt
        lookahead_idx = int(center_idx + yaw_change * center_idx)
        lookahead_idx = int(np.clip(lookahead_idx, 0, len(scan) - 1))
        lookahead_range = float(scan[lookahead_idx])
        if lookahead_range < max(1.2, 0.4 * min_scan):
            steering *= 0.5
        return steering

    def _detect_u_shape(self, scan: np.ndarray, center_idx: int, min_scan: float) -> bool:
        threshold = getattr(self, "u_shape_threshold", 0.5)
        if min_scan > threshold * self.max_distance:
            return False
        window = max(center_idx // 4, 1)
        left = float(np.min(scan[max(0, center_idx - window):center_idx])) if center_idx > 0 else np.inf
        right = float(np.min(scan[center_idx:center_idx + window])) if center_idx < len(scan) else np.inf
        front = float(scan[center_idx])
        return left < threshold * self.max_distance and right < threshold * self.max_distance and front < threshold * self.max_distance

    def set_action_space(self, action_space) -> None:
        self.action_space = action_space

    def act(self, obs: Dict[str, Any], deterministic: bool = False, aid: Optional[str] = None) -> np.ndarray:
        return self.get_action(self.action_space, obs)

    def store(self, *args, **kwargs) -> None:
        return None

    def finish_path(self, **kwargs) -> None:
        return None

    def update(self) -> Optional[Dict[str, float]]:
        return None

    def save(self, path: str) -> None:
        """Save policy configuration to a JSON file.

        Args:
            path: Path to save the configuration file.

        Raises:
            ValueError: If the parent directory doesn't exist.
            IOError: If there's a permission error or other I/O error.
        """
        # Check parent directory exists
        parent_dir = os.path.dirname(path)
        if parent_dir and not os.path.exists(parent_dir):
            raise ValueError(
                f"Cannot save to '{path}': parent directory '{parent_dir}' does not exist"
            )

        config = {key: getattr(self, key) for key in self.CONFIG_DEFAULTS if hasattr(self, key)}
        payload = {"config": config}

        try:
            with open(path, "w") as f:
                json.dump(payload, f, indent=2)
        except PermissionError as e:
            raise IOError(f"Permission denied when writing to '{path}': {e}") from e
        except OSError as e:
            raise IOError(f"Failed to write configuration to '{path}': {e}") from e

    def load(self, path: str) -> None:
        """Load policy configuration from a JSON file.

        Args:
            path: Path to the configuration file to load.

        Raises:
            FileNotFoundError: If the configuration file doesn't exist.
            ValueError: If the file contains invalid JSON or configuration.
            IOError: If there's a permission error or other I/O error.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found: '{path}'")

        try:
            with open(path, "r") as f:
                payload = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file '{path}': {e}") from e
        except PermissionError as e:
            raise IOError(f"Permission denied when reading '{path}': {e}") from e
        except OSError as e:
            raise IOError(f"Failed to read configuration from '{path}': {e}") from e

        config = payload.get("config", payload)
        if config:
            self.apply_config(config)


class FTGAgent(FollowTheGapPolicy):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        params = dict(config or {})
        if "params" in params and isinstance(params["params"], dict):
            params = dict(params["params"])
        action_space = params.pop("action_space", None)
        policy = FollowTheGapPolicy.from_config(params)
        self.__dict__.update(policy.__dict__)
        self.action_space = action_space
