import math
from typing import Optional, Tuple

import numpy as np

class FollowTheGapPolicy:
    CENTRAL_STATE_KEYS = (
        "poses_x",
        "poses_y",
        "poses_theta",
        "linear_vels_x",
        "linear_vels_y",
        "ang_vels_z",
        "collisions",
    )

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
        "secondary_warning_border": 0.35,
        "secondary_hard_border": 0.5,
        "secondary_safe_distance": 1.0,
        "secondary_turn_gain": 1.5,
        "secondary_border_speed_scale": 0.5,
        "secondary_max_turn": 1.0,
        "secondary_target_slot": 0,
        "secondary_target_agent": None,
        "secondary_lane_center": None,
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
                 secondary_warning_border=0.35,
                 secondary_hard_border=0.5,
                 secondary_safe_distance=1.0,
                 secondary_turn_gain=1.5,
                 secondary_border_speed_scale=0.5,
                 secondary_max_turn=1.0,
                 secondary_target_slot=0,
                 secondary_target_agent=None,
                 secondary_lane_center=None):
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
        self.secondary_warning_border = float(secondary_warning_border)
        self.secondary_hard_border = float(secondary_hard_border)
        self.secondary_safe_distance = float(secondary_safe_distance)
        self.secondary_turn_gain = float(secondary_turn_gain)
        self.secondary_border_speed_scale = float(secondary_border_speed_scale)
        self.secondary_max_turn = float(secondary_max_turn)
        self.secondary_target_slot = int(secondary_target_slot)
        self.secondary_target_agent = (str(secondary_target_agent).strip() if secondary_target_agent else None)
        self.secondary_lane_center = None if secondary_lane_center is None else float(secondary_lane_center)
        self.agent_slot: Optional[int] = None
        self._secondary_lane_center: Optional[float] = None

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

    def preprocess_lidar(self, ranges):
        """Smooth LiDAR with moving average + clip to max_distance."""
        N = len(ranges)
        half = self.window_size // 2
        proc = []
        for i in range(N):
            start = max(0, i - half)
            end = min(N - 1, i + half)
            avg = np.mean(np.clip(ranges[start:end+1], 0, self.max_distance))
            proc.append(avg)
        return np.array(proc)

    def create_bubble(self, proc):
        """Zero out a bubble around the closest obstacle."""
        closest = np.argmin(proc)
        start = max(0, closest - self.bubble_radius)
        end = min(len(proc) - 1, closest + self.bubble_radius)
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
        mode = getattr(self, "mode", "lidar")
        if isinstance(mode, str) and mode.strip().lower() in {"secondary", "secondary_vicon", "convoy"}:
            return self._get_action_secondary(action_space, obs)
        return self._get_action_lidar(action_space, obs)

    def _get_action_lidar(self, action_space, obs: dict):
        scan = np.array(obs["scans"])
        if self.normalized:
            scan = scan * self.max_distance

        N = len(scan)
        center_idx = N // 2

        # 1. Base gap-following
        proc = self.preprocess_lidar(scan)
        proc = self.create_bubble(proc)
        gap = self.find_max_gap(proc)
        best = self.best_point_midgap(gap)
        offset = (best - center_idx) / center_idx
        steering = offset * self.steering_gain * self.max_steer

        # 2. Sector-based danger weighting
        left_min = np.min(scan[:center_idx]) if center_idx > 0 else np.inf
        right_min = np.min(scan[center_idx:]) if center_idx < N else np.inf
        min_scan = float(np.min(scan))

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
        steering = np.clip(steering, -self.max_steer, self.max_steer)
        steering = self.steer_smooth * self.last_steer + (1 - self.steer_smooth) * steering
        self.last_steer = steering

        # 3. Speed schedule (more conservative)
        free_ahead = scan[center_idx]
        if min_scan < 2.0:
            speed = max(self.min_speed, 0.8)
        elif free_ahead > 8.0:
            speed = self.max_speed
        elif free_ahead > 4.0:
            speed = 0.7 * self.max_speed
        else:
            speed = self.min_speed

        action = np.array([steering, speed], dtype=np.float32)
        if action_space is not None:
            action = np.clip(action, action_space.low, action_space.high)
        return action

    def _get_action_secondary(self, action_space, obs: dict):
        state_raw = obs.get("state")
        if state_raw is None:
            return self._get_action_lidar(action_space, obs)

        state_vec = np.asarray(state_raw, dtype=np.float32).flatten()
        key_count = len(self.CENTRAL_STATE_KEYS)
        if state_vec.size == 0 or state_vec.size % key_count != 0:
            return self._get_action_lidar(action_space, obs)

        n_agents = state_vec.size // key_count
        if n_agents <= 0:
            return self._get_action_lidar(action_space, obs)

        ego_pose_arr = np.asarray(obs.get("pose", ()), dtype=np.float32).flatten()
        agent_slot = self.agent_slot
        if agent_slot is None or not (0 <= agent_slot < n_agents):
            agent_slot = self._infer_agent_slot_from_state(ego_pose_arr, state_vec, n_agents)
        if agent_slot is None or not (0 <= agent_slot < n_agents):
            agent_slot = max(0, min(n_agents - 1, self.secondary_target_slot if 0 <= self.secondary_target_slot < n_agents else n_agents - 1))
        self.agent_slot = agent_slot

        target_slot = self.secondary_target_slot
        if target_slot == agent_slot or not (0 <= target_slot < n_agents):
            alternatives = [idx for idx in range(n_agents) if idx != agent_slot]
            target_slot = alternatives[0] if alternatives else agent_slot

        ego_pose = self._extract_pose_from_state(state_vec, agent_slot, n_agents)
        target_pose = self._extract_pose_from_state(state_vec, target_slot, n_agents)
        if ego_pose is None or target_pose is None:
            return self._get_action_lidar(action_space, obs)

        x2, y2, _ = ego_pose
        x1, y1, _ = target_pose

        speed = float(self.max_speed)
        steering = 0.0

        lane_center = self.secondary_lane_center
        if lane_center is None:
            if np.isfinite(y1):
                # Mirror the primary vehicle's lateral position so the follower
                # hugs the same corridor unless an explicit centre is supplied.
                self._secondary_lane_center = float(y1)
            lane_center = self._secondary_lane_center if self._secondary_lane_center is not None else 0.0

        # Lateral offset relative to the desired lane centre (defaults to the
        # primary's lateral position when unspecified).
        lane_offset = y2 - lane_center

        if abs(y2) > self.secondary_hard_border:
            speed = self.max_speed * max(0.1, self.secondary_border_speed_scale * 0.5)
            steering = -self.secondary_max_turn if y2 > 0.0 else self.secondary_max_turn
        elif abs(y2) > self.secondary_warning_border:
            speed = self.max_speed * self.secondary_border_speed_scale
            turn_direction = -1.0 if y2 > 0.0 else 1.0
            steering = turn_direction * self.secondary_max_turn
        else:
            distance = float(math.hypot(x1 - x2, y1 - y2))
            if distance < self.secondary_safe_distance:
                if y1 > 0.0:
                    angle = math.atan2(-self.secondary_warning_border - y1, x1 - x2)
                else:
                    angle = math.atan2(self.secondary_warning_border - y1, x1 - x2)
                angle = float(np.clip(angle, -self.secondary_max_turn, self.secondary_max_turn))
                steering = angle * self.secondary_turn_gain
            else:
                steering = np.clip(lane_offset * self.secondary_turn_gain, -self.secondary_max_turn, self.secondary_max_turn)

        speed = max(0.0, speed)
        action = np.array([steering, speed], dtype=np.float32)
        if action_space is not None:
            action = np.clip(action, action_space.low, action_space.high)
        return action

    def _infer_agent_slot_from_state(
        self,
        ego_pose: np.ndarray,
        state_vec: np.ndarray,
        n_agents: int,
    ) -> Optional[int]:
        if ego_pose.size < 2 or n_agents <= 0:
            return None
        px = state_vec[0:n_agents]
        py = state_vec[n_agents:2 * n_agents]
        diffs = np.column_stack((px - ego_pose[0], py - ego_pose[1]))
        distances = np.sum(diffs ** 2, axis=1)
        idx = int(np.argmin(distances))
        return idx

    def _extract_pose_from_state(
        self,
        state_vec: np.ndarray,
        slot: int,
        n_agents: int,
    ) -> Optional[Tuple[float, float, float]]:
        if not (0 <= slot < n_agents):
            return None
        px = state_vec[0:n_agents]
        py = state_vec[n_agents:2 * n_agents]
        headings = state_vec[2 * n_agents:3 * n_agents]
        return float(px[slot]), float(py[slot]), float(headings[slot])
