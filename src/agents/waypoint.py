"""Centerline waypoint-following controllers for F110 MARL.

Provides heuristic agents that follow the track centerline without learning.
Each class implements the .act(obs) interface used by SB3RoleWrapper.

Observation dict keys used:
    "pose"     — np.ndarray [x, y, theta]
    "velocity" — np.ndarray [vx, vy]  (optional, for speed smoothing)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _is_closed_path(pts: np.ndarray) -> bool:
    """Return whether the endpoint gap is consistent with a closed centerline."""
    if len(pts) < 3:
        return False
    segment_lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    positive = segment_lengths[segment_lengths > 1e-6]
    if positive.size == 0:
        return False
    endpoint_gap = float(np.linalg.norm(pts[0] - pts[-1]))
    return endpoint_gap <= 1.25 * float(np.median(positive))


def _find_nearest(pts: np.ndarray, pos: np.ndarray, last_idx: int, window: int = 60) -> int:
    """Return the index of the centerline point closest to pos.

    Uses a sliding window around last_idx for O(window) cost when the agent
    is already on the centerline; falls back to a full O(N) scan when last_idx
    is -1 or out of bounds.
    """
    n = len(pts)
    if last_idx < 0 or last_idx >= n:
        dists = np.sum((pts - pos) ** 2, axis=1)
        return int(np.argmin(dists))
    if _is_closed_path(pts):
        # Modular candidates let the incremental search cross N-1 -> 0 at
        # the start/finish seam without abandoning its O(window) behavior.
        candidates = np.arange(last_idx - window, last_idx + window + 1) % n
        dists = np.sum((pts[candidates] - pos) ** 2, axis=1)
        return int(candidates[int(np.argmin(dists))])
    lo = max(0, last_idx - window)
    hi = min(n, last_idx + window + 1)
    dists = np.sum((pts[lo:hi] - pos) ** 2, axis=1)
    return int(lo + np.argmin(dists))


def _lookahead_point(pts: np.ndarray, start: int, dist: float) -> tuple[np.ndarray, int]:
    """Walk forward along centerline until arc length >= dist from pts[start].

    Returns (goal_xy, goal_idx). Closed centerlines wrap at the seam; open
    centerlines retain the previous endpoint clamp.
    """
    n = len(pts)
    idx = start
    accumulated = 0.0
    closed = _is_closed_path(pts)
    max_segments = n if closed else max(n - 1 - start, 0)
    for _ in range(max_segments):
        next_idx = (idx + 1) % n if closed else idx + 1
        seg = float(np.linalg.norm(pts[next_idx] - pts[idx]))
        if accumulated + seg >= dist:
            # Interpolate exactly at the requested distance
            frac = (dist - accumulated) / max(seg, 1e-9)
            goal = pts[idx] + frac * (pts[next_idx] - pts[idx])
            return goal, idx
        accumulated += seg
        idx = next_idx
    return pts[-1].copy(), n - 1


def _path_curvature(pts: np.ndarray, start: int, horizon_m: float) -> float:
    """Mean absolute curvature (rad/m) over horizon_m ahead of pts[start]."""
    n = len(pts)
    if n < 3:
        return 0.0
    dx = np.diff(pts[:, 0].astype(np.float64))
    dy = np.diff(pts[:, 1].astype(np.float64))
    seg_len = np.hypot(dx, dy)
    heading = np.arctan2(dy, np.maximum(np.abs(dx), 1e-12) * np.sign(dx + 1e-12))
    heading = np.unwrap(np.arctan2(dy, dx))
    dtheta = np.abs(np.diff(np.append(heading, heading[-1])))
    curvature = dtheta / np.maximum(seg_len, 1e-6)  # rad/m per segment

    acc = 0.0
    total_curv = 0.0
    count = 0
    for i in range(start, min(n - 1, start + 500)):
        seg = seg_len[i] if i < len(seg_len) else 0.0
        total_curv += curvature[i] if i < len(curvature) else 0.0
        count += 1
        acc += seg
        if acc >= horizon_m:
            break
    return float(total_curv / max(count, 1))


# ---------------------------------------------------------------------------
# Pure Pursuit
# ---------------------------------------------------------------------------

class PurePursuitPolicy:
    """Centerline-following agent using the pure pursuit steering law.

    Parameters
    ----------
    centerline_fn:
        Zero-argument callable that returns the current centerline as an
        np.ndarray of shape (N, 2) or (N, 3).  Called every step so it
        automatically tracks map cycles.  Example::

            lambda: env.centerline_points

    lookahead:
        Arc-length lookahead distance in metres.  Larger values → smoother
        but less accurate path-tracking; smaller values → tighter but can
        oscillate on high-speed corners.
    min_speed / max_speed:
        Speed range in m/s.  Speed is linearly reduced when upcoming
        curvature exceeds ``curvature_slowdown_threshold``.
    max_steer:
        Maximum steering angle in radians (clamped to vehicle limits).
    wheelbase:
        Effective wheelbase L = lf + lr.  Default matches vehicle_params in
        marl_6car.yaml (0.15875 + 0.17145 = 0.3302 m).
    curvature_slowdown_threshold:
        Path curvature (rad/m) at which speed begins to drop toward min_speed.
    speed_horizon:
        Arc-length (m) ahead used to estimate upcoming curvature for speed.
    search_window:
        Half-window (in waypoints) for incremental nearest-point search.
    """

    def __init__(
        self,
        centerline_fn: Callable[[], Optional[np.ndarray]],
        *,
        lookahead: float = 1.5,
        min_speed: float = 1.0,
        max_speed: float = 3.5,
        max_steer: float = 0.46,
        wheelbase: float = 0.33020,
        curvature_slowdown_threshold: float = 0.3,
        speed_horizon: float = 3.0,
        search_window: int = 60,
        action_space=None,
    ) -> None:
        self._cl_fn = centerline_fn
        self.lookahead = float(lookahead)
        self.min_speed = float(min_speed)
        self.max_speed = float(max_speed)
        self.max_steer = float(max_steer)
        self.wheelbase = float(wheelbase)
        self.curvature_slowdown_threshold = float(curvature_slowdown_threshold)
        self.speed_horizon = float(speed_horizon)
        self.search_window = int(search_window)
        self.action_space = action_space

        self._last_idx: int = -1

    def reset(self) -> None:
        """Call on each episode reset to restart the nearest-point search."""
        self._last_idx = -1

    def act(
        self,
        obs: Dict[str, Any],
        deterministic: bool = False,
        aid: Optional[str] = None,
    ) -> np.ndarray:
        cl = self._cl_fn()
        if cl is None or len(cl) < 2:
            return np.array([0.0, self.min_speed], dtype=np.float32)

        pose = obs.get("pose")
        if pose is None:
            return np.array([0.0, self.min_speed], dtype=np.float32)

        x, y, theta = float(pose[0]), float(pose[1]), float(pose[2])
        pos = np.array([x, y], dtype=np.float32)
        pts = np.asarray(cl[:, :2], dtype=np.float32)

        # 1. Nearest centerline point (windowed search)
        self._last_idx = _find_nearest(pts, pos, self._last_idx, self.search_window)

        # 2. Lookahead goal point
        goal, goal_idx = _lookahead_point(pts, self._last_idx, self.lookahead)

        # 3. Pure pursuit steering
        dx = goal[0] - x
        dy = goal[1] - y
        # Rotate into vehicle frame (x = forward, y = left)
        alpha = np.arctan2(
            -np.sin(theta) * dx + np.cos(theta) * dy,
             np.cos(theta) * dx + np.sin(theta) * dy,
        )
        # Actual Euclidean distance to goal (more accurate than arc lookahead)
        L_d = max(float(np.hypot(dx, dy)), 1e-3)
        steer = float(np.arctan2(2.0 * self.wheelbase * np.sin(alpha), L_d))
        steer = float(np.clip(steer, -self.max_steer, self.max_steer))

        # 4. Speed: drop linearly as curvature exceeds threshold
        curv = _path_curvature(pts, goal_idx, self.speed_horizon)
        t = np.clip(
            (curv - self.curvature_slowdown_threshold) / max(self.curvature_slowdown_threshold, 1e-3),
            0.0, 1.0,
        )
        speed = float(self.max_speed - t * (self.max_speed - self.min_speed))

        action = np.array([steer, speed], dtype=np.float32)
        if self.action_space is not None:
            action = np.clip(action, self.action_space.low, self.action_space.high)
        return action

    # Compatibility shims so this can sit alongside FTG in other_agents
    def store(self, *args, **kwargs) -> None:
        return None

    def finish_path(self, **kwargs) -> None:
        return None

    def update(self) -> Optional[Dict[str, float]]:
        return None

    def save(self, path: str) -> None:
        import json
        cfg = {
            "lookahead": self.lookahead,
            "min_speed": self.min_speed,
            "max_speed": self.max_speed,
            "max_steer": self.max_steer,
            "wheelbase": self.wheelbase,
            "curvature_slowdown_threshold": self.curvature_slowdown_threshold,
            "speed_horizon": self.speed_horizon,
        }
        with open(path, "w") as f:
            import json
            json.dump({"config": cfg}, f, indent=2)


# ---------------------------------------------------------------------------
# Stanley Controller
# ---------------------------------------------------------------------------

class StanleyPolicy:
    """Stanley steering controller for centerline following.

    Combines heading error with lateral (cross-track) error:
        δ = θ_e + arctan(k · e_y / (v + k_s))

    where θ_e is the heading error, e_y is signed lateral distance to the
    nearest centerline point, v is forward speed, and k / k_s are gains.

    Stanley is better than pure pursuit at low speeds and straight sections
    where the cross-track term dominates. At high speeds and sharp corners
    it becomes aggressive; the ``max_steer`` clip prevents instability.

    Parameters
    ----------
    centerline_fn:
        Callable returning current centerline (N, 2) or (N, 3).
    k:
        Cross-track gain (higher → more aggressive lateral correction).
    k_s:
        Softening constant (prevents division-by-zero at zero speed).
    min_speed / max_speed:
        Speed range in m/s (curvature-based schedule same as PurePursuit).
    curvature_slowdown_threshold:
        Path curvature (rad/m) at which speed starts dropping.
    speed_horizon:
        Arc-length ahead (m) used to estimate curvature for speed.
    """

    def __init__(
        self,
        centerline_fn: Callable[[], Optional[np.ndarray]],
        *,
        k: float = 1.0,
        k_s: float = 0.5,
        min_speed: float = 1.0,
        max_speed: float = 3.5,
        max_steer: float = 0.46,
        curvature_slowdown_threshold: float = 0.3,
        speed_horizon: float = 3.0,
        search_window: int = 60,
        action_space=None,
    ) -> None:
        self._cl_fn = centerline_fn
        self.k = float(k)
        self.k_s = float(k_s)
        self.min_speed = float(min_speed)
        self.max_speed = float(max_speed)
        self.max_steer = float(max_steer)
        self.curvature_slowdown_threshold = float(curvature_slowdown_threshold)
        self.speed_horizon = float(speed_horizon)
        self.search_window = int(search_window)
        self.action_space = action_space

        self._last_idx: int = -1

    def reset(self) -> None:
        self._last_idx = -1

    def act(
        self,
        obs: Dict[str, Any],
        deterministic: bool = False,
        aid: Optional[str] = None,
    ) -> np.ndarray:
        cl = self._cl_fn()
        if cl is None or len(cl) < 2:
            return np.array([0.0, self.min_speed], dtype=np.float32)

        pose = obs.get("pose")
        if pose is None:
            return np.array([0.0, self.min_speed], dtype=np.float32)

        x, y, theta = float(pose[0]), float(pose[1]), float(pose[2])
        pos = np.array([x, y], dtype=np.float32)
        pts = np.asarray(cl[:, :2], dtype=np.float32)

        self._last_idx = _find_nearest(pts, pos, self._last_idx, self.search_window)
        nearest = pts[self._last_idx]

        # Tangent heading at nearest point (position-derived)
        prev_i = max(0, self._last_idx - 1)
        next_i = min(len(pts) - 1, self._last_idx + 1)
        delta = pts[next_i] - pts[prev_i]
        path_theta = float(np.arctan2(delta[1], delta[0]))

        # Heading error: wrap to (-pi, pi)
        theta_e = float(np.arctan2(np.sin(path_theta - theta), np.cos(path_theta - theta)))

        # Cross-track error: signed distance (positive = car is to the right of path)
        normal = np.array([-np.sin(path_theta), np.cos(path_theta)], dtype=np.float32)
        e_y = float(np.dot(pos - nearest, normal))

        # Forward speed for softening
        vel = obs.get("velocity")
        if vel is not None:
            v = float(np.linalg.norm(np.asarray(vel, dtype=np.float32)))
        else:
            v = self.min_speed

        steer = theta_e + np.arctan2(self.k * e_y, v + self.k_s)
        steer = float(np.clip(steer, -self.max_steer, self.max_steer))

        # Speed: curvature-based
        curv = _path_curvature(pts, self._last_idx, self.speed_horizon)
        t = np.clip(
            (curv - self.curvature_slowdown_threshold) / max(self.curvature_slowdown_threshold, 1e-3),
            0.0, 1.0,
        )
        speed = float(self.max_speed - t * (self.max_speed - self.min_speed))

        action = np.array([steer, speed], dtype=np.float32)
        if self.action_space is not None:
            action = np.clip(action, self.action_space.low, self.action_space.high)
        return action

    def store(self, *args, **kwargs) -> None:
        return None

    def finish_path(self, **kwargs) -> None:
        return None

    def update(self) -> Optional[Dict[str, float]]:
        return None


# ---------------------------------------------------------------------------
# Hybrid: Pure Pursuit steering + FTG lidar obstacle avoidance
# ---------------------------------------------------------------------------

class HybridPPFTGPolicy:
    """Pure pursuit for on-track direction, FTG for obstacle avoidance.

    Uses pure pursuit to pick a steering direction from the centerline, then
    blends in FTG's obstacle-reactive steering when LiDAR sees something close.
    When ``obstacle_distance`` is large the output is pure pursuit; when
    something is within ``blend_distance`` the output tilts toward FTG.

    Parameters
    ----------
    centerline_fn:
        Callable returning current centerline (N, 2) or (N, 3).
    ftg:
        A FollowTheGapPolicy instance (from src.agents.ftg).
    blend_distance:
        LiDAR range (m) at which FTG starts blending in.
    full_ftg_distance:
        LiDAR range (m) at or below which FTG has full authority.
    """

    def __init__(
        self,
        centerline_fn: Callable[[], Optional[np.ndarray]],
        ftg,
        *,
        lookahead: float = 1.5,
        min_speed: float = 1.0,
        max_speed: float = 3.5,
        max_steer: float = 0.46,
        wheelbase: float = 0.33020,
        curvature_slowdown_threshold: float = 0.3,
        speed_horizon: float = 3.0,
        blend_distance: float = 3.0,
        full_ftg_distance: float = 1.5,
        action_space=None,
    ) -> None:
        self._pp = PurePursuitPolicy(
            centerline_fn,
            lookahead=lookahead,
            min_speed=min_speed,
            max_speed=max_speed,
            max_steer=max_steer,
            wheelbase=wheelbase,
            curvature_slowdown_threshold=curvature_slowdown_threshold,
            speed_horizon=speed_horizon,
        )
        self._ftg = ftg
        self.blend_distance = float(blend_distance)
        self.full_ftg_distance = float(full_ftg_distance)
        self.action_space = action_space

    def reset(self) -> None:
        self._pp.reset()

    def act(
        self,
        obs: Dict[str, Any],
        deterministic: bool = False,
        aid: Optional[str] = None,
    ) -> np.ndarray:
        pp_action = self._pp.act(obs, deterministic=deterministic, aid=aid)

        scan = obs.get("scans")
        if scan is None:
            scan = obs.get("lidar")
        if scan is None:
            return pp_action

        scan = np.asarray(scan, dtype=np.float32)
        scan = np.where(np.isfinite(scan) & (scan > 0), scan, self._ftg.max_distance)
        min_range = float(np.min(scan))

        if min_range >= self.blend_distance:
            # No nearby obstacles — pure pursuit only
            action = pp_action
        elif min_range <= self.full_ftg_distance:
            # Very close obstacle — full FTG authority
            action = self._ftg.act(obs)
        else:
            # Linear blend
            t = (self.blend_distance - min_range) / max(
                self.blend_distance - self.full_ftg_distance, 1e-3
            )
            t = float(np.clip(t, 0.0, 1.0))
            ftg_action = self._ftg.act(obs)
            action = (1.0 - t) * pp_action + t * ftg_action

        if self.action_space is not None:
            action = np.clip(action, self.action_space.low, self.action_space.high)
        return action.astype(np.float32)

    def store(self, *args, **kwargs) -> None:
        return None

    def finish_path(self, **kwargs) -> None:
        return None

    def update(self) -> Optional[Dict[str, float]]:
        return None


# ---------------------------------------------------------------------------
# AgentFactory-compatible adapters
#
# These wrap the policy classes above to match the FTGAgent pattern:
#   - Constructor takes a single config dict (from YAML params)
#   - set_env(env) injects the live env reference after factory creation
#   - Falls back to straight/slow action until set_env() is called
# ---------------------------------------------------------------------------

_DEFAULTS_PP = {
    "lookahead": 1.5,
    "min_speed": 1.0,
    "max_speed": 3.5,
    "max_steer": 0.46,
    "wheelbase": 0.33020,
    "curvature_slowdown_threshold": 0.3,
    "speed_horizon": 3.0,
    "search_window": 60,
}

_DEFAULTS_STANLEY = {
    "k": 1.0,
    "k_s": 0.5,
    "min_speed": 1.0,
    "max_speed": 3.5,
    "max_steer": 0.46,
    "curvature_slowdown_threshold": 0.3,
    "speed_horizon": 3.0,
    "search_window": 60,
}

_DEFAULTS_HYBRID = {
    "lookahead": 1.5,
    "min_speed": 1.0,
    "max_speed": 3.5,
    "max_steer": 0.46,
    "wheelbase": 0.33020,
    "curvature_slowdown_threshold": 0.3,
    "speed_horizon": 3.0,
    "blend_distance": 3.0,
    "full_ftg_distance": 1.5,
}


def _extract(cfg: dict, defaults: dict) -> dict:
    """Pull known keys from cfg, coercing to the default's type."""
    out = {}
    for key, default in defaults.items():
        if key in cfg:
            try:
                out[key] = type(default)(cfg[key])
            except (TypeError, ValueError):
                out[key] = default
        else:
            out[key] = default
    return out


class PurePursuitAgent:
    """AgentFactory-compatible Pure Pursuit agent.

    YAML usage::

        algorithm: pure_pursuit
        params:
          lookahead: 1.5
          min_speed: 1.0
          max_speed: 3.5
          max_steer: 0.46
          curvature_slowdown_threshold: 0.3
          speed_horizon: 3.0
    """

    def __init__(self, config: dict) -> None:
        params = dict(config.get("params", config))
        kw = _extract(params, _DEFAULTS_PP)
        self._policy: Optional[PurePursuitPolicy] = None
        self._kw = kw
        self._fallback_speed = float(kw["min_speed"])

    def set_env(self, env) -> None:
        """Inject live env reference so centerline updates on map cycle."""
        self._policy = PurePursuitPolicy(
            lambda: getattr(env, "centerline_points", None),
            **self._kw,
        )

    def act(self, obs: Dict[str, Any], deterministic: bool = False, aid=None) -> np.ndarray:
        if self._policy is None:
            return np.array([0.0, self._fallback_speed], dtype=np.float32)
        return self._policy.act(obs, deterministic=deterministic, aid=aid)

    def reset(self) -> None:
        if self._policy is not None:
            self._policy.reset()

    def store(self, *args, **kwargs) -> None:
        return None

    def finish_path(self, **kwargs) -> None:
        return None

    def update(self) -> Optional[Dict[str, float]]:
        return None


class StanleyAgent:
    """AgentFactory-compatible Stanley controller agent.

    YAML usage::

        algorithm: stanley
        params:
          k: 1.0
          k_s: 0.5
          min_speed: 1.0
          max_speed: 3.5
          max_steer: 0.46
          curvature_slowdown_threshold: 0.3
          speed_horizon: 3.0
    """

    def __init__(self, config: dict) -> None:
        params = dict(config.get("params", config))
        kw = _extract(params, _DEFAULTS_STANLEY)
        self._policy: Optional[StanleyPolicy] = None
        self._kw = kw
        self._fallback_speed = float(kw["min_speed"])

    def set_env(self, env) -> None:
        self._policy = StanleyPolicy(
            lambda: getattr(env, "centerline_points", None),
            **self._kw,
        )

    def act(self, obs: Dict[str, Any], deterministic: bool = False, aid=None) -> np.ndarray:
        if self._policy is None:
            return np.array([0.0, self._fallback_speed], dtype=np.float32)
        return self._policy.act(obs, deterministic=deterministic, aid=aid)

    def reset(self) -> None:
        if self._policy is not None:
            self._policy.reset()

    def store(self, *args, **kwargs) -> None:
        return None

    def finish_path(self, **kwargs) -> None:
        return None

    def update(self) -> Optional[Dict[str, float]]:
        return None


class HybridPPFTGAgent:
    """AgentFactory-compatible Hybrid Pure-Pursuit + FTG agent.

    Pure pursuit drives on-track; FTG blends in when LiDAR detects obstacles.

    YAML usage::

        algorithm: hybrid_pp_ftg
        params:
          lookahead: 1.5
          min_speed: 1.0
          max_speed: 3.5
          blend_distance: 3.0
          full_ftg_distance: 1.5
          ftg:                      # nested FTG params (optional)
            max_distance: 10.0
            bubble_radius: 2.5
            steering_gain: 0.30
            steer_smooth: 0.5
    """

    def __init__(self, config: dict) -> None:
        from agents.ftg import FollowTheGapPolicy
        params = dict(config.get("params", config))
        kw = _extract(params, _DEFAULTS_HYBRID)
        ftg_params = params.get("ftg", {})
        self._ftg = FollowTheGapPolicy.from_config(ftg_params) if ftg_params else FollowTheGapPolicy()
        self._policy: Optional[HybridPPFTGPolicy] = None
        self._kw = kw
        self._fallback_speed = float(kw["min_speed"])

    def set_env(self, env) -> None:
        self._policy = HybridPPFTGPolicy(
            lambda: getattr(env, "centerline_points", None),
            self._ftg,
            **self._kw,
        )

    def act(self, obs: Dict[str, Any], deterministic: bool = False, aid=None) -> np.ndarray:
        if self._policy is None:
            return np.array([0.0, self._fallback_speed], dtype=np.float32)
        return self._policy.act(obs, deterministic=deterministic, aid=aid)

    def reset(self) -> None:
        if self._policy is not None:
            self._policy.reset()

    def store(self, *args, **kwargs) -> None:
        return None

    def finish_path(self, **kwargs) -> None:
        return None

    def update(self) -> Optional[Dict[str, float]]:
        return None
