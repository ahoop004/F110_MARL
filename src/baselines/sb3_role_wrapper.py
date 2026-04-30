"""SB3 role wrapper for multi-agent F110 training.

Wraps N same-role agents (e.g. all defenders, or all attackers) as a single
Gymnasium environment for one SB3 model.  Training cycles the "focal" agent
each episode; non-focal same-role agents run the shared model in inference
mode so the environment stays realistic.  Fixed-policy agents from other
roles (e.g. FTG opponents) step via their .act() interface.

Usage
-----
    role_wrapper = SB3RoleWrapper(
        env,
        role_agent_ids=["car_0", "car_1", "car_2"],
        obs_dim=57,
        action_low=np.array([-0.46, -1.0]),
        action_high=np.array([0.46, 4.0]),
        observation_preset="centerline",
        reward_strategy=reward_strategy,
        action_repeat=2,
    )
    role_wrapper.set_other_agents({"car_3": ftg3, "car_4": ftg4, "car_5": ftg5})

    model = PPO("MlpPolicy", role_wrapper, ...)
    role_wrapper.set_shared_model(model)   # must be called before model.learn()
    model.learn(...)
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.core.obs_flatten import flatten_observation
from src.rewards.base import RewardStrategy
from src.metrics.outcomes import determine_outcome
from src.utils.centerline import project_to_centerline


class SB3RoleWrapper(gym.Env):
    """Single-policy wrapper for N same-role agents in a PettingZoo ParallelEnv.

    Each episode one agent is "focal" (generates the SB3 training transition).
    The other same-role agents run the shared model in inference mode so the
    environment dynamics remain multi-agent realistic.  Fixed-policy agents
    (other roles) use their own .act() method.

    The focal agent cycles deterministically: episode 0 → role_agents[0],
    episode 1 → role_agents[1], ..., wrapping around.

    Args:
        env: PettingZoo ParallelEnv (F110ParallelEnv).
        role_agent_ids: Ordered list of agent IDs that share this policy.
        obs_dim: Flat observation dimension (must match flatten output).
        action_low: Physical action lower bounds.
        action_high: Physical action upper bounds.
        observation_preset: 'centerline' or 'gaplock'; used for flattening.
        reward_strategy: RewardStrategy applied to the focal agent each step.
        action_repeat: Number of env steps per decision step.
        action_constraints: Optional dict with keys:
            - prevent_reverse (bool): clip negative speed to 0
            - speed_index (int): which action dim is speed (default 1)
            - max_v (float): hard cap on physical speed
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        env,
        role_agent_ids: List[str],
        obs_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        observation_preset: Optional[str] = None,
        reward_strategy: Optional[RewardStrategy] = None,
        action_repeat: int = 1,
        action_constraints: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        if not role_agent_ids:
            raise ValueError("role_agent_ids must contain at least one agent ID")

        self.env = env
        self.render_mode = getattr(env, "render_mode", None)
        self.metadata = getattr(env, "metadata", {"render_modes": []})

        self._role_ids: List[str] = list(role_agent_ids)
        self.observation_preset = observation_preset
        self.reward_strategy = reward_strategy

        try:
            self.action_repeat = max(1, int(action_repeat))
        except (TypeError, ValueError):
            self.action_repeat = 1

        constraints = action_constraints or {}
        self._prevent_reverse = bool(constraints.get("prevent_reverse", False))
        try:
            self._speed_index = int(constraints.get("speed_index", 1))
        except (TypeError, ValueError):
            self._speed_index = 1
        try:
            self._max_v = float(constraints["max_v"]) if "max_v" in constraints else None
        except (TypeError, ValueError):
            self._max_v = None

        self._action_low = np.asarray(action_low, dtype=np.float32)
        self._action_high = np.asarray(action_high, dtype=np.float32)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=self._action_low.shape, dtype=np.float32
        )

        # Shared SB3 model — set by runner after instantiation
        self._shared_model: Optional[Any] = None
        # Fixed-policy agents from other roles
        self._other_agents: Dict[str, Any] = {}

        # Per-episode state
        self._focal_episode: int = 0          # increments each reset()
        self._current_obs_dict: Optional[Dict[str, Any]] = None
        self._episode_steps: int = 0
        self._prev_actions: Dict[str, np.ndarray] = {
            aid: np.zeros(self._action_low.shape, dtype=np.float32)
            for aid in self._role_ids
        }
        # Cached centerline projection indices for fast windowed search
        self._progress_indices: Dict[str, int] = {aid: -1 for aid in self._role_ids}
        # Cached track length (computed once per map load; reset on env.reset)
        self._track_length: Optional[float] = None
        # RNG for grid spawn selection (seeded independently of the env)
        self._spawn_rng = np.random.default_rng()

        # Observation scales resolved from underlying env
        self._obs_scales = self._resolve_obs_scales()
        # Spawn candidate cache keyed by map name — survives across episodes on same map
        self._spawn_cache: Dict[str, Optional[np.ndarray]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_shared_model(self, model: Any) -> None:
        """Register the SB3 model so non-focal role agents can run inference."""
        self._shared_model = model

    def set_other_agents(self, agents: Dict[str, Any]) -> None:
        """Register fixed-policy agents for other roles."""
        self._other_agents = {
            aid: agent for aid, agent in agents.items()
            if aid not in self._role_ids
        }

    @property
    def focal_id(self) -> str:
        return self._role_ids[self._focal_episode % len(self._role_ids)]

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        # Advance focal agent for this episode
        self._focal_episode += 1
        focal = self.focal_id

        # Seed the spawn RNG if a seed is provided (first reset only)
        if seed is not None:
            self._spawn_rng = np.random.default_rng(seed)

        # Phase 1: reset with no custom poses so the env can cycle the map.
        # _maybe_cycle_map() fires inside env.reset() and updates centerline_points
        # to the new map's geometry before returning.
        base_options: Dict[str, Any] = dict(options or {})
        obs_dict, info_dict = self.env.reset(
            seed=seed, options=base_options if base_options else None
        )

        # Phase 2: now env.centerline_points belongs to the NEW map.
        # Compute a 2×3 start grid from its geometry, then re-place the agents
        # without triggering another map cycle.
        if "poses" not in base_options:
            grid_poses = self._sample_grid_poses()
            if grid_poses is not None:
                old_cycle = getattr(self.env, "_map_cycle_mode", "")
                try:
                    self.env._map_cycle_mode = ""   # suppress cycle on second reset
                    obs_dict, info_dict = self.env.reset(
                        seed=None, options={"poses": grid_poses}
                    )
                finally:
                    self.env._map_cycle_mode = old_cycle

        self._current_obs_dict = obs_dict
        self._episode_steps = 0

        # Reset per-agent state
        self._track_length = None  # recompute after map may have changed
        for aid in self._role_ids:
            self._prev_actions[aid] = np.zeros(self._action_low.shape, dtype=np.float32)
            self._progress_indices[aid] = -1  # force full centerline search on first step

        obs = self._get_obs(focal)
        return obs, info_dict.get(focal, {})

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        focal = self.focal_id
        phys_focal = self._denormalize(action)
        self._prev_actions[focal] = np.clip(
            np.asarray(action, dtype=np.float32), -1.0, 1.0
        )

        # Build actions for all agents once (sticky across action_repeat)
        actions = self._build_all_actions(focal, phys_focal)

        prev_obs_dict = self._current_obs_dict
        total_reward = 0.0
        reward_components: Dict[str, float] = {}
        terminated = False
        truncated = False
        obs_dict: Optional[Dict] = None
        info_dict: Dict[str, Any] = {}

        for _ in range(self.action_repeat):
            obs_dict, reward_dict, done_dict, truncated_dict, info_dict = self.env.step(actions)
            step_terminated = bool(done_dict.get(focal, False))
            step_truncated = bool(truncated_dict.get(focal, False))

            if self.reward_strategy and prev_obs_dict is not None:
                reward_info = self._build_reward_info(
                    prev_obs=prev_obs_dict,
                    next_obs=obs_dict,
                    info=info_dict,
                    terminated=step_terminated,
                    truncated=step_truncated,
                )
                step_reward, components = self.reward_strategy.compute(reward_info)
                for name, val in (components or {}).items():
                    reward_components[name] = reward_components.get(name, 0.0) + float(val)
            else:
                step_reward = float(reward_dict.get(focal, 0.0))

            total_reward += float(step_reward)
            self._current_obs_dict = obs_dict
            prev_obs_dict = obs_dict
            self._episode_steps += 1
            terminated = step_terminated
            truncated = step_truncated

            if terminated or truncated:
                break

        if obs_dict is None:
            raise RuntimeError("Environment returned no observations during step.")

        obs = self._get_obs(focal)
        info = dict(info_dict.get(focal, {}))

        # --- Collision fix: only the focal agent's OWN collision counts as
        # terminated.  If another car crashed and the env ended the episode
        # for everyone, that's a truncation from focal's perspective — no
        # terminal penalty fires, and the trajectory is still useful.
        focal_collided = bool(info.get("collision", False))
        if terminated and not focal_collided:
            # Env terminated focal due to another car's collision → reclassify
            terminated = False
            truncated = True

        if terminated or truncated:
            outcome = determine_outcome(info, truncated=truncated)
            info["outcome"] = outcome.value
            info["is_success"] = outcome.is_success()

        if reward_components:
            info["reward_components"] = reward_components

        info["focal_agent"] = focal
        info["role_episode"] = self._focal_episode

        return obs, total_reward, terminated, truncated, info

    def render(self):
        if hasattr(self.env, "render"):
            return self.env.render()

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_all_actions(
        self, focal: str, phys_focal: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Compute actions for every agent in the env for one repeat block."""
        actions: Dict[str, np.ndarray] = {focal: phys_focal}

        # Non-focal same-role agents: run shared model in inference mode
        for aid in self._role_ids:
            if aid == focal:
                continue
            obs = (self._current_obs_dict or {}).get(aid)
            if obs is not None and self._shared_model is not None:
                flat_obs = self._flatten_role_obs(aid, obs)
                norm_act, _ = self._shared_model.predict(flat_obs, deterministic=False)
                actions[aid] = self._denormalize(norm_act)
                self._prev_actions[aid] = np.clip(
                    np.asarray(norm_act, dtype=np.float32), -1.0, 1.0
                )
            else:
                # Model not yet set (first episode): drive straight at half speed
                safe = np.zeros(self._action_low.shape, dtype=np.float32)
                actions[aid] = self._denormalize(safe)

        # Fixed-policy agents (other roles)
        for aid, agent in self._other_agents.items():
            obs = (self._current_obs_dict or {}).get(aid)
            if obs is not None and hasattr(agent, "act"):
                actions[aid] = agent.act(obs)

        return actions

    def _get_obs(self, agent_id: str) -> np.ndarray:
        """Extract and flatten the focal agent's current observation."""
        raw = (self._current_obs_dict or {}).get(agent_id, {})
        return self._flatten_role_obs(agent_id, raw)

    def _flatten_role_obs(self, agent_id: str, raw_obs: Any) -> np.ndarray:
        """Flatten one role agent's raw observation dict."""
        if isinstance(raw_obs, np.ndarray):
            return raw_obs.astype(np.float32)

        if isinstance(raw_obs, dict) and self.observation_preset:
            obs_with_extras = dict(raw_obs)
            obs_with_extras["prev_action"] = self._prev_actions.get(
                agent_id, np.zeros(self._action_low.shape, dtype=np.float32)
            )
            # Inject centerline progress for presets that use it
            if self.observation_preset == "centerline_racing":
                obs_with_extras["progress"] = self._compute_progress(agent_id, raw_obs)
            return flatten_observation(
                obs_with_extras,
                preset=self.observation_preset,
                scales=self._obs_scales,
            )

        try:
            return np.asarray(raw_obs, dtype=np.float32)
        except Exception:
            return np.zeros(self.observation_space.shape, dtype=np.float32)

    # Grid layout constants — 2 columns × 3 rows, matching original circle grid
    _GRID_ROW_SPACING = 1.6    # metres between rows (along track, backward)
    _GRID_COL_OFFSET  = 0.35   # metres lateral from centreline (±)
    _GRID_MIN_HALF_WIDTH = 0.6 # min wall half-distance required at spawn point
    _GRID_CURVATURE_MAX  = 0.5 # max curvature (rad/m) over grid footprint — rejects corners

    def _sample_grid_poses(self) -> Optional[np.ndarray]:
        """Pick a random straight, wide section and return 6 poses in a 2×3 grid."""
        centerline = getattr(self.env, "centerline_points", None)
        if centerline is None or centerline.shape[0] < 10:
            return None

        n = centerline.shape[0]
        candidates = self._get_spawn_candidates(centerline, n)
        if candidates is None or len(candidates) == 0:
            # Fallback: any point in 5–95 % of the lap
            candidates = np.arange(int(0.05 * n), int(0.95 * n))

        idx = int(self._spawn_rng.choice(candidates))
        return self._build_grid_poses(centerline, idx, n)

    def _get_spawn_candidates(
        self, centerline: np.ndarray, n: int
    ) -> Optional[np.ndarray]:
        """Return cached array of valid spawn indices for the current map.

        Cache is keyed by map name so the expensive computation runs once per
        unique map, not once per episode.
        """
        map_key = getattr(self.env, "map_name", None) or str(id(centerline))
        if map_key in self._spawn_cache:
            return self._spawn_cache[map_key]

        from scipy.ndimage import maximum_filter1d

        pts = centerline[:, :2].astype(np.float32)

        # --- curvature filter -------------------------------------------------
        seg_len = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        avg_seg = float(seg_len.mean()) if len(seg_len) > 0 else 0.2

        dx = np.diff(pts[:, 0].astype(np.float64))
        dy = np.diff(pts[:, 1].astype(np.float64))
        heading_seg = np.arctan2(dy, dx)
        thetas_raw = np.concatenate([heading_seg, [heading_seg[-1]]])

        thetas_unwrap = np.unwrap(thetas_raw)
        dtheta = np.abs(np.diff(thetas_unwrap))
        curv_seg = dtheta / np.maximum(seg_len, 1e-6)

        # Vectorized rolling max over grid footprint (3 rows × row_spacing + margin)
        footprint_m = 3.0 * self._GRID_ROW_SPACING + 1.0
        window = max(3, int(footprint_m / avg_seg))
        rolling_max = maximum_filter1d(curv_seg, size=window, mode="nearest")
        rolling_max_n = np.append(rolling_max, rolling_max[-1])
        straight = rolling_max_n < self._GRID_CURVATURE_MAX

        # --- width filter via walls ------------------------------------------
        walls = getattr(self.env, "walls", None)
        if walls is not None:
            try:
                from scipy.spatial import cKDTree
                all_wall_pts = np.vstack([
                    np.asarray(v, dtype=np.float32)
                    for v in (walls.values() if isinstance(walls, dict) else walls)
                    if len(v) > 0
                ])
                wall_tree = cKDTree(all_wall_pts)
                # KD-tree query at every 10th centerline point, then interpolate
                sample_idxs = np.arange(0, n, 10)
                sampled_pts = pts[sample_idxs]
                dists, _ = wall_tree.query(sampled_pts, k=1)
                nearest_sample_idx = np.argmin(
                    np.abs(np.arange(n)[:, None] - sample_idxs[None, :]), axis=1
                )
                half_widths = dists[nearest_sample_idx]
                wide_enough = half_widths >= self._GRID_MIN_HALF_WIDTH
            except Exception:
                wide_enough = np.ones(n, dtype=bool)
        else:
            wide_enough = np.ones(n, dtype=bool)

        # --- lap position filter (avoid finish line) --------------------------
        lo = int(0.05 * n)
        hi = int(0.95 * n)
        lap_ok = np.zeros(n, dtype=bool)
        lap_ok[lo:hi] = True

        valid = np.where(straight & wide_enough & lap_ok)[0]

        # Adaptive fallback: if the hard curvature threshold leaves fewer than
        # 10 % of lap points valid (e.g. uniformly circular tracks), relax to
        # the straightest 20 % of points that are at least wide enough.
        lap_indices = np.where(wide_enough & lap_ok)[0]
        if len(valid) < max(5, int(0.10 * len(lap_indices))) and len(lap_indices) > 0:
            curv_lap = rolling_max_n[lap_indices]
            threshold = np.percentile(curv_lap, 20)
            valid = lap_indices[curv_lap <= threshold]

        result = valid if len(valid) > 0 else None
        self._spawn_cache[map_key] = result
        return result

    def _build_grid_poses(
        self, centerline: np.ndarray, idx: int, n: int
    ) -> np.ndarray:
        """Compute 6 poses in a 2-column × 3-row racing grid at centerline[idx].

        Each row is anchored to the ACTUAL centerline point at that row's
        arc-length offset, so the grid correctly follows track curvature rather
        than projecting all rows from a single tangent.
        """
        pts = centerline[:, :2].astype(np.float32)
        seg_len = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        avg_seg = float(seg_len.mean()) if len(seg_len) > 0 else 0.2
        steps_per_row = max(1, int(round(self._GRID_ROW_SPACING / avg_seg)))

        poses = np.zeros((6, 3), dtype=np.float32)
        k = 0
        for row in range(3):
            row_idx = max(0, idx - row * steps_per_row)
            pt = pts[row_idx]

            # Position-derived heading at this row's centerline point
            prev_i = max(0, row_idx - 1)
            next_i = min(n - 1, row_idx + 1)
            delta = pts[next_i] - pts[prev_i]
            row_theta = float(np.arctan2(delta[1], delta[0]))

            cos_t, sin_t = np.cos(row_theta), np.sin(row_theta)
            tangent = np.array([cos_t,  sin_t])
            normal  = np.array([-sin_t, cos_t])

            for col_sign in (-1, +1):   # -1 = right, +1 = left
                pos = pt + col_sign * self._GRID_COL_OFFSET * normal
                poses[k] = [pos[0], pos[1], row_theta]
                k += 1
        return poses

    def _get_track_length(self) -> Optional[float]:
        """Return centerline arc length in metres, cached per episode."""
        if self._track_length is not None:
            return self._track_length
        centerline = getattr(self.env, "centerline_points", None)
        if centerline is None or centerline.shape[0] < 2:
            return None
        cl = np.asarray(centerline, dtype=np.float32)
        self._track_length = float(np.linalg.norm(np.diff(cl[:, :2], axis=0), axis=1).sum())
        return self._track_length

    def _compute_progress(self, agent_id: str, obs: Dict[str, Any]) -> float:
        """Return normalized lap progress [0, 1] via centerline projection."""
        centerline = getattr(self.env, "centerline_points", None)
        if centerline is None or centerline.shape[0] == 0:
            return 0.0
        pose = obs.get("pose")
        if pose is None:
            return 0.0
        pose_arr = np.asarray(pose, dtype=np.float32).reshape(-1)
        if pose_arr.size < 2:
            return 0.0
        position = pose_arr[:2]
        heading = float(pose_arr[2]) if pose_arr.size >= 3 else 0.0
        last_idx = self._progress_indices.get(agent_id, -1)
        try:
            proj = project_to_centerline(
                centerline, position, heading, last_index=last_idx if last_idx >= 0 else None
            )
            self._progress_indices[agent_id] = proj.index
            return float(proj.progress)
        except Exception:
            return 0.0

    def _denormalize(self, norm_action: np.ndarray) -> np.ndarray:
        """Map normalized action [-1,1] → physical action space."""
        norm = np.clip(np.asarray(norm_action, dtype=np.float32), -1.0, 1.0)
        if self._prevent_reverse and 0 <= self._speed_index < norm.shape[0]:
            if norm[self._speed_index] < 0.0:
                norm = norm.copy()
                norm[self._speed_index] = 0.0
        phys = self._action_low + (norm + 1.0) * 0.5 * (self._action_high - self._action_low)
        if self._max_v is not None and 0 <= self._speed_index < phys.shape[0]:
            if phys[self._speed_index] > self._max_v:
                phys = phys.copy()
                phys[self._speed_index] = self._max_v
        return phys

    def _build_reward_info(
        self,
        prev_obs: Dict[str, Any],
        next_obs: Dict[str, Any],
        info: Dict[str, Any],
        terminated: bool,
        truncated: bool,
    ) -> Dict[str, Any]:
        focal = self.focal_id
        return {
            "obs": prev_obs.get(focal, {}),
            "next_obs": next_obs.get(focal, {}),
            "info": info.get(focal, {}) if isinstance(info, dict) else {},
            "step": self._episode_steps,
            "done": terminated or truncated,
            "truncated": truncated,
            "timestep": getattr(self.env, "timestep", 0.01),
            "action": self._prev_actions.get(focal, np.zeros(2, dtype=np.float32)),
            "centerline": getattr(self.env, "centerline_points", None),
            "track_length": self._get_track_length(),
            "walls": getattr(self.env, "walls", None),
            # Expose all agents' info so reward can read cross-agent state
            "all_info": info,
        }

    def _resolve_obs_scales(self) -> Dict[str, float]:
        scales: Dict[str, float] = {}
        lidar_range = getattr(self.env, "lidar_range", None)
        if lidar_range is not None:
            try:
                scales["lidar_range"] = float(lidar_range)
            except (TypeError, ValueError):
                pass
        params = getattr(self.env, "params", None)
        if isinstance(params, dict):
            candidates = [
                abs(float(v))
                for v in (params.get("v_max"), params.get("v_min"))
                if v is not None
            ]
            if candidates:
                speed_scale = max(candidates)
                if speed_scale > 0.0:
                    scales["speed"] = speed_scale
        return scales


__all__ = ["SB3RoleWrapper"]
