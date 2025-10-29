from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces
import yaml
from PIL import Image

from pettingzoo import ParallelEnv

# base classes
from f110x.physics import Simulator, Integrator
from f110x.render import EnvRenderer
from f110x.envs.start_pose_state import StartPoseState
from f110x.envs.collision import build_terminations
from f110x.envs.state_buffer import StateBuffers
from f110x.utils.centerline import progress_from_spacing
from f110x.utils.config_schema import _default_vehicle_params


# others
import numpy as np
import os
import time
import math
import logging

# gl
import pyglet
pyglet.options['debug_gl'] = False
from pyglet import gl
from pyglet import image as pyg_img

# constants
from f110x.wrappers.observation import _sector_from_angle, _radial_gain, _SECTOR_NAMES

# rendering
# VIDEO_W = 600
# VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800

logger = logging.getLogger(__name__)

DEFAULT_AGENT_SENSORS = (
    "lidar",
    "pose",
    "velocity",
    "angular_velocity",
    "lap",
    "collision",
)

class F110ParallelEnv(ParallelEnv):

    metadata = {"name": "F110ParallelEnv", "render_modes": ["human", "rgb_array"]}

    # rendering
    def __init__(self, **kwargs):
        map_data = kwargs.pop("map_data", None)
        env_config = kwargs.get("env", {})
        merged = {**env_config, **kwargs}
        
        self._configure_rendering(merged)
        self._configure_basic_environment(merged)
   
        self.timestep: float = float(merged.get("timestep", 0.01))
        self.integrator = self._resolve_integrator(merged)

        self._configure_map_paths(merged, map_data)
        self.start_poses = np.array(merged.get("start_poses", []),dtype=np.float32)

        self.params = self._configure_vehicle_params(merged)
        
        self.lidar_beams = int(merged.get("lidar_beams", 1080))
        if self.lidar_beams <= 0:
            self.lidar_beams = 1080
        self.lidar_range = float(merged.get("lidar_range", 30.0))
        self._lidar_beam_count = max(int(self.lidar_beams), 1)

        self.lidar_dist: float = float(merged.get("lidar_dist", 0.0))
        
        self.state_buffers = StateBuffers.build(self.n_agents)
        self._bind_state_views()

        default_terminate = bool(merged.get("terminate_on_collision", True))
        self.terminate_on_collision = {
            aid: default_terminate for aid in self.possible_agents
        }
        self.terminate_on_any_done = bool(merged.get("terminate_on_any_done", False))

        self._agent_sensor_spec: Dict[str, Tuple[str, ...]] = {
            aid: DEFAULT_AGENT_SENSORS for aid in self.possible_agents
        }
        self._agent_target_index: Dict[str, Optional[int]] = {
            aid: None for aid in self.possible_agents
        }

        raw_target_laps = merged.get("target_laps") or merged.get("laps")
        try:
            laps_val = int(raw_target_laps) if raw_target_laps is not None else 1
        except (TypeError, ValueError):
            laps_val = 1
        if laps_val <= 0:
            laps_val = 1
        self.target_laps: int = int(laps_val)

        self.current_time = 0.0
        self._elapsed_steps = 0

        # Start pose state machine
        self.lap_forward_vel_epsilon = float(merged.get("lap_forward_vel_epsilon", 0.1))
        self.start_state = StartPoseState.build(
            self.possible_agents,
            self.start_poses,
            self.lap_forward_vel_epsilon,
        )
        self.lap_counts = self.start_state.lap_counts
        self.lap_times = self.start_state.lap_times

        # initiate stuff
        self.sim = Simulator(
            self.params,
            self.n_agents,
            self.seed,
            time_step=self.timestep,
            integrator=self.integrator,
            lidar_dist=self.lidar_dist,
            num_beams=self._lidar_beam_count,
        )

        self.sim.set_map(str(self.yaml_path), self.map_ext)
        meta, img_path, (width, height) = self._load_map_metadata(merged, map_data)

        self.map_meta = meta
        self.map_image_path = img_path
        self._spawn_points = self._extract_spawn_points(map_data, meta)
        random_spawn_cfg = merged.get("random_spawn")
        if random_spawn_cfg is None:
            random_spawn_cfg = merged.get("spawn_random")
        allow_reuse_flag = False
        if isinstance(random_spawn_cfg, Mapping):
            enabled_val = random_spawn_cfg.get("enabled", True)
            self._random_spawn_enabled = bool(enabled_val) and bool(self._spawn_points)
            allow_reuse_flag = bool(random_spawn_cfg.get("allow_reuse", False))
        else:
            self._random_spawn_enabled = bool(random_spawn_cfg) and bool(self._spawn_points)
            allow_reuse_flag = bool(merged.get("random_spawn_allow_reuse", False))
        if not self._spawn_points:
            self._random_spawn_enabled = False
        self._random_spawn_allow_reuse = allow_reuse_flag
        self._spawn_point_names: List[str] = sorted(self._spawn_points.keys())
        self._last_spawn_selection: Dict[str, str] = {}

        R = float(meta.get("resolution", 1.0))
        x0, y0, _ = meta.get('origin', (0.0, 0.0, 0.0))
        x_min = x0
        x_max = x0 + width * R
        y_min = y0
        y_max = y0 + height * R

        self._build_observation_spaces(x_min, x_max, y_min, y_max)

        # stateful observations for rendering
        self.render_obs = None
        self._reward_ring_config: Optional[Dict[str, Any]] = None
        self._reward_ring_focus_agent: Optional[str] = None
        self._reward_ring_target: Optional[str] = None
        self._reward_ring_dirty: bool = False
        self._reward_ring_target_dirty: bool = False
        self._render_metrics_payload: Optional[Dict[str, Any]] = None
        self._render_metrics_dirty: bool = False
        self._render_ticker: deque[str] = deque(maxlen=64)
        self._render_ticker_dirty: bool = False
        self._render_wrapped_obs: Dict[str, np.ndarray] = {}
        default_lidar_skip = int(merged.get("lidar_beams", self._lidar_beam_count))
        if default_lidar_skip < 0:
            default_lidar_skip = 0
        self._render_lidar_skip_default = default_lidar_skip
        self._render_lidar_skip: Dict[str, int] = {aid: default_lidar_skip for aid in self.possible_agents}
        self._render_callbacks: List[Callable[[EnvRenderer], None]] = []

        self._single_action_space = spaces.Box(
            low=np.array([self.params["s_min"], self.params["v_min"]], dtype=np.float32),
            high=np.array([self.params["s_max"], self.params["v_max"]], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_spaces = {
            aid: self._single_action_space for aid in self.possible_agents
        }

    def _configure_rendering(self, cfg: Mapping[str, Any]) -> None:
        self.render_mode = cfg.get("render_mode", "human")
        self.metadata = {"render_modes": ["human", "rgb_array"], "name": "F110ParallelEnv"}
        self.renderer: Optional[EnvRenderer] = None
        headless_env = str(os.environ.get("PYGLET_HEADLESS", "")).lower()
        self._headless = pyglet.options.get("headless", False) or headless_env in {"1", "true", "yes", "on"}
        mode = (self.render_mode or "").lower()
        self._collect_render_data = mode == "rgb_array" or (mode == "human" and not self._headless)

        render_user_override = bool(cfg.pop("_centerline_render_user_override", False))
        feature_user_override = bool(cfg.pop("_centerline_features_user_override", False))
        autoload_user_override = bool(cfg.pop("_centerline_autoload_user_override", False))

        render_cfg_value = cfg.get("centerline_render")
        features_cfg_value = cfg.get("centerline_features")
        autoload_cfg_value = cfg.get("centerline_autoload")

        render_cfg = render_cfg_value if render_user_override else None
        features_cfg = features_cfg_value if feature_user_override else None
        autoload_cfg = autoload_cfg_value if autoload_user_override else None

        self._centerline_render_auto = not render_user_override
        self._centerline_feature_auto = not feature_user_override
        self._centerline_autoload_auto = not autoload_user_override

        self.centerline_render_enabled = bool(render_cfg) if render_cfg is not None else False
        self._centerline_feature_requested = bool(features_cfg) if features_cfg is not None else False
        self.centerline_features_enabled = self._centerline_feature_requested

        self.centerline_points: Optional[np.ndarray] = None
        self.centerline_path: Optional[Path] = None
        self.centerline_render_progress = self._normalize_progress_fractions(
            cfg.get("centerline_render_progress")
        )
        spacing_value = cfg.get("centerline_render_spacing")
        try:
            self.centerline_render_spacing = max(float(spacing_value), 0.0)
        except (TypeError, ValueError):
            self.centerline_render_spacing = 0.0
        raw_connect = cfg.get("centerline_render_connect")
        if raw_connect is None:
            self.centerline_render_connect = not bool(self.centerline_render_progress)
        else:
            self.centerline_render_connect = bool(raw_connect)
        self._render_centerline_points: Optional[np.ndarray] = None

    def _configure_basic_environment(self, cfg: Mapping[str, Any]) -> None:
        self.seed = int(cfg.get("seed", 42))
        self.rng = np.random.default_rng(self.seed)
        self.max_steps = int(cfg.get("max_steps", 5000))
        self.n_agents = int(cfg.get("n_agents", 2))
        self._central_state_keys = (
            "poses_x",
            "poses_y",
            "poses_theta",
            "linear_vels_x",
            "linear_vels_y",
            "ang_vels_z",
            "collisions",
        )
        self._central_state_dim = self.n_agents * len(self._central_state_keys)
        self.possible_agents = [f"car_{i}" for i in range(self.n_agents)]
        self._agent_id_to_index = {aid: idx for idx, aid in enumerate(self.possible_agents)}
        self.agents = self.possible_agents.copy()

    def _resolve_integrator(self, cfg: Mapping[str, Any]) -> str:
        integrator_cfg = cfg.get("integrator", Integrator.RK4)
        if isinstance(integrator_cfg, Integrator):
            integrator_name = integrator_cfg.value
        else:
            integrator_name = str(integrator_cfg)
        integrator_name = integrator_name.strip()
        if integrator_name.lower() == "rk4":
            return "RK4"
        if integrator_name.lower() == "euler":
            return "Euler"
        return "RK4"

    @staticmethod
    def _normalize_map_identifier(identifier: Optional[Any]) -> Optional[str]:
        if identifier is None:
            return None
        identifier = str(identifier)
        return identifier if Path(identifier).suffix else f"{identifier}.yaml"

    def _configure_map_paths(self, cfg: Mapping[str, Any], map_data: Optional[Any]) -> None:
        map_dir_value = cfg.get("map_dir")
        if map_dir_value is not None:
            self.map_dir = Path(map_dir_value)
        elif map_data is not None:
            self.map_dir = Path(map_data.yaml_path).parent  # type: ignore[attr-defined]
        else:
            self.map_dir = Path.cwd()

        map_ext_value = cfg.get("map_ext")
        if map_ext_value is not None:
            self.map_ext = map_ext_value
        elif map_data is not None:
            self.map_ext = map_data.image_path.suffix or ".png"  # type: ignore[attr-defined]
        else:
            self.map_ext = ".png"

        raw_map_name = cfg.get("map")
        raw_map_yaml = cfg.get("map_yaml")
        self.map_name = self._normalize_map_identifier(raw_map_name)
        self.map_yaml = self._normalize_map_identifier(raw_map_yaml)

        if self.map_name is None and self.map_yaml is not None:
            self.map_name = self.map_yaml
        elif self.map_yaml is None and self.map_name is not None:
            self.map_yaml = self.map_name

        self.map_path = (self.map_dir / f"{self.map_name}").resolve()
        self.yaml_path = (self.map_dir / f"{self.map_yaml}").resolve()

    def _configure_vehicle_params(self, cfg: Mapping[str, Any]) -> Dict[str, float]:
        base_vehicle_params = _default_vehicle_params()
        vehicle_params = cfg.get("vehicle_params")
        if vehicle_params is None:
            vehicle_params = cfg.get("params")
        if vehicle_params is not None:
            if not isinstance(vehicle_params, Mapping):
                raise TypeError("env.vehicle_params must be a mapping")
            overrides = {str(key): float(value) for key, value in vehicle_params.items()}
            base_vehicle_params.update(overrides)
        return base_vehicle_params

    def _load_map_metadata(
        self,
        cfg: Mapping[str, Any],
        map_data: Optional[Any],
    ) -> Tuple[Dict[str, Any], Path, Tuple[int, int]]:
        meta = cfg.get("map_meta")
        if meta is None and map_data is not None:
            meta = dict(map_data.metadata)  # type: ignore[attr-defined]
        elif isinstance(meta, Mapping):
            meta = dict(meta)
        if meta is None:
            with open(self.map_path, "r") as f:
                meta = yaml.safe_load(f) or {}

        preloaded_image_path = cfg.get("map_image_path")
        if preloaded_image_path is None and map_data is not None:
            preloaded_image_path = map_data.image_path  # type: ignore[attr-defined]
        image_rel = meta.get("image")
        if preloaded_image_path is not None:
            img_path = Path(preloaded_image_path).resolve()
        elif image_rel:
            img_path = (self.map_path.parent / image_rel).resolve()
        else:
            img_filename = cfg.get("map_image")
            if img_filename is not None:
                img_path = (self.map_dir / img_filename).resolve()
            elif map_data is not None:
                img_path = Path(map_data.image_path).resolve()  # type: ignore[attr-defined]
            else:
                img_path = self.map_path.with_suffix(self.map_ext)

        image_size = cfg.get("map_image_size")
        if image_size is None and map_data is not None:
            image_size = map_data.image_size  # type: ignore[attr-defined]
        if image_size is not None:
            width, height = map(int, image_size)
        else:
            with Image.open(img_path) as img:
                width, height = img.size

        return meta, img_path, (width, height)

    @staticmethod
    def _extract_spawn_points(
        map_data: Optional[Any],
        metadata: Mapping[str, Any],
    ) -> Dict[str, np.ndarray]:
        spawn_points: Dict[str, np.ndarray] = {}
        if map_data is not None:
            candidate = getattr(map_data, "spawn_points", None)
            if isinstance(candidate, Mapping):
                for name, value in candidate.items():
                    arr = np.asarray(value, dtype=np.float32)
                    if arr.ndim == 1 and arr.shape[0] >= 2:
                        spawn_points[str(name)] = arr
        if spawn_points:
            return spawn_points

        annotations = metadata.get("annotations", {})
        if isinstance(annotations, Mapping):
            points = annotations.get("spawn_points")
        else:
            points = None
        if isinstance(points, Mapping):
            iterable = points.items()
        elif isinstance(points, list):
            iterable = ((entry.get("name"), entry.get("pose")) for entry in points if isinstance(entry, Mapping))
        else:
            iterable = []

        for name, value in iterable:
            if name is None:
                continue
            arr = np.asarray(value, dtype=np.float32)
            if arr.ndim == 1 and arr.shape[0] >= 2:
                spawn_points[str(name)] = arr
        return spawn_points

    def _sample_random_spawn(self) -> Optional[Tuple[Dict[str, str], np.ndarray]]:
        if not self._random_spawn_enabled or not self._spawn_point_names:
            return None

        agent_ids = self.possible_agents
        count = len(agent_ids)
        if count == 0:
            return None

        pool = self._spawn_point_names
        replace = self._random_spawn_allow_reuse or len(pool) < count

        rng = getattr(self, "rng", None)
        if rng is None:
            rng = np.random.default_rng(self.seed)
            self.rng = rng

        selected_names = rng.choice(pool, size=count, replace=replace)
        if isinstance(selected_names, np.ndarray):
            selected_list = [str(name) for name in selected_names.tolist()]
        else:
            selected_list = [str(name) for name in selected_names]

        pose_rows: List[np.ndarray] = []
        spawn_mapping: Dict[str, str] = {}
        for idx, name in enumerate(selected_list):
            raw = self._spawn_points.get(name)
            if raw is None:
                continue
            if raw.shape[0] < 3:
                pose = np.zeros(3, dtype=np.float32)
                pose[: raw.shape[0]] = raw
            else:
                pose = np.asarray(raw[:3], dtype=np.float32)
            pose_rows.append(pose)
            if idx < len(agent_ids):
                spawn_mapping[agent_ids[idx]] = name

        if len(pose_rows) != count:
            return None

        poses = np.stack(pose_rows, axis=0)
        return spawn_mapping, poses

    def action_space(self, agent: str):
        return self.action_spaces[agent]

    def _update_state(self, obs_dict):
        self.state_buffers.update(obs_dict)

    def _refresh_render_observations(self, obs: Dict[str, Dict[str, Any]]) -> None:
        if not self._collect_render_data:
            self.render_obs = {}
            return

        render_obs: Dict[str, Dict[str, Any]] = {}
        agent_index = self._agent_id_to_index
        for aid in self.agents:
            idx = agent_index[aid]
            entry = {
                "poses_x": float(self.poses_x[idx]),
                "poses_y": float(self.poses_y[idx]),
                "poses_theta": float(self.poses_theta[idx]),
                "pose": np.array([
                    float(self.poses_x[idx]),
                    float(self.poses_y[idx]),
                    float(self.poses_theta[idx])
                ], dtype=np.float32),
                "lap_time": float(self.lap_times[idx]),
                "lap_count": int(self.lap_counts[idx]),
                "collision": bool(self.collisions[idx]),
            }
            agent_obs = obs.get(aid) if obs is not None else None
            if agent_obs is not None:
                scan_entry = agent_obs.get("scans")
                if scan_entry is not None:
                    entry["scans"] = scan_entry
                    self._render_lidar_skip[aid] = self._render_lidar_skip_default
                target_collision_val = agent_obs.get("target_collision")
                if target_collision_val is not None:
                    try:
                        entry["target_collision"] = bool(target_collision_val)
                    except Exception:
                        entry["target_collision"] = False
                components: Dict[str, Any] = {}
                for key, value in agent_obs.items():
                    if key in {"scans", "lidar", "state"}:
                        continue
                    if key in entry and key not in {"collision", "target_collision"}:
                        continue
                    cleaned: Any
                    if key == "relative_sector":
                        arr = np.asarray(value, dtype=np.float32).flatten()
                        sector_flags: Dict[str, bool] = {}
                        for idx, name in enumerate(_SECTOR_NAMES):
                            flag = False
                            if idx < arr.size:
                                flag = bool(arr[idx] > 0.5)
                            sector_flags[name] = flag
                        cleaned = sector_flags
                    elif isinstance(value, np.ndarray):
                        cleaned = value.astype(np.float32, copy=False).flatten()
                        if cleaned.size > 6:
                            cleaned = cleaned[:6]
                        cleaned = cleaned.tolist()
                    elif isinstance(value, (np.bool_, bool)):
                        cleaned = bool(value)
                    elif isinstance(value, (np.floating, np.integer)):
                        cleaned = float(value)
                    else:
                        cleaned = value
                    components[key] = cleaned
                if components:
                    entry["obs_components"] = components
            snapshot = self._compute_relative_snapshot(aid)
            if snapshot is not None:
                entry["relative"] = snapshot
            wrapped_vector = self._render_wrapped_obs.get(aid)
            if wrapped_vector is not None:
                entry["wrapped_obs"] = np.asarray(wrapped_vector, dtype=np.float32)
                entry["wrapped_skip"] = int(self._render_lidar_skip.get(aid, self._render_lidar_skip_default))
            render_obs[aid] = entry

        self.render_obs = render_obs

    def _compute_relative_snapshot(self, agent_id: str) -> Optional[Dict[str, Any]]:
        agent_idx = self._agent_id_to_index.get(agent_id)
        target_idx = self._agent_target_index.get(agent_id)
        if agent_idx is None or target_idx is None:
            return None
        if not (0 <= agent_idx < len(self.poses_x)) or not (0 <= target_idx < len(self.poses_x)):
            return None

        agent_x = float(self.poses_x[agent_idx])
        agent_y = float(self.poses_y[agent_idx])
        agent_heading = float(self.poses_theta[agent_idx]) if agent_idx < len(self.poses_theta) else 0.0

        target_x = float(self.poses_x[target_idx])
        target_y = float(self.poses_y[target_idx])

        dx = target_x - agent_x
        dy = target_y - agent_y
        distance = float(math.hypot(dx, dy))

        if dx == 0.0 and dy == 0.0:
            sector = "front"
        else:
            rel_angle = math.degrees(math.atan2(dy, dx) - agent_heading)
            sector = _sector_from_angle(rel_angle)

        cfg = self._reward_ring_config or {}
        preferred = float(cfg.get("preferred_radius", 0.0) or 0.0)
        inner = float(cfg.get("inner_tolerance", 0.0) or 0.0)
        outer = float(cfg.get("outer_tolerance", 0.0) or 0.0)
        falloff = str(cfg.get("falloff", "linear") or "linear").lower()

        gain = _radial_gain(distance, preferred, inner, outer, falloff)
        sector_active = gain > 0.0

        if preferred > 0.0 or inner > 0.0 or outer > 0.0:
            lower = max(preferred - inner, 0.0)
            upper = preferred + outer
            in_ring = float(lower) <= distance <= float(upper)
        else:
            in_ring = sector_active

        weights_raw = cfg.get("weights")
        sector_weight = 0.0
        if isinstance(weights_raw, dict):
            try:
                sector_weight = float(weights_raw.get(sector, 0.0))
            except (TypeError, ValueError):
                sector_weight = 0.0
        reward_sector = sector_active and sector_weight > 0.0

        sector_code = "".join(word[0].upper() for word in sector.split("_")) if sector else "--"

        return {
            "distance": distance,
            "sector": sector,
            "sector_code": sector_code,
            "sector_active": bool(sector_active),
            "in_ring": bool(in_ring),
            "sector_weight": sector_weight,
            "reward_sector": bool(reward_sector),
        }

    def update_render_metrics(
        self,
        phase: str,
        metrics: Mapping[str, Any],
        *,
        step: Optional[float] = None,
    ) -> None:
        """
        Cache the latest logger metrics so the renderer HUD can surface them.
        """
        if not phase or metrics is None:
            return
        try:
            snapshot = dict(metrics)
        except Exception:
            return
        payload: Dict[str, Any] = {
            "phase": str(phase).strip().lower(),
            "metrics": snapshot,
            "timestamp": time.time(),
        }
        if step is not None:
            payload["step"] = float(step)
        self._render_metrics_payload = payload
        self._render_metrics_dirty = True

    def update_render_wrapped_observations(self, wrapped: Mapping[str, np.ndarray]) -> None:
        if not wrapped:
            return
        store: Dict[str, np.ndarray] = {}
        for agent_id, vector in wrapped.items():
            if vector is None:
                continue
            try:
                array = np.asarray(vector, dtype=np.float32)
            except Exception:
                continue
            store[str(agent_id)] = array.copy()
        if not store:
            return
        self._render_wrapped_obs = store

    def append_render_ticker(
        self,
        agent_id: str,
        *,
        step: int,
        reward: float,
        components: Optional[Mapping[str, Any]] = None,
    ) -> None:
        snapshot = self._compute_relative_snapshot(agent_id)
        if snapshot is None:
            return

        relative_reward = None
        if components:
            value = components.get("relative_position")
            try:
                relative_reward = float(value)
            except (TypeError, ValueError):
                relative_reward = None

        total_reward = float(reward)
        distance = snapshot["distance"]
        sector_code = snapshot.get("sector_code", "--")
        sector_active = snapshot.get("sector_active", False)
        in_ring = snapshot.get("in_ring", False)
        reward_sector = snapshot.get("reward_sector", False)

        rel_text = f"{relative_reward:+.3f}" if relative_reward is not None else "--"
        line = (
            f"{int(step):04d} {agent_id} "
            f"r={total_reward:+.3f} "
            f"rel={rel_text} "
            f"d={distance:.2f} "
            f"{sector_code:<2} "
            f"S={1 if sector_active else 0} "
            f"R={1 if in_ring else 0} "
            f"W={1 if reward_sector else 0}"
        )

        self._render_ticker.appendleft(line)
        self._render_ticker_dirty = True

    def configure_reward_ring(self, config: Optional[Dict[str, Any]], *, agent_id: Optional[str] = None) -> None:
        if config is None:
            self._reward_ring_config = None
            self._reward_ring_focus_agent = None
            self._reward_ring_target = None
            self._reward_ring_dirty = True
            self._reward_ring_target_dirty = True
            return

        stored = {
            "preferred_radius": max(float(config.get("preferred_radius", 0.0)), 0.0),
            "inner_tolerance": max(float(config.get("inner_tolerance", 0.0)), 0.0),
            "outer_tolerance": max(float(config.get("outer_tolerance", 0.0)), 0.0),
            "segments": max(int(config.get("segments", 96) or 96), 8),
        }
        for key in ("fill_color", "border_color", "preferred_color"):
            if key in config and isinstance(config[key], (list, tuple)):
                stored[key] = tuple(float(component) for component in config[key])
        falloff_val = config.get("falloff")
        if falloff_val is not None:
            stored["falloff"] = str(falloff_val).lower()
        weights_val = config.get("weights")
        if isinstance(weights_val, Mapping):
            safe_weights: Dict[str, float] = {}
            for name, value in weights_val.items():
                if value is None:
                    continue
                try:
                    safe_weights[str(name)] = float(value)
                except (TypeError, ValueError):
                    continue
            if safe_weights:
                stored["weights"] = safe_weights

        if self._reward_ring_config != stored or self._reward_ring_focus_agent != agent_id:
            self._reward_ring_config = stored
            self._reward_ring_focus_agent = agent_id
            self._reward_ring_dirty = True
            self._reward_ring_target_dirty = True

    def update_reward_ring_target(self, agent_id: str, target_id: Optional[str]) -> None:
        if self._reward_ring_config is None:
            return

        focus = self._reward_ring_focus_agent
        if focus is not None and agent_id != focus:
            return

        normalized = str(target_id) if target_id else None
        if self._reward_ring_target != normalized:
            self._reward_ring_target = normalized
            self._reward_ring_target_dirty = True

    def _apply_reward_ring_to_renderer(self) -> None:
        if self.renderer is None:
            return

        if self._reward_ring_dirty:
            cfg = self._reward_ring_config
            if cfg:
                renderer_payload: Dict[str, Any] = {
                    "enabled": True,
                    "preferred_radius": cfg["preferred_radius"],
                    "inner_tolerance": cfg["inner_tolerance"],
                    "outer_tolerance": cfg["outer_tolerance"],
                    "segments": cfg.get("segments", 96),
                }
                for key in ("fill_color", "border_color", "preferred_color"):
                    if key in cfg:
                        renderer_payload[key] = cfg[key]
                self.renderer.configure_reward_ring(**renderer_payload)
            else:
                self.renderer.configure_reward_ring(enabled=False)
            self._reward_ring_dirty = False

        if self._reward_ring_target_dirty:
            self.renderer.set_reward_ring_target(self._reward_ring_target)
            self._reward_ring_target_dirty = False

    def _update_start_from_poses(self, poses: np.ndarray):
        if poses is None or poses.size == 0:
            return
        self.start_poses = np.asarray(poses, dtype=np.float32)
        self.start_state.apply_start_poses(self.start_poses)
        self.start_state.reset()

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            seed_value = int(seed)
            self.seed = seed_value
            self.rng = np.random.default_rng(seed_value)
            reseed_sim = getattr(self.sim, "reseed", None)
            if callable(reseed_sim):
                reseed_sim(seed_value)
        self.agents = self.possible_agents.copy()
        self._elapsed_steps = 0
        self.current_time = 0.0
        self.start_state.reset()
        self.state_buffers.reset()
        self._render_ticker.clear()
        self._render_ticker_dirty = True
        self._render_wrapped_obs.clear()
        poses = None
        spawn_mapping: Dict[str, str] = {}
        if self.renderer is not None:
            self.renderer.reset_state()
            self._update_renderer_centerline()
            self._reward_ring_dirty = True
            self._reward_ring_target_dirty = True
    # Case 1: Explicit override via options
        if options is not None:
            if isinstance(options, dict) and "poses" in options:
                poses = np.array(options["poses"], dtype=np.float32)
                if poses.ndim == 1:
                    poses = np.expand_dims(poses, axis=0)
                self._update_start_from_poses(poses)
                spawn_mapping = {}
        # Case 2: Default to config start_poses
        if poses is None and self._random_spawn_enabled:
            sampled = self._sample_random_spawn()
            if sampled is not None:
                spawn_mapping, sampled_poses = sampled
                self._update_start_from_poses(sampled_poses)
                poses = self.start_poses
        if poses is None and hasattr(self, "start_poses") and len(self.start_poses) > 0:
            poses = self.start_poses
            spawn_mapping = {}
        self._last_spawn_selection = dict(spawn_mapping)

        # options: (N,3) poses (x,y,theta). If None, caller must set internally.
        # poses = options if options is not None else np.zeros((self.n_agents, 3), dtype=np.float32)
        obs_joint = self.sim.reset(poses)
        obs = self._split_obs(obs_joint)
        self._attach_central_state(obs, obs_joint)
        self._update_state(obs_joint)
        self._refresh_render_observations(obs)

        infos = {aid: {} for aid in self.agents}
        if spawn_mapping:
            for aid, name in spawn_mapping.items():
                if aid in infos:
                    infos[aid]["spawn_point"] = name
        return obs, infos

    def step(self, actions: Dict[str, np.ndarray]):

        joint = np.zeros((self.n_agents, 2), dtype=np.float32)
        agent_index = self._agent_id_to_index
        for aid in self.agents:
            if aid in actions:
                joint[agent_index[aid]] = np.asarray(actions[aid], dtype=np.float32)
                

        obs_joint = self.sim.step(joint)
        obs = self._split_obs(obs_joint)
        self._attach_central_state(obs, obs_joint)
        self._update_state(obs_joint)
        self._refresh_render_observations(obs)

        self.current_time += self.timestep
        lap_completion = self.start_state.update_progress(
            self.poses_x,
            self.poses_y,
            self.linear_vels_x_curr,
            self.linear_vels_y_curr,
            self.current_time,
            self.target_laps,
        )
        # simple per-step reward (customize as needed)
        rewards = {aid: float(self.timestep * 0.0) for aid in self.agents}

        # terminations/truncations
        collisions = obs_joint["collisions"]
        terminations = build_terminations(
            self.possible_agents,
            collisions,
            lap_completion,
            self.terminate_on_collision,
        )

        trunc_flag = (self.max_steps > 0) and (self._elapsed_steps + 1 >= self.max_steps)
        truncations = {aid: bool(trunc_flag) for aid in self.possible_agents}
        infos = {aid: {} for aid in self.possible_agents}

        # advance and cull finished agents
        self._elapsed_steps += 1
        self.agents = [aid for aid in self.possible_agents if not (terminations[aid] or truncations[aid])]

        return obs, rewards, terminations, truncations, infos

    def update_map(self, map_path, map_ext):
        path = Path(map_path).resolve()

        self.map_dir = path.parent
        self.map_ext = map_ext
        self.map_name = path.name
        self.map_yaml = path.name
        self.map_path = path
        self.yaml_path = path

        with open(path, "r") as f:
            meta = yaml.safe_load(f)

        image_rel = meta.get("image")
        if image_rel:
            img_path = (path.parent / image_rel).resolve()
        else:
            img_path = path.with_suffix(map_ext)
        self.map_meta = meta
        self.map_image_path = img_path

        self.sim.set_map(str(path), map_ext)

        if self.renderer is not None:
            self.renderer.update_map(
                str(path.with_suffix("")),
                map_ext,
                map_meta=self.map_meta,
                map_image_path=self.map_image_path,
            )
            self._update_renderer_centerline()

    def update_params(self, params, index=-1):

        self.sim.update_params(params, agent_idx=index)

    def render(self):
        assert self.render_mode in ["human", "rgb_array"]

        self._collect_render_data = True
        if self._headless and self.render_mode == "human":
            # Nothing to do when headless; keep API contract intact.
            return None

        if self.renderer is None:
            self.renderer = EnvRenderer(WINDOW_W, WINDOW_H,
                                        lidar_fov=4.7,
                                        max_range=30.0)
            # use self.map_path (without extension) and self.map_ext
            self.renderer.update_map(
                str(self.map_path.with_suffix("")),
                self.map_ext,
                map_meta=self.map_meta,
                map_image_path=self.map_image_path,
                centerline_points=self._render_centerline_points if self.centerline_render_enabled else None,
                centerline_connect=self.centerline_render_connect,
            )
            self._reward_ring_dirty = True
            self._reward_ring_target_dirty = True

        self._apply_reward_ring_to_renderer()

        if self.renderer is not None and self._render_metrics_payload is not None and self._render_metrics_dirty:
            payload = self._render_metrics_payload
            self.renderer.update_metrics(
                phase=payload.get("phase", ""),
                metrics=payload.get("metrics", {}),
                step=payload.get("step"),
                timestamp=payload.get("timestamp"),
            )
            self._render_metrics_dirty = False

        if self.renderer is not None and self._render_ticker_dirty:
            lines = list(self._render_ticker)
            self.renderer.update_ticker(lines[:16])
            self._render_ticker_dirty = False

        if self.render_obs:
            self.renderer.update_obs(self.render_obs)
        if self.renderer is not None and self._render_callbacks:
            for callback in list(self._render_callbacks):
                try:
                    callback(self.renderer)
                except Exception:
                    logger.exception("Render callback failed")

        self.renderer.dispatch_events()
        self.renderer.on_draw()
        self.renderer.flip()

        if self.render_mode == "human":
            time.sleep(0.005)
        elif self.render_mode == "rgb_array":
            buf = pyg_img.get_buffer_manager().get_color_buffer()
            w, h = buf.width, buf.height
            img = buf.get_image_data()
            data = img.get_data("RGB", -w * 3)
            frame = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3).copy()
            return frame

    def add_render_callback(self, callback: Callable[[EnvRenderer], None]) -> None:
        if not callable(callback):
            raise TypeError("Render callback must be callable")
        if callback not in self._render_callbacks:
            self._render_callbacks.append(callback)

    def clear_render_callbacks(self) -> None:
        self._render_callbacks.clear()

    @staticmethod
    def _normalize_progress_fractions(raw: Optional[Any]) -> Tuple[float, ...]:
        if raw is None:
            return ()
        if isinstance(raw, (float, int)):
            candidates: List[Any] = [raw]
        elif isinstance(raw, str):
            candidates = [raw]
        else:
            try:
                candidates = list(raw)  # type: ignore[arg-type]
            except TypeError:
                candidates = [raw]

        fractions: List[float] = []
        for value in candidates:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(numeric):
                continue
            frac = numeric % 1.0
            if frac <= 0.0:
                continue
            fractions.append(frac)
        if not fractions:
            return ()
        return tuple(sorted(set(fractions)))

    def _build_render_centerline_points(self) -> Optional[np.ndarray]:
        base = self.centerline_points
        if base is None:
            return None
        if base.ndim != 2 or base.shape[0] == 0:
            return base
        fractions: List[float] = list(self.centerline_render_progress)
        if self.centerline_render_spacing > 0.0:
            spacing_fracs = progress_from_spacing(base, self.centerline_render_spacing)
            if spacing_fracs:
                fractions.extend(spacing_fracs)
        if not fractions:
            return base
        unique = sorted(set(frac for frac in fractions if 0.0 < frac < 1.0))
        if not unique:
            return base
        denom = max(base.shape[0] - 1, 1)
        indices = set()
        for frac in unique:
            idx = int(round(frac * denom)) % base.shape[0]
            indices.add(idx)
        if not indices:
            return base
        ordered = sorted(indices)
        return base[ordered]

    def _update_renderer_centerline(self) -> None:
        if self.renderer is None:
            return
        if self.centerline_render_enabled:
            self.renderer.update_centerline(
                self._render_centerline_points,
                connect=self.centerline_render_connect,
            )
        else:
            self.renderer.update_centerline(None)

    def register_centerline_usage(self, *, require_render: bool = False, require_features: bool = False) -> None:
        changed = False
        if require_features and self._centerline_feature_auto and not self.centerline_features_enabled:
            self.centerline_features_enabled = True
            self._centerline_feature_requested = True
            changed = True
        if require_render and self._centerline_render_auto and not self.centerline_render_enabled:
            self.centerline_render_enabled = True
            changed = True
        if changed:
            self._render_centerline_points = self._build_render_centerline_points()
            self._update_renderer_centerline()

    def set_centerline(self, centerline: Optional[np.ndarray], *, path: Optional[Path] = None) -> None:
        if centerline is not None:
            array = np.asarray(centerline, dtype=np.float32)
            array.setflags(write=False)
        else:
            array = None
        self.centerline_points = array
        self.centerline_path = path.resolve() if path is not None else None
        self.centerline_features_enabled = self._centerline_feature_requested and array is not None
        self._render_centerline_points = self._build_render_centerline_points()
        self._update_renderer_centerline()
    
    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def _build_observation_spaces(self, x_min: float, x_max: float, y_min: float, y_max: float) -> None:
        pose_low = np.array([x_min, y_min, -np.pi], dtype=np.float32)
        pose_high = np.array([x_max, y_max, np.pi], dtype=np.float32)
        v_min = float(self.params.get("v_min", -5.0))
        v_max = float(self.params.get("v_max", 20.0))
        vel_low = np.array([v_min, v_min], dtype=np.float32)
        vel_high = np.array([v_max, v_max], dtype=np.float32)

        accel_cap = float(self.params.get("a_max", 10.0))
        accel_low = np.array([-accel_cap, -accel_cap], dtype=np.float32)
        accel_high = np.array([accel_cap, accel_cap], dtype=np.float32)

        ang_cap = float(self.params.get("ang_vel_max", 10.0))

        lap_cap = float(getattr(self, "target_laps", 1))
        lap_low = np.array([0.0, 0.0], dtype=np.float32)
        lap_high = np.array([lap_cap, 1e5], dtype=np.float32)

        obs_spaces: Dict[str, gym.Space] = {}
        for aid in self.possible_agents:
            sensors = self._agent_sensor_spec.get(aid, DEFAULT_AGENT_SENSORS)
            components: Dict[str, gym.Space] = {}

            if "lidar" in sensors:
                components["lidar"] = spaces.Box(
                    low=0.0,
                    high=self.lidar_range,
                    shape=(self._lidar_beam_count,),
                    dtype=np.float32,
                )
            if "pose" in sensors:
                components["pose"] = spaces.Box(pose_low, pose_high, dtype=np.float32)
            if "velocity" in sensors:
                components["velocity"] = spaces.Box(vel_low, vel_high, dtype=np.float32)
            if "acceleration" in sensors:
                components["acceleration"] = spaces.Box(accel_low, accel_high, dtype=np.float32)
            if "angular_velocity" in sensors:
                components["angular_velocity"] = spaces.Box(
                    low=-ang_cap,
                    high=ang_cap,
                    shape=(),
                    dtype=np.float32,
                )
            if "target_pose" in sensors:
                components["target_pose"] = spaces.Box(pose_low, pose_high, dtype=np.float32)
            if "target_collision" in sensors:
                components["target_collision"] = spaces.Box(0.0, 1.0, shape=(), dtype=np.float32)
            if "lap" in sensors:
                components["lap"] = spaces.Box(lap_low, lap_high, dtype=np.float32)
            if "collision" in sensors:
                components["collision"] = spaces.Box(0.0, 1.0, shape=(), dtype=np.float32)

            components["state"] = spaces.Box(
                -np.inf,
                np.inf,
                shape=(self._central_state_dim,),
                dtype=np.float32,
            )

            obs_spaces[aid] = spaces.Dict(components)

        if obs_spaces:
            self.observation_spaces = obs_spaces
        else:
            self.observation_spaces = {
                aid: spaces.Dict(
                    {
                        "state": spaces.Box(
                            -np.inf,
                            np.inf,
                            shape=(self._central_state_dim,),
                            dtype=np.float32,
                        )
                    }
                )
                for aid in self.possible_agents
            }

    def _bind_state_views(self) -> None:
        """Expose state buffer arrays as legacy attributes expected by callers."""
        buffers = self.state_buffers
        self.poses_x = buffers.poses_x
        self.poses_y = buffers.poses_y
        self.poses_theta = buffers.poses_theta
        self.collisions = buffers.collisions
        self.linear_vels_x_prev = buffers.linear_vels_x_prev
        self.linear_vels_y_prev = buffers.linear_vels_y_prev
        self.angular_vels_prev = buffers.angular_vels_prev
        self.linear_vels_x_curr = buffers.linear_vels_x_curr
        self.linear_vels_y_curr = buffers.linear_vels_y_curr
        self.angular_vels_curr = buffers.angular_vels_curr

    def _central_state_tensor(self, joint: Dict[str, np.ndarray]) -> np.ndarray:
        n = self.n_agents
        central = np.zeros((self._central_state_dim,), dtype=np.float32)
        if n == 0:
            return central

        span = n
        offset = 0
        for key in self._central_state_keys:
            arr = joint.get(key)
            if arr is None:
                offset += span
                continue
            view = np.asarray(arr, dtype=np.float32).reshape(-1)
            if view.size >= span:
                central[offset:offset + span] = view[:span]
            else:
                central[offset:offset + view.size] = view
            offset += span
        return central

    def _attach_central_state(self, obs: Dict[str, Dict[str, np.ndarray]], joint: Dict[str, np.ndarray]) -> None:
        central_state = self._central_state_tensor(joint)
        for aid in self.possible_agents:
            if aid in obs:
                obs[aid]["state"] = central_state

    def apply_initial_speeds(self, speed_map: Mapping[str, float]) -> Optional[Dict[str, Dict[str, np.ndarray]]]:
        """Adjust simulator state to honour per-agent initial speed requests."""
        if not speed_map:
            return None
        updated = False
        for agent_id, raw_value in speed_map.items():
            idx = self._agent_id_to_index.get(agent_id)
            if idx is None:
                continue
            try:
                speed = float(raw_value)
            except (TypeError, ValueError):
                speed = 0.0
            self.sim.set_agent_speed(idx, speed)
            updated = True
        if not updated:
            return None

        joint = self.sim.current_observation()
        self._update_state(joint)
        obs = self._split_obs(joint)
        self._attach_central_state(obs, joint)
        self._refresh_render_observations(obs)
        return obs

    # helper: joint->per-agent dicts expected by PZ Parallel API
    def _split_obs(self, joint: Dict[str, np.ndarray]) -> Dict[str, Dict[str, np.ndarray]]:
        out: Dict[str, Dict[str, np.ndarray]] = {}

        scans = joint.get("scans")
        poses_x = joint.get("poses_x")
        poses_y = joint.get("poses_y")
        poses_theta = joint.get("poses_theta")
        linear_vels_x = joint.get("linear_vels_x")
        linear_vels_y = joint.get("linear_vels_y")
        ang_vels_z = joint.get("ang_vels_z")
        collisions = joint.get("collisions")

        lap_counts = getattr(self, "lap_counts", np.zeros((self.n_agents,), dtype=np.float32))
        lap_times = getattr(self, "lap_times", np.zeros((self.n_agents,), dtype=np.float32))

        timestep = max(self.timestep, 1e-6)

        for i, aid in enumerate(self.possible_agents):
            sensors = self._agent_sensor_spec.get(aid, DEFAULT_AGENT_SENSORS)
            agent_obs: Dict[str, np.ndarray] = {}
            target_idx = self._agent_target_index.get(aid)

            has_entry = isinstance(scans, np.ndarray) and i < scans.shape[0]

            if "lidar" in sensors:
                if has_entry:
                    lidar_reading = np.asarray(scans[i], dtype=np.float32)
                else:
                    lidar_reading = np.zeros((scans.shape[1] if isinstance(scans, np.ndarray) and scans.ndim == 2 else self._lidar_beam_count,), dtype=np.float32)
                agent_obs["lidar"] = lidar_reading
                agent_obs["scans"] = lidar_reading

            if "pose" in sensors:
                if isinstance(poses_x, np.ndarray) and i < poses_x.shape[0]:
                    agent_obs["pose"] = np.array([
                        np.float32(poses_x[i]),
                        np.float32(poses_y[i]),
                        np.float32(poses_theta[i]),
                    ], dtype=np.float32)
                else:
                    agent_obs["pose"] = np.zeros(3, dtype=np.float32)

            curr_vx = float(linear_vels_x[i]) if isinstance(linear_vels_x, np.ndarray) and i < linear_vels_x.shape[0] else 0.0
            curr_vy = float(linear_vels_y[i]) if isinstance(linear_vels_y, np.ndarray) and i < linear_vels_y.shape[0] else 0.0

            if "velocity" in sensors:
                agent_obs["velocity"] = np.array([curr_vx, curr_vy], dtype=np.float32)

            if "acceleration" in sensors:
                if self.state_buffers.velocity_initialized and i < self.linear_vels_x_curr.shape[0]:
                    prev_vx = float(self.linear_vels_x_curr[i])
                    prev_vy = float(self.linear_vels_y_curr[i])
                    ax = (curr_vx - prev_vx) / timestep
                    ay = (curr_vy - prev_vy) / timestep
                else:
                    ax = 0.0
                    ay = 0.0
                agent_obs["acceleration"] = np.array([ax, ay], dtype=np.float32)

            if "angular_velocity" in sensors:
                if isinstance(ang_vels_z, np.ndarray) and i < ang_vels_z.shape[0]:
                    agent_obs["angular_velocity"] = np.float32(ang_vels_z[i])
                else:
                    agent_obs["angular_velocity"] = np.float32(0.0)

            if "target_pose" in sensors:
                if target_idx is not None and isinstance(poses_x, np.ndarray) and target_idx < poses_x.shape[0]:
                    agent_obs["target_pose"] = np.array([
                        np.float32(poses_x[target_idx]),
                        np.float32(poses_y[target_idx]),
                        np.float32(poses_theta[target_idx]),
                    ], dtype=np.float32)
                else:
                    agent_obs["target_pose"] = np.zeros(3, dtype=np.float32)

            if "target_collision" in sensors:
                if target_idx is not None and isinstance(collisions, np.ndarray) and target_idx < collisions.shape[0]:
                    target_col_val = float(collisions[target_idx])
                else:
                    target_col_val = 0.0
                agent_obs["target_collision"] = np.float32(target_col_val)

            if "lap" in sensors:
                lap_count_val = float(lap_counts[i]) if i < len(lap_counts) else 0.0
                lap_time_val = float(lap_times[i]) if i < len(lap_times) else 0.0
                agent_obs["lap"] = np.array([lap_count_val, lap_time_val], dtype=np.float32)

            if "collision" in sensors:
                if isinstance(collisions, np.ndarray) and i < collisions.shape[0]:
                    col_val = float(collisions[i])
                else:
                    col_val = float(self.collisions[i]) if i < self.collisions.shape[0] else 0.0
                agent_obs["collision"] = np.float32(col_val)

            out[aid] = agent_obs

        return out
    
