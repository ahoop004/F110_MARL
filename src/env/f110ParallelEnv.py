from collections import deque
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Tuple, Sequence
import gymnasium as gym
from gymnasium import spaces
import yaml
from PIL import Image

from pettingzoo import ParallelEnv

# base classes
from src.physics import Simulator, Integrator
# Lazy import to avoid pyglet initialization on HPC without display
# from src.render import EnvRenderer  # Moved to render() method
from src.env.start_pose_state import StartPoseState
from src.env.collision import build_terminations
from src.env.state_buffer import StateBuffers
from src.utils.centerline import progress_from_spacing, centerline_arc_length, centerline_pose
from src.utils.map_loader import MapLoader

# Type checking only imports (don't execute at runtime)
if TYPE_CHECKING:
    from src.render import EnvRenderer


def _default_vehicle_params() -> Dict[str, float]:
    """Default vehicle dynamics parameters used across experiments."""
    return {
        "mu": 1.0489,
        "C_Sf": 4.718,
        "C_Sr": 5.4562,
        "lf": 0.15875,
        "lr": 0.17145,
        "h": 0.074,
        "m": 3.74,
        "I": 0.04712,
        "s_min": -0.4189,
        "s_max": 0.4189,
        "sv_min": -3.2,
        "sv_max": 3.2,
        "v_switch": 7.319,
        "a_max": 9.51,
        "v_min": -5.0,
        "v_max": 10.0,
        "width": 0.225,
        "length": 0.32,
    }


# others
import numpy as np
import os
import time
import math
import logging

# gl - Lazy import for headless system compatibility
# Pyglet will be imported only when rendering is actually needed
_PYGLET_AVAILABLE = None
pyglet = None
gl = None
pyg_img = None

def _ensure_pyglet():
    """Lazy load pyglet modules. Returns True if successful, False if not available."""
    global _PYGLET_AVAILABLE, pyglet, gl, pyg_img
    if _PYGLET_AVAILABLE is not None:
        return _PYGLET_AVAILABLE

    try:
        import pyglet as _pyglet
        pyglet = _pyglet
        pyglet.options['debug_gl'] = False
        from pyglet import gl as _gl
        from pyglet import image as _pyg_img
        gl = _gl
        pyg_img = _pyg_img
        _PYGLET_AVAILABLE = True
        return True
    except Exception as e:
        _PYGLET_AVAILABLE = False
        logger.warning(f"Pyglet not available (headless system?): {e}")
        logger.warning("Rendering will be disabled. This is normal for HPC/headless systems.")
        return False

# constants
from wrappers.observation import _sector_from_angle, _radial_gain, _SECTOR_NAMES

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
        self._map_loader = MapLoader(base_dir=Path.cwd())
        self._map_root = Path(merged.get("map_dir") or merged.get("map_root") or Path.cwd()).resolve()
        self._map_cycle_mode = str(merged.get("map_cycle", "")).strip().lower()
        self._map_pick_mode = str(merged.get("map_pick", "first")).strip().lower()
        self._map_epoch_shuffle = bool(merged.get("epoch_shuffle", False))
        self._map_split_mode = str(merged.get("map_split_mode", "train")).strip().lower()
        self._map_bundles_train = list(merged.get("map_bundles_train") or [])
        self._map_bundles_eval = list(merged.get("map_bundles_eval") or [])
        self._map_bundles = list(merged.get("map_bundles") or [])
        if not self._map_bundles_train and self._map_bundles:
            self._map_bundles_train = list(self._map_bundles)
        if not self._map_bundles_eval and self._map_bundles:
            self._map_bundles_eval = []
        self._map_bundle_active = merged.get("map_bundle_active") or merged.get("map_bundle")
        self._map_cycle_indices = {"train": 0, "eval": 0}
        self._map_cycle_order = {
            "train": list(self._map_bundles_train),
            "eval": list(self._map_bundles_eval),
        }
        if self._map_epoch_shuffle:
            for key, order in self._map_cycle_order.items():
                if order:
                    self.rng.shuffle(order)
                    self._map_cycle_indices[key] = 0
        self.walls = getattr(map_data, "walls", None) if map_data is not None else None
        self.walls_path = getattr(map_data, "walls_path", None) if map_data is not None else None
        self._wall_points: Optional[np.ndarray] = None
        self._refresh_wall_points()
        self._track_mask = getattr(map_data, "track_mask", None) if map_data is not None else None
        self.spawn_policy = merged.get("spawn_policy")
        self.spawn_centerline_cfg = merged.get("spawn_centerline", {}) or {}
        self.spawn_offsets_cfg = merged.get("spawn_offsets", {}) or {}
        self.spawn_target_cfg = merged.get("spawn_target", {}) or {}
        self.spawn_ego_cfg = merged.get("spawn_ego", {}) or {}
        self._track_threshold = merged.get("track_threshold")
        self._track_inverted = bool(merged.get("track_inverted", False))
        self._centerline_csv = merged.get("centerline_csv")
        self._walls_csv = merged.get("walls_csv")
        self._walls_autoload = bool(merged.get("walls_autoload", True))
        self._spawn_centerline_index = 0
        self._last_spawn_metadata: Dict[str, Any] = {}
        self.start_poses = np.array(merged.get("start_poses", []),dtype=np.float32)

        self.params = self._configure_vehicle_params(merged)
        
        self.lidar_beams = int(merged.get("lidar_beams", 1080))
        if self.lidar_beams <= 0:
            self.lidar_beams = 1080
        self.lidar_range = float(merged.get("lidar_range", 12.0))
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

        # Persistent collision tracking (like v1)
        # Once an agent collides, they stay collided for the episode
        self._collision_flags = np.zeros(self.n_agents, dtype=bool)
        self._collision_steps = np.full(self.n_agents, -1, dtype=np.int32)

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
        self._map_data = map_data
        self._spawn_points = self._extract_spawn_points(map_data, meta)
        random_spawn_cfg = merged.get("random_spawn")
        if random_spawn_cfg is None:
            random_spawn_cfg = merged.get("spawn_random")
        allow_reuse_flag = False
        self._random_spawn_cfg = random_spawn_cfg if isinstance(random_spawn_cfg, Mapping) else {}
        self._random_spawn_mode = "map"
        self._random_spawn_target_agent = None
        self._random_spawn_offsets: Dict[str, Any] = {
            "x_range": (0.0, 0.0),
            "y_range": (0.0, 0.0),
            "theta_range": (0.0, 0.0),
        }
        self._random_spawn_frame = "world"
        self._random_spawn_target_on_centerline = False
        self._random_spawn_min_separation = 0.0
        self._random_spawn_min_wall_distance = 0.0
        self._random_spawn_reject_offtrack = False
        self._random_spawn_max_attempts = 50
        if isinstance(random_spawn_cfg, Mapping):
            enabled_val = random_spawn_cfg.get("enabled", True)
            self._random_spawn_requested = bool(enabled_val)
            self._random_spawn_enabled = self._random_spawn_requested and bool(self._spawn_points)
            allow_reuse_flag = bool(random_spawn_cfg.get("allow_reuse", False))
            mode_raw = random_spawn_cfg.get("mode")
            if mode_raw:
                self._random_spawn_mode = str(mode_raw).strip().lower()
            target_agent = random_spawn_cfg.get("target_agent")
            if target_agent:
                self._random_spawn_target_agent = str(target_agent)
            target_on_centerline = random_spawn_cfg.get("target_on_centerline")
            if target_on_centerline is not None:
                self._random_spawn_target_on_centerline = bool(target_on_centerline)
            min_sep_raw = random_spawn_cfg.get("min_separation")
            if min_sep_raw is not None:
                try:
                    self._random_spawn_min_separation = max(0.0, float(min_sep_raw))
                except (TypeError, ValueError):
                    self._random_spawn_min_separation = 0.0
            min_wall_raw = random_spawn_cfg.get("min_wall_distance")
            if min_wall_raw is None:
                min_wall_raw = random_spawn_cfg.get("min_wall_clearance")
            if min_wall_raw is not None:
                try:
                    self._random_spawn_min_wall_distance = max(0.0, float(min_wall_raw))
                except (TypeError, ValueError):
                    self._random_spawn_min_wall_distance = 0.0
            offsets_cfg = random_spawn_cfg.get("offsets")
            if isinstance(offsets_cfg, Mapping):
                self._random_spawn_offsets = self._parse_spawn_offsets(offsets_cfg)
                frame_raw = offsets_cfg.get("frame") or random_spawn_cfg.get("frame")
                if frame_raw:
                    self._random_spawn_frame = str(frame_raw).strip().lower()
            elif random_spawn_cfg.get("frame"):
                self._random_spawn_frame = str(random_spawn_cfg.get("frame")).strip().lower()
            if self._random_spawn_mode in {"target_relative", "target-relative", "target"} and not random_spawn_cfg.get("frame"):
                if not (isinstance(offsets_cfg, Mapping) and offsets_cfg.get("frame")):
                    self._random_spawn_frame = "target"
            reject_offtrack_raw = random_spawn_cfg.get("reject_offtrack")
            if reject_offtrack_raw is not None:
                self._random_spawn_reject_offtrack = bool(reject_offtrack_raw)
            max_attempts_raw = random_spawn_cfg.get("max_attempts")
            if max_attempts_raw is not None:
                try:
                    self._random_spawn_max_attempts = max(1, int(max_attempts_raw))
                except (TypeError, ValueError):
                    self._random_spawn_max_attempts = 50
        else:
            self._random_spawn_requested = bool(random_spawn_cfg)
            self._random_spawn_enabled = self._random_spawn_requested and bool(self._spawn_points)
            allow_reuse_flag = bool(merged.get("random_spawn_allow_reuse", False))
        if not self._spawn_points:
            self._random_spawn_enabled = False
        self._random_spawn_allow_reuse = allow_reuse_flag
        self._spawn_point_names: List[str] = sorted(self._spawn_points.keys())
        self._last_spawn_selection: Dict[str, str] = {}

        self._finish_line_data = self._parse_finish_line(merged.get("finish_line"))
        if self._finish_line_data is not None:
            self._finish_signed_prev = np.zeros((self.n_agents,), dtype=np.float32)
            self._finish_crossed = np.zeros((self.n_agents,), dtype=bool)
        else:
            self._finish_signed_prev = None
            self._finish_crossed = None

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
        self._reward_ring_marker_states: Dict[str, List[bool]] = {}
        self._reward_ring_marker_dirty: bool = False
        self._reward_overlays: List[Dict[str, Any]] = []
        self._reward_overlay_dirty: bool = False
        self._reward_overlay_enabled = False
        self._reward_overlay_applied = False
        overlay_cfg = merged.get("reward_overlay")
        if isinstance(overlay_cfg, Mapping):
            enabled_raw = overlay_cfg.get("enabled", merged.get("reward_overlay_enabled", False))
            overlay_alpha = overlay_cfg.get("alpha", merged.get("reward_overlay_alpha", 0.25))
            overlay_scale = overlay_cfg.get(
                "value_scale",
                overlay_cfg.get("scale", merged.get("reward_overlay_value_scale", 1.0)),
            )
            overlay_segments = overlay_cfg.get("segments", merged.get("reward_overlay_segments", 48))
        else:
            enabled_raw = merged.get("reward_overlay_enabled", False)
            overlay_alpha = merged.get("reward_overlay_alpha", 0.25)
            overlay_scale = merged.get("reward_overlay_value_scale", 1.0)
            overlay_segments = merged.get("reward_overlay_segments", 48)

        self._reward_overlay_enabled = self._coerce_bool_flag(enabled_raw, default=False)
        try:
            self._reward_overlay_alpha = float(overlay_alpha)
        except (TypeError, ValueError):
            self._reward_overlay_alpha = 0.25
        self._reward_overlay_alpha = float(min(max(self._reward_overlay_alpha, 0.0), 1.0))
        try:
            self._reward_overlay_value_scale = float(overlay_scale)
        except (TypeError, ValueError):
            self._reward_overlay_value_scale = 1.0
        if self._reward_overlay_value_scale <= 0.0 or not np.isfinite(self._reward_overlay_value_scale):
            self._reward_overlay_value_scale = 1.0
        try:
            self._reward_overlay_segments = int(overlay_segments)
        except (TypeError, ValueError):
            self._reward_overlay_segments = 48
        if self._reward_overlay_segments < 8:
            self._reward_overlay_segments = 8

        self._reward_heatmap_payload: Optional[Dict[str, Any]] = None
        self._reward_heatmap_dirty: bool = False
        self._reward_heatmap_enabled = False
        self._reward_heatmap_applied = False
        heatmap_cfg = merged.get("reward_heatmap")
        if isinstance(heatmap_cfg, Mapping):
            enabled_raw = heatmap_cfg.get("enabled", merged.get("reward_heatmap_enabled", False))
            heatmap_alpha = heatmap_cfg.get("alpha", merged.get("reward_heatmap_alpha", 0.22))
            heatmap_scale = heatmap_cfg.get(
                "value_scale",
                heatmap_cfg.get("scale", merged.get("reward_heatmap_value_scale", 1.0)),
            )
            heatmap_extent = heatmap_cfg.get(
                "extent_m",
                heatmap_cfg.get("extent", merged.get("reward_heatmap_extent_m", merged.get("reward_heatmap_extent", 6.0))),
            )
            heatmap_cell_size = heatmap_cfg.get(
                "cell_size_m",
                heatmap_cfg.get(
                    "cell_size",
                    merged.get("reward_heatmap_cell_size_m", merged.get("reward_heatmap_cell_size", 0.25)),
                ),
            )
        else:
            enabled_raw = merged.get("reward_heatmap_enabled", False)
            heatmap_alpha = merged.get("reward_heatmap_alpha", 0.22)
            heatmap_scale = merged.get("reward_heatmap_value_scale", 1.0)
            heatmap_extent = merged.get("reward_heatmap_extent_m", merged.get("reward_heatmap_extent", 6.0))
            heatmap_cell_size = merged.get("reward_heatmap_cell_size_m", merged.get("reward_heatmap_cell_size", 0.25))

        self._reward_heatmap_enabled = self._coerce_bool_flag(enabled_raw, default=False)
        try:
            self._reward_heatmap_alpha = float(heatmap_alpha)
        except (TypeError, ValueError):
            self._reward_heatmap_alpha = 0.22
        self._reward_heatmap_alpha = float(min(max(self._reward_heatmap_alpha, 0.0), 1.0))
        try:
            self._reward_heatmap_value_scale = float(heatmap_scale)
        except (TypeError, ValueError):
            self._reward_heatmap_value_scale = 1.0
        if self._reward_heatmap_value_scale <= 0.0 or not np.isfinite(self._reward_heatmap_value_scale):
            self._reward_heatmap_value_scale = 1.0
        try:
            self._reward_heatmap_extent_m = float(heatmap_extent)
        except (TypeError, ValueError):
            self._reward_heatmap_extent_m = 6.0
        if self._reward_heatmap_extent_m <= 0.0 or not np.isfinite(self._reward_heatmap_extent_m):
            self._reward_heatmap_extent_m = 6.0
        try:
            self._reward_heatmap_cell_size_m = float(heatmap_cell_size)
        except (TypeError, ValueError):
            self._reward_heatmap_cell_size_m = 0.25
        if self._reward_heatmap_cell_size_m <= 0.0 or not np.isfinite(self._reward_heatmap_cell_size_m):
            self._reward_heatmap_cell_size_m = 0.25
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
        self._render_callbacks: List[Callable[["EnvRenderer"], None]] = []

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
        self.renderer: Optional["EnvRenderer"] = None
        headless_env = str(os.environ.get("PYGLET_HEADLESS", "")).lower()
        if headless_env in {"1", "true", "yes", "on"}:
            self._headless = True
        else:
            if pyglet is None and not _ensure_pyglet():
                self._headless = True
            else:
                self._headless = bool(pyglet.options.get("headless", False)) if pyglet is not None else True
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

    @staticmethod
    def _parse_spawn_offsets(offsets_cfg: Mapping[str, Any]) -> Dict[str, Any]:
        def _coerce_range(value: Any, min_key: str, max_key: str) -> Tuple[float, float]:
            if isinstance(value, (list, tuple)) and len(value) == 2:
                low, high = value
            elif value is None:
                low = offsets_cfg.get(min_key, 0.0)
                high = offsets_cfg.get(max_key, 0.0)
            else:
                try:
                    val = float(value)
                except (TypeError, ValueError):
                    val = 0.0
                low, high = -abs(val), abs(val)
            try:
                low_val = float(low)
            except (TypeError, ValueError):
                low_val = 0.0
            try:
                high_val = float(high)
            except (TypeError, ValueError):
                high_val = 0.0
            if low_val > high_val:
                low_val, high_val = high_val, low_val
            return low_val, high_val

        x_range = _coerce_range(offsets_cfg.get("x_range"), "x_min", "x_max")
        y_range = _coerce_range(offsets_cfg.get("y_range"), "y_min", "y_max")
        theta_range = _coerce_range(offsets_cfg.get("theta_range"), "theta_min", "theta_max")

        x_ranges: List[Tuple[float, float]] = []
        raw_ranges = offsets_cfg.get("x_ranges")
        if isinstance(raw_ranges, (list, tuple)):
            for entry in raw_ranges:
                if isinstance(entry, (list, tuple)) and len(entry) == 2:
                    x_ranges.append(_coerce_range(entry, "x_min", "x_max"))
        for key in ("x_range_behind", "x_range_front"):
            if key in offsets_cfg:
                x_ranges.append(_coerce_range(offsets_cfg.get(key), "x_min", "x_max"))

        parsed = {
            "x_range": x_range,
            "y_range": y_range,
            "theta_range": theta_range,
        }
        if x_ranges:
            parsed["x_ranges"] = x_ranges
        return parsed

    def _sample_offset_from_ranges(
        self,
        rng: np.random.Generator,
        ranges: List[Tuple[float, float]],
    ) -> float:
        if not ranges:
            return 0.0
        low, high = ranges[int(rng.integers(0, len(ranges)))]
        return self._sample_offset(rng, low, high)

    def _world_to_track_rc(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        mask = self._track_mask
        if mask is None:
            return None
        meta = self.map_meta or {}
        try:
            resolution = float(meta.get("resolution", 1.0))
        except (TypeError, ValueError):
            resolution = 1.0
        origin = meta.get("origin", (0.0, 0.0, 0.0))
        try:
            ox = float(origin[0])
            oy = float(origin[1])
        except (TypeError, ValueError, IndexError):
            ox, oy = 0.0, 0.0
        try:
            otheta = float(origin[2])
        except (TypeError, ValueError, IndexError):
            otheta = 0.0
        cos_o = math.cos(otheta)
        sin_o = math.sin(otheta)
        x_trans = x - ox
        y_trans = y - oy
        x_rot = x_trans * cos_o + y_trans * sin_o
        y_rot = -x_trans * sin_o + y_trans * cos_o
        if x_rot < 0.0 or y_rot < 0.0:
            return None
        try:
            col = int(x_rot / resolution)
            row = int(y_rot / resolution)
        except (TypeError, ValueError, ZeroDivisionError):
            return None
        height, width = mask.shape
        if row < 0 or row >= height or col < 0 or col >= width:
            return None
        return row, col

    def _point_on_track(self, x: float, y: float) -> bool:
        mask = self._track_mask
        if mask is None:
            return True
        rc = self._world_to_track_rc(x, y)
        if rc is None:
            return False
        row, col = rc
        return bool(mask[row, col])

    def _refresh_wall_points(self) -> None:
        if not self.walls:
            self._wall_points = None
            return
        arrays = [pts for pts in self.walls.values() if isinstance(pts, np.ndarray) and pts.size > 0]
        if not arrays:
            self._wall_points = None
            return
        self._wall_points = np.vstack(arrays).astype(np.float32, copy=False)

    def _distance_to_walls(self, x: float, y: float) -> Optional[float]:
        if self._wall_points is None or self._wall_points.size == 0:
            return None
        point = np.array([x, y], dtype=np.float32)
        diffs = self._wall_points[:, :2] - point
        d2 = np.einsum("ij,ij->i", diffs, diffs)
        if d2.size == 0:
            return None
        return float(np.sqrt(np.min(d2)))

    def _sample_offset(self, rng: np.random.Generator, low: float, high: float) -> float:
        if low > high:
            low, high = high, low
        if low == high:
            return float(low)
        return float(rng.uniform(low, high))

    def _sample_target_relative_spawn(self) -> Optional[Tuple[Dict[str, str], np.ndarray]]:
        agent_ids = list(self.possible_agents)
        if not agent_ids:
            return None

        target_id = self._random_spawn_target_agent
        if target_id is None:
            if "car_1" in agent_ids:
                target_id = "car_1"
            elif len(agent_ids) > 1:
                target_id = agent_ids[1]
            else:
                target_id = agent_ids[0]
        if target_id not in agent_ids:
            target_id = agent_ids[0]

        rng = getattr(self, "rng", None)
        if rng is None:
            rng = np.random.default_rng(self.seed)
            self.rng = rng

        use_centerline = bool(self._random_spawn_target_on_centerline)
        candidates = self._spawn_point_names
        cfg_points = self._random_spawn_cfg.get("spawn_points")
        if isinstance(cfg_points, (list, tuple)):
            filtered = [str(name) for name in cfg_points if str(name) in self._spawn_points]
            if filtered:
                candidates = filtered
        if not candidates and not use_centerline:
            return None

        x_range = self._random_spawn_offsets.get("x_range", (0.0, 0.0))
        x_ranges = self._random_spawn_offsets.get("x_ranges")
        y_range = self._random_spawn_offsets.get("y_range", (0.0, 0.0))
        theta_range = self._random_spawn_offsets.get("theta_range", (0.0, 0.0))
        frame = str(self._random_spawn_frame or "world").strip().lower()
        use_target_frame = frame in {"target", "target_frame", "target-relative", "target_relative", "relative", "local"}
        min_sep = float(self._random_spawn_min_separation or 0.0)
        min_sep_sq = min_sep * min_sep
        min_wall = float(self._random_spawn_min_wall_distance or 0.0)

        for _ in range(self._random_spawn_max_attempts):
            spawn_meta: Optional[Dict[str, float]] = None
            if use_centerline:
                sampled = self._sample_centerline_index()
                if sampled is None:
                    if not candidates:
                        return None
                    use_centerline = False
                    continue
                centerline, n_points, idx = sampled
                target_pose = centerline_pose(centerline, idx)
                spawn_name = "centerline"
                spawn_meta = {
                    "spawn_s": float(idx) / max(n_points - 1, 1),
                    "spawn_d": 0.0,
                }
            else:
                spawn_name = str(rng.choice(candidates))
                raw = self._spawn_points.get(spawn_name)
                if raw is None:
                    continue
                if raw.shape[0] < 3:
                    target_pose = np.zeros(3, dtype=np.float32)
                    target_pose[: raw.shape[0]] = raw
                else:
                    target_pose = np.asarray(raw[:3], dtype=np.float32)

            target_x = float(target_pose[0])
            target_y = float(target_pose[1])
            target_theta = float(target_pose[2])
            if self._random_spawn_reject_offtrack and not self._point_on_track(target_x, target_y):
                continue
            if min_wall > 0.0:
                wall_dist = self._distance_to_walls(target_x, target_y)
                if wall_dist is not None and wall_dist < min_wall:
                    continue

            poses = np.zeros((len(agent_ids), 3), dtype=np.float32)
            spawn_mapping: Dict[str, str] = {}

            target_idx = agent_ids.index(target_id)
            poses[target_idx] = target_pose
            for aid in agent_ids:
                spawn_mapping[aid] = spawn_name

            ok = True
            cos_t = math.cos(target_theta)
            sin_t = math.sin(target_theta)
            placed_indices = [target_idx]
            for idx, aid in enumerate(agent_ids):
                if aid == target_id:
                    continue
                placed = False
                for _ in range(self._random_spawn_max_attempts):
                    if isinstance(x_ranges, list) and x_ranges:
                        dx = self._sample_offset_from_ranges(rng, x_ranges)
                    else:
                        dx = self._sample_offset(rng, x_range[0], x_range[1])
                    dy = self._sample_offset(rng, y_range[0], y_range[1])
                    dtheta = self._sample_offset(rng, theta_range[0], theta_range[1])

                    if use_target_frame:
                        x = target_x + cos_t * dx - sin_t * dy
                        y = target_y + sin_t * dx + cos_t * dy
                    else:
                        x = target_x + dx
                        y = target_y + dy
                    theta = target_theta + dtheta

                    if self._random_spawn_reject_offtrack and not self._point_on_track(x, y):
                        continue
                    if min_wall > 0.0:
                        wall_dist = self._distance_to_walls(x, y)
                        if wall_dist is not None and wall_dist < min_wall:
                            continue
                    if min_sep_sq > 0.0:
                        too_close = False
                        for placed_idx in placed_indices:
                            px, py = poses[placed_idx, 0], poses[placed_idx, 1]
                            dxp = x - float(px)
                            dyp = y - float(py)
                            if dxp * dxp + dyp * dyp < min_sep_sq:
                                too_close = True
                                break
                        if too_close:
                            continue
                    poses[idx] = np.array([x, y, theta], dtype=np.float32)
                    placed = True
                    break
                if not placed:
                    ok = False
                    break
                placed_indices.append(idx)
            if ok:
                if use_centerline and spawn_meta:
                    self._last_spawn_metadata = dict(spawn_meta)
                return spawn_mapping, poses

        return None

    def _sample_random_spawn(self) -> Optional[Tuple[Dict[str, str], np.ndarray]]:
        if not self._random_spawn_enabled or not self._spawn_point_names:
            return None

        if self._random_spawn_mode in {"target_relative", "target-relative", "target"}:
            return self._sample_target_relative_spawn()

        agent_ids = self.possible_agents
        count = len(agent_ids)
        if count == 0:
            return None

        pool = self._spawn_point_names
        cfg_points = self._random_spawn_cfg.get("spawn_points")
        if isinstance(cfg_points, (list, tuple)):
            filtered = [str(name) for name in cfg_points if str(name) in self._spawn_points]
            if filtered:
                pool = filtered
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

    def _select_map_bundle(self) -> Optional[str]:
        if self._map_cycle_mode != "per_episode":
            return None
        mode = "eval" if self._map_split_mode == "eval" else "train"
        bundles = self._map_cycle_order.get(mode) or []
        if not bundles:
            return None
        pick = self._map_pick_mode
        if pick == "random":
            return bundles[int(self.rng.integers(0, len(bundles)))]
        if pick == "first":
            return bundles[0]
        idx = int(self._map_cycle_indices.get(mode, 0))
        bundle = bundles[idx % len(bundles)]
        idx += 1
        if idx >= len(bundles):
            if self._map_epoch_shuffle:
                self.rng.shuffle(bundles)
            idx = 0
        self._map_cycle_indices[mode] = idx
        return bundle

    def _apply_map_data(self, map_data: Any, bundle: Optional[str] = None) -> None:
        if map_data is None:
            return
        self._map_data = map_data
        self.map_dir = Path(map_data.yaml_path).parent
        self.map_ext = map_data.image_path.suffix or ".png"
        self.map_name = map_data.yaml_path.name
        self.map_yaml = map_data.yaml_path.name
        self.map_path = map_data.yaml_path
        self.yaml_path = map_data.yaml_path
        self.map_meta = dict(map_data.metadata)
        self.map_image_path = map_data.image_path
        self._track_mask = map_data.track_mask
        self.walls = map_data.walls
        self.walls_path = map_data.walls_path
        self._refresh_wall_points()
        self._spawn_points = self._extract_spawn_points(map_data, self.map_meta)
        self._spawn_point_names = sorted(self._spawn_points.keys())
        if not self._spawn_points:
            self._random_spawn_enabled = False
        else:
            self._random_spawn_enabled = bool(getattr(self, "_random_spawn_requested", False))

        # Update simulation + renderer
        self.sim.set_map(str(self.yaml_path), self.map_ext)
        if self.renderer is not None:
            self.renderer.update_map(
                str(self.yaml_path.with_suffix("")),
                self.map_ext,
                map_meta=self.map_meta,
                map_image_path=self.map_image_path,
            )

        # Update centerline + render
        self.set_centerline(map_data.centerline, path=map_data.centerline_path)
        self._update_renderer_centerline()

        # Update observation bounds based on new map
        width, height = map_data.image_size
        R = float(self.map_meta.get("resolution", 1.0))
        x0, y0, _ = self.map_meta.get('origin', (0.0, 0.0, 0.0))
        x_min = x0
        x_max = x0 + width * R
        y_min = y0
        y_max = y0 + height * R
        self._build_observation_spaces(x_min, x_max, y_min, y_max)

        if bundle is not None:
            self._map_bundle_active = bundle

    def _maybe_cycle_map(self) -> None:
        bundle = self._select_map_bundle()
        if bundle is None:
            return
        if bundle == self._map_bundle_active:
            return
        map_cfg = {
            "map_dir": str(self._map_root),
            "map_bundle": bundle,
            "map_ext": self.map_ext,
            "centerline_autoload": True,
            "centerline_csv": self._centerline_csv,
            "centerline_render": self.centerline_render_enabled,
            "centerline_features": self.centerline_features_enabled,
            "walls_autoload": self._walls_autoload,
            "walls_csv": self._walls_csv,
            "track_threshold": self._track_threshold,
            "track_inverted": self._track_inverted,
        }
        map_data = self._map_loader.load(map_cfg)
        self._apply_map_data(map_data, bundle=bundle)

    def _resolve_centerline_sampling(self) -> Optional[Tuple[np.ndarray, int, int, int, str]]:
        centerline = self.centerline_points
        if centerline is None or centerline.shape[0] == 0:
            return None

        mode = str(self.spawn_centerline_cfg.get("mode", "random")).lower()
        if self._map_split_mode == "eval":
            mode_eval = self.spawn_centerline_cfg.get("mode_eval")
            if mode_eval is not None:
                mode = str(mode_eval).lower()
            elif mode == "random":
                mode = "round_robin"

        min_progress = float(self.spawn_centerline_cfg.get("min_progress", 0.05))
        max_progress = float(self.spawn_centerline_cfg.get("max_progress", 0.95))
        avoid_finish = bool(self.spawn_centerline_cfg.get("avoid_finish", True))

        if avoid_finish:
            min_progress = max(min_progress, 0.05)
            max_progress = min(max_progress, 0.95)

        n_points = centerline.shape[0]
        min_idx = int(max(0, min_progress * (n_points - 1)))
        max_idx = int(min(n_points - 1, max_progress * (n_points - 1)))
        min_distance = self.spawn_centerline_cfg.get("min_distance_m")
        max_distance = self.spawn_centerline_cfg.get("max_distance_m")
        if min_distance is None:
            min_distance = self.spawn_centerline_cfg.get("min_distance")
        if max_distance is None:
            max_distance = self.spawn_centerline_cfg.get("max_distance")
        if min_distance is not None or max_distance is not None:
            try:
                total_length = float(centerline_arc_length(centerline))
            except (TypeError, ValueError):
                total_length = 0.0
            spacing = total_length / max(n_points - 1, 1)
            if spacing > 0.0:
                try:
                    min_dist_val = float(min_distance) if min_distance is not None else 0.0
                except (TypeError, ValueError):
                    min_dist_val = 0.0
                try:
                    max_dist_val = float(max_distance) if max_distance is not None else total_length
                except (TypeError, ValueError):
                    max_dist_val = total_length
                if min_dist_val < 0.0:
                    min_dist_val = 0.0
                if max_dist_val < 0.0:
                    max_dist_val = 0.0
                if min_dist_val > max_dist_val:
                    min_dist_val, max_dist_val = max_dist_val, min_dist_val
                min_idx = int(max(0, math.floor(min_dist_val / spacing)))
                max_idx = int(min(n_points - 1, math.floor(max_dist_val / spacing)))
                if avoid_finish:
                    min_idx = max(min_idx, int(0.05 * (n_points - 1)))
                    max_idx = min(max_idx, int(0.95 * (n_points - 1)))
        if max_idx <= min_idx:
            min_idx = 0
            max_idx = n_points - 1

        return centerline, n_points, min_idx, max_idx, mode

    def _sample_centerline_index(self) -> Optional[Tuple[np.ndarray, int, int]]:
        resolved = self._resolve_centerline_sampling()
        if resolved is None:
            return None
        centerline, n_points, min_idx, max_idx, mode = resolved

        if mode == "round_robin":
            idx = self._spawn_centerline_index
            if idx < min_idx or idx > max_idx:
                idx = min_idx
            self._spawn_centerline_index = idx + 1
            if self._spawn_centerline_index > max_idx:
                self._spawn_centerline_index = min_idx
        else:
            idx = int(self.rng.integers(min_idx, max_idx + 1))

        return centerline, n_points, idx

    def _sample_centerline_spawn(self) -> Optional[Tuple[np.ndarray, Dict[str, float], Dict[str, float]]]:
        if str(self.spawn_policy or "").lower() != "centerline_relative":
            return None
        sampled = self._sample_centerline_index()
        if sampled is None:
            return None
        centerline, n_points, idx = sampled

        s_offset = float(self.spawn_offsets_cfg.get("s_offset", 0.0))
        d_offset = float(self.spawn_offsets_cfg.get("d_offset", 0.0))
        d_max = self.spawn_offsets_cfg.get("d_max")
        d_max = float(d_max) if d_max is not None else None

        spacing = centerline_arc_length(centerline) / max(n_points - 1, 1)
        offset_idx = int(round(s_offset / spacing)) if spacing > 0.0 else 0
        ego_idx = (idx + offset_idx) % n_points

        target_enabled = bool(self.spawn_target_cfg.get("enabled", True))
        target_pose = centerline_pose(centerline, idx)
        ego_pose = centerline_pose(centerline, ego_idx)
        d_offset = self._clamp_d_offset(centerline, ego_idx, d_offset, d_max)

        normal = np.array([-np.sin(ego_pose[2]), np.cos(ego_pose[2])], dtype=np.float32)
        ego_pose[:2] = ego_pose[:2] + normal * d_offset

        agent_order = self.possible_agents
        poses = np.zeros((len(agent_order), 3), dtype=np.float32)
        for i, aid in enumerate(agent_order):
            if target_enabled and aid == "car_1":
                poses[i] = target_pose
            else:
                poses[i] = ego_pose

        speeds = {}
        target_speed = float(self.spawn_target_cfg.get("speed", 0.0))
        ego_speed = float(self.spawn_ego_cfg.get("speed", target_speed))
        for aid in agent_order:
            if target_enabled and aid == "car_1":
                speeds[aid] = target_speed
            else:
                speeds[aid] = ego_speed

        metadata = {
            "spawn_s": float(idx) / max(n_points - 1, 1),
            "spawn_d": float(d_offset),
        }
        return poses, speeds, metadata

    def _clamp_d_offset(
        self,
        centerline: np.ndarray,
        index: int,
        d_offset: float,
        d_max: Optional[float],
    ) -> float:
        limit = d_max if d_max is not None else None
        if self.walls:
            point = centerline[index, :2].astype(np.float32)
            distances = []
            for wall_points in self.walls.values():
                if wall_points is None or len(wall_points) == 0:
                    continue
                diffs = wall_points[:, :2] - point
                dist = float(np.min(np.linalg.norm(diffs, axis=1)))
                distances.append(dist)
            if distances:
                wall_limit = max(min(distances) - 0.1, 0.0)
                limit = wall_limit if limit is None else min(limit, wall_limit)
        if limit is None:
            return float(d_offset)
        return float(np.clip(d_offset, -limit, limit))

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
                "linear_vels_x": float(self.linear_vels_x_curr[idx]),
                "linear_vels_y": float(self.linear_vels_y_curr[idx]),
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
            self._reward_ring_marker_states.clear()
            self._reward_ring_dirty = True
            self._reward_ring_target_dirty = True
            self._reward_ring_marker_dirty = True
            return

        stored = {
            "preferred_radius": max(float(config.get("preferred_radius", 0.0)), 0.0),
            "inner_tolerance": max(float(config.get("inner_tolerance", 0.0)), 0.0),
            "outer_tolerance": max(float(config.get("outer_tolerance", 0.0)), 0.0),
            "segments": max(int(config.get("segments", 96) or 96), 8),
            "marker_radius": max(float(config.get("marker_radius", 0.0)), 0.0),
            "marker_segments": max(int(config.get("marker_segments", 12) or 12), 4),
            "offsets_only": bool(config.get("offsets_only", False)),
        }
        for key in ("fill_color", "border_color", "preferred_color"):
            if key in config and isinstance(config[key], (list, tuple)):
                stored[key] = tuple(float(component) for component in config[key])
        falloff_val = config.get("falloff")
        if falloff_val is not None:
            stored["falloff"] = str(falloff_val).lower()
        marker_color_val = config.get("marker_color")
        if isinstance(marker_color_val, (list, tuple)):
            stored["marker_color"] = tuple(float(component) for component in marker_color_val)
        offsets_val = config.get("offsets")
        if offsets_val:
            cleaned_offsets: List[Tuple[float, float]] = []
            for entry in offsets_val:
                if entry is None:
                    continue
                try:
                    pair = tuple(float(v) for v in entry)  # type: ignore[arg-type]
                except Exception:
                    continue
                if len(pair) >= 2:
                    cleaned_offsets.append((pair[0], pair[1]))
            if cleaned_offsets:
                stored["offsets"] = cleaned_offsets
        marker_color_active_val = config.get("marker_color_active")
        if isinstance(marker_color_active_val, (list, tuple)):
            stored["marker_color_active"] = tuple(float(component) for component in marker_color_active_val)
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
            self._reward_ring_marker_states.clear()
            self._reward_ring_dirty = True
            self._reward_ring_target_dirty = True
            self._reward_ring_marker_dirty = True

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
            self._reward_ring_marker_dirty = True

        if normalized:
            pending = self._reward_ring_marker_states.get(agent_id)
            if pending is not None:
                self._reward_ring_marker_states[normalized] = list(pending)
                if agent_id != normalized:
                    self._reward_ring_marker_states.pop(agent_id, None)
                self._reward_ring_marker_dirty = True

    def update_reward_ring_markers(self, agent_id: str, states: Optional[Sequence[bool]]) -> None:
        if self._reward_ring_config is None:
            return
        focus = self._reward_ring_focus_agent
        if focus is not None and agent_id != focus:
            return
        target_key = self._reward_ring_target or agent_id
        if states is None:
            if target_key in self._reward_ring_marker_states:
                self._reward_ring_marker_states.pop(target_key, None)
                self._reward_ring_marker_dirty = True
        else:
            snapshot = [bool(s) for s in states]
            if self._reward_ring_marker_states.get(target_key) != snapshot:
                self._reward_ring_marker_states[target_key] = snapshot
                self._reward_ring_marker_dirty = True

    def configure_agent_targets(self, target_mapping: Dict[str, str]) -> None:
        """Configure which agent is the target of which other agent.

        Args:
            target_mapping: Dict mapping agent_id -> target_agent_id
                           For example: {'car_0': 'car_1'} means car_0 is targeting car_1
        """
        for agent_id, target_id in target_mapping.items():
            if agent_id not in self._agent_id_to_index:
                continue
            if target_id not in self._agent_id_to_index:
                continue

            target_idx = self._agent_id_to_index[target_id]
            self._agent_target_index[agent_id] = target_idx

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
                    "marker_radius": cfg.get("marker_radius", 0.0),
                    "marker_segments": cfg.get("marker_segments", 12),
                    "offsets_only": cfg.get("offsets_only", False),
                }
                for key in ("fill_color", "border_color", "preferred_color", "marker_color"):
                    if key in cfg:
                        renderer_payload[key] = cfg[key]
                if "offsets" in cfg:
                    renderer_payload["offsets"] = cfg["offsets"]
                self.renderer.configure_reward_ring(**renderer_payload)
            else:
                self.renderer.configure_reward_ring(enabled=False)
            self._reward_ring_dirty = False

        if self._reward_ring_target_dirty:
            self.renderer.set_reward_ring_target(self._reward_ring_target)
            self._reward_ring_target_dirty = False

        if self._reward_ring_marker_dirty:
            target_id = self._reward_ring_target
            if target_id:
                states = self._reward_ring_marker_states.get(target_id)
                try:
                    self.renderer.set_reward_ring_marker_state(target_id, states)
                except Exception:
                    pass
            self._reward_ring_marker_dirty = False

    def update_reward_overlays(
        self,
        overlays: Optional[Sequence[Mapping[str, Any]]],
        *,
        enabled: Optional[bool] = None,
        alpha: Optional[float] = None,
        value_scale: Optional[float] = None,
        segments: Optional[int] = None,
    ) -> None:
        """Update translucent circle overlays used to visualise reward regions."""
        if enabled is not None:
            enabled_val = self._coerce_bool_flag(enabled, default=self._reward_overlay_enabled)
            if enabled_val != self._reward_overlay_enabled:
                self._reward_overlay_enabled = enabled_val
                self._reward_overlay_dirty = True
        if overlays is None:
            if self._reward_overlays:
                self._reward_overlays = []
                self._reward_overlay_dirty = True
        else:
            cleaned: List[Dict[str, Any]] = []
            for entry in overlays:
                if not isinstance(entry, Mapping):
                    continue
                cleaned.append(dict(entry))
            self._reward_overlays = cleaned
            self._reward_overlay_dirty = True

        if alpha is not None:
            try:
                alpha_val = float(alpha)
            except (TypeError, ValueError):
                alpha_val = self._reward_overlay_alpha
            self._reward_overlay_alpha = float(min(max(alpha_val, 0.0), 1.0))
            self._reward_overlay_dirty = True

        if value_scale is not None:
            try:
                scale_val = float(value_scale)
            except (TypeError, ValueError):
                scale_val = self._reward_overlay_value_scale
            if scale_val > 0.0 and np.isfinite(scale_val):
                self._reward_overlay_value_scale = float(scale_val)
                self._reward_overlay_dirty = True

        if segments is not None:
            try:
                seg_val = int(segments)
            except (TypeError, ValueError):
                seg_val = self._reward_overlay_segments
            seg_val = max(seg_val, 8)
            if seg_val != self._reward_overlay_segments:
                self._reward_overlay_segments = seg_val
                self._reward_overlay_dirty = True

    def update_reward_heatmap(
        self,
        heatmap: Optional[Mapping[str, Any]],
        *,
        enabled: Optional[bool] = None,
        alpha: Optional[float] = None,
        value_scale: Optional[float] = None,
        extent_m: Optional[float] = None,
        cell_size_m: Optional[float] = None,
    ) -> None:
        """Update the cached potential-field heatmap renderer state."""
        if enabled is not None:
            enabled_val = self._coerce_bool_flag(enabled, default=self._reward_heatmap_enabled)
            if enabled_val != self._reward_heatmap_enabled:
                self._reward_heatmap_enabled = enabled_val
                self._reward_heatmap_dirty = True

        if heatmap is None:
            if self._reward_heatmap_payload is not None:
                self._reward_heatmap_payload = None
                self._reward_heatmap_dirty = True
        elif isinstance(heatmap, Mapping):
            try:
                payload = dict(heatmap)
            except Exception:
                payload = None
            if payload is not None and payload != self._reward_heatmap_payload:
                self._reward_heatmap_payload = payload
                self._reward_heatmap_dirty = True

        if alpha is not None:
            try:
                alpha_val = float(alpha)
            except (TypeError, ValueError):
                alpha_val = self._reward_heatmap_alpha
            alpha_val = float(min(max(alpha_val, 0.0), 1.0))
            if alpha_val != self._reward_heatmap_alpha:
                self._reward_heatmap_alpha = alpha_val
                self._reward_heatmap_dirty = True

        if value_scale is not None:
            try:
                scale_val = float(value_scale)
            except (TypeError, ValueError):
                scale_val = self._reward_heatmap_value_scale
            if scale_val > 0.0 and np.isfinite(scale_val) and scale_val != self._reward_heatmap_value_scale:
                self._reward_heatmap_value_scale = float(scale_val)
                self._reward_heatmap_dirty = True

        if extent_m is not None:
            try:
                extent_val = float(extent_m)
            except (TypeError, ValueError):
                extent_val = self._reward_heatmap_extent_m
            if extent_val > 0.0 and np.isfinite(extent_val) and extent_val != self._reward_heatmap_extent_m:
                self._reward_heatmap_extent_m = float(extent_val)
                self._reward_heatmap_dirty = True

        if cell_size_m is not None:
            try:
                cell_val = float(cell_size_m)
            except (TypeError, ValueError):
                cell_val = self._reward_heatmap_cell_size_m
            if cell_val > 0.0 and np.isfinite(cell_val) and cell_val != self._reward_heatmap_cell_size_m:
                self._reward_heatmap_cell_size_m = float(cell_val)
                self._reward_heatmap_dirty = True

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
        self._maybe_cycle_map()
        self.agents = self.possible_agents.copy()
        self._elapsed_steps = 0
        self.current_time = 0.0

        # Reset persistent collision tracking
        self._collision_flags.fill(False)
        self._collision_steps.fill(-1)

        self.start_state.reset()
        self.state_buffers.reset()
        self._render_ticker.clear()
        self._render_ticker_dirty = True
        self._render_wrapped_obs.clear()
        self._last_spawn_metadata = {}
        poses = None
        velocities = None
        spawn_mapping: Dict[str, str] = {}

        # Speed locking for curriculum
        self._lock_speed_steps = 0
        self._locked_velocities = {}
        self._episode_step_count = 0
        if self.renderer is not None:
            self.renderer.reset_state()
            self._update_renderer_centerline()
            self._reward_ring_dirty = True
            self._reward_ring_target_dirty = True
            self._reward_overlay_dirty = True
            self._reward_overlay_applied = False
            self._reward_heatmap_payload = None
            self._reward_heatmap_dirty = True
            self._reward_heatmap_applied = False
    # Case 1: Explicit override via options
        if options is not None:
            if isinstance(options, dict) and "poses" in options:
                poses = np.array(options["poses"], dtype=np.float32)
                if poses.ndim == 1:
                    poses = np.expand_dims(poses, axis=0)
                self._update_start_from_poses(poses)
                spawn_mapping = {}
            # Extract velocities if provided (dict mapping agent_id -> velocity)
            if isinstance(options, dict) and "velocities" in options:
                vel_dict = options["velocities"]
                if isinstance(vel_dict, dict):
                    # Convert per-agent velocities to array (None for agents not in dict)
                    velocities = np.full(self.n_agents, np.nan, dtype=np.float32)
                    agent_index = self._agent_id_to_index
                    for agent_id, vel in vel_dict.items():
                        if agent_id in agent_index:
                            velocities[agent_index[agent_id]] = float(vel)
                    # Store velocities for speed locking
                    self._locked_velocities = vel_dict.copy()
                else:
                    # Backward compatibility: array format
                    velocities = np.array(vel_dict, dtype=np.float32)
                    if velocities.ndim == 0:
                        velocities = np.array([velocities])

            # Extract speed locking parameter
            if isinstance(options, dict) and "lock_speed_steps" in options:
                self._lock_speed_steps = max(0, int(options["lock_speed_steps"]))
        # Case 2: Centerline-based spawn policy
        if poses is None:
            spawn_policy = self._sample_centerline_spawn()
            if spawn_policy is not None:
                poses, velocities, meta = spawn_policy
                self._update_start_from_poses(poses)
                spawn_mapping = {}
                self._last_spawn_metadata = dict(meta)

        # Case 3: Default to config start_poses
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

        if isinstance(velocities, dict):
            vel_array = np.full(self.n_agents, np.nan, dtype=np.float32)
            agent_index = self._agent_id_to_index
            for agent_id, vel in velocities.items():
                if agent_id in agent_index:
                    vel_array[agent_index[agent_id]] = float(vel)
            velocities = vel_array

        # options: (N,3) poses (x,y,theta). If None, caller must set internally.
        # poses = options if options is not None else np.zeros((self.n_agents, 3), dtype=np.float32)
        obs_joint = self.sim.reset(poses, velocities=velocities)
        obs = self._split_obs(obs_joint)
        self._attach_central_state(obs, obs_joint)
        self._update_state(obs_joint)
        self._refresh_render_observations(obs)
        self._reset_finish_line_tracking()

        infos = {aid: {} for aid in self.agents}
        if self._map_bundle_active:
            for aid in infos:
                infos[aid]["map_bundle"] = str(self._map_bundle_active)
        if spawn_mapping:
            for aid, name in spawn_mapping.items():
                if aid in infos:
                    infos[aid]["spawn_point"] = name
        if self._last_spawn_metadata:
            for aid in infos:
                infos[aid].update(self._last_spawn_metadata)
        self._inject_finish_line_info(infos)
        return obs, infos

    def step(self, actions: Dict[str, np.ndarray]):

        joint = np.zeros((self.n_agents, 2), dtype=np.float32)
        agent_index = self._agent_id_to_index
        for aid in self.agents:
            if aid in actions:
                joint[agent_index[aid]] = np.asarray(actions[aid], dtype=np.float32)


        # Increment episode step counter
        self._episode_step_count += 1

        obs_joint = self.sim.step(joint)

        # Apply speed locking AFTER simulation step (restore locked velocities)
        if self._lock_speed_steps > 0 and self._episode_step_count <= self._lock_speed_steps:
            for agent_id, locked_vel in self._locked_velocities.items():
                if agent_id in agent_index:
                    idx = agent_index[agent_id]
                    # Directly set the velocity in the simulation state
                    # state[3] is v_long (longitudinal velocity)
                    self.sim.agents[idx].state[3] = float(locked_vel)
                    # Update the velocity attributes if they exist
                    if hasattr(self.sim.agents[idx], 'v_long'):
                        self.sim.agents[idx].v_long = float(locked_vel)
                    # Update the observation to reflect locked velocity
                    obs_joint['linear_vels_x'][idx] = float(locked_vel)

        obs = self._split_obs(obs_joint)
        self._attach_central_state(obs, obs_joint)
        self._update_state(obs_joint)
        self._refresh_render_observations(obs)
        finish_completion = self._update_finish_line_progress()

        self.current_time += self.timestep
        lap_completion = self.start_state.update_progress(
            self.poses_x,
            self.poses_y,
            self.linear_vels_x_curr,
            self.linear_vels_y_curr,
            self.current_time,
            self.target_laps,
        )
        if finish_completion:
            for aid, done in finish_completion.items():
                if done:
                    lap_completion[aid] = True
        # simple per-step reward (customize as needed)
        rewards = {aid: float(self.timestep * 0.0) for aid in self.agents}

        # terminations/truncations
        collisions = obs_joint["collisions"]

        # Update persistent collision flags (like v1)
        # Once an agent collides, they stay collided for the episode
        for idx, agent_id in enumerate(self.possible_agents):
            if bool(collisions[idx]) and not self._collision_flags[idx]:
                self._collision_flags[idx] = True
                self._collision_steps[idx] = self._elapsed_steps

        # Use persistent collision flags for termination
        terminations = build_terminations(
            self.possible_agents,
            self._collision_flags,  # Use persistent flags instead of current step
            lap_completion,
            self.terminate_on_collision,
        )

        # Global termination: if ANY agent terminates due to collision, end episode for ALL agents
        # This ensures accurate outcome tracking (target_crash vs self_crash vs collision)
        any_collision_termination = any(
            self._collision_flags[idx] and self.terminate_on_collision.get(agent_id, True)
            for idx, agent_id in enumerate(self.possible_agents)
        )
        if any_collision_termination:
            for agent_id in self.possible_agents:
                terminations[agent_id] = True

        trunc_flag = (self.max_steps > 0) and (self._elapsed_steps + 1 >= self.max_steps)
        truncations = {aid: bool(trunc_flag) for aid in self.possible_agents}
        infos = {aid: {} for aid in self.possible_agents}
        if trunc_flag:
            for aid in infos:
                infos[aid]["time_limit"] = True
        self._inject_finish_line_info(infos)
        if self._map_bundle_active:
            for aid in infos:
                infos[aid]["map_bundle"] = str(self._map_bundle_active)
        if self._last_spawn_metadata:
            for aid in infos:
                infos[aid].update(self._last_spawn_metadata)

        # Add collision info for outcome determination
        # The training loop needs this to categorize episode outcomes
        for idx, agent_id in enumerate(self.possible_agents):
            if agent_id in infos:
                # Add own collision status
                infos[agent_id]["collision"] = bool(self._collision_flags[idx])

                # Add target collision status if this agent has a target
                target_idx = self._agent_target_index.get(agent_id)
                if target_idx is not None and target_idx < len(self._collision_flags):
                    infos[agent_id]["target_collision"] = bool(self._collision_flags[target_idx])
                else:
                    infos[agent_id]["target_collision"] = False

                # Add target finish status if finish line tracking is enabled
                finish_crossed = self._finish_crossed
                if (
                    target_idx is not None
                    and finish_crossed is not None
                    and target_idx < len(finish_crossed)
                ):
                    infos[agent_id]["target_finished"] = bool(finish_crossed[target_idx])
                else:
                    infos[agent_id]["target_finished"] = False

                # Add locked speed info for curriculum-based velocity control
                if agent_id in self._locked_velocities:
                    infos[agent_id]["locked_velocity"] = float(self._locked_velocities[agent_id])
                    infos[agent_id]["lock_speed_active"] = bool(
                        self._lock_speed_steps > 0 and self._episode_step_count <= self._lock_speed_steps
                    )
                else:
                    infos[agent_id]["locked_velocity"] = None
                    infos[agent_id]["lock_speed_active"] = False

        # advance and cull finished agents
        self._elapsed_steps += 1
        self.agents = [aid for aid in self.possible_agents if not (terminations[aid] or truncations[aid])]

        return obs, rewards, terminations, truncations, infos

    # ------------------------------------------------------------------
    # Finish line helpers
    # ------------------------------------------------------------------
    def _parse_finish_line(self, cfg: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
        if not isinstance(cfg, Mapping):
            return None
        start_raw = cfg.get("start")
        end_raw = cfg.get("end")
        if start_raw is None or end_raw is None:
            return None
        try:
            start = np.asarray(start_raw, dtype=np.float32).reshape(-1)
            end = np.asarray(end_raw, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            return None
        if start.size < 2 or end.size < 2:
            return None
        start = start[:2]
        end = end[:2]
        segment = end - start
        length = float(np.linalg.norm(segment))
        if length <= 1e-6:
            return None
        segment_unit = segment / length
        length_sq = float(length * length)
        tol_value = cfg.get("tolerance", cfg.get("thickness", cfg.get("width", 1.0)))
        try:
            tolerance = max(float(tol_value), 1e-3)
        except (TypeError, ValueError):
            tolerance = 1.0
        pad_value = cfg.get("padding", 0.5)
        try:
            padding = float(pad_value)
        except (TypeError, ValueError):
            padding = 0.5
        dir_vec = cfg.get("direction")
        direction_unit: Optional[np.ndarray] = None
        if dir_vec is not None:
            try:
                dir_arr = np.asarray(dir_vec, dtype=np.float32).reshape(-1)
                if dir_arr.size >= 2:
                    norm = float(np.linalg.norm(dir_arr[:2]))
                    if norm > 0.0:
                        direction_unit = dir_arr[:2] / norm
            except (TypeError, ValueError):
                direction_unit = None
        min_speed_raw = cfg.get("trigger_speed", cfg.get("min_speed", 0.0))
        try:
            min_speed = max(float(min_speed_raw), 0.0)
        except (TypeError, ValueError):
            min_speed = 0.0
        return {
            "start": start,
            "end": end,
            "segment": segment,
            "segment_unit": segment_unit,
            "segment_length": length,
            "segment_length_sq": length_sq,
            "tolerance": tolerance,
            "padding": max(padding, 0.0),
            "direction": direction_unit,
            "min_speed": min_speed,
        }

    def _reset_finish_line_tracking(self) -> None:
        if self._finish_line_data is None or self._finish_signed_prev is None or self._finish_crossed is None:
            return
        self._finish_crossed.fill(False)
        for idx in range(self.n_agents):
            point = np.array([self.poses_x[idx], self.poses_y[idx]], dtype=np.float32)
            self._finish_signed_prev[idx] = self._signed_distance_to_finish(point)

    def _signed_distance_to_finish(self, point: np.ndarray) -> float:
        data = self._finish_line_data
        if data is None:
            return 0.0
        rel = point - data["start"]
        seg_unit = data["segment_unit"]
        cross = seg_unit[0] * rel[1] - seg_unit[1] * rel[0]
        return cross

    def _update_finish_line_progress(self) -> Dict[str, bool]:
        data = self._finish_line_data
        if (
            data is None
            or self._finish_signed_prev is None
            or self._finish_crossed is None
            or not self.possible_agents
        ):
            return {}
        completed: Dict[str, bool] = {}
        segment = data["segment"]
        len_sq = data["segment_length_sq"]
        tolerance = data["tolerance"]
        padding = data["padding"]
        direction = data["direction"]
        min_speed = data["min_speed"]
        for idx, aid in enumerate(self.possible_agents):
            if self._finish_crossed[idx]:
                continue
            point = np.array([self.poses_x[idx], self.poses_y[idx]], dtype=np.float32)
            rel = point - data["start"]
            proj = float(np.dot(rel, segment) / len_sq)
            if proj < -padding or proj > 1.0 + padding:
                self._finish_signed_prev[idx] = self._signed_distance_to_finish(point)
                continue
            curr_signed = self._signed_distance_to_finish(point)
            prev_signed = self._finish_signed_prev[idx]
            self._finish_signed_prev[idx] = curr_signed
            sign_switch = (prev_signed <= 0.0 < curr_signed) or (prev_signed >= 0.0 > curr_signed)
            if not sign_switch:
                continue
            if abs(curr_signed) > tolerance and abs(prev_signed) > tolerance:
                continue
            if direction is not None:
                vel = np.array(
                    [self.linear_vels_x_curr[idx], self.linear_vels_y_curr[idx]],
                    dtype=np.float32,
                )
                if float(np.dot(vel, direction)) < min_speed:
                    continue
            self._finish_crossed[idx] = True
            completed[aid] = True
        return completed

    def _inject_finish_line_info(self, infos: Mapping[str, Dict[str, Any]]) -> None:
        if self._finish_line_data is None or self._finish_crossed is None:
            return
        for aid, idx in self._agent_id_to_index.items():
            payload = infos.get(aid)
            if payload is None:
                continue
            payload["finish_line"] = bool(self._finish_crossed[idx])

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

        if self._headless and self.render_mode == "human":
            # Nothing to do when headless; keep API contract intact.
            return None

        # Check if pyglet is available before rendering
        if not _ensure_pyglet():
            logger.warning("Cannot render: pyglet not available (headless system)")
            return None

        self._collect_render_data = True

        if self.renderer is None:
            # Lazy import to avoid pyglet initialization when rendering disabled
            from src.render import EnvRenderer

            self.renderer = EnvRenderer(WINDOW_W, WINDOW_H,
                                        lidar_fov=4.7,
                                        max_range=30.0,
                                        lidar_offset=self.lidar_dist)
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
        if self.renderer is not None:
            if self._reward_heatmap_enabled:
                if self._reward_heatmap_dirty or (not self._reward_heatmap_applied and self._reward_heatmap_payload is not None):
                    try:
                        self.renderer.update_reward_heatmap(
                            self._reward_heatmap_payload,
                            alpha=self._reward_heatmap_alpha,
                            value_scale=self._reward_heatmap_value_scale,
                            extent_m=self._reward_heatmap_extent_m,
                            cell_size_m=self._reward_heatmap_cell_size_m,
                        )
                        self._reward_heatmap_applied = self._reward_heatmap_payload is not None
                    except Exception:
                        pass
                    self._reward_heatmap_dirty = False
            elif self._reward_heatmap_applied:
                try:
                    self.renderer.update_reward_heatmap(
                        None,
                        alpha=self._reward_heatmap_alpha,
                        value_scale=self._reward_heatmap_value_scale,
                        extent_m=self._reward_heatmap_extent_m,
                        cell_size_m=self._reward_heatmap_cell_size_m,
                    )
                except Exception:
                    pass
                self._reward_heatmap_applied = False
            if self._reward_overlay_enabled:
                if self._reward_overlay_dirty or self._reward_overlays:
                    try:
                        self.renderer.update_reward_overlays(
                            self._reward_overlays,
                            alpha=self._reward_overlay_alpha,
                            value_scale=self._reward_overlay_value_scale,
                            segments=self._reward_overlay_segments,
                        )
                        self._reward_overlay_applied = bool(self._reward_overlays)
                    except Exception:
                        pass
                    self._reward_overlay_dirty = False
            elif self._reward_overlay_applied:
                try:
                    self.renderer.update_reward_overlays(
                        [],
                        alpha=self._reward_overlay_alpha,
                        value_scale=self._reward_overlay_value_scale,
                        segments=self._reward_overlay_segments,
                    )
                except Exception:
                    pass
                self._reward_overlay_applied = False
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

    def add_render_callback(self, callback: Callable[["EnvRenderer"], None]) -> None:
        if not callable(callback):
            raise TypeError("Render callback must be callable")
        if callback not in self._render_callbacks:
            self._render_callbacks.append(callback)

    def clear_render_callbacks(self) -> None:
        self._render_callbacks.clear()

    @staticmethod
    def _coerce_bool_flag(value: Any, *, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "y", "1", "on"}:
                return True
            if lowered in {"false", "no", "n", "0", "off"}:
                return False
            return default
        return bool(value)

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
    
