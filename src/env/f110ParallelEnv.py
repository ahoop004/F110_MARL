from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Tuple, Sequence


# base classes
from src.physics import Simulator, Integrator
# Lazy import to avoid pyglet initialization on HPC without display
# from src.render import EnvRenderer  # Moved to render() method
from src.env.start_pose_state import StartPoseState
from src.env.centerline_state import (
    CenterlineProgressTracker,
    CenterlineRuntimeState,
    apply_centerline_to_renderer,
    inject_finish_line_info,
    parse_finish_line,
    reset_finish_line_tracking,
    signed_distance_to_finish,
    update_finish_line_progress,
)
from src.env.collision_state import (
    apply_episode_termination_policy,
    build_step_terminations,
    build_truncations,
    normalize_episode_termination_mode,
    update_collision_flags,
)
from src.env.info_builder import (
    add_episode_metadata,
    add_step_info_fields,
    add_time_limit_info,
    build_reset_info_payloads,
    build_step_facts,
    filter_info_payloads,
)
from src.env.map_config import normalize_map_identifier, resolve_map_runtime_config
from src.env.map_schedule import MapScheduler
from src.env.obs_assembly import split_joint_obs
from src.env.render_adapter import (
    build_render_observations,
    compute_relative_snapshot,
    flush_render_state,
)
from src.env.spawn_manager import SpawnManager
from src.env.spaces_builder import build_action_spaces, build_observation_spaces
from src.env.state_views import build_agent_state, build_global_state, central_state_tensor
from src.env.state_buffer import StateBuffers
from src.env.types import AgentState, GlobalState
from src.render.render_state import RenderRuntimeState, parse_heatmap_config, parse_overlay_config

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

# rendering
# VIDEO_W = 600
# VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800

logger = logging.getLogger(__name__)


def _parse_vehicle_colors(color_map: Mapping[str, Any]) -> Dict[str, tuple]:
    """Convert a scenario ``vehicle_colors`` dict to normalized RGBA float tuples.

    Accepted per-agent formats
    --------------------------
    - Hex string: ``"#e8503c"`` or ``"#e8503cff"`` (3-byte or 4-byte)
    - RGB list/tuple: ``[0.91, 0.31, 0.23]`` (values in [0, 1])
    - RGBA list/tuple: ``[0.91, 0.31, 0.23, 1.0]``

    Returns a dict of ``{agent_id: (r, g, b, a)}`` with floats in ``[0, 1]``.
    Malformed entries are skipped with a warning.
    """
    result: Dict[str, tuple] = {}
    for aid, raw in color_map.items():
        try:
            result[str(aid)] = _color_to_rgba(raw)
        except (ValueError, TypeError) as exc:
            logger.warning("vehicle_colors: skipping agent %r — %s", aid, exc)
    return result


def _color_to_rgba(raw: Any) -> tuple:
    """Convert a single color spec to a normalized (r, g, b, a) float tuple."""
    if isinstance(raw, str):
        # Hex string
        s = raw.strip().lstrip("#")
        if len(s) == 6:
            r, g, b = int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
            return (r / 255.0, g / 255.0, b / 255.0, 1.0)
        if len(s) == 8:
            r, g, b, a = int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16), int(s[6:8], 16)
            return (r / 255.0, g / 255.0, b / 255.0, a / 255.0)
        raise ValueError(f"Hex color must be 6 or 8 hex digits, got {len(s)}: {raw!r}")
    if isinstance(raw, (list, tuple)):
        if len(raw) == 3:
            return (float(raw[0]), float(raw[1]), float(raw[2]), 1.0)
        if len(raw) == 4:
            return (float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))
        raise ValueError(f"Color list must have 3 (RGB) or 4 (RGBA) elements, got {len(raw)}")
    raise TypeError(f"Unsupported color type {type(raw).__name__!r}: {raw!r}")

DEFAULT_AGENT_SENSORS = (
    "lidar",
    "pose",
    "velocity",
    "angular_velocity",
    "lap",
    "collision",
)

class F110ParallelEnv:

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
        self._map_split_mode = str(merged.get("map_split_mode", "train")).strip().lower()
        self._map_scheduler = MapScheduler(merged, rng=self.rng)
        self._map_bundle_active = self._map_scheduler.active_bundle
        self.walls = getattr(map_data, "walls", None) if map_data is not None else None
        self.walls_path = getattr(map_data, "walls_path", None) if map_data is not None else None
        self._track_mask = getattr(map_data, "track_mask", None) if map_data is not None else None
        self.info_level = str(merged.get("info_level", "training")).strip().lower()
        self.last_step_facts = None
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
        episode_termination = merged.get("episode_termination", {}) or {}
        if not isinstance(episode_termination, Mapping):
            raise TypeError("environment.episode_termination must be a mapping")
        legacy_any_done = merged.get("terminate_on_any_done")
        default_mode = "any_agent" if legacy_any_done is not False else "all_agents"
        self.episode_termination_mode = normalize_episode_termination_mode(
            episode_termination.get("mode", default_mode)
        )
        self.episode_done = False

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

        # Episode step counters — also set in reset(); initialised here so
        # any accidental pre-reset step() call raises a clean error rather
        # than an AttributeError.
        self._episode_step_count = 0
        self._lock_speed_steps = 0
        self._locked_velocities: Dict[str, float] = {}

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
        self._spawn_manager = SpawnManager(
            merged, map_data,
            possible_agents=self.possible_agents,
            agent_index=self._agent_id_to_index,
            rng=self.rng,
            seed=self.seed,
            map_split_mode=self._map_split_mode,
            init_metadata=meta,
        )

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
        default_lidar_skip = int(merged.get("lidar_beams", self._lidar_beam_count))
        if default_lidar_skip < 0:
            default_lidar_skip = 0
        self._render_state = RenderRuntimeState(
            lidar_skip_default=default_lidar_skip,
            lidar_skip={aid: default_lidar_skip for aid in self.possible_agents},
        )
        overlay = parse_overlay_config(merged)
        self._render_state.apply_overlay_config(overlay)
        heatmap = parse_heatmap_config(merged)
        self._render_state.apply_heatmap_config(heatmap)

        self._single_action_space, self.action_spaces = build_action_spaces(
            self.possible_agents,
            self.params,
        )

        # Centerline progress tracker — computes per-step projection facts when
        # centerline_features is enabled.  _last_centerline_facts is updated each
        # step and consumed by get_agent_state() and info injection.
        self._centerline_progress_tracker = CenterlineProgressTracker(
            agent_ids=self.possible_agents,
        )
        self._last_centerline_facts: Dict[str, Dict[str, float]] = {}

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

        # Parse scenario-level vehicle color overrides from environment.rendering.vehicle_colors
        rendering_cfg = cfg.get("rendering", {}) or {}
        self._vehicle_colors: Dict[str, tuple] = _parse_vehicle_colors(
            rendering_cfg.get("vehicle_colors", {}) or {}
        )

        self._centerline_state = CenterlineRuntimeState.from_config(cfg)

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
        self.episode_done = False
        self.controlled_agents = list(cfg.get("controlled_agents") or self.possible_agents)
        self.trainable_agents = list(cfg.get("trainable_agents") or [])
        self.fixed_policy_agents = list(cfg.get("fixed_policy_agents") or [])

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
        return normalize_map_identifier(identifier)

    def _configure_map_paths(self, cfg: Mapping[str, Any], map_data: Optional[Any]) -> None:
        runtime = resolve_map_runtime_config(cfg, map_data)
        self._map_runtime = runtime
        self.map_dir = runtime.map_dir
        self.map_ext = runtime.map_ext
        self.map_name = runtime.map_name
        self.map_yaml = runtime.map_yaml
        self.map_path = runtime.map_path
        self.yaml_path = runtime.yaml_path

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
        runtime = getattr(self, "_map_runtime", None)
        if runtime is None:
            runtime = resolve_map_runtime_config(cfg, map_data)
            self._map_runtime = runtime
        return runtime.metadata, runtime.image_path, runtime.image_size

    def _apply_map_data(
        self,
        map_data: Any,
        bundle: Optional[str] = None,
        *,
        keep_centerline: bool = False,
    ) -> None:
        """Apply a loaded MapData object to the env, sim, and renderer.

        Parameters
        ----------
        map_data:
            Populated map-data object from :class:`~src.utils.map_loader.MapLoader`.
        bundle:
            Bundle name to record as the active bundle (when cycling maps).
        keep_centerline:
            When ``True``, preserve the existing in-memory centerline rather
            than loading the new map's centerline.  Used by :meth:`update_map`
            which hot-swaps the map surface but keeps the loaded centerline.
        """
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
        self._spawn_manager.update_map_data(map_data, self.map_meta)

        # Update simulation + renderer
        self.sim.set_map(str(self.yaml_path), self.map_ext)
        if self.renderer is not None:
            self.renderer.update_map(
                str(self.yaml_path.with_suffix("")),
                self.map_ext,
                map_meta=self.map_meta,
                map_image_path=self.map_image_path,
            )

        # Update centerline
        if keep_centerline:
            # Preserve in-memory centerline; re-apply to new renderer surface.
            self._update_renderer_centerline()
        else:
            self.set_centerline(map_data.centerline, path=map_data.centerline_path)

        # Update observation bounds based on new map
        width, height = map_data.image_size
        R = float(self.map_meta.get("resolution", 1.0))
        x0, y0, _ = self.map_meta.get("origin", (0.0, 0.0, 0.0))
        self._build_observation_spaces(x0, x0 + width * R, y0, y0 + height * R)

        if bundle is not None:
            self._map_bundle_active = bundle
            self._map_scheduler.active_bundle = bundle

    def _maybe_cycle_map(self) -> None:
        bundle = self._map_scheduler.select_next_bundle(self._map_split_mode)
        if bundle is None or bundle == self._map_scheduler.active_bundle:
            return
        map_data = self._map_scheduler.load_bundle(
            bundle,
            map_ext=self.map_ext,
            centerline_render=self.centerline_render_enabled,
            centerline_features=self.centerline_features_enabled,
        )
        self._apply_map_data(map_data, bundle=bundle)

    def action_space(self, agent: str):
        return self.action_spaces[agent]

    def _update_state(self, obs_dict):
        self.state_buffers.update(obs_dict)

    def _refresh_render_observations(self, obs: Dict[str, Dict[str, Any]]) -> None:
        if not self._collect_render_data:
            self._render_state.render_obs = {}
            return
        self._render_state.render_obs = build_render_observations(
            self.agents,
            obs,
            agent_index=self._agent_id_to_index,
            agent_target_index=self._agent_target_index,
            poses_x=self.poses_x,
            poses_y=self.poses_y,
            poses_theta=self.poses_theta,
            linear_vels_x=self.linear_vels_x_curr,
            linear_vels_y=self.linear_vels_y_curr,
            lap_times=self.lap_times,
            lap_counts=self.lap_counts,
            collisions=self.collisions,
            render_state=self._render_state,
        )

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
        self._render_state.metrics_payload = payload
        self._render_state.metrics_dirty = True

    def update_render_wrapped_observations(self, wrapped: Mapping[str, np.ndarray]) -> None:
        self._render_state.set_wrapped_observations(wrapped)

    def append_render_ticker(
        self,
        agent_id: str,
        *,
        step: int,
        reward: float,
        components: Optional[Mapping[str, Any]] = None,
    ) -> None:
        snapshot = compute_relative_snapshot(
            agent_id,
            agent_index=self._agent_id_to_index,
            agent_target_index=self._agent_target_index,
            poses_x=self.poses_x,
            poses_y=self.poses_y,
            poses_theta=self.poses_theta,
            reward_ring_config=self._render_state.reward_ring_config,
        )
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

        self._render_state.append_ticker_line(line)

    def configure_reward_ring(self, config: Optional[Dict[str, Any]], *, agent_id: Optional[str] = None) -> None:
        self._render_state.configure_reward_ring(config, agent_id=agent_id)

    def update_reward_ring_target(self, agent_id: str, target_id: Optional[str]) -> None:
        self._render_state.update_reward_ring_target(agent_id, target_id)

    def update_reward_ring_markers(self, agent_id: str, states: Optional[Sequence[bool]]) -> None:
        self._render_state.update_reward_ring_markers(agent_id, states)

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

    def get_target_id(self, agent_id: str) -> Optional[str]:
        """Return the configured target agent ID for *agent_id*, if any."""

        target_idx = self._agent_target_index.get(agent_id)
        if target_idx is None:
            return None
        if target_idx < 0 or target_idx >= len(self.possible_agents):
            return None
        return self.possible_agents[target_idx]

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
        self._render_state.update_reward_overlays(
            overlays,
            enabled=enabled,
            alpha=alpha,
            value_scale=value_scale,
            segments=segments,
        )

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
        self._render_state.update_reward_heatmap(
            heatmap,
            enabled=enabled,
            alpha=alpha,
            value_scale=value_scale,
            extent_m=extent_m,
            cell_size_m=cell_size_m,
        )

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
            self._spawn_manager.reseed(seed_value, self.rng)
        self._maybe_cycle_map()
        self.agents = self.possible_agents.copy()
        self.episode_done = False
        self._elapsed_steps = 0
        self.current_time = 0.0

        # Reset persistent collision tracking
        self._collision_flags.fill(False)
        self._collision_steps.fill(-1)

        self.start_state.reset()
        self.state_buffers.reset()
        self._render_state.reset_episode()
        self._spawn_manager.reset_episode()
        self._last_centerline_facts = {}
        self._centerline_progress_tracker.reset()

        # Speed locking for curriculum
        self._lock_speed_steps = 0
        self._locked_velocities = {}
        self._episode_step_count = 0
        if self.renderer is not None:
            self.renderer.reset_state()
            self._update_renderer_centerline()
            self._render_state.reset_renderer_payloads()

        # Extract a deterministic SpawnPlan if provided via options (e.g. curriculum).
        _spawn_plan = None
        if isinstance(options, dict) and "spawn_plan" in options:
            _spawn_plan = options["spawn_plan"]

        spawn_result = self._spawn_manager.resolve(
            options,
            centerline=self.centerline_points,
            walls=self.walls,
            start_poses=getattr(self, "start_poses", None),
            spawn_plan=_spawn_plan,
        )
        poses = spawn_result.poses
        velocities = spawn_result.velocities
        spawn_mapping = dict(spawn_result.spawn_mapping)
        self._locked_velocities = dict(spawn_result.locked_velocities)
        self._lock_speed_steps = int(spawn_result.lock_speed_steps)
        if spawn_result.update_start_poses and poses is not None:
            self._update_start_from_poses(poses)
            poses = self.start_poses

        # options: (N,3) poses (x,y,theta). If None, caller must set internally.
        # poses = options if options is not None else np.zeros((self.n_agents, 3), dtype=np.float32)
        obs_joint = self.sim.reset(poses, velocities=velocities)
        obs = self._split_obs(obs_joint)
        self._attach_central_state(obs, obs_joint)
        self._update_state(obs_joint)
        self._refresh_render_observations(obs)
        self._reset_finish_line_tracking()

        infos = build_reset_info_payloads(
            agent_ids=self.agents,
            map_bundle=self._map_bundle_active,
            spawn_mapping=spawn_mapping,
            spawn_metadata=self._spawn_manager.last_spawn_metadata,
            finish_line_data=self._finish_line_data,
            finish_crossed=self._finish_crossed,
            agent_id_to_index=self._agent_id_to_index,
            info_level=self.info_level,
        )
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

        update_collision_flags(
            self.possible_agents,
            collisions,
            self._collision_flags,
            self._collision_steps,
            self._elapsed_steps,
        )
        active_before_step = tuple(self.agents)
        terminations = build_step_terminations(
            self.possible_agents,
            self._collision_flags,
            lap_completion,
            self.terminate_on_collision,
        )
        truncations, trunc_flag = build_truncations(
            self.possible_agents,
            max_steps=self.max_steps,
            elapsed_steps=self._elapsed_steps,
        )
        terminations, self.episode_done = apply_episode_termination_policy(
            terminations,
            truncations,
            active_agents=active_before_step,
            possible_agents=self.possible_agents,
            trainable_agents=self.trainable_agents,
            mode=self.episode_termination_mode,
        )
        infos = {aid: {} for aid in self.possible_agents}
        add_time_limit_info(infos, truncated=trunc_flag)
        self._inject_finish_line_info(infos)
        add_episode_metadata(
            infos,
            map_bundle=self._map_bundle_active,
            spawn_metadata=self._spawn_manager.last_spawn_metadata,
        )

        add_step_info_fields(
            infos,
            possible_agents=self.possible_agents,
            agent_target_index=self._agent_target_index,
            collision_flags=self._collision_flags,
            finish_crossed=self._finish_crossed,
            locked_velocities=self._locked_velocities,
            lock_speed_steps=self._lock_speed_steps,
            episode_step_count=self._episode_step_count,
        )

        # Centerline projection facts — injected before filtering so that both
        # CenterlineRewardComponent and ProgressComponent can read info["centerline"].
        if self.centerline_features_enabled and self.centerline_points is not None:
            self._last_centerline_facts = self._centerline_progress_tracker.update(
                self.centerline_points,
                self.poses_x,
                self.poses_y,
                self.poses_theta,
                self.linear_vels_x_curr,
                self.linear_vels_y_curr,
                self._agent_id_to_index,
            )
            for agent_id, facts in self._last_centerline_facts.items():
                if agent_id in infos:
                    infos[agent_id]["centerline"] = facts
        else:
            self._last_centerline_facts = {}

        infos = filter_info_payloads(infos, info_level=self.info_level)
        self.last_step_facts = build_step_facts(
            agent_ids=self.possible_agents,
            agent_states={
                agent_id: self.get_agent_state(agent_id)
                for agent_id in self.possible_agents
            },
            global_state=self.get_global_state(),
            collision_flags=self._collision_flags,
            terminations=terminations,
            truncations=truncations,
            infos=infos,
        )

        # advance and cull finished agents
        self._elapsed_steps += 1
        self.agents = [
            aid
            for aid in active_before_step
            if not (terminations[aid] or truncations[aid])
        ]
        if self.episode_done:
            self.agents = []

        return obs, rewards, terminations, truncations, infos

    # ------------------------------------------------------------------
    # Finish line helpers
    # ------------------------------------------------------------------
    def _parse_finish_line(self, cfg: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
        return parse_finish_line(cfg)

    def _reset_finish_line_tracking(self) -> None:
        reset_finish_line_tracking(
            self._finish_line_data,
            self._finish_signed_prev,
            self._finish_crossed,
            self.poses_x,
            self.poses_y,
            self.n_agents,
        )

    def _signed_distance_to_finish(self, point: np.ndarray) -> float:
        return signed_distance_to_finish(self._finish_line_data, point)

    def _update_finish_line_progress(self) -> Dict[str, bool]:
        return update_finish_line_progress(
            self._finish_line_data,
            self._finish_signed_prev,
            self._finish_crossed,
            self.possible_agents,
            self.poses_x,
            self.poses_y,
            self.linear_vels_x_curr,
            self.linear_vels_y_curr,
        )

    def _inject_finish_line_info(self, infos: Mapping[str, Dict[str, Any]]) -> None:
        inject_finish_line_info(
            self._finish_line_data,
            self._finish_crossed,
            self._agent_id_to_index,
            infos,
        )

    def update_map(self, map_path: str, map_ext: str) -> None:
        """Hot-swap the map at runtime (public API, legacy compatible).

        Preserves the in-memory centerline — the map surface (image, YAML)
        and simulator state are updated but the loaded centerline is kept.
        """
        map_data = self._map_scheduler.load_from_path(map_path, map_ext)
        self._apply_map_data(map_data, keep_centerline=True)

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
            # Apply scenario-configurable vehicle color overrides
            if self._vehicle_colors:
                self.renderer.set_agent_colors(self._vehicle_colors)
            # use self.map_path (without extension) and self.map_ext
            self.renderer.update_map(
                str(self.map_path.with_suffix("")),
                self.map_ext,
                map_meta=self.map_meta,
                map_image_path=self.map_image_path,
                centerline_points=(
                    self._centerline_state.render_points
                    if self._centerline_state.render_enabled
                    else None
                ),
                centerline_connect=self.centerline_render_connect,
            )
            self._render_state.reward_ring_dirty = True
            self._render_state.reward_ring_target_dirty = True

        flush_render_state(self.renderer, self._render_state, _logger=logger)

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
        self._render_state.add_callback(callback)

    def clear_render_callbacks(self) -> None:
        self._render_state.callbacks.clear()

    def _build_render_centerline_points(self) -> Optional[np.ndarray]:
        return self._centerline_state.build_render_points()

    def _update_renderer_centerline(self) -> None:
        apply_centerline_to_renderer(self.renderer, self._centerline_state)

    def register_centerline_usage(self, *, require_render: bool = False, require_features: bool = False) -> None:
        if self._centerline_state.register_usage(
            require_render=require_render,
            require_features=require_features,
        ):
            self._update_renderer_centerline()

    def set_centerline(self, centerline: Optional[np.ndarray], *, path: Optional[Path] = None) -> None:
        self._centerline_state.set_centerline(centerline, path=path)
        self._update_renderer_centerline()

    @property
    def centerline_points(self) -> Optional[np.ndarray]:
        return self._centerline_state.points

    @property
    def centerline_path(self) -> Optional[Path]:
        return self._centerline_state.path

    @property
    def centerline_render_enabled(self) -> bool:
        return self._centerline_state.render_enabled

    @property
    def centerline_features_enabled(self) -> bool:
        return self._centerline_state.features_enabled

    @property
    def centerline_render_connect(self) -> bool:
        return self._centerline_state.render_connect
    
    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def _build_observation_spaces(self, x_min: float, x_max: float, y_min: float, y_max: float) -> None:
        self.observation_spaces = build_observation_spaces(
            possible_agents=self.possible_agents,
            agent_sensor_spec=self._agent_sensor_spec,
            default_sensors=DEFAULT_AGENT_SENSORS,
            central_state_dim=self._central_state_dim,
            lidar_beam_count=self._lidar_beam_count,
            lidar_range=self.lidar_range,
            vehicle_params=self.params,
            target_laps=getattr(self, "target_laps", 1),
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
        )

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
        return central_state_tensor(
            joint,
            n_agents=self.n_agents,
            central_state_keys=self._central_state_keys,
        )

    def _attach_central_state(self, obs: Dict[str, Dict[str, np.ndarray]], joint: Dict[str, np.ndarray]) -> None:
        central_state = self._central_state_tensor(joint)
        for aid in self.possible_agents:
            if aid in obs:
                obs[aid]["state"] = central_state

    def get_agent_state(self, agent_id: str) -> AgentState:
        if agent_id not in self._agent_id_to_index:
            raise KeyError(f"unknown agent_id: {agent_id}")
        return build_agent_state(
            agent_id,
            agent_index=self._agent_id_to_index,
            poses_x=self.poses_x,
            poses_y=self.poses_y,
            poses_theta=self.poses_theta,
            linear_vels_x=self.linear_vels_x_curr,
            linear_vels_y=self.linear_vels_y_curr,
            angular_vels=self.angular_vels_curr,
            collision_flags=self._collision_flags,
            lap_counts=self.lap_counts,
            lap_times=self.lap_times,
            finish_crossed=self._finish_crossed,
            centerline_facts=self._last_centerline_facts.get(agent_id),
            metadata={
                "map_bundle": self._map_bundle_active,
                **self._spawn_manager.last_spawn_metadata,
            },
        )

    def get_global_state(self) -> GlobalState:
        joint = self.sim.current_observation()
        central = self._central_state_tensor(joint)
        return build_global_state(
            possible_agents=self.possible_agents,
            active_agents=self.agents,
            central_vector=central,
            controlled_agents=self.controlled_agents,
            trainable_agents=self.trainable_agents,
            metadata={
                "map_bundle": self._map_bundle_active,
                **self._spawn_manager.last_spawn_metadata,
            },
        )

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
        return split_joint_obs(
            joint,
            possible_agents=self.possible_agents,
            agent_sensor_spec=self._agent_sensor_spec,
            agent_target_index=self._agent_target_index,
            default_sensors=DEFAULT_AGENT_SENSORS,
            lidar_beam_count=self._lidar_beam_count,
            timestep=self.timestep,
            lap_counts=self.lap_counts,
            lap_times=self.lap_times,
            prev_vels_x=self.linear_vels_x_curr,
            prev_vels_y=self.linear_vels_y_curr,
            velocity_initialized=self.state_buffers.velocity_initialized,
            fallback_collisions=self.collisions,
        )
