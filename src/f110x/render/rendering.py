# MIT License
# Rendering engine for F1TENTH-style env using pyglet + OpenGL
# Refactored for PettingZoo MARL with per-agent render_obs dict.

import os
import time

_PYGLET_HEADLESS = os.environ.get("PYGLET_HEADLESS", "").strip().lower() in {"1", "true", "yes", "on"}

import pyglet

if _PYGLET_HEADLESS:
    pyglet.options["headless"] = True
    try:
        pyglet.options["shadow_window"] = False
    except Exception:
        pass
    try:
        pyglet.options["debug_gl"] = False
    except Exception:
        pass

_PYGLET_IMPORT_ERROR = None

try:
    from pyglet.gl import *  # noqa: F401,F403 - re-exported GL symbols
    from pyglet.math import Mat4
    from pyglet.graphics import ShaderGroup
    from pyglet.window import key
    _PYGLET_AVAILABLE = True
except Exception as exc:  # pragma: no cover - headless fallback
    _PYGLET_AVAILABLE = False
    _PYGLET_IMPORT_ERROR = exc
    Mat4 = None  # type: ignore[assignment]
    ShaderGroup = None  # type: ignore[assignment]
    key = None  # type: ignore[assignment]

import numpy as np
from array import array
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence
from PIL import Image
import yaml
import pandas as pd

# Adjust import to your project layout:
# from f110_gym.envs.collision_models import get_vertices
from f110x.physics.collision_models import get_vertices
from .shader import get_default_shader

# zoom constants
ZOOM_IN_FACTOR = 1.2
ZOOM_OUT_FACTOR = 1.0 / ZOOM_IN_FACTOR

# vehicle shape constants (meters)
CAR_LENGTH = 0.58
CAR_WIDTH  = 0.31

# colors (normalized RGBA)
def _rgba255(r: float, g: float, b: float, a: float = 255.0) -> tuple[float, float, float, float]:
    return (
        float(r) / 255.0,
        float(g) / 255.0,
        float(b) / 255.0,
        float(a) / 255.0,
    )


CAR_LEARNER = _rgba255(183, 193, 222)
CAR_OTHER   = _rgba255( 99,  52,  94)
MAP_COLOR   = _rgba255(255, 193,  50)
CENTERLINE_COLOR = _rgba255(102, 255, 102)
CENTERLINE_POINT_COLOR = _rgba255(255, 255, 255)
LIDAR_COLOR_HIT = (1.0, 0.0, 0.0, 1.0)
LIDAR_COLOR_MAX = _rgba255(180, 180, 180)
REWARD_RING_FILL_COLOR = _rgba255(120, 220, 120, 96)
REWARD_RING_BORDER_COLOR = _rgba255(56, 182, 86, 220)
REWARD_RING_PREFERRED_COLOR = _rgba255(255, 255, 255, 255)

if not _PYGLET_AVAILABLE:
    class EnvRenderer:
        def __init__(self, *_, **__):
            raise RuntimeError(
                "pyglet is unavailable or failed to initialize (set render_mode=null for headless runs): "
                + str(_PYGLET_IMPORT_ERROR)
            )

else:
    class EnvRenderer(pyglet.window.Window):
        """
        Pyglet-based renderer for a multi-agent racecar environment.
        Consumes per-agent render_obs dicts: {agent_id: {poses_x, poses_y, poses_theta, scans, lap_time?, lap_count?}}
        """
        def __init__(self, width, height, lidar_fov=4.7, max_range=30.0, *args, **kwargs):
            conf = Config(sample_buffers=1, samples=4, depth_size=16, double_buffer=True)
            super().__init__(width, height, config=conf, resizable=True, vsync=False,  *args, **kwargs)

            # GL init
            glClearColor(9/255, 32/255, 87/255, 1.0)

            # camera
            self.left = -width / 2
            self.right = width / 2
            self.bottom = -height / 2
            self.top = height / 2
            self.zoom_level = 1.2
            self.zoomed_width = width
            self.zoomed_height = height
            self._pan_step_pixels = 50.0
            self._follow_enabled = True

            # shader + batch
            self.shader = get_default_shader()
            self.shader_group = ShaderGroup(self.shader)
            self.batch = pyglet.graphics.Batch()
            self.point_size = 4.0
            self.shader['point_size'] = float(self.point_size)

            # map
            self.map_points = None
            self.map_vlist = None
            self._map_vertex_count = 0
            self._centerline_vlist = None
            self._centerline_points_vlist = None

            # per-agent drawables and cached state
            self.cars_vlist = {}       # aid -> vertex_list (GL_QUADS)
            self.scan_hits_vlist = {}  # aid -> vertex_list (GL_POINTS)
            self.agent_infos = {}      # aid -> last render_obs snapshot
            self.agent_ids = []        # agents updated this frame (sorted)
            self._camera_target = None
            self._user_camera_target = None
            self._follow_padding_m = 16.0
            self._lidar_buffer_cache = {}
            self._reward_ring_enabled = False
            self._reward_ring_config = {}
            self._reward_ring_target = None
            self._reward_ring_segments = 0
            self._reward_ring_angles = None
            self._reward_ring_angles_ext = None
            self._reward_ring_cos = None
            self._reward_ring_sin = None
            self._reward_ring_cos_ext = None
            self._reward_ring_sin_ext = None
            self._reward_ring_fill_vlist = None
            self._reward_ring_outer_vlist = None
            self._reward_ring_inner_vlist = None
            self._reward_ring_pref_vlist = None
            self._reward_ring_fill_positions = None
            self._reward_ring_fill_view = None
            self._reward_ring_outer_positions = None
            self._reward_ring_outer_view = None
            self._reward_ring_inner_positions = None
            self._reward_ring_inner_view = None
            self._reward_ring_pref_positions = None
            self._reward_ring_pref_view = None

            # options
            self.lidar_fov = lidar_fov
            self.max_range = max_range
            self.render_scale = 50.0  # meters->pixels

            # HUD overlays drawn in screen space
            self.fps_display = pyglet.window.FPSDisplay(self)
            self.hud_label = pyglet.text.Label(
                'Agents: 0',
                font_size=16,
                x=10,
                y=height - 10,
                anchor_x='left',
                anchor_y='top',
                color=(255, 255, 255, 255),
                multiline=True,
                width=max(width - 20, 100),
            )
            self.ticker_label = pyglet.text.Label(
                '',
                font_size=12,
                x=10,
                y=10,
                anchor_x='left',
                anchor_y='bottom',
                color=(200, 200, 200, 255),
                multiline=True,
                width=max(width - 20, 100),
            )
            self._telemetry_phase: Optional[str] = None
            self._telemetry_metrics: Dict[str, Any] = {}
            self._telemetry_step: Optional[float] = None
            self._telemetry_timestamp: Optional[float] = None
            self._ticker_lines: List[str] = []

        # ---------- Map ----------

        def update_map(
            self,
            map_path_no_ext: str,
            map_ext: str,
            *,
            map_meta=None,
            map_image_path=None,
            centerline_points: Optional[np.ndarray] = None,
            centerline_connect: bool = True,
        ):
            """
            Update map geometry.
            map_path_no_ext: absolute path WITHOUT extension (e.g., '/.../maps/levine')
            map_ext: image extension (e.g., '.png')
            """
            yaml_path = map_path_no_ext + '.yaml'
            if map_meta is None:
                raise ValueError('update_map requires preloaded map_meta metadata')
            meta = map_meta

            if map_image_path is not None:
                img_path = str(map_image_path)
            else:
                image_field = meta.get('image')
                if image_field:
                    img_path = str((Path(yaml_path).parent / image_field).resolve())
                else:
                    img_path = map_path_no_ext + map_ext
            res = float(meta['resolution'])
            origin = meta['origin']
            ox, oy = origin[0], origin[1]

            with Image.open(img_path) as pil_img:
                pil_img = pil_img.convert('L').transpose(Image.FLIP_TOP_BOTTOM)
                img_gray = np.asarray(pil_img, dtype=np.float32)

            img_norm = img_gray / 255.0
            if int(meta.get('negate', 0)):
                # negate==1 flips semantics in ROS; do not invert again so walls stay dark
                occ_prob = img_norm
            else:
                # ROS map convention: dark pixels are obstacles -> invert intensity
                occ_prob = 1.0 - img_norm

            occ_thresh = float(meta.get('occupied_thresh', 0.65))
            free_thresh = float(meta.get('free_thresh', 0.196))
            if occ_thresh <= free_thresh:
                occ_thresh = free_thresh + np.finfo(np.float32).eps

            H, W = occ_prob.shape[0], occ_prob.shape[1]

            xs = (np.arange(W) * res + ox)
            ys = (np.arange(H) * res + oy)
            gx, gy = np.meshgrid(xs, ys)

            mask = occ_prob >= occ_thresh  # obstacle pixels based on occupancy probability
            flat_idx = np.flatnonzero(mask)
            max_points = 200_000
            if flat_idx.size > max_points:
                stride = (flat_idx.size + max_points - 1) // max_points
                flat_idx = flat_idx[::stride]
                if flat_idx.size > max_points:
                    flat_idx = flat_idx[:max_points]

            pts = np.empty((flat_idx.size, 3), dtype=np.float32)
            if flat_idx.size:
                gx_flat = gx.ravel()
                gy_flat = gy.ravel()
                pts[:, 0] = gx_flat[flat_idx]
                pts[:, 1] = gy_flat[flat_idx]
                pts[:, 2] = 0.0
            pts *= self.render_scale

            N = pts.shape[0]
            positions = pts[:, :2].astype(np.float32, copy=False).ravel().tolist()
            colors = list(MAP_COLOR) * N

            reuse = self.map_vlist is not None
            if not reuse:
                self.map_vlist = self.shader.vertex_list(
                    N, pyglet.gl.GL_POINTS, batch=self.batch, group=self.shader_group,
                    position=('f', positions),
                    color=('f', colors)
                )
            else:
                if N != self._map_vertex_count:
                    self.map_vlist.resize(N)
            if self.map_vlist is not None:
                self.map_vlist.position[:] = positions
                self.map_vlist.color[:] = colors
            self._map_vertex_count = N
            self.map_points = pts

            self.update_centerline(centerline_points, connect=centerline_connect)

        def update_centerline(
            self,
            centerline_points: Optional[np.ndarray],
            *,
            connect: bool = True,
        ) -> None:
            if self._centerline_vlist is not None:
                try:
                    self._centerline_vlist.delete()
                except Exception:
                    pass
                self._centerline_vlist = None
            if self._centerline_points_vlist is not None:
                try:
                    self._centerline_points_vlist.delete()
                except Exception:
                    pass
                self._centerline_points_vlist = None

            if centerline_points is None:
                return

            points_np = np.asarray(centerline_points, dtype=np.float32)
            if points_np.ndim != 2 or points_np.shape[0] == 0:
                return

            verts = (points_np[:, :2] * self.render_scale).astype(np.float32, copy=False).ravel().tolist()
            if connect:
                colors = list(CENTERLINE_COLOR) * points_np.shape[0]
                try:
                    self._centerline_vlist = self.shader.vertex_list(
                        points_np.shape[0],
                        pyglet.gl.GL_LINE_STRIP,
                        batch=self.batch,
                        group=self.shader_group,
                        position=('f', verts),
                        color=('f', colors),
                    )
                except Exception:
                    self._centerline_vlist = None
            try:
                self._centerline_points_vlist = self.shader.vertex_list(
                    points_np.shape[0],
                    pyglet.gl.GL_POINTS,
                    batch=self.batch,
                    group=self.shader_group,
                    position=('f', verts),
                    color=('f', list(CENTERLINE_POINT_COLOR) * points_np.shape[0]),
                )
            except Exception:
                self._centerline_points_vlist = None

        def reset_state(self):
            for v in self.cars_vlist.values():
                try:
                    v.delete()
                except Exception:
                    pass
            for v in self.scan_hits_vlist.values():
                try:
                    v.delete()
                except Exception:
                    pass
            self.cars_vlist.clear()
            self.scan_hits_vlist.clear()
            self.agent_infos.clear()
            self.agent_ids = []
            self._camera_target = None
            self.hud_label.text = 'Agents: 0'
            self.ticker_label.text = ''
            self._ticker_lines.clear()
            if self._centerline_vlist is not None:
                try:
                    self._centerline_vlist.delete()
                except Exception:
                    pass
                self._centerline_vlist = None
            if self._centerline_points_vlist is not None:
                try:
                    self._centerline_points_vlist.delete()
                except Exception:
                    pass
                self._centerline_points_vlist = None
            self._clear_reward_ring_geometry()
            self._reward_ring_target = None

        # ---------- Window / Camera ----------

        def on_resize(self, width, height):
            super().on_resize(width, height)
            glViewport(0, 0, width, height)
            self.left = -self.zoom_level * width / 2
            self.right = self.zoom_level * width / 2
            self.bottom = -self.zoom_level * height / 2
            self.top = self.zoom_level * height / 2
            self.zoomed_width = self.zoom_level * width
            self.zoomed_height = self.zoom_level * height
            max_width = max(width - 20, 100)
            self.hud_label.y = height - 10
            self.hud_label.width = max_width
            self.ticker_label.width = max_width

        def on_mouse_drag(self, x, y, dx, dy, _buttons, _modifiers):
            if self._follow_enabled:
                self._follow_enabled = False
                self._user_camera_target = None
            self._pan_view(dx, dy)

        def on_mouse_scroll(self, x, y, dx, dy):
            factor = ZOOM_IN_FACTOR if dy > 0 else ZOOM_OUT_FACTOR if dy < 0 else 1.0
            if factor != 1.0:
                w, h = self.get_size()
                self._apply_zoom(factor, (x / max(w, 1)), (y / max(h, 1)))

        def on_close(self):
            super().on_close()
            # Let the env catch this by calling renderer.close() instead of raising here.

        def close(self):
            # explicit closer for env.close()
            try:
                self.reset_state()
                super().close()
            except Exception:
                pass

        # ---------- Draw ----------

        def on_draw(self):
            if self.map_points is None:
                raise Exception('Map not set for renderer.')
            # cars_vlist created on first update
            self.clear()
            proj = Mat4.orthogonal_projection(self.left, self.right, self.bottom, self.top, -1, 1)
            self.shader.use()
            self.shader['projection'] = proj
            self.shader['point_size'] = float(self.point_size)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glEnable(GL_PROGRAM_POINT_SIZE)
            glPointSize(4)

            self.batch.draw()
            self.shader.stop()

            self.hud_label.draw()
            self.ticker_label.draw()
            self.fps_display.draw()

        # ---------- Per-Agent Update ----------

        def update_obs(self, render_obs: dict):
            """
            render_obs: {agent_id: {"scans","poses_x","poses_y","poses_theta", optional "lap_time","lap_count"}}
            """
            if render_obs is None:
                return

            active_ids = sorted(render_obs.keys())

            for cached in self.agent_infos.values():
                cached["__active__"] = False

            for aid in active_ids:
                state = render_obs[aid]
                cached = self.agent_infos.get(aid, {})
                for key, value in state.items():
                    if isinstance(value, np.ndarray):
                        cached[key] = np.array(value, copy=True)
                    else:
                        cached[key] = value
                cached["__active__"] = True
                self.agent_infos[aid] = cached

            if active_ids:
                self.agent_ids = active_ids
            elif not self.agent_ids and self.agent_infos:
                self.agent_ids = sorted(self.agent_infos.keys())

            if not self._follow_enabled:
                self._camera_target = None
            else:
                if self._user_camera_target and self._user_camera_target in self.agent_infos:
                    self._camera_target = self._user_camera_target
                elif self.agent_ids:
                    self._camera_target = self.agent_ids[0]
                elif self.agent_infos:
                    self._camera_target = sorted(self.agent_infos.keys())[0]
                else:
                    self._camera_target = None

            all_ids = sorted(self.agent_infos.keys())

            for aid in all_ids:
                st = self.agent_infos[aid]
                x = float(st.get("poses_x", 0.0))
                y = float(st.get("poses_y", 0.0))
                th = float(st.get("poses_theta", 0.0))

                verts_np = self.render_scale * get_vertices(np.array([x, y, th]), CAR_LENGTH, CAR_WIDTH)
                positions = verts_np.astype(np.float32, copy=False).flatten().tolist()

                car_vlist = self.cars_vlist.get(aid)
                if car_vlist is None:
                    color = CAR_LEARNER if len(self.cars_vlist) == 0 else CAR_OTHER
                    color_array = list(color) * 4
                    self.cars_vlist[aid] = self.shader.vertex_list(
                        4, pyglet.gl.GL_QUADS, batch=self.batch, group=self.shader_group,
                        position=('f', positions),
                        color=('f', color_array)
                    )
                else:
                    car_vlist.position[:] = positions

                scans = st.get("scans")
                if scans is None:
                    continue

                scans = np.asarray(scans, dtype=np.float32)
                n = scans.shape[0]
                if n == 0:
                    continue

                key = (n, float(self.lidar_fov))
                buffers = self._lidar_buffer_cache.get(key)
                if buffers is None:
                    base_angles = np.linspace(-self.lidar_fov / 2.0, self.lidar_fov / 2.0, n, dtype=np.float32)
                    angle_buffer = np.empty_like(base_angles)
                    cos_buffer = np.empty_like(base_angles)
                    sin_buffer = np.empty_like(base_angles)
                    positions_array = array('f', [0.0] * (2 * n))
                    positions_view = np.frombuffer(positions_array, dtype=np.float32)
                    colors_array = array('f', [0.0] * (4 * n))
                    colors_view = np.frombuffer(colors_array, dtype=np.float32)
                    buffers = {
                        'base_angles': base_angles,
                        'angles': angle_buffer,
                        'cos': cos_buffer,
                        'sin': sin_buffer,
                        'positions_array': positions_array,
                        'positions_view': positions_view,
                        'colors_array': colors_array,
                        'colors_matrix': colors_view.reshape((n, 4)),
                    }
                    self._lidar_buffer_cache[key] = buffers

                base_angles = buffers['base_angles']
                angles = buffers['angles']
                cos_vals = buffers['cos']
                sin_vals = buffers['sin']
                positions_array = buffers['positions_array']
                positions_flat = buffers['positions_view']
                colors_array = buffers['colors_array']
                colors_matrix = buffers['colors_matrix']

                np.add(base_angles, th, out=angles)
                np.cos(angles, out=cos_vals)
                np.sin(angles, out=sin_vals)

                np.multiply(scans, cos_vals, out=cos_vals)
                np.multiply(scans, sin_vals, out=sin_vals)

                np.add(cos_vals, x, out=cos_vals)
                np.add(sin_vals, y, out=sin_vals)

                np.multiply(cos_vals, self.render_scale, out=cos_vals)
                np.multiply(sin_vals, self.render_scale, out=sin_vals)

                positions_flat[0::2] = cos_vals
                positions_flat[1::2] = sin_vals

                colors_matrix[:] = LIDAR_COLOR_MAX
                hit_mask = scans < self.max_range * 0.99
                if np.any(hit_mask):
                    colors_matrix[hit_mask] = LIDAR_COLOR_HIT

                scan_vlist = self.scan_hits_vlist.get(aid)
                if scan_vlist is None or len(scan_vlist.position) != positions_flat.size:
                    if scan_vlist is not None:
                        try:
                            scan_vlist.delete()
                        except Exception:
                            pass
                    self.scan_hits_vlist[aid] = self.shader.vertex_list(
                        n, pyglet.gl.GL_POINTS, batch=self.batch, group=self.shader_group,
                        position=('f/dynamic', positions_array),
                        color=('f/dynamic', colors_array)
                    )
                else:
                    scan_vlist.position[:] = positions_array
                    scan_vlist.color[:] = colors_array

            if not self._follow_enabled:
                camera_line = 'Camera: free'
            else:
                target_label = str(self._camera_target) if (self._camera_target is not None) else 'auto'
                camera_line = f'Camera: follow {target_label}'

            hud_lines = [f'Agents: {len(all_ids)}', camera_line]
            for aid in all_ids:
                st = self.agent_infos[aid]
                lap_val = st.get("lap_count")
                lap_str = "lap ?"
                if lap_val is not None:
                    try:
                        lap_str = f"lap {int(lap_val)}"
                    except (ValueError, TypeError):
                        pass

                time_val = st.get("lap_time")
                time_str = "t=?"
                if time_val is not None:
                    try:
                        time_str = f"t={float(time_val):.1f}s"
                    except (ValueError, TypeError):
                        pass

                status_suffix = "" if st.get("__active__", False) else " (inactive)"
                extras: List[str] = []
                relative_info = st.get("relative")
                if isinstance(relative_info, dict):
                    sector_code = relative_info.get("sector_code")
                    if sector_code:
                        extras.append(str(sector_code))
                    sector_active = relative_info.get("sector_active")
                    if sector_active is not None:
                        extras.append(f"S={1 if sector_active else 0}")
                    in_ring = relative_info.get("in_ring")
                    if in_ring is not None:
                        extras.append(f"R={1 if in_ring else 0}")
                collision_value = st.get("collision")
                if collision_value is not None:
                    extras.append(f"C={1 if bool(collision_value) else 0}")
                target_collision = st.get("target_collision")
                if target_collision is not None:
                    extras.append(f"T={1 if bool(target_collision) else 0}")
                extra_text = f" | {' '.join(extras)}" if extras else ""
                hud_lines.append(f"{aid}: {lap_str} {time_str}{status_suffix}{extra_text}")
                components = st.get("obs_components")
                if isinstance(components, dict):
                    comp_text = self._format_obs_components(components)
                    if comp_text:
                        hud_lines.append(f"    {comp_text}")
                wrapped_obs = st.get("wrapped_obs")
                if wrapped_obs is not None:
                    skip = st.get("wrapped_skip")
                    try:
                        lidar_len = int(skip)
                    except (TypeError, ValueError):
                        lidar_len = 0
                    vector_text = self._format_observation_vector(wrapped_obs, lidar_len=lidar_len)
                    if vector_text:
                        hud_lines.append(f"    obs={vector_text}")

            telemetry_lines = self._format_telemetry_lines()
            if telemetry_lines:
                hud_lines.append("")
                hud_lines.extend(telemetry_lines)

            self.hud_label.text = "\n".join(hud_lines)

            self._update_reward_ring_overlay()

            if self._follow_enabled:
                self._camera_follow_first()

        def update_metrics(
            self,
            *,
            phase: str,
            metrics: Mapping[str, Any],
            step: Optional[float] = None,
            timestamp: Optional[float] = None,
        ) -> None:
            phase_label = str(phase).strip().lower() if phase else ""
            self._telemetry_phase = phase_label or None
            if metrics is None:
                snapshot: Dict[str, Any] = {}
            else:
                try:
                    snapshot = dict(metrics)
                except Exception:
                    snapshot = {}
            self._telemetry_metrics = snapshot
            self._telemetry_step = float(step) if step is not None else None
            if timestamp is not None:
                try:
                    self._telemetry_timestamp = float(timestamp)
                except (TypeError, ValueError):
                    self._telemetry_timestamp = time.time()
            else:
                self._telemetry_timestamp = time.time()

        def update_ticker(self, entries: Sequence[str]) -> None:
            if not entries:
                if self._ticker_lines:
                    self._ticker_lines = []
                    self.ticker_label.text = ""
                return
            new_lines = list(entries)
            if new_lines == self._ticker_lines:
                return
            self._ticker_lines = new_lines
            self.ticker_label.text = "\n".join(new_lines)

        def _format_telemetry_lines(self) -> List[str]:
            metrics = self._telemetry_metrics
            if not metrics:
                return []

            phase = self._telemetry_phase or ""
            prefix = f"{phase}/" if phase else ""
            base: Dict[str, Any] = {}
            agents: Dict[str, Dict[str, Any]] = {}
            for raw_key, value in metrics.items():
                if prefix:
                    if not raw_key.startswith(prefix):
                        continue
                    key = raw_key[len(prefix):]
                else:
                    key = raw_key

                if key.startswith("agent/"):
                    remainder = key[len("agent/") :]
                    agent_id, _, metric_name = remainder.partition("/")
                    if not metric_name:
                        continue
                    entry = agents.setdefault(agent_id, {})
                    entry[metric_name] = value
                    continue
                if key.startswith("reward/"):
                    continue
                base[key] = value

            if self._telemetry_step is not None and "episode" not in base:
                base["episode"] = self._telemetry_step

            lines: List[str] = []
            header = f"Telemetry [{phase.upper()}]" if phase else "Telemetry"
            lines.append(header)

            episode = self._coerce_int(base.get("episode"))
            total = self._coerce_int(base.get("episodes_total"))
            if episode is not None:
                label = f"Episode {episode}"
                if total is not None and total > 0:
                    label += f"/{total}"
                lines.append(label)
            elif total is not None:
                lines.append(f"Episodes total {total}")

            steps = self._coerce_int(base.get("steps"))
            if steps is not None:
                lines.append(f"Steps {steps}")

            return_parts: List[str] = []
            primary_return = self._coerce_float(base.get("return"))
            if primary_return is not None:
                return_parts.append(f"Return {primary_return:.2f}")
            return_mean = self._coerce_float(base.get("return_mean"))
            if return_mean is not None:
                window = self._coerce_int(base.get("return_window"))
                if window and window > 1:
                    return_parts.append(f"Avg{window} {return_mean:.2f}")
                else:
                    return_parts.append(f"Avg {return_mean:.2f}")
            return_best = self._coerce_float(base.get("return_best"))
            if return_best is not None:
                return_parts.append(f"Best {return_best:.2f}")
            if return_parts:
                lines.append("  ".join(return_parts))

            collision_parts: List[str] = []
            collisions = self._coerce_float(base.get("collisions"))
            if collisions is not None:
                collision_parts.append(f"Coll {collisions:.0f}")
            collision_rate = self._coerce_float(base.get("collision_rate"))
            if collision_rate is not None:
                collision_parts.append(f"Rate {collision_rate:.2f}")
            if collision_parts:
                lines.append("  ".join(collision_parts))

            success = self._coerce_bool(base.get("success"))
            success_rate = self._coerce_float(base.get("success_rate"))
            if success is not None or success_rate is not None:
                status_parts: List[str] = []
                if success is not None:
                    status_parts.append(f"Success {'yes' if success else 'no'}")
                if success_rate is not None:
                    status_parts.append(f"{success_rate * 100:.1f}%")
                lines.append("  ".join(status_parts))

            epsilon = self._coerce_float(base.get("epsilon"))
            buffer_fraction = self._coerce_float(base.get("buffer_fraction"))
            idle = self._coerce_bool(base.get("idle"))
            misc_parts: List[str] = []
            if epsilon is not None:
                misc_parts.append(f"Eps {epsilon:.3f}")
            if buffer_fraction is not None:
                misc_parts.append(f"Buffer {buffer_fraction * 100:.0f}%")
            if idle is not None:
                misc_parts.append(f"Idle {'yes' if idle else 'no'}")
            if misc_parts:
                lines.append("  ".join(misc_parts))

            defender_crashed = self._coerce_bool(base.get("defender_crashed"))
            attacker_crashed = self._coerce_bool(base.get("attacker_crashed"))
            defender_survival = self._coerce_float(base.get("defender_survival_steps"))
            crash_parts: List[str] = []
            if defender_crashed is not None:
                crash_parts.append(f"Def {'X' if defender_crashed else 'ok'}")
            if attacker_crashed is not None:
                crash_parts.append(f"Atk {'X' if attacker_crashed else 'ok'}")
            if crash_parts:
                if defender_survival is not None:
                    crash_parts.append(f"Surv {defender_survival:.0f}")
                lines.append("  ".join(crash_parts))
            elif defender_survival is not None:
                lines.append(f"Defender survival {defender_survival:.0f}")

            primary_agent = base.get("primary_agent")
            if isinstance(primary_agent, str) and primary_agent:
                lines.append(f"Primary {primary_agent}")

            if agents:
                for agent_id in sorted(agents):
                    entry = agents[agent_id]
                    row_parts: List[str] = [agent_id]
                    ret_val = self._coerce_float(entry.get("return"))
                    if ret_val is not None:
                        row_parts.append(f"R {ret_val:.2f}")
                    coll_val = self._coerce_float(entry.get("collisions"))
                    if coll_val is not None:
                        row_parts.append(f"C {coll_val:.0f}")
                    speed_val = self._coerce_float(entry.get("avg_speed") or entry.get("speed"))
                    if speed_val is not None:
                        row_parts.append(f"V {speed_val:.2f}")
                    collision_step = self._coerce_float(entry.get("collision_step"))
                    if collision_step is not None:
                        row_parts.append(f"Hit {collision_step:.0f}")
                    lap_count = self._coerce_float(entry.get("lap_count"))
                    if lap_count is not None:
                        row_parts.append(f"Lap {lap_count:.0f}")
                    lines.append("  ".join(row_parts))

            if self._telemetry_timestamp is not None:
                age = max(time.time() - self._telemetry_timestamp, 0.0)
                lines.append(f"Updated {age:.1f}s ago")

            return lines

        @staticmethod
        def _format_obs_components(components: Mapping[str, Any]) -> Optional[str]:
            if not components:
                return None
            chunks: List[str] = []
            for key in sorted(components):
                value = components[key]
                if isinstance(value, np.ndarray):
                    arr = value.flatten()
                    if arr.size > 6:
                        arr = arr[:6]
                    val_str = ",".join(f"{float(item):.2f}" for item in arr)
                    val_repr = f"[{val_str}]"
                elif isinstance(value, (list, tuple)):
                    subset = list(value[:6])
                    if subset and all(isinstance(item, (int, float)) for item in subset):
                        val_repr = "[" + ",".join(f"{float(item):.2f}" for item in subset) + "]"
                    else:
                        val_repr = str(subset)
                elif isinstance(value, Mapping):
                    entries = []
                    max_items = 8 if len(value) >= 8 else len(value)
                    idx = 0
                    for sub_key, sub_val in value.items():
                        flag = "T" if bool(sub_val) else "F"
                        entries.append(f"{sub_key[:2]}={flag}")
                        idx += 1
                        if idx >= max_items:
                            break
                    if len(value) > max_items:
                        entries.append("…")
                    val_repr = "[" + ", ".join(entries) + "]"
                elif isinstance(value, (np.bool_, bool)):
                    val_repr = "T" if bool(value) else "F"
                elif isinstance(value, (float, int, np.floating, np.integer)):
                    val_repr = f"{float(value):.2f}"
                else:
                    val_repr = str(value)
                chunks.append(f"{key}={val_repr}")
                if len(chunks) >= 4:
                    break
            if not chunks:
                return None
            return " | ".join(chunks)

        @staticmethod
        def _format_observation_vector(vector: Any, *, lidar_len: int = 0) -> Optional[str]:
            try:
                arr = np.asarray(vector, dtype=np.float32).flatten()
            except Exception:
                return None
            if arr.size == 0:
                return "[]"
            if lidar_len > 0 and arr.size > lidar_len:
                arr = arr[lidar_len:]
            max_items = 24
            display = arr[:max_items]
            values = ", ".join(f"{value:+.3f}" for value in display)
            if arr.size > max_items:
                values += ", …"
            return f"[{values}] (len={arr.size})"

        @staticmethod
        def _coerce_float(value: Any) -> Optional[float]:
            if value is None or isinstance(value, bool):
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        @staticmethod
        def _coerce_int(value: Any) -> Optional[int]:
            if value is None or isinstance(value, bool):
                return None
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        @staticmethod
        def _coerce_bool(value: Any) -> Optional[bool]:
            if isinstance(value, bool):
                return value
            return None

        # ---------- Helpers ----------

        def set_camera_target(self, agent_id):
            """Persistently select which agent the camera should follow."""
            self._follow_enabled = True
            self._user_camera_target = agent_id
            if agent_id is None:
                if self.agent_ids:
                    self._camera_target = self.agent_ids[0]
                elif self.agent_infos:
                    self._camera_target = sorted(self.agent_infos.keys())[0]
                else:
                    self._camera_target = None
                return
            if agent_id in self.agent_infos:
                self._camera_target = agent_id
            else:
                self._camera_target = self.agent_ids[0] if self.agent_ids else None

        def _camera_follow_first(self):
            target = self._camera_target
            if target is None:
                return
            v = self.cars_vlist.get(target, None)
            if v is None:
                return
            xs = v.position[::2]
            ys = v.position[1::2]
            top, bottom, left, right = max(ys), min(ys), min(xs), max(xs)
            padding = self._follow_padding_m * self.render_scale
            cx = (left + right) * 0.5
            cy = (top + bottom) * 0.5
            half_w = (right - left) * 0.5 + padding
            half_h = (top - bottom) * 0.5 + padding
            self.left = cx - half_w
            self.right = cx + half_w
            self.bottom = cy - half_h
            self.top = cy + half_h

        def on_key_press(self, symbol, modifiers):
            if symbol in (key.PLUS, key.EQUAL, key.NUM_ADD):
                self._apply_zoom(ZOOM_IN_FACTOR)
                return True
            if symbol in (key.MINUS, key.NUM_SUBTRACT):
                self._apply_zoom(ZOOM_OUT_FACTOR)
                return True
            if symbol == key.T:
                self._follow_enabled = not self._follow_enabled
                if not self._follow_enabled:
                    self._user_camera_target = None
                    self._camera_target = None
            elif symbol == key.SPACE:
                self._reset_view()
            elif symbol == key.PAGEUP:
                self._cycle_camera_target(1)
            elif symbol == key.PAGEDOWN:
                self._cycle_camera_target(-1)
            elif symbol == key.RIGHT or symbol == key.D:
                self._manual_pan(-self._pan_step_pixels, 0.0, modifiers)
            elif symbol == key.LEFT or symbol == key.A:
                self._manual_pan(self._pan_step_pixels, 0.0, modifiers)
            elif symbol == key.UP or symbol == key.W:
                self._manual_pan(0.0, -self._pan_step_pixels, modifiers)
            elif symbol == key.DOWN or symbol == key.S:
                self._manual_pan(0.0, self._pan_step_pixels, modifiers)
            elif symbol == key.HOME:
                self._reset_view()
            elif symbol == key.TAB:
                direction = -1 if modifiers & key.MOD_SHIFT else 1
                self._cycle_camera_target(direction)
            else:
                return super().on_key_press(symbol, modifiers)
            return True

        def _manual_pan(self, dx_pixels: float, dy_pixels: float, modifiers: int) -> None:
            if self._follow_enabled:
                self._follow_enabled = False
                self._user_camera_target = None
            factor = 4.0 if modifiers & key.MOD_SHIFT else 1.0
            self._pan_view(dx_pixels * factor, dy_pixels * factor)

        def _cycle_camera_target(self, direction: int) -> None:
            if not self.agent_ids:
                return
            if self._user_camera_target in self.agent_ids:
                idx = self.agent_ids.index(self._user_camera_target)
            elif self._camera_target in self.agent_ids:
                idx = self.agent_ids.index(self._camera_target)
            else:
                idx = 0
            idx = (idx + direction) % len(self.agent_ids)
            self._user_camera_target = self.agent_ids[idx]
            self._follow_enabled = True
            self._camera_target = self._user_camera_target

        def _apply_zoom(self, factor: float, anchor_x: float = 0.5, anchor_y: float = 0.5) -> None:
            new_zoom = self.zoom_level * factor
            if not 0.01 < new_zoom < 10.0:
                return
            w, h = self.get_size()
            if w <= 0 or h <= 0:
                return
            mx_world = self.left + anchor_x * self.zoomed_width
            my_world = self.bottom + anchor_y * self.zoomed_height
            self.zoom_level = new_zoom
            self.zoomed_width *= factor
            self.zoomed_height *= factor
            self.left = mx_world - anchor_x * self.zoomed_width
            self.right = self.left + self.zoomed_width
            self.bottom = my_world - anchor_y * self.zoomed_height
            self.top = self.bottom + self.zoomed_height

        def _pan_view(self, dx_pixels: float, dy_pixels: float) -> None:
            self.left -= dx_pixels * self.zoom_level
            self.right -= dx_pixels * self.zoom_level
            self.bottom -= dy_pixels * self.zoom_level
            self.top -= dy_pixels * self.zoom_level

        def _reset_view(self) -> None:
            self._follow_enabled = True
            self._user_camera_target = None
            w, h = self.get_size()
            if w <= 0 or h <= 0:
                return
            self.zoom_level = 1.2
            self.zoomed_width = self.zoom_level * w
            self.zoomed_height = self.zoom_level * h
            self.left = -self.zoomed_width / 2
            self.right = self.zoomed_width / 2
            self.bottom = -self.zoomed_height / 2
            self.top = self.zoomed_height / 2

        # ---------- Optional: utilities for extra overlays ----------

        @staticmethod
        def make_centerline_callback(centerline_csv_path: str, point_size=1):
            df = pd.read_csv(centerline_csv_path, comment='#', header=None)
            waypoints = df[[0, 1]].values
            def callback(env_renderer: "EnvRenderer"):
                glPointSize(point_size)
                pts = (waypoints * env_renderer.render_scale).flatten().tolist()
                color = [0, 255, 0, 255]
                if not hasattr(env_renderer, '_centerline_vlist'):
                    n = waypoints.shape[0]
                    env_renderer._centerline_vlist = env_renderer.shader.vertex_list(
                        n, pyglet.gl.GL_POINTS, batch=env_renderer.batch, group=env_renderer.shader_group,
                        position=('f', pts), color=('B', color * n)
                    )
            return callback

        @staticmethod
        def make_waypoints_callback(waypoints_csv_path: str, passed_flags=None, point_size=3):
            df = pd.read_csv(waypoints_csv_path, header=None, comment='#')
            if df.shape[1] < 2:
                raise ValueError(f"{waypoints_csv_path} must have at least two columns for x and y.")
            waypoints = df.iloc[:, :2].values
            def callback(env_renderer: "EnvRenderer"):
                glPointSize(point_size)
                # delete previous if exists
                if hasattr(env_renderer, '_waypoints_vlist'):
                    try:
                        env_renderer._waypoints_vlist.delete()
                    except Exception:
                        pass
                    del env_renderer._waypoints_vlist
                # build colors
                num = waypoints.shape[0]
                colors = []
                current_idx = None
                if passed_flags is not None:
                    for i, p in enumerate(passed_flags):
                        if not p:
                            current_idx = i
                            break
                for i in range(num):
                    if passed_flags and i < len(passed_flags) and passed_flags[i]:
                        colors.extend([255, 0, 0, 255])     # passed
                    elif current_idx is not None and i == current_idx:
                        colors.extend([255, 255, 255, 255]) # current
                    else:
                        colors.extend([255, 255, 0, 255])   # pending
                pos = (waypoints * env_renderer.render_scale).flatten().tolist()
                env_renderer._waypoints_vlist = env_renderer.shader.vertex_list(
                    num, pyglet.gl.GL_POINTS, batch=env_renderer.batch, group=env_renderer.shader_group,
                    position=('f', pos), color=('B', colors)
                )
            return callback

        # ---------- Reward Ring Overlay ----------

        def configure_reward_ring(
            self,
            *,
            enabled: bool,
            preferred_radius: float = 0.0,
            inner_tolerance: float = 0.0,
            outer_tolerance: float = 0.0,
            segments: int = 96,
            fill_color: Optional[tuple[float, float, float, float]] = None,
            border_color: Optional[tuple[float, float, float, float]] = None,
            preferred_color: Optional[tuple[float, float, float, float]] = None,
        ) -> None:
            desired_enabled = bool(enabled) and int(segments) >= 8
            if not desired_enabled:
                self._reward_ring_enabled = False
                self._reward_ring_config = {}
                self._reward_ring_segments = 0
                self._reward_ring_target = None
                self._clear_reward_ring_geometry()
                return

            segments = max(int(segments), 8)
            config = {
                "preferred_radius": max(float(preferred_radius), 0.0),
                "inner_tolerance": max(float(inner_tolerance), 0.0),
                "outer_tolerance": max(float(outer_tolerance), 0.0),
                "segments": segments,
                "fill_color": tuple(fill_color) if fill_color is not None else REWARD_RING_FILL_COLOR,
                "border_color": tuple(border_color) if border_color is not None else REWARD_RING_BORDER_COLOR,
                "preferred_color": tuple(preferred_color) if preferred_color is not None else REWARD_RING_PREFERRED_COLOR,
            }

            if self._reward_ring_enabled and self._reward_ring_config == config:
                return

            self._reward_ring_enabled = True
            self._reward_ring_config = config
            self._reward_ring_segments = segments
            angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False, dtype=np.float32)
            self._reward_ring_angles = angles
            self._reward_ring_cos = np.cos(angles).astype(np.float32)
            self._reward_ring_sin = np.sin(angles).astype(np.float32)
            angles_ext = np.concatenate([angles, angles[:1]])
            self._reward_ring_angles_ext = angles_ext.astype(np.float32)
            self._reward_ring_cos_ext = np.cos(self._reward_ring_angles_ext).astype(np.float32)
            self._reward_ring_sin_ext = np.sin(self._reward_ring_angles_ext).astype(np.float32)
            self._clear_reward_ring_geometry()

        def set_reward_ring_target(self, agent_id: Optional[str]) -> None:
            if agent_id == self._reward_ring_target:
                return
            self._reward_ring_target = agent_id
            if agent_id is None:
                self._clear_reward_ring_geometry()

        def _ensure_reward_ring_geometry(self, *, need_inner: bool, need_preferred: bool) -> None:
            if not self._reward_ring_enabled or not self._reward_ring_config:
                return
            segments = self._reward_ring_segments
            if segments <= 0:
                return

            fill_vertices = 2 * (segments + 1)
            if self._reward_ring_fill_vlist is None:
                self._reward_ring_fill_positions = array('f', [0.0] * (2 * fill_vertices))
                self._reward_ring_fill_view = np.frombuffer(self._reward_ring_fill_positions, dtype=np.float32)
                fill_color = list(self._reward_ring_config.get("fill_color", REWARD_RING_FILL_COLOR)) * fill_vertices
                self._reward_ring_fill_vlist = self.shader.vertex_list(
                    fill_vertices,
                    pyglet.gl.GL_TRIANGLE_STRIP,
                    batch=self.batch,
                    group=self.shader_group,
                    position=('f/dynamic', self._reward_ring_fill_positions),
                    color=('f', fill_color),
                )

            if self._reward_ring_outer_vlist is None:
                self._reward_ring_outer_positions = array('f', [0.0] * (2 * segments))
                self._reward_ring_outer_view = np.frombuffer(self._reward_ring_outer_positions, dtype=np.float32)
                border_color = list(self._reward_ring_config.get("border_color", REWARD_RING_BORDER_COLOR)) * segments
                self._reward_ring_outer_vlist = self.shader.vertex_list(
                    segments,
                    pyglet.gl.GL_LINE_LOOP,
                    batch=self.batch,
                    group=self.shader_group,
                    position=('f/dynamic', self._reward_ring_outer_positions),
                    color=('f', border_color),
                )

            if need_inner:
                if self._reward_ring_inner_vlist is None:
                    self._reward_ring_inner_positions = array('f', [0.0] * (2 * segments))
                    self._reward_ring_inner_view = np.frombuffer(self._reward_ring_inner_positions, dtype=np.float32)
                    border_color = list(self._reward_ring_config.get("border_color", REWARD_RING_BORDER_COLOR)) * segments
                    self._reward_ring_inner_vlist = self.shader.vertex_list(
                        segments,
                        pyglet.gl.GL_LINE_LOOP,
                        batch=self.batch,
                        group=self.shader_group,
                        position=('f/dynamic', self._reward_ring_inner_positions),
                        color=('f', border_color),
                    )
            elif self._reward_ring_inner_vlist is not None:
                try:
                    self._reward_ring_inner_vlist.delete()
                except Exception:
                    pass
                self._reward_ring_inner_vlist = None
                self._reward_ring_inner_positions = None
                self._reward_ring_inner_view = None

            if need_preferred:
                if self._reward_ring_pref_vlist is None:
                    self._reward_ring_pref_positions = array('f', [0.0] * (2 * segments))
                    self._reward_ring_pref_view = np.frombuffer(self._reward_ring_pref_positions, dtype=np.float32)
                    pref_color = list(self._reward_ring_config.get("preferred_color", REWARD_RING_PREFERRED_COLOR)) * segments
                    self._reward_ring_pref_vlist = self.shader.vertex_list(
                        segments,
                        pyglet.gl.GL_LINE_LOOP,
                        batch=self.batch,
                        group=self.shader_group,
                        position=('f/dynamic', self._reward_ring_pref_positions),
                        color=('f', pref_color),
                    )
            elif self._reward_ring_pref_vlist is not None:
                try:
                    self._reward_ring_pref_vlist.delete()
                except Exception:
                    pass
                self._reward_ring_pref_vlist = None
                self._reward_ring_pref_positions = None
                self._reward_ring_pref_view = None

        def _clear_reward_ring_geometry(self) -> None:
            for attr in (
                "_reward_ring_fill_vlist",
                "_reward_ring_outer_vlist",
                "_reward_ring_inner_vlist",
                "_reward_ring_pref_vlist",
            ):
                vlist = getattr(self, attr, None)
                if vlist is not None:
                    try:
                        vlist.delete()
                    except Exception:
                        pass
                    setattr(self, attr, None)

            self._reward_ring_fill_positions = None
            self._reward_ring_fill_view = None
            self._reward_ring_outer_positions = None
            self._reward_ring_outer_view = None
            self._reward_ring_inner_positions = None
            self._reward_ring_inner_view = None
            self._reward_ring_pref_positions = None
            self._reward_ring_pref_view = None

        def _update_reward_ring_overlay(self) -> None:
            if not self._reward_ring_enabled or not self._reward_ring_config:
                return
            if self._reward_ring_cos is None:
                return

            target_id = self._reward_ring_target
            if not target_id:
                self._clear_reward_ring_geometry()
                return

            state = self.agent_infos.get(target_id)
            if state is None:
                self._clear_reward_ring_geometry()
                return

            cx = float(state.get("poses_x", 0.0)) * self.render_scale
            cy = float(state.get("poses_y", 0.0)) * self.render_scale

            cfg = self._reward_ring_config
            preferred = max(float(cfg.get("preferred_radius", 0.0)), 0.0)
            inner_tol = max(float(cfg.get("inner_tolerance", 0.0)), 0.0)
            outer_tol = max(float(cfg.get("outer_tolerance", 0.0)), 0.0)

            inner = max(preferred - inner_tol, 0.0)
            outer = max(preferred + outer_tol, 0.0)
            if outer <= 0.0:
                self._clear_reward_ring_geometry()
                return

            px_outer = outer * self.render_scale
            px_inner = inner * self.render_scale
            px_pref = preferred * self.render_scale

            need_inner = px_inner > 1e-4
            need_preferred = px_pref > 1e-4
            self._ensure_reward_ring_geometry(need_inner=need_inner, need_preferred=need_preferred)

            if self._reward_ring_outer_view is None or self._reward_ring_fill_view is None:
                return

            outer_x = cx + px_outer * self._reward_ring_cos
            outer_y = cy + px_outer * self._reward_ring_sin
            self._reward_ring_outer_view[0::2] = outer_x
            self._reward_ring_outer_view[1::2] = outer_y
            if self._reward_ring_outer_vlist is not None and self._reward_ring_outer_positions is not None:
                self._reward_ring_outer_vlist.position[:] = self._reward_ring_outer_positions

            if self._reward_ring_inner_view is not None and self._reward_ring_inner_vlist is not None:
                inner_x = cx + px_inner * self._reward_ring_cos
                inner_y = cy + px_inner * self._reward_ring_sin
                self._reward_ring_inner_view[0::2] = inner_x
                self._reward_ring_inner_view[1::2] = inner_y
                if self._reward_ring_inner_positions is not None:
                    self._reward_ring_inner_vlist.position[:] = self._reward_ring_inner_positions

            if self._reward_ring_pref_view is not None and self._reward_ring_pref_vlist is not None:
                pref_x = cx + px_pref * self._reward_ring_cos
                pref_y = cy + px_pref * self._reward_ring_sin
                self._reward_ring_pref_view[0::2] = pref_x
                self._reward_ring_pref_view[1::2] = pref_y
                if self._reward_ring_pref_positions is not None:
                    self._reward_ring_pref_vlist.position[:] = self._reward_ring_pref_positions

            if self._reward_ring_cos_ext is None or self._reward_ring_sin_ext is None:
                return

            fill_outer_x = cx + px_outer * self._reward_ring_cos_ext
            fill_outer_y = cy + px_outer * self._reward_ring_sin_ext
            fill_inner_x = cx + px_inner * self._reward_ring_cos_ext
            fill_inner_y = cy + px_inner * self._reward_ring_sin_ext

            self._reward_ring_fill_view[0::4] = fill_outer_x
            self._reward_ring_fill_view[1::4] = fill_outer_y
            self._reward_ring_fill_view[2::4] = fill_inner_x
            self._reward_ring_fill_view[3::4] = fill_inner_y
            if self._reward_ring_fill_positions is not None and self._reward_ring_fill_vlist is not None:
                self._reward_ring_fill_vlist.position[:] = self._reward_ring_fill_positions
