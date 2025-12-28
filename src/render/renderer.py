# MIT License
# Minimal v2 rendering engine for F1TENTH multi-agent environments
# Simplified from v1's 2,141 lines to ~350 lines core

import os

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
    from pyglet.gl import *  # noqa: F401,F403
    from pyglet.math import Mat4
    from pyglet.graphics import ShaderGroup
    _PYGLET_AVAILABLE = True
except Exception as exc:
    _PYGLET_AVAILABLE = False
    _PYGLET_IMPORT_ERROR = exc
    Mat4 = None  # type: ignore[assignment]
    ShaderGroup = None  # type: ignore[assignment]

import numpy as np
from array import array
from pathlib import Path
from typing import Any, Dict, List, Optional
from PIL import Image

from physics.collision_models import get_vertices
from .shader import get_default_shader

# Zoom constants
ZOOM_IN_FACTOR = 1.2
ZOOM_OUT_FACTOR = 1.0 / ZOOM_IN_FACTOR

# Vehicle shape constants (meters)
CAR_LENGTH = 0.32
CAR_WIDTH = 0.225

# Colors (normalized RGBA)
def _rgba255(r: float, g: float, b: float, a: float = 255.0) -> tuple[float, float, float, float]:
    return (r / 255.0, g / 255.0, b / 255.0, a / 255.0)


CAR_LEARNER = _rgba255(183, 193, 222)
CAR_OTHER = _rgba255(99, 52, 94)
MAP_COLOR = _rgba255(255, 193, 50)
LIDAR_COLOR_HIT = (1.0, 0.0, 0.0, 1.0)
LIDAR_COLOR_MAX = _rgba255(180, 180, 180)


if not _PYGLET_AVAILABLE:
    class EnvRenderer:
        def __init__(self, *_, **__):
            raise RuntimeError(
                "pyglet is unavailable or failed to initialize (set render_mode=null for headless runs): "
                + str(_PYGLET_IMPORT_ERROR)
            )

else:
    class EnvRenderer(pyglet.window.Window):
        """Minimal multi-agent renderer with extension support.

        Core features:
        - Map rendering from occupancy grid
        - Multi-agent car visualization (colored by ID)
        - LiDAR scan visualization (red=hit, gray=max)
        - Camera controls (zoom, pan, follow)

        Extension support:
        - Add optional visualizations via add_extension()
        - Examples: HUD, reward zones, heatmaps, telemetry
        """

        def __init__(self, width, height, lidar_fov=4.7, max_range=30.0, lidar_offset=0.0, *args, **kwargs):
            """Initialize renderer.

            Args:
                width: Window width in pixels
                height: Window height in pixels
                lidar_fov: LiDAR field of view in radians
                max_range: Maximum LiDAR range in meters
                lidar_offset: LiDAR offset from car center in meters
            """
            conf = Config(sample_buffers=1, samples=4, depth_size=16, double_buffer=True)
            super().__init__(width, height, config=conf, resizable=True, vsync=False, *args, **kwargs)

            # GL init
            glClearColor(9/255, 32/255, 87/255, 1.0)

            # Camera
            self.left = -width / 2
            self.right = width / 2
            self.bottom = -height / 2
            self.top = height / 2
            self.zoom_level = 1.2
            self.zoomed_width = width
            self.zoomed_height = height
            self.lidar_offset = float(lidar_offset)
            self._follow_enabled = True

            # Shader + batch
            self.shader = get_default_shader()
            self.shader_group = ShaderGroup(self.shader)
            self.batch = pyglet.graphics.Batch()
            self.point_size = 4.0
            self.shader['point_size'] = float(self.point_size)

            # Map
            self.map_points = None
            self.map_vlist = None
            self._map_vertex_count = 0

            # Per-agent drawables and cached state
            self.cars_vlist = {}  # aid -> vertex_list (GL_QUADS)
            self.scan_hits_vlist = {}  # aid -> vertex_list (GL_POINTS)
            self.agent_infos = {}  # aid -> last render_obs snapshot
            self.agent_ids = []  # sorted agent IDs updated this frame
            self._camera_target = None
            self._user_camera_target = None
            self._follow_padding_m = 16.0
            self._lidar_buffer_cache = {}

            # Rendering options
            self.lidar_fov = lidar_fov
            self.max_range = max_range
            self.render_scale = 50.0  # meters -> pixels

            # Extension system
            self._extensions = []

        def add_extension(self, extension):
            """Add optional visualization extension.

            Args:
                extension: RenderExtension instance
            """
            self._extensions.append(extension)

        # ---------- Map ----------

        def update_map(
            self,
            map_path_no_ext: str,
            map_ext: str,
            *,
            map_meta=None,
            map_image_path=None,
            centerline_points=None,
            centerline_connect=True,
        ):
            """Update map geometry from occupancy grid.

            Args:
                map_path_no_ext: Path WITHOUT extension (e.g., 'maps/levine')
                map_ext: Image extension (e.g., '.png')
                map_meta: Preloaded map metadata dict
                map_image_path: Optional override for image path
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

            # Load and process image
            with Image.open(img_path) as pil_img:
                pil_img = pil_img.convert('L').transpose(Image.FLIP_TOP_BOTTOM)
                img_gray = np.asarray(pil_img, dtype=np.float32)

            img_norm = img_gray / 255.0
            if int(meta.get('negate', 0)):
                occ_prob = img_norm
            else:
                occ_prob = 1.0 - img_norm  # Dark pixels = obstacles

            occ_thresh = float(meta.get('occupied_thresh', 0.65))
            free_thresh = float(meta.get('free_thresh', 0.196))
            if occ_thresh <= free_thresh:
                occ_thresh = free_thresh + np.finfo(np.float32).eps

            H, W = occ_prob.shape[0], occ_prob.shape[1]

            # Convert to world coordinates
            xs = (np.arange(W) * res + ox)
            ys = (np.arange(H) * res + oy)
            gx, gy = np.meshgrid(xs, ys)

            # Extract obstacle pixels
            mask = occ_prob >= occ_thresh
            flat_idx = np.flatnonzero(mask)

            # Subsample if too many points (max 200k)
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

            # Create/update vertex list
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
                self.map_vlist.position[:] = positions
                self.map_vlist.color[:] = colors

            self._map_vertex_count = N
            self.map_points = pts

            # Centerline rendering (stub for now - can be implemented as extension)
            if centerline_points is not None:
                self.update_centerline(centerline_points, connect=centerline_connect)

        def reset_state(self):
            """Clean up all agent state."""
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

            # Clean up extensions
            for ext in self._extensions:
                ext.cleanup()

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

        def on_key_press(self, symbol, modifiers):
            """Handle keyboard input for visualization controls.

            Keyboard shortcuts:
            - T: Cycle telemetry display mode
            - R: Toggle reward ring
            - H: Toggle reward heatmap
            - F: Toggle camera follow mode
            - 1-9: Focus telemetry on specific agent
            - 0: Show all agents in telemetry
            """
            # Find telemetry, reward ring, and heatmap extensions
            telemetry_ext = None
            ring_ext = None
            heatmap_ext = None

            for ext in self._extensions:
                ext_name = ext.__class__.__name__
                if ext_name == 'TelemetryHUD':
                    telemetry_ext = ext
                elif ext_name == 'RewardRingExtension':
                    ring_ext = ext
                elif ext_name == 'RewardHeatmap':
                    heatmap_ext = ext

            # Import key constants
            from pyglet.window import key

            # T: Cycle telemetry mode
            if symbol == key.T:
                print("[Keyboard] T pressed - cycling telemetry mode")
                if telemetry_ext:
                    telemetry_ext.cycle_mode()
                    print(f"[Keyboard] Telemetry mode: {telemetry_ext._mode}")
                else:
                    print("[Keyboard] No telemetry extension found")

            # R: Toggle reward ring
            elif symbol == key.R:
                print("[Keyboard] R pressed - toggling reward ring")
                if ring_ext:
                    ring_ext._enabled = not ring_ext._enabled
                    print(f"[Keyboard] Reward ring: {'ON' if ring_ext._enabled else 'OFF'}")
                else:
                    print("[Keyboard] No reward ring extension found")

            # H: Toggle reward heatmap
            elif symbol == key.H:
                print("[Keyboard] H pressed - toggling heatmap")
                if heatmap_ext:
                    heatmap_ext._enabled = not heatmap_ext._enabled
                    print(f"[Keyboard] Heatmap: {'ON' if heatmap_ext._enabled else 'OFF'}")
                else:
                    print("[Keyboard] No heatmap extension found")

            # F: Toggle camera follow
            elif symbol == key.F:
                self._follow_enabled = not self._follow_enabled
                if not self._follow_enabled:
                    self._user_camera_target = None
                print(f"[Keyboard] F pressed - Camera follow: {'ON' if self._follow_enabled else 'OFF'}")

            # 1-9: Focus on specific agent
            elif key._1 <= symbol <= key._9:
                agent_idx = symbol - key._1
                if telemetry_ext and self.agent_ids:
                    if agent_idx < len(self.agent_ids):
                        agent_id = self.agent_ids[agent_idx]
                        telemetry_ext.set_focused_agent(agent_id)
                        print(f"[Keyboard] Focused on agent: {agent_id}")

            # 0: Show all agents
            elif symbol == key._0:
                if telemetry_ext:
                    telemetry_ext.set_focused_agent(None)
                    print("[Keyboard] Showing all agents")

        def on_close(self):
            super().on_close()

        def close(self):
            """Explicit closer for env.close()."""
            try:
                self.reset_state()
                super().close()
            except Exception:
                pass

        def _apply_zoom(self, factor: float, anchor_x: float = 0.5, anchor_y: float = 0.5) -> None:
            """Zoom camera around anchor point.

            Args:
                factor: Zoom factor (>1 = zoom in, <1 = zoom out)
                anchor_x: Anchor x in normalized window coords [0,1]
                anchor_y: Anchor y in normalized window coords [0,1]
            """
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
            """Pan camera by pixel offset.

            Args:
                dx_pixels: Horizontal pan in pixels
                dy_pixels: Vertical pan in pixels
            """
            self.left -= dx_pixels * self.zoom_level
            self.right -= dx_pixels * self.zoom_level
            self.bottom -= dy_pixels * self.zoom_level
            self.top -= dy_pixels * self.zoom_level

        # ---------- Draw ----------

        def on_draw(self):
            """Render frame."""
            if self.map_points is None:
                raise Exception('Map not set for renderer.')

            self.clear()
            proj = Mat4.orthogonal_projection(self.left, self.right, self.bottom, self.top, -1, 1)
            self.shader.use()
            self.shader['projection'] = proj
            self.shader['point_size'] = float(self.point_size)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glEnable(GL_PROGRAM_POINT_SIZE)
            glPointSize(4)

            # Draw core geometry (map, cars, lidar)
            self.batch.draw()

            # Draw extensions
            for ext in self._extensions:
                ext.draw_geometry(self.batch, self.shader_group)

            self.shader.stop()

        # ---------- Per-Agent Update ----------

        def update_obs(self, render_obs: dict):
            """Update agent positions and LiDAR scans.

            Args:
                render_obs: Dict mapping agent_id -> observation dict
                    Required fields per agent: poses_x, poses_y, poses_theta
                    Optional: scans (LiDAR array)
            """
            if render_obs is None:
                return

            active_ids = sorted(render_obs.keys())

            # Mark cached agents as inactive
            for cached in self.agent_infos.values():
                cached["__active__"] = False

            # Update agent states
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

            # Update camera target
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

            # Update car and LiDAR geometry
            for aid in all_ids:
                st = self.agent_infos[aid]
                x = float(st.get("poses_x", 0.0))
                y = float(st.get("poses_y", 0.0))
                th = float(st.get("poses_theta", 0.0))

                # Update car vertices
                verts_np = self.render_scale * get_vertices(np.array([x, y, th]), CAR_LENGTH, CAR_WIDTH)
                positions = verts_np.astype(np.float32, copy=False).flatten().tolist()

                car_vlist = self.cars_vlist.get(aid)
                if car_vlist is None:
                    # Color: first agent = blue, others = purple
                    color = CAR_LEARNER if len(self.cars_vlist) == 0 else CAR_OTHER
                    color_array = list(color) * 4
                    self.cars_vlist[aid] = self.shader.vertex_list(
                        4, pyglet.gl.GL_QUADS, batch=self.batch, group=self.shader_group,
                        position=('f', positions),
                        color=('f', color_array)
                    )
                else:
                    car_vlist.position[:] = positions

                # Update LiDAR scans
                scans = st.get("scans")
                if scans is None:
                    continue

                scans = np.asarray(scans, dtype=np.float32)
                n = scans.shape[0]
                if n == 0:
                    continue

                # Get/create cached buffers for this (beam_count, fov) combo
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

                # Compute scan endpoints in world coordinates
                np.add(base_angles, th, out=angles)
                np.cos(angles, out=cos_vals)
                np.sin(angles, out=sin_vals)

                np.multiply(scans, cos_vals, out=cos_vals)
                np.multiply(scans, sin_vals, out=sin_vals)

                origin_x = x + self.lidar_offset * np.cos(th)
                origin_y = y + self.lidar_offset * np.sin(th)

                np.add(cos_vals, origin_x, out=cos_vals)
                np.add(sin_vals, origin_y, out=sin_vals)

                np.multiply(cos_vals, self.render_scale, out=cos_vals)
                np.multiply(sin_vals, self.render_scale, out=sin_vals)

                positions_flat[0::2] = cos_vals
                positions_flat[1::2] = sin_vals

                # Color: red for hits, gray for max range
                colors_matrix[:] = LIDAR_COLOR_MAX
                hit_mask = scans < self.max_range * 0.99
                if np.any(hit_mask):
                    colors_matrix[hit_mask] = LIDAR_COLOR_HIT

                # Update/create vertex list
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

            # Update extensions
            for ext in self._extensions:
                ext.update(render_obs)

            # Camera follow
            if self._follow_enabled:
                self._camera_follow_first()

        def _camera_follow_first(self) -> None:
            """Center camera on target agent."""
            if not self._camera_target or self._camera_target not in self.agent_infos:
                return

            st = self.agent_infos[self._camera_target]
            x = float(st.get("poses_x", 0.0))
            y = float(st.get("poses_y", 0.0))

            cx = x * self.render_scale
            cy = y * self.render_scale

            self.left = cx - self.zoomed_width / 2
            self.right = cx + self.zoomed_width / 2
            self.bottom = cy - self.zoomed_height / 2
            self.top = cy + self.zoomed_height / 2

        # ---------- Compatibility Stubs (for v1 features not yet implemented as extensions) ----------

        def update_centerline(self, centerline_points, *, connect: bool = True) -> None:
            """Stub for centerline rendering (can be implemented as extension later)."""
            pass

        def configure_reward_ring(self, **kwargs) -> None:
            """Stub for reward ring configuration (can be implemented as extension later)."""
            pass

        def set_reward_ring_target(self, target_id: Optional[str]) -> None:
            """Stub for setting reward ring target (can be implemented as extension later)."""
            pass

        def set_reward_ring_marker_state(self, agent_id: str, states: Any) -> None:
            """Stub for setting reward ring marker state (can be implemented as extension later)."""
            pass

        def update_metrics(self, *, phase: str = "", metrics=None, step=None, timestamp=None) -> None:
            """Stub for metrics display (can be implemented as extension later)."""
            pass

        def update_ticker(self, entries) -> None:
            """Stub for ticker messages (can be implemented as extension later)."""
            pass

        def update_reward_heatmap(self, heatmap=None, **kwargs) -> None:
            """Stub for reward heatmap (can be implemented as extension later)."""
            pass

        def update_reward_overlays(self, overlays, **kwargs) -> None:
            """Stub for reward overlays (can be implemented as extension later)."""
            pass
