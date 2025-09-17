# MIT License
# Rendering engine for F1TENTH-style env using pyglet + OpenGL
# Refactored for PettingZoo MARL with per-agent render_obs dict.

import pyglet
from pyglet.gl import *
from pyglet.math import Mat4
from pyglet.graphics import Group, ShaderGroup

import numpy as np
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

# colors
CAR_LEARNER = (183, 193, 222)
CAR_OTHER   = ( 99,  52,  94)
LIDAR_COLOR_HIT = (255,   0,   0)
LIDAR_COLOR_MAX = (180, 180, 180)

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

        # shader + batch
        self.shader = get_default_shader()
        self.shader_group = ShaderGroup(self.shader)
        self.batch = pyglet.graphics.Batch()

        # map
        self.map_points = None
        self.map_vlist = None

        # per-agent drawables and cached state
        self.cars_vlist = {}       # aid -> vertex_list (GL_QUADS)
        self.scan_hits_vlist = {}  # aid -> vertex_list (GL_POINTS)
        self.agent_infos = {}      # aid -> last render_obs snapshot
        self.agent_ids = []        # agents updated this frame (sorted)
        self._camera_target = None
        self._lidar_angle_cache = {}

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
            color=(255, 255, 255, 255)
        )

    # ---------- Map ----------

    def update_map(self, map_path_no_ext: str, map_ext: str):
        """
        Update map geometry.
        map_path_no_ext: absolute path WITHOUT extension (e.g., '/.../maps/levine')
        map_ext: image extension (e.g., '.png')
        """
        yaml_path = map_path_no_ext + '.yaml'
        img_path  = map_path_no_ext + map_ext

        with open(yaml_path, 'r') as f:
            meta = yaml.safe_load(f)
        res = float(meta['resolution'])
        origin = meta['origin']
        ox, oy = origin[0], origin[1]

        img = np.array(Image.open(img_path).transpose(Image.FLIP_TOP_BOTTOM)).astype(np.float64)
        H, W = img.shape[0], img.shape[1]

        xs = (np.arange(W) * res + ox)
        ys = (np.arange(H) * res + oy)
        gx, gy = np.meshgrid(xs, ys)
        gz = np.zeros_like(gx)

        mask = (img == 0.0)  # obstacle pixels are black
        pts = np.vstack((gx[mask], gy[mask], gz[mask])).T  # (N,3)
        pts *= self.render_scale

        N = pts.shape[0]
        positions = pts[:, :2].flatten().tolist()
        colors = [255, 193, 50] * N

        if self.map_vlist is not None:
            self.map_vlist.delete()
            self.map_vlist = None

        self.map_vlist = self.shader.vertex_list(
            N, pyglet.gl.GL_POINTS, batch=self.batch, group=self.shader_group,
            position=('f', positions),
            color=('B', colors)
        )
        self.map_points = pts

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
        self.hud_label.y = height - 10

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        self.left -= dx * self.zoom_level
        self.right -= dx * self.zoom_level
        self.bottom -= dy * self.zoom_level
        self.top += dy * self.zoom_level

    def on_mouse_scroll(self, x, y, dx, dy):
        f = ZOOM_IN_FACTOR if dy > 0 else ZOOM_OUT_FACTOR if dy < 0 else 1.0
        if 0.01 < self.zoom_level * f < 10.0:
            self.zoom_level *= f
            w, h = self.get_size()
            mx, my = x / w, y / h
            mx_world = self.left + mx * self.zoomed_width
            my_world = self.bottom + my * self.zoomed_height
            self.zoomed_width *= f
            self.zoomed_height *= f
            self.left = mx_world - mx * self.zoomed_width
            self.right = mx_world + (1 - mx) * self.zoomed_width
            self.bottom = my_world - my * self.zoomed_height
            self.top = my_world + (1 - my) * self.zoomed_height

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
        glEnable(GL_PROGRAM_POINT_SIZE)
        glPointSize(4)

        self.batch.draw()
        self.shader.stop()

        self.fps_display.draw()
        self.hud_label.draw()

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
            self._camera_target = active_ids[0]
        elif not self.agent_ids and self.agent_infos:
            self.agent_ids = sorted(self.agent_infos.keys())
            self._camera_target = self.agent_ids[0]

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
                self.cars_vlist[aid] = self.shader.vertex_list(
                    4, pyglet.gl.GL_QUADS, batch=self.batch, group=self.shader_group,
                    position=('f', positions),
                    color=('B', color * 4)
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

            base_angles = self._lidar_angle_cache.get(n)
            if base_angles is None:
                base_angles = np.linspace(-self.lidar_fov / 2.0, self.lidar_fov / 2.0, n, dtype=np.float32)
                self._lidar_angle_cache[n] = base_angles

            beam_angles = base_angles + th
            cos_vals = np.cos(beam_angles)
            sin_vals = np.sin(beam_angles)

            xs = (x + scans * cos_vals) * self.render_scale
            ys = (y + scans * sin_vals) * self.render_scale

            positions_hits = np.empty(2 * n, dtype=np.float32)
            positions_hits[0::2] = xs
            positions_hits[1::2] = ys

            colors_hits = np.tile(LIDAR_COLOR_MAX, (n, 1)).astype(np.uint8)
            hit_mask = scans < self.max_range * 0.99
            colors_hits[hit_mask] = LIDAR_COLOR_HIT
            colors_hits_flat = colors_hits.reshape(-1)

            scan_vlist = self.scan_hits_vlist.get(aid)
            if scan_vlist is None or len(scan_vlist.position) != positions_hits.size:
                if scan_vlist is not None:
                    try:
                        scan_vlist.delete()
                    except Exception:
                        pass
                self.scan_hits_vlist[aid] = self.shader.vertex_list(
                    n, pyglet.gl.GL_POINTS, batch=self.batch, group=self.shader_group,
                    position=('f', positions_hits.tolist()),
                    color=('B', colors_hits_flat.tolist())
                )
            else:
                scan_vlist.position[:] = positions_hits.tolist()
                scan_vlist.color[:] = colors_hits_flat.tolist()

        hud_lines = [f'Agents: {len(all_ids)}']
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
            hud_lines.append(f"{aid}: {lap_str} {time_str}{status_suffix}")

        self.hud_label.text = "\n".join(hud_lines)

        self._camera_follow_first()

    # ---------- Helpers ----------

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
        self.left   = left  - 800
        self.right  = right + 800
        self.top    = top   + 800
        self.bottom = bottom - 800

    # ---------- Optional: utilities for extra overlays ----------

    @staticmethod
    def make_centerline_callback(centerline_csv_path: str, point_size=1):
        df = pd.read_csv(centerline_csv_path, comment='#', header=None)
        waypoints = df[[0, 1]].values
        def callback(env_renderer: "EnvRenderer"):
            glPointSize(point_size)
            pts = (waypoints * env_renderer.render_scale).flatten().tolist()
            color = [0, 255, 0]
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
                    colors.extend([255, 0, 0])     # passed
                elif current_idx is not None and i == current_idx:
                    colors.extend([255, 255, 255]) # current
                else:
                    colors.extend([255, 255, 0])   # pending
            pos = (waypoints * env_renderer.render_scale).flatten().tolist()
            env_renderer._waypoints_vlist = env_renderer.shader.vertex_list(
                num, pyglet.gl.GL_POINTS, batch=env_renderer.batch, group=env_renderer.shader_group,
                position=('f', pos), color=('B', colors)
            )
        return callback
