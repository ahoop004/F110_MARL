# MIT License

# Copyright (c) 2020 Joseph Auckley, Matthew O'Kelly, Aman Sinha, Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.



"""
Rendering engine for f1tenth gym env based on pyglet and OpenGL
Author: Hongrui Zheng
"""

# opengl stuff
import pyglet
from pyglet.gl import *
from pyglet.math import Mat4
from pyglet.graphics import Group, ShaderGroup

# other
import numpy as np
from PIL import Image
import yaml
import pandas as pd
# helpers
from f110_gym.envs.collision_models import get_vertices
from .shader import get_default_shader

# zooming constants
ZOOM_IN_FACTOR = 1.2
ZOOM_OUT_FACTOR = 1/ZOOM_IN_FACTOR
R=183
G=193
B=222
R2=99
G2=52
B2=94
# vehicle shape constants
CAR_LENGTH = 0.58
CAR_WIDTH = 0.31

class EnvRenderer(pyglet.window.Window):
    """
    A window class inherited from pyglet.window.Window, handles the camera/projection interaction, resizing window, and rendering the environment
    """
    def __init__(self, width, height, *args, **kwargs):
        """
        Class constructor

        Args:
            width (int): width of the window
            height (int): height of the window

        Returns:
            None
        """
        conf = Config(sample_buffers=1,
                      samples=4,
                      depth_size=16,
                      double_buffer=True)
        super().__init__(width, height, config=conf, resizable=True, vsync=False, *args, **kwargs)

        # gl init
        glClearColor(9/255, 32/255, 87/255, 1.)

        # initialize camera values
        self.left = -width/2
        self.right = width/2
        self.bottom = -height/2
        self.top = height/2
        self.zoom_level = 1.2
        self.zoomed_width = width
        self.zoomed_height = height

        # current batch that keeps track of all graphics
        self.shader = get_default_shader()
        self.shader_group = ShaderGroup(self.shader)
        self.batch = pyglet.graphics.Batch()

        # current env map
        self.map_points = None
        
        # current env agent poses, (num_agents, 3), columns are (x, y, theta)
        self.poses = None

        # current env agent vertices, (num_agents, 4, 2), 2nd and 3rd dimensions are the 4 corners in 2D
        self.vertices = None

        # current score label
        self.score_label = pyglet.text.Label(
                'Lap Time: {laptime:.2f}, Ego Lap Count: {count:.0f}'.format(
                    laptime=0.0, count=0.0),
                font_size=36,
                x=0,
                y=-800,
                anchor_x='center',
                anchor_y='center',
                width=0.01,
                height=0.01,
                color=(255, 255, 255, 255),
                batch=self.batch)

        self.fps_display = pyglet.window.FPSDisplay(self)
        self.scan_lines = None
        self.lidar_fov = kwargs.get('lidar_fov', 2 * np.pi)
        self.max_range = kwargs.get('max_range', 4.0)

    def update_map(self, map_path, map_ext):
        """
        Update the map being drawn by the renderer. Converts image to a list of 3D points representing each obstacle pixel in the map.

        Args:
            map_path (str): absolute path to the map without extensions
            map_ext (str): extension for the map image file

        Returns:
            None
        """
        # print("update_map() called with", map_path, map_ext)
        # load map metadata
        with open(map_path + '.yaml', 'r') as yaml_stream:
            try:
                map_metadata = yaml.safe_load(yaml_stream)
                map_resolution = map_metadata['resolution']
                origin = map_metadata['origin']
                origin_x = origin[0]
                origin_y = origin[1]
            except yaml.YAMLError as ex:
                print(ex)

        # load map image
        map_img = np.array(Image.open(map_path + map_ext).transpose(Image.FLIP_TOP_BOTTOM)).astype(np.float64)
        map_height = map_img.shape[0]
        map_width = map_img.shape[1]

        # convert map pixels to coordinates
        range_x = np.arange(map_width)
        range_y = np.arange(map_height)
        map_x, map_y = np.meshgrid(range_x, range_y)
        map_x = (map_x * map_resolution + origin_x).flatten()
        map_y = (map_y * map_resolution + origin_y).flatten()
        map_z = np.zeros(map_y.shape)
        map_coords = np.vstack((map_x, map_y, map_z))

        # mask and only leave the obstacle points
        map_mask = map_img == 0.0
        map_mask_flat = map_mask.flatten()
        map_points = 50. * map_coords[:, map_mask_flat].T
        
        N = map_points.shape[0]
        positions = map_points[:, :2].flatten().tolist()  
        colors = [255, 193, 50]
        self.map_vlist = self.shader.vertex_list(
            N, pyglet.gl.GL_POINTS, batch=self.batch,
            group=self.shader_group,
            position=('f', positions),
            color=('B', colors*N)
        )
        # print("Map vertex list created with", N, "points")
        # print("Positions length:", len(positions))
        # print("Colors length:", len(colors * N))
        self.map_points = map_points

    def on_resize(self, width, height):
        """
        Callback function on window resize, overrides inherited method, and updates camera values on top of the inherited on_resize() method.

        Potential improvements on current behavior: zoom/pan resets on window resize.

        Args:
            width (int): new width of window
            height (int): new height of window

        Returns:
            None
        """

        # call overrided function
        super().on_resize(width, height)

        # update camera value
        width, height = self.get_size()
        glViewport(0, 0, width, height)
        self.left = -self.zoom_level * width/2
        self.right = self.zoom_level * width/2
        self.bottom = -self.zoom_level * height/2
        self.top = self.zoom_level * height/2
        self.zoomed_width = self.zoom_level * width
        self.zoomed_height = self.zoom_level * height

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        """
        Callback function on mouse drag, overrides inherited method.

        Args:
            x (int): Distance in pixels from the left edge of the window.
            y (int): Distance in pixels from the bottom edge of the window.
            dx (int): Relative X position from the previous mouse position.
            dy (int): Relative Y position from the previous mouse position.
            buttons (int): Bitwise combination of the mouse buttons currently pressed.
            modifiers (int): Bitwise combination of any keyboard modifiers currently active.

        Returns:
            None
        """

        # pan camera
        self.left -= dx * self.zoom_level
        self.right -= dx * self.zoom_level
        self.bottom -= dy * self.zoom_level
        self.top -= dy * self.zoom_level

    def on_mouse_scroll(self, x, y, dx, dy):
        """
        Callback function on mouse scroll, overrides inherited method.

        Args:
            x (int): Distance in pixels from the left edge of the window.
            y (int): Distance in pixels from the bottom edge of the window.
            scroll_x (float): Amount of movement on the horizontal axis.
            scroll_y (float): Amount of movement on the vertical axis.

        Returns:
            None
        """

        # Get scale factor
        f = ZOOM_IN_FACTOR if dy > 0 else ZOOM_OUT_FACTOR if dy < 0 else 1

        # If zoom_level is in the proper range
        if .01 < self.zoom_level * f < 10:

            self.zoom_level *= f

            (width, height) = self.get_size()

            mouse_x = x/width
            mouse_y = y/height

            mouse_x_in_world = self.left + mouse_x*self.zoomed_width
            mouse_y_in_world = self.bottom + mouse_y*self.zoomed_height

            self.zoomed_width *= f
            self.zoomed_height *= f

            self.left = mouse_x_in_world - mouse_x * self.zoomed_width
            self.right = mouse_x_in_world + (1 - mouse_x) * self.zoomed_width
            self.bottom = mouse_y_in_world - mouse_y * self.zoomed_height
            self.top = mouse_y_in_world + (1 - mouse_y) * self.zoomed_height

    def on_close(self):
        """
        Callback function when the 'x' is clicked on the window, overrides inherited method. Also throws exception to end the python program when in a loop.

        Args:
            None

        Returns:
            None

        Raises:
            Exception: with a message that indicates the rendering window was closed
        """

        super().on_close()
        raise Exception('Rendering window was closed.')

    def on_draw(self):
        """
        Function when the pyglet is drawing. The function draws the batch created that includes the map points, the agent polygons, and the information text, and the fps display.
        
        Args:
            None

        Returns:
            None
        """

        # if map and poses doesn't exist, raise exception
        if self.map_points is None:
            raise Exception('Map not set for renderer.')
        if self.poses is None:
            raise Exception('Agent poses not updated for renderer.')

       
        # Clear window with ClearColor
        self.clear()  # clears color/depth buffers
        
        proj = Mat4.orthogonal_projection(
            self.left, self.right,
            self.bottom, self.top,
            -1, 1
        )
        self.shader.use()
        self.shader['projection'] = proj
        glEnable(GL_PROGRAM_POINT_SIZE)
        glPointSize(4) 

 
        self.batch.draw()
        self.fps_display.draw()
      
        self.shader.stop()
      

    def update_obs(self, obs):
        """
        Updates the renderer with the latest observation from the gym environment, including the agent poses, and the information text.

        Args:
            obs (dict): observation dict from the gym env

        Returns:
            None
        """

        self.ego_idx = obs['ego_idx']
        poses_x = obs['poses_x']
        poses_y = obs['poses_y']
        poses_theta = obs['poses_theta']

        num_agents = len(poses_x)
        if self.poses is None:
            self.cars = []

            for i in range(num_agents):
                vertices = get_vertices(np.array([0., 0., 0.]), CAR_LENGTH, CAR_WIDTH).flatten().tolist()
                color = [R,G,B] if i==self.ego_idx else [R2,G2,B2]
                vlist = self.shader.vertex_list(
                    4, pyglet.gl.GL_QUADS, batch=self.batch,
                    group=self.shader_group,
                    position=('f', vertices),
                    color=('B', color*4)
                )
                self.cars.append(vlist)

        poses = np.stack((poses_x, poses_y, poses_theta)).T
        for j in range(poses.shape[0]):
            vertices_np = 50. * get_vertices(poses[j, :], CAR_LENGTH, CAR_WIDTH)
            vertices = list(vertices_np.flatten())
            self.cars[j].position[:] = vertices
        self.poses = poses

        self._draw_lidar_endpoints(obs)
        self.render_callback()
        self.score_label.text = 'Lap Time: {laptime:.2f}, Ego Lap Count: {count:.0f}'.format(laptime=obs['lap_times'][0], count=obs['lap_counts'][obs['ego_idx']])

    def _draw_lidar_beams(self, obs):
        # 1) grab & optionally subsample
        full_n = obs['scans'][obs['ego_idx']].shape[0]
        idxs     = np.arange(0, full_n, 60)   
        scan = obs['scans'][obs['ego_idx']] # e.g. every 30th beam
        dists    = scan[idxs]
        n        = len(dists)

        # 2) compute beam angles around ego heading
        theta_0      = self.poses[obs['ego_idx'], 2]
        angles  = (np.linspace(-self.lidar_fov/2,
                               self.lidar_fov/2,
                               full_n)[idxs] + theta_0)

        # 3) world→pixel coords
        x0, y0      = self.poses[obs['ego_idx'], :2]
        scale       = 50.0
        x0_pix, y0_pix = x0*scale, y0*scale
        xs_pix    = (x0 + dists * np.cos(angles)) * scale
        ys_pix    = (y0 + dists * np.sin(angles)) * scale

        # 4) build line‐segment vertex list
        positions = []
        for xi, yi in zip(xs_pix, ys_pix):
            positions.extend([x0_pix, y0_pix, xi, yi])


        colors = [0, 255, 0] * (2*n)

        if self.scan_lines is None:
            self.scan_lines = self.shader.vertex_list(
                2*n,
                pyglet.gl.GL_LINES,
                batch=self.batch,
                group=self.shader_group,
                position=('f', positions),
                color=('B', colors)
            )
        else:
            self.scan_lines.position[:] = positions
            self.scan_lines.color[:]   = colors
    
    
    def _draw_lidar_endpoints(self, obs):
        """
        Draws lidar scan endpoints as colored points:
        - Red if they hit an obstacle (within max range)
        - Gray if they go to max range (no collision)
        """
        # full_n = obs['scans'][obs['ego_idx']].shape[0]
        # idxs = np.arange(0, full_n, 60)  # sample every 60th beam for speed; tweak as needed
        dists = obs['scans'][obs['ego_idx']]
        n = len(dists)

        # Compute beam angles
        theta_0 = self.poses[obs['ego_idx'], 2]
        angles = (np.linspace(-self.lidar_fov/2, self.lidar_fov/2, n) + theta_0)

        # Get vehicle position (world/pixels)
        x0, y0 = self.poses[obs['ego_idx'], :2]
        scale = 50.0
        xs_pix = (x0 + dists * np.cos(angles)) * scale
        ys_pix = (y0 + dists * np.sin(angles)) * scale

        # Draw endpoints as colored points
        if hasattr(self, 'scan_hits'):
            self.scan_hits.delete()
        hit_positions = []
        hit_colors = []
        for xi, yi, dist in zip(xs_pix, ys_pix, dists):
            hit_positions.extend([xi, yi])
            if dist < self.max_range * 0.99:
                hit_colors.extend([255, 0, 0])  # red for hit
            else:
                hit_colors.extend([180, 180, 180])  # gray for max range
        self.scan_hits = self.shader.vertex_list(
            n,
            pyglet.gl.GL_POINTS,
            batch=self.batch,
            group=self.shader_group,
            position=('f', hit_positions),
            color=('B', hit_colors)
        )
    def meters_to_map_px(x, y, origin_x, origin_y, resolution):
        """
        Convert (x, y) in world meters to (px, py) in map PNG pixels.
        """
        px = (x - origin_x) / resolution
        py = (y - origin_y) / resolution
        return px, py
    
    def make_centerline_callback(centerline_csv_path, point_size = 1):
    # Read the CSV, skipping the header/comment row(s)
        df = pd.read_csv(centerline_csv_path, comment='#', header=None)
        # Only x and y columns
        waypoints = df[[0, 1]].values  # shape (N,2)

        scale = 50.0  # Match EnvRenderer scaling
        waypoints_scaled = waypoints * scale

        def callback(env_renderer):
            glPointSize(point_size)
            if not hasattr(env_renderer, '_centerline_vlist'):
                n_points = waypoints_scaled.shape[0]
                positions = waypoints_scaled.flatten().tolist()
                color = [0, 255, 0]  

                env_renderer._centerline_vlist = env_renderer.shader.vertex_list(
                    n_points, pyglet.gl.GL_POINTS, batch=env_renderer.batch,
                    group=env_renderer.shader_group,
                    position=('f', positions),
                    color=('B', color * n_points)
                )
        return callback
    
    def make_waypoints_callback(waypoints_csv_path, passed_flags=None, point_size=3):
        """
        Create a callback function that draws a set of waypoints and updates their colors
        based on whether they've been passed.

        Args:
            waypoints_csv_path (str): Path to a CSV file with at least two columns (x, y).
            passed_flags (list or None): Boolean list indicating which waypoints have been passed.
                Waypoints set to True are drawn red. If None, colors stay static.
            point_size (int): Size of the rendered points.

        Returns:
            Callable[[EnvRenderer], None]: A callback usable with add_render_callback().
        """

        df = pd.read_csv(waypoints_csv_path, header=None, comment='#')
        if df.shape[1] < 2:
            raise ValueError(f"{waypoints_csv_path} must have at least two columns for x and y.")
        waypoints = df.iloc[:, :2].values

        scale = 50.0
        waypoints_scaled = waypoints * scale
        num_points = len(waypoints_scaled)

        # initial colors: first waypoint white, others yellow
        initial_colors = []
        for i in range(num_points):
            if i == 0:
                initial_colors.extend([255, 255, 255])  # white
            else:
                initial_colors.extend([255, 255, 0])    # yellow

        def callback(env_renderer):
            glPointSize(point_size)

            # Determine current target: the first False in passed_flags
            current_idx = None
            if passed_flags is not None:
                for i, passed in enumerate(passed_flags):
                    if not passed:
                        current_idx = i
                        break

            # Delete the previous waypoint vertex list if it exists
            if hasattr(env_renderer, '_waypoints_vlist'):
                try:
                    env_renderer._waypoints_vlist.delete()
                except Exception:
                    pass
                del env_renderer._waypoints_vlist

            # Build a color array: passed → red, current → white, unpassed → yellow
            color_data = []
            for idx in range(num_points):
                if passed_flags and idx < len(passed_flags) and passed_flags[idx]:
                    color_data.extend([255, 0, 0])      # red
                elif current_idx is not None and idx == current_idx:
                    color_data.extend([255, 255, 255])  # white
                else:
                    color_data.extend([255, 255, 0])    # yellow

            # Create a new vertex list with updated colors
            positions = waypoints_scaled.flatten().tolist()
            env_renderer._waypoints_vlist = env_renderer.shader.vertex_list(
                num_points,
                pyglet.gl.GL_POINTS,
                batch=env_renderer.batch,
                group=env_renderer.shader_group,
                position=('f', positions),
                color=('B', color_data)
            )
        return callback


    def render_callback(self):
    # custom extra drawing function

        # e = env_renderer

        # update camera to follow car
        x = self.cars[0].position[::2]
        y = self.cars[0].position[1::2]
        top, bottom, left, right = max(y), min(y), min(x), max(x)
        self.score_label.x = left
        self.score_label.y = top - 700
        self.left = left - 800
        self.right = right + 800
        self.top = top + 800
        self.bottom = bottom - 800