import functools
from typing import Dict, Tuple, Optional, Any, List
import gymnasium as gym
from gymnasium import error, spaces, utils
import yaml
from PIL import Image

from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers

# base classes
from ..physics.base_classes import Simulator, Integrator

# others
import numpy as np
import os
import time

# gl
import pyglet
pyglet.options['debug_gl'] = False
from pyglet import gl
from pyglet import image as pyg_img

# constants

# rendering
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800

class F110Env(ParallelEnv):

    metadata = {"name": "F110ParallelEnv", "render_modes": ["human", "human_fast", "rgb_array"]}

    # rendering
    renderer = None
    current_obs = None
    render_callbacks = []

    def __init__(self, **kwargs):        
        # kwargs extraction
        try:
            self.seed = kwargs['seed']
        except:
            self.seed = 42
            
        try:
            self.max_steps = kwargs['max_steps']
        except:
            self.max_steps = 5000
        try:
            self.map_name = kwargs['map']
            # different default maps
            if self.map_name == 'berlin':
                self.map_path = os.path.dirname(os.path.abspath(__file__)) + '/maps/berlin.yaml'
            elif self.map_name == 'skirk':
                self.map_path = os.path.dirname(os.path.abspath(__file__)) + '/maps/skirk.yaml'
            elif self.map_name == 'levine':
                self.map_path = os.path.dirname(os.path.abspath(__file__)) + '/maps/levine.yaml'
            else:
                self.map_path = self.map_name + '.yaml'
        except:
            self.map_path = os.path.dirname(os.path.abspath(__file__)) + '/maps/vegas.yaml'

        try:
            self.map_ext = kwargs['map_ext']
        except:
            self.map_ext = '.png'

        try:
            self.params = kwargs['params']
        except:
            self.params = {'mu': 1.0489,
                           'C_Sf': 4.718,
                           'C_Sr': 5.4562,
                           'lf': 0.15875,
                           'lr': 0.17145,
                           'h': 0.074,
                           'm': 3.74,
                           'I': 0.04712,
                           's_min': -0.4189,
                           's_max': 0.4189,
                           'sv_min': -3.2,
                           'sv_max': 3.2,
                           'v_switch': 7.319,
                           'a_max': 9.51,
                           'v_min':-5.0,
                           'v_max': 20.0,
                           'width': 0.31,
                           'length': 0.58}

        # simulation parameters
        self.n_agents = int(kwargs.get("num_agents", 2))
        self.possible_agents = [f"car_{i}" for i in range(self.n_agents)]
        self.agents = self.possible_agents.copy()


        try:
            self.timestep = kwargs['timestep']
        except:
            self.timestep = 0.01

        # default ego index
        try:
            self.ego_idx = kwargs['ego_idx']
        except:
            self.ego_idx = 0

        # default integrator
        try:
            self.integrator = kwargs['integrator']
        except:
            self.integrator = Integrator.RK4
            
        # default LiDAR position
        try:
            self.lidar_dist = kwargs['lidar_dist']
        except:
            self.lidar_dist = 0.0

        # radius to consider done
        self.start_thresh = 0.5  # 10cm

        # env states
        self.poses_x = []
        self.poses_y = []
        self.poses_theta = []
        self.collisions = np.zeros((self.n_agents, ))
       
        self.max_steps = int(kwargs.get('max_steps', 0))  # 0 = no limit
        self._elapsed_steps = 0

        # loop completion
        self.near_start = True
        self.num_toggles = 0

        # race info
        self.lap_times = np.zeros((self.n_agents, ))
        self.lap_counts = np.zeros((self.n_agents, ))
        self.current_time = 0.0

        # finish line info
        self.num_toggles = 0
        self.near_start = True
        self.near_starts = np.array([True]*self.n_agents)
        self.toggle_list = np.zeros((self.n_agents,))
        self.start_xs = np.zeros((self.n_agents, ))
        self.start_ys = np.zeros((self.n_agents, ))
        self.start_thetas = np.zeros((self.n_agents, ))
        self.start_rot = np.eye(2)

        # initiate stuff
        self.sim = Simulator(self.params, self.n_agents, self.seed, time_step=self.timestep, integrator=self.integrator, lidar_dist=self.lidar_dist)
        self.sim.set_map(self.map_path, self.map_ext)
        
        meta = yaml.safe_load(open('/home/aaron/f110_gymnasium_ros2_jazzy/assets/maps/levine.yaml'))
        R = meta['resolution']
        x0, y0, _ = meta.get('origin', (0.0, 0.0, 0.0))
        img = Image.open('/home/aaron/f110_gymnasium_ros2_jazzy/assets/maps/' + meta['image'])
        width, height = img.size
        x_min = x0
        x_max = x0 + width * R
        y_min = y0
        y_max = y0 + height * R

        # stateful observations for rendering
        self.render_obs = None
        
        self._single_action_space = spaces.Box(
            low=np.array([self.params["s_min"], self.params["v_min"]], dtype=np.float32),
            high=np.array([self.params["s_max"], self.params["v_max"]], dtype=np.float32),
            dtype=np.float32,
        )
 
        
        self._single_observation_space = spaces.Dict({
            "scans":         spaces.Box(0.0, 30.0, shape=(1080,), dtype=np.float32),
            "poses_x":       spaces.Box(x_min, x_max, shape=(), dtype=np.float32),
            "poses_y":       spaces.Box(y_min, y_max, shape=(), dtype=np.float32),
            "poses_theta":   spaces.Box(-np.pi, np.pi, shape=(), dtype=np.float32),
            "linear_vels_x": spaces.Box(self.params["v_min"], self.params["v_max"], shape=(), dtype=np.float32),
            "linear_vels_y": spaces.Box(self.params["v_min"], self.params["v_max"], shape=(), dtype=np.float32),
            "ang_vels_z":    spaces.Box(0.0, 10.0, shape=(), dtype=np.float32),
            "collisions":    spaces.Discrete(2),
            "lap_times":     spaces.Box(0.0, 1e5, shape=(), dtype=np.float32),
            "lap_counts":    spaces.Box(0, 10, shape=(), dtype=np.float32),
        })
        
        

    def __del__(self):

        pass
    
    def observation_space(self, agent: str):
        return self._single_observation_space  # per-agent space, required by Parallel API. :contentReference[oaicite:1]{index=1}

    def action_space(self, agent: str):
        return self._single_action_space

    def _check_done(self):

        left_t = 2
        right_t = 2
        
        poses_x = np.array(self.poses_x)-self.start_xs
        poses_y = np.array(self.poses_y)-self.start_ys
        delta_pt = np.dot(self.start_rot, np.stack((poses_x, poses_y), axis=0))
        temp_y = delta_pt[1,:]
        idx1 = temp_y > left_t
        idx2 = temp_y < -right_t
        temp_y[idx1] -= left_t
        temp_y[idx2] = -right_t - temp_y[idx2]
        temp_y[np.invert(np.logical_or(idx1, idx2))] = 0

        dist2 = delta_pt[0, :]**2 + temp_y**2
        closes = dist2 <= 0.1
        for i in range(self.num_agents):
            if closes[i] and not self.near_starts[i]:
                self.near_starts[i] = True
                self.toggle_list[i] += 1
            elif not closes[i] and self.near_starts[i]:
                self.near_starts[i] = False
                self.toggle_list[i] += 1
            self.lap_counts[i] = self.toggle_list[i] // 2
            if self.toggle_list[i] < 4:
                self.lap_times[i] = self.current_time
        
        done = (self.collisions[self.ego_idx]) or np.all(self.toggle_list >= 4)
        
        return bool(done), self.toggle_list >= 4

    def _update_state(self, obs_dict):

        self.poses_x = obs_dict['poses_x']
        self.poses_y = obs_dict['poses_y']
        self.poses_theta = obs_dict['poses_theta']
        self.collisions = obs_dict['collisions']

   
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None: self._seed = seed
        self.agents = self.possible_agents.copy()
        self._elapsed_steps = 0

        # options: (N,3) poses (x,y,theta). If None, caller must set internally.
        poses = options if options is not None else np.zeros((self.num_agents, 3), dtype=np.float32)
        obs_joint = self.sim.reset(poses)
        obs = self._split_obs(obs_joint)
        self.render_obs = {
            "ego_idx":     self.ego_idx,
            "poses_x":     obs_joint["poses_x"],
            "poses_y":     obs_joint["poses_y"],
            "poses_theta": obs_joint["poses_theta"],
            "lap_times":   self.lap_times,
            "lap_counts":  self.lap_counts,
            "scans":       obs_joint["scans"],
        }
        infos = {aid: {} for aid in self.agents}
        return obs, infos

    def step(self, actions: Dict[str, np.ndarray]):
        # dict->joint action in consistent order
        joint = np.zeros((self.num_agents, 2), dtype=np.float32)
        alive_mask = np.zeros(self.num_agents, dtype=bool)
        for i, aid in enumerate(self.possible_agents):
            if aid in self.agents and aid in actions:
                joint[i] = np.asarray(actions[aid], dtype=np.float32)
                alive_mask[i] = True

        obs_joint = self.sim.step(joint)
        obs = self._split_obs(obs_joint)
        self.render_obs = {
            "ego_idx":     self.ego_idx,
            "poses_x":     obs_joint["poses_x"],
            "poses_y":     obs_joint["poses_y"],
            "poses_theta": obs_joint["poses_theta"],
            "lap_times":   self.lap_times,
            "lap_counts":  self.lap_counts,
            "scans":       obs_joint["scans"],
        }

        # simple per-step reward (customize as needed)
        rewards = {aid: float(self.timestep) for aid in self.agents}

        # terminations/truncations
        terminations = {aid: bool(obs_joint["collisions"][i]) for i, aid in enumerate(self.possible_agents)}
        trunc_flag = (self.max_steps > 0) and (self._elapsed_steps + 1 >= self.max_steps)
        truncations = {aid: bool(trunc_flag) for aid in self.possible_agents}
        infos = {aid: {} for aid in self.possible_agents}

        # advance and cull finished agents
        self._elapsed_steps += 1
        self.agents = [aid for aid in self.possible_agents if not (terminations[aid] or truncations[aid])]

        return obs, rewards, terminations, truncations, infos

    def update_map(self, map_path, map_ext):

        self.sim.set_map(map_path, map_ext)

    def update_params(self, params, index=-1):

        self.sim.update_params(params, agent_idx=index)

    def add_render_callback(self, callback_func):
        self.render_callbacks.append(callback_func)

    def render(self, mode='human'):
 
        assert mode in ['human', 'human_fast']
        
        if self.renderer is None:
            # first call, initialize everything
            from f110_gym.envs.rendering import EnvRenderer
            fov = 4.7
            max_range=30.0
            self.renderer = EnvRenderer(WINDOW_W,
                                           WINDOW_H,
                                           
                                          )
            self.renderer.lidar_fov = fov       # use the same FOV as your scanner, e.g. 4.7 rad
            self.renderer.max_range = max_range
            self.renderer.update_map(self.map_name, self.map_ext)
            
        self.renderer.update_obs(self.render_obs)

        for render_callback in self.render_callbacks:
            render_callback(self.renderer)
        
        self.renderer.dispatch_events()
        self.renderer.on_draw()
        self.renderer.flip()
        if mode == 'human':
            time.sleep(0.005)
        elif mode == 'human_fast':
            pass
        elif mode == "rgb_array":
            # Grab the current GL color buffer and return HxWx3 RGB (uint8)
            buf = pyg_img.get_buffer_manager().get_color_buffer()
            w, h = buf.width, buf.height
            img = buf.get_image_data()
            # flip vertically via negative pitch so the array is top-to-bottom
            data = img.get_data("RGB", -w * 3)
            frame = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3).copy()
            return frame
    
    def close(self): pass

    # helper: joint->per-agent dicts expected by PZ Parallel API
    def _split_obs(self, j: Dict[str, np.ndarray]) -> Dict[str, Dict[str, np.ndarray]]:
        out: Dict[str, Dict[str, np.ndarray]] = {}
        for i, aid in enumerate(self.possible_agents):
            out[aid] = {
                "lidar":         j["scans"][i].astype(np.float32),
                "poses_x":       np.float32(j["poses_x"][i]),
                "poses_y":       np.float32(j["poses_y"][i]),
                "poses_theta":   np.float32(j["poses_theta"][i]),
                "linear_vels_x": np.float32(j["linear_vels_x"][i]),
                "linear_vels_y": np.float32(j["linear_vels_y"][i]),
                "ang_vels_z":    np.float32(j["ang_vels_z"][i]),
                "collisions":    np.int32(j["collisions"][i]),
            }
        return out