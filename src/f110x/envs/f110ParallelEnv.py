import functools
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List
import gymnasium as gym
from gymnasium import error, spaces, utils
import yaml
from PIL import Image

from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers

# base classes
from f110x.physics import Simulator, Integrator
from f110x.render import EnvRenderer


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
# VIDEO_W = 600
# VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800

class F110ParallelEnv(ParallelEnv):

    metadata = {"name": "F110ParallelEnv", "render_modes": ["human", "rgb_array"]}

    # rendering
    def __init__(self, **kwargs):
      
        env_config = kwargs.get("env", {})
        merged = {**env_config, **kwargs}
        
        self.render_mode = merged.get("render_mode", "human")
        self.metadata = {"render_modes": ["human", "rgb_array"], "name": "F110ParallelEnv"}     
        self.renderer: Optional[EnvRenderer] = None
        self.current_obs = None
        self.render_callbacks: List[Any] = []

        self.seed: int = int(merged.get("seed", 42))
        self.max_steps: int = int(merged.get("max_steps", 5000))
        self.n_agents: int = int(merged.get("n_agents", 2))
        self.possible_agents = [f"car_{i}" for i in range(self.n_agents)]
        self._agent_id_to_index = {aid: idx for idx, aid in enumerate(self.possible_agents)}
        self.agents = self.possible_agents.copy()
   
        self.timestep: float = float(merged.get("timestep", 0.01))
        self.integrator = merged.get("integrator", Integrator.RK4)

        self.map_dir = Path(merged.get("map_dir", None))

        def _normalize_map_id(identifier: Optional[str]) -> Optional[str]:
            if identifier is None:
                return None
            identifier = str(identifier)
            return identifier if Path(identifier).suffix else f"{identifier}.yaml"

        raw_map_name = merged.get("map", None)
        raw_map_yaml = merged.get("map_yaml", None)
        self.map_ext = merged.get("map_ext", ".png")

        self.map_name = _normalize_map_id(raw_map_name)
        self.map_yaml = _normalize_map_id(raw_map_yaml)

        if self.map_name is None and self.map_yaml is not None:
            self.map_name = self.map_yaml
        elif self.map_yaml is None and self.map_name is not None:
            self.map_yaml = self.map_name

        self.map_path = (self.map_dir / f"{self.map_name}").resolve()
        self.yaml_path = (self.map_dir / f"{self.map_yaml}").resolve()
        self.map_base = self.yaml_path.with_suffix("")
        self.start_poses = np.array(merged.get("start_poses", []),dtype=np.float32)
        defaults = {'mu': 1.0489,
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
                    'v_min': -5.0,
                    'v_max': 20.0,
                    'width': 0.31,
                    'length': 0.58}
        vehicle_params = merged.get("vehicle_params")
        if vehicle_params is None:
            vehicle_params = merged.get("params")
        self.params = defaults if vehicle_params is None else {**defaults, **vehicle_params}
        
        self.lidar_dist: float = float(merged.get("lidar_dist", 0.0))
        self.start_thresh: float = float(merged.get("start_thresh", 0.5))
        
        # env states
        self.poses_x = np.zeros((self.n_agents, ))
        self.poses_y = np.zeros((self.n_agents, ))
        self.poses_theta = np.zeros((self.n_agents, ))
        self.collisions = np.zeros((self.n_agents, ))
        default_terminate = bool(merged.get("terminate_on_collision", True))
        flags = [default_terminate] * self.n_agents
        agent_index = {aid: idx for idx, aid in enumerate(self.possible_agents)}

        scenario_agents = merged.get("agents")
        if isinstance(scenario_agents, list):
            for idx, agent_cfg in enumerate(scenario_agents):
                if idx >= self.n_agents or not isinstance(agent_cfg, dict):
                    continue
                for key in ("id", "name", "agent_id"):
                    alias = agent_cfg.get(key)
                    if isinstance(alias, str) and alias not in agent_index:
                        agent_index[alias] = idx

        raw_overrides = merged.get("terminate_on_collision_overrides")
        if isinstance(raw_overrides, dict):
            for key, value in raw_overrides.items():
                if value is None:
                    continue
                flag = bool(value)
                if isinstance(key, int):
                    if 0 <= key < self.n_agents:
                        flags[key] = flag
                else:
                    idx = agent_index.get(str(key))
                    if idx is not None:
                        flags[idx] = flag
        elif isinstance(raw_overrides, (list, tuple, np.ndarray)):
            for idx, value in enumerate(raw_overrides):
                if idx >= self.n_agents or value is None:
                    continue
                flags[idx] = bool(value)

        if isinstance(scenario_agents, list):
            for idx, agent_cfg in enumerate(scenario_agents):
                if not isinstance(agent_cfg, dict):
                    continue
                override = agent_cfg.get("terminate_on_collision")
                task_cfg = agent_cfg.get("task")
                if override is None and isinstance(task_cfg, dict):
                    override = task_cfg.get("terminate_on_collision")
                if override is None:
                    continue
                flag = bool(override)
                alias_keys: List[str] = []
                if idx < self.n_agents:
                    alias_keys.append(self.possible_agents[idx])
                for key in ("id", "name", "agent_id"):
                    alias = agent_cfg.get(key)
                    if isinstance(alias, str):
                        alias_keys.append(alias)
                for alias in alias_keys:
                    mapped = agent_index.get(alias)
                    if mapped is not None:
                        flags[mapped] = flag

        self.terminate_on_collision = {
            aid: flags[idx] for idx, aid in enumerate(self.possible_agents)
        }

        raw_target_laps = merged.get("target_laps")
        if raw_target_laps is None:
            raw_target_laps = merged.get("laps")
        target_laps_val: Optional[int] = None
        if raw_target_laps is not None:
            try:
                parsed = int(raw_target_laps)
            except (TypeError, ValueError):
                parsed = 0
            if parsed > 0:
                target_laps_val = parsed

        if target_laps_val is None and isinstance(scenario_agents, list):
            max_agent_laps = 0
            for agent_cfg in scenario_agents:
                if not isinstance(agent_cfg, dict):
                    continue
                candidate = agent_cfg.get("target_laps")
                if candidate is None:
                    candidate = agent_cfg.get("laps")
                if candidate is None:
                    task_cfg = agent_cfg.get("task")
                    if isinstance(task_cfg, dict):
                        candidate = task_cfg.get("target_laps")
                        if candidate is None:
                            candidate = task_cfg.get("laps")
                if candidate is None:
                    continue
                try:
                    laps_val = int(candidate)
                except (TypeError, ValueError):
                    continue
                if laps_val > max_agent_laps:
                    max_agent_laps = laps_val
            if max_agent_laps > 0:
                target_laps_val = max_agent_laps

        if target_laps_val is None or target_laps_val <= 0:
            target_laps_val = 1
        self.target_laps: int = int(target_laps_val)
    
        # race info
        self.lap_times = np.zeros((self.n_agents, ))
        self.lap_counts = np.zeros((self.n_agents, ))
        self.current_time = 0.0
        self._elapsed_steps = 0

        # finish line info
        self.num_toggles = 0
        self.near_start = True
        self.near_starts = np.array([True]*self.n_agents)
        self.toggle_list = np.zeros((self.n_agents,))
        self.start_xs = np.zeros((self.n_agents, ))
        self.start_ys = np.zeros((self.n_agents, ))
        self.start_thetas = np.zeros((self.n_agents, ))
        start_angles = np.zeros(self.n_agents, dtype=np.float32)
        if self.start_thetas.size == self.n_agents:
            start_angles[:] = np.asarray(self.start_thetas, dtype=np.float32)
        start_pose_array = None
        if self.start_poses.size > 0:
            start_pose_array = np.atleast_2d(self.start_poses)
            if start_pose_array.shape[1] >= 3:
                count = min(self.n_agents, start_pose_array.shape[0])
                start_angles[:count] = start_pose_array[:count, 2]

        if start_angles.size == 0:
            self.start_rot = np.eye(2, dtype=np.float32)
        elif np.allclose(start_angles, start_angles[0]):
            angle = float(start_angles[0])
            cos_a = float(np.cos(angle))
            sin_a = float(np.sin(angle))
            self.start_rot = np.array(
                [[cos_a, -sin_a],
                 [sin_a,  cos_a]],
                dtype=np.float32,
            )
        else:
            cos_vals = np.cos(start_angles).astype(np.float32)
            sin_vals = np.sin(start_angles).astype(np.float32)
            rot = np.empty((self.n_agents, 2, 2), dtype=np.float32)
            rot[:, 0, 0] = cos_vals
            rot[:, 0, 1] = -sin_vals
            rot[:, 1, 0] = sin_vals
            rot[:, 1, 1] = cos_vals
            self.start_rot = rot
        if start_pose_array is not None:
            count = min(self.n_agents, start_pose_array.shape[0])
            if count:
                cols = start_pose_array.shape[1]
                if cols >= 1:
                    self.start_xs[:count] = start_pose_array[:count, 0]
                if cols >= 2:
                    self.start_ys[:count] = start_pose_array[:count, 1]
                if cols >= 3:
                    self.start_thetas[:count] = start_pose_array[:count, 2]
                else:
                    self.start_thetas[:count] = start_angles[:count]

        # initiate stuff
        self.sim = Simulator(self.params,
                             self.n_agents,
                             self.seed,
                             time_step=self.timestep,
                             lidar_dist=self.lidar_dist)
        
        self.sim.set_map(str(self.yaml_path), self.map_ext)
        
        meta = merged.get("map_meta")
        if meta is None:
            with open(self.map_path, "r") as f:
                meta = yaml.safe_load(f)

        preloaded_image_path = merged.get("map_image_path")
        image_rel = meta.get("image")
        if preloaded_image_path is not None:
            img_path = Path(preloaded_image_path).resolve()
        elif image_rel:
            img_path = (self.map_path.parent / image_rel).resolve()
        else:
            img_filename = merged.get("map_image", None)
            img_path = (self.map_dir / img_filename).resolve()

        image_size = merged.get("map_image_size")
        if image_size is not None:
            width, height = map(int, image_size)
        else:
            with Image.open(img_path) as img:
                width, height = img.size

        self.map_meta = meta
        self.map_image_path = img_path

        R = float(meta.get("resolution", 1.0))
        x0, y0, _ = meta.get('origin', (0.0, 0.0, 0.0))
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
        
        self.observation_spaces = {
            aid: self._single_observation_space for aid in self.possible_agents
        }
        self.action_spaces = {
            aid: self._single_action_space for aid in self.possible_agents
        }
        # TODO: expose MARLlib env registration helpers (env_info, policy mapping) so trainers can auto-configure agents.
        
        
    def __del__(self):

        pass
    
    def observation_space(self, agent: str):
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        return self.action_spaces[agent]

    def _check_done(self):

        left_t = 2
        right_t = 2
        
        poses_x = np.array(self.poses_x) - self.start_xs
        poses_y = np.array(self.poses_y) - self.start_ys
        offsets = np.stack((poses_x, poses_y), axis=-1).astype(np.float32, copy=False)
        rotation = self.start_rot
        if rotation.ndim == 2:
            delta_pt = rotation @ offsets.T
        else:
            rotated = np.einsum("aij,aj->ai", rotation, offsets)
            delta_pt = rotated.T
        temp_y = delta_pt[1, :]
        temp_y = delta_pt[1,:]
        idx1 = temp_y > left_t
        idx2 = temp_y < -right_t
        temp_y[idx1] -= left_t
        temp_y[idx2] = -right_t - temp_y[idx2]
        temp_y[np.invert(np.logical_or(idx1, idx2))] = 0

        dist2 = delta_pt[0, :]**2 + temp_y**2
        closes = dist2 <= 0.1
        for i, agent in enumerate(self.possible_agents):
            if closes[i] and not self.near_starts[i]:
                self.near_starts[i] = True
                self.toggle_list[i] += 1
            elif not closes[i] and self.near_starts[i]:
                self.near_starts[i] = False
                self.toggle_list[i] += 1
            self.lap_counts[i] = self.toggle_list[i] // 2
            if self.toggle_list[i] < 4:
                self.lap_times[i] = self.current_time
        
        terminations = {}
        target_laps = self.target_laps
        for i, agent in enumerate(self.agents):
            terminations[agent] = bool(self.collisions[i]) or self.lap_counts[i] >= target_laps

        return terminations

    def _update_state(self, obs_dict):

        self.poses_x = obs_dict['poses_x']
        self.poses_y = obs_dict['poses_y']
        self.poses_theta = obs_dict['poses_theta']
        self.collisions = obs_dict['collisions']

   
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None: 
            self.seed = seed
            self.rng = np.random.default_rng(seed)
        self.agents = self.possible_agents.copy()
        self._elapsed_steps = 0
        poses = None
        if self.renderer is not None:
            self.renderer.reset_state()
    # Case 1: Explicit override via options
        if options is not None:
            if isinstance(options, dict) and "poses" in options:
                poses = np.array(options["poses"], dtype=np.float32)
        # Case 2: Default to config start_poses
        if poses is None and hasattr(self, "start_poses") and len(self.start_poses) > 0:
            poses = self.start_poses

        # options: (N,3) poses (x,y,theta). If None, caller must set internally.
        # poses = options if options is not None else np.zeros((self.n_agents, 3), dtype=np.float32)
        obs_joint = self.sim.reset(poses)
        obs = self._split_obs(obs_joint)
        self._update_state(obs_joint)
        self.render_obs = {}
        agent_index = self._agent_id_to_index
        for aid in self.agents:
            idx = agent_index[aid]
            self.render_obs[aid] = {
                **obs[aid],  # lidar, poses, velocities, collisions
                "lap_time":  float(self.lap_times[idx]),
                "lap_count": int(self.lap_counts[idx]),
            }

        infos = {aid: {} for aid in self.agents}
        return obs, infos

    def step(self, actions: Dict[str, np.ndarray]):

        joint = np.zeros((self.n_agents, 2), dtype=np.float32)
        agent_index = self._agent_id_to_index
        for aid in self.agents:
            if aid in actions:
                joint[agent_index[aid]] = np.asarray(actions[aid], dtype=np.float32)
                

        obs_joint = self.sim.step(joint)
        obs = self._split_obs(obs_joint)
        # TODO: emit a centralized state tensor alongside per-agent obs for MARLlib centralized training pipelines.
        self._update_state(obs_joint)
        self.render_obs = {}
        agent_index = self._agent_id_to_index
        for aid in self.agents:
            idx = agent_index[aid]
            self.render_obs[aid] = {
                **obs[aid],  # lidar, poses, velocities, collisions
                "lap_time":  float(self.lap_times[idx]),
                "lap_count": int(self.lap_counts[idx]),
            }

        self.current_time += self.timestep
        done_flags = self._check_done()
        # simple per-step reward (customize as needed)
        rewards = {aid: float(self.timestep) for aid in self.agents}

        # terminations/truncations
        terminations = {}
        collisions = obs_joint["collisions"]
        target_laps = self.target_laps
        terminate_on_collision = self.terminate_on_collision
        agent_index = self._agent_id_to_index
        for aid in self.possible_agents:
            idx = agent_index[aid]
            collided = bool(collisions[idx])
            laps_done = self.lap_counts[idx] >= target_laps
            extra_done = done_flags.get(aid, False)
            collision_done = collided and terminate_on_collision.get(aid, True)
            non_collision_done = laps_done or (extra_done and not collided)
            terminations[aid] = collision_done or non_collision_done


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

    def update_params(self, params, index=-1):

        self.sim.update_params(params, agent_idx=index)

    def add_render_callback(self, callback_func):
        self.render_callbacks.append(callback_func)

    def render(self):
        assert self.render_mode in ["human", "rgb_array"]

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
            )

        if self.render_obs:
            self.renderer.update_obs(self.render_obs)

        for render_callback in self.render_callbacks:
            render_callback(self.renderer)

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
    
    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    # helper: joint->per-agent dicts expected by PZ Parallel API
    def _split_obs(self, joint: Dict[str, np.ndarray]) -> Dict[str, Dict[str, np.ndarray]]:
        out: Dict[str, Dict[str, np.ndarray]] = {}
        for i, aid in enumerate(self.possible_agents):
            if i < joint["scans"].shape[0]:
                # normal agent data
                out[aid] = {
                    "scans":         joint["scans"][i].astype(np.float32),
                    "poses_x":       np.float32(joint["poses_x"][i]),
                    "poses_y":       np.float32(joint["poses_y"][i]),
                    "poses_theta":   np.float32(joint["poses_theta"][i]),
                    "linear_vels_x": np.float32(joint["linear_vels_x"][i]),
                    "linear_vels_y": np.float32(joint["linear_vels_y"][i]),
                    "ang_vels_z":    np.float32(joint["ang_vels_z"][i]),
                    "collisions":    np.int32(joint["collisions"][i]),
                }
            else:
                # fill dummy obs for agents not in sim output
                out[aid] = {
                    "scans":         np.zeros(1080, dtype=np.float32),
                    "poses_x":       0.0,
                    "poses_y":       0.0,
                    "poses_theta":   0.0,
                    "linear_vels_x": 0.0,
                    "linear_vels_y": 0.0,
                    "ang_vels_z":    0.0,
                    "collisions":    1,  # mark as crashed
                }
        # TODO: attach lap_counts, lap_times, and other scoreboard data promised in `observation_space`.
        return out
    
    def _zero_joint_actions(self) -> np.ndarray:
        """
        Returns a joint action array of shape (n_agents, action_dim)
        filled with zeros or neutral actions so that sim.step() produces
        valid observations without causing movement or collisions.
        """

        # determine dims, using current agents list
        n = len(self.possible_agents)
        # assuming actions are 2-dimensional: [steer, throttle]
        action_dim = 2

        joint = np.zeros((n, action_dim), dtype=np.float32)

        # If your action space has different structure or normalization, adapt here.
        # For example, if throttle neutral is 0.5, or steer neutral is mid-value.
        # E.g. if action[1] in [-1,1] means throttle, maybe use 0.
        # If your action_space sample() method gives a neutral, you can do:
        #   joint[i] = self.action_space(self.possible_agents[i]).sample() * 0
        #
        return joint
