from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List
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
      
        env_config = kwargs.get("env", {})
        merged = {**env_config, **kwargs}
        
        self.render_mode = merged.get("render_mode", "human")
        self.metadata = {"render_modes": ["human", "rgb_array"], "name": "F110ParallelEnv"}
        self.renderer: Optional[EnvRenderer] = None
        headless_env = str(os.environ.get("PYGLET_HEADLESS", "")).lower()
        self._headless = pyglet.options.get("headless", False) or headless_env in {"1", "true", "yes", "on"}
        mode = (self.render_mode or "").lower()
        self._collect_render_data = mode == "rgb_array" or (mode == "human" and not self._headless)

        self.seed: int = int(merged.get("seed", 42))
        self.max_steps: int = int(merged.get("max_steps", 5000))
        self.n_agents: int = int(merged.get("n_agents", 2))
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
   
        self.timestep: float = float(merged.get("timestep", 0.01))
        integrator_cfg = merged.get("integrator", Integrator.RK4)
        if isinstance(integrator_cfg, Integrator):
            integrator_name = integrator_cfg.value
        else:
            integrator_name = str(integrator_cfg)
        integrator_name = integrator_name.strip()
        if integrator_name.lower() == "rk4":
            integrator_name = "RK4"
        elif integrator_name.lower() == "euler":
            integrator_name = "Euler"
        else:
            integrator_name = "RK4"
        self.integrator = integrator_name

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

        self._build_observation_spaces(x_min, x_max, y_min, y_max)

        # stateful observations for rendering
        self.render_obs = None

        self._single_action_space = spaces.Box(
            low=np.array([self.params["s_min"], self.params["v_min"]], dtype=np.float32),
            high=np.array([self.params["s_max"], self.params["v_max"]], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_spaces = {
            aid: self._single_action_space for aid in self.possible_agents
        }

    def action_space(self, agent: str):
        return self.action_spaces[agent]

    def _update_state(self, obs_dict):
        self.state_buffers.update(obs_dict)

   
    def _update_start_from_poses(self, poses: np.ndarray):
        if poses is None or poses.size == 0:
            return
        self.start_poses = np.asarray(poses, dtype=np.float32)
        self.start_state.apply_start_poses(self.start_poses)
        self.start_state.reset()

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None: 
            self.seed = seed
            self.rng = np.random.default_rng(seed)
        self.agents = self.possible_agents.copy()
        self._elapsed_steps = 0
        self.current_time = 0.0
        self.start_state.reset()
        self.state_buffers.reset()
        poses = None
        if self.renderer is not None:
            self.renderer.reset_state()
    # Case 1: Explicit override via options
        if options is not None:
            if isinstance(options, dict) and "poses" in options:
                poses = np.array(options["poses"], dtype=np.float32)
                if poses.ndim == 1:
                    poses = np.expand_dims(poses, axis=0)
                self._update_start_from_poses(poses)
        # Case 2: Default to config start_poses
        if poses is None and hasattr(self, "start_poses") and len(self.start_poses) > 0:
            poses = self.start_poses

        # options: (N,3) poses (x,y,theta). If None, caller must set internally.
        # poses = options if options is not None else np.zeros((self.n_agents, 3), dtype=np.float32)
        obs_joint = self.sim.reset(poses)
        obs = self._split_obs(obs_joint)
        self._attach_central_state(obs, obs_joint)
        self._update_state(obs_joint)
        if self._collect_render_data:
            self.render_obs = {}
            agent_index = self._agent_id_to_index
            for aid in self.agents:
                idx = agent_index[aid]
                render_entry = {
                    "poses_x": float(self.poses_x[idx]),
                    "poses_y": float(self.poses_y[idx]),
                    "poses_theta": float(self.poses_theta[idx]),
                    "pose": np.array([
                        float(self.poses_x[idx]),
                        float(self.poses_y[idx]),
                        float(self.poses_theta[idx])
                    ], dtype=np.float32),
                    "lap_time":  float(self.lap_times[idx]),
                    "lap_count": int(self.lap_counts[idx]),
                    "collision": bool(self.collisions[idx])
                }
                scan_entry = obs[aid].get("scans")
                if scan_entry is not None:
                    render_entry["scans"] = scan_entry
                self.render_obs[aid] = render_entry
        else:
            self.render_obs = {}

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
        self._attach_central_state(obs, obs_joint)
        self._update_state(obs_joint)
        if self._collect_render_data:
            self.render_obs = {}
            agent_index = self._agent_id_to_index
            for aid in self.agents:
                idx = agent_index[aid]
                render_entry = {
                    "poses_x": float(self.poses_x[idx]),
                    "poses_y": float(self.poses_y[idx]),
                    "poses_theta": float(self.poses_theta[idx]),
                    "pose": np.array([
                        float(self.poses_x[idx]),
                        float(self.poses_y[idx]),
                        float(self.poses_theta[idx])
                    ], dtype=np.float32),
                    "lap_time":  float(self.lap_times[idx]),
                    "lap_count": int(self.lap_counts[idx]),
                    "collision": bool(self.collisions[idx])
                }
                scan_entry = obs[aid].get("scans")
                if scan_entry is not None:
                    render_entry["scans"] = scan_entry
                self.render_obs[aid] = render_entry
        else:
            self.render_obs = {}

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
            )

        if self.render_obs:
            self.renderer.update_obs(self.render_obs)

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
    
