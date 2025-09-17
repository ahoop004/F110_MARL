
from enum import Enum
import warnings

import numpy as np
from numba import njit

from f110x.physics.vehicle import RaceCar
from f110x.physics import collision_models
from f110x.physics.collision_models import get_vertices, collision_multiple


class Integrator(Enum):
    RK4 = "RK4"
    Euler = "Euler"


class Simulator(object):


    def __init__(self, params, num_agents, seed, time_step=0.01, integrator="RK4", lidar_dist=0.0):
        """
        Init function

        Args:
            params (dict): vehicle parameter dictionary, includes {'mu', 'C_Sf', 'C_Sr', 'lf', 'lr', 'h', 'm', 'I', 's_min', 's_max', 'sv_min', 'sv_max', 'v_switch', 'a_max', 'v_min', 'v_max', 'length', 'width'}
            num_agents (int): number of agents in the environment
            seed (int): seed of the rng in scan simulation
            time_step (float, default=0.01): physics time step

            lidar_dist (float, default=0): vertical distance between LiDAR and backshaft

        Returns:
            None
        """
        self.num_agents = num_agents
        self.seed = seed
        self.time_step = time_step

        self.params = params
        self.agent_poses = np.empty((self.num_agents, 3))
        self.agents = []
        self.collisions = np.zeros((self.num_agents, ))
        self.collision_idx = -1 * np.ones((self.num_agents, ))
        

        # initializing agents
        for i in range(self.num_agents):
            agent = RaceCar(params, self.seed, time_step=self.time_step, integrator=integrator, lidar_dist=lidar_dist)
            self.agents.append(agent)



    def set_map(self, map_path, map_ext):
        """
        Sets the map of the environment and sets the map for scan simulator of each agent

        Args:
            map_path (str): path to the map yaml file
            map_ext (str): extension for the map image file

        Returns:
            None
        """
        for agent in self.agents:
            agent.set_map(map_path, map_ext)


    def update_params(self, params, agent_idx=-1):
        """
        Updates the params of agents, if an index of an agent is given, update only that agent's params

        Args:
            params (dict): dictionary of params, see details in docstring of __init__
            agent_idx (int, default=-1): index for agent that needs param update, if negative, update all agents

        Returns:
            None
        """
        if agent_idx < 0:
            # update params for all
            for agent in self.agents:
                agent.update_params(params)
        elif agent_idx >= 0 and agent_idx < self.num_agents:
            # only update one agent's params
            self.agents[agent_idx].update_params(params)
        else:
            # index out of bounds, throw error
            raise IndexError('Index given is out of bounds for list of agents.')

    def check_collision(self):
        """
        Checks for collision between agents using GJK and agents' body vertices

        Args:
            None

        Returns:
            None
        """
        # get vertices of all agents
        all_vertices = np.empty((self.num_agents, 4, 2))
        for i in range(self.num_agents):
            all_vertices[i, :, :] = get_vertices(np.append(self.agents[i].state[0:2],self.agents[i].state[4]), self.params['length'], self.params['width'])
        self.collisions, self.collision_idx = collision_multiple(all_vertices)


    def step(self, control_inputs: np.ndarray) -> dict:
        """
        Advance all agents one physics step and return vectorized observations.

        Args:
            control_inputs: (N,2) array of (steer_cmd, vel_cmd) in ENV UNITS.

        Returns:
            obs_dict dict[str, np.ndarray] with shapes:
                scans:(N, num_beams), poses_x/y/theta:(N,), linear_vels_x/y:(N,), ang_vels_z:(N,), collisions:(N,)
            collisions are 0/1 for THIS step only (per-step semantics).
        """
        N = self.num_agents
        if not isinstance(control_inputs, np.ndarray) or control_inputs.shape != (N, 2):
            raise ValueError(f"control_inputs must have shape ({N}, 2)")

        # --- 0) per-step collision flags reset
        self.collisions[:] = 0
        self.collision_idx[:] = -1

        scans_list = []
        # store prev states for revert if needed
        prev_states = [agent.state.copy() for agent in self.agents]

        # --- 1) advance dynamics, collect scans + detect wall/environment collision (iTTC)
        for i, agent in enumerate(self.agents):
            current_scan = agent.update_pose(float(control_inputs[i, 0]), float(control_inputs[i, 1]))
            # agent.update_pose internally should call agent.check_ttc(...) to set agent.in_collision

            if getattr(agent, "in_collision", False):
                # revert penetration
                agent.state = prev_states[i]
                # zero velocities
                # assume state vector indices: 3 -> longitudinal vel, 5 -> angular vel
                agent.state[3] = 0.0
                agent.state[5] = 0.0
                self.collisions[i] = 1  # mark environment collision
                # TODO: re-run the scan for the reverted pose so lidar data stays consistent with the restored state.

            # ensure scan is freshest
            agent.scan = current_scan
            scans_list.append(current_scan)

            # update pose arrays
            self.agent_poses[i, 0] = agent.state[0]
            self.agent_poses[i, 1] = agent.state[1]
            self.agent_poses[i, 2] = agent.state[4]  # theta

        # --- 2) agent-agent collisions (GJK)
        verts = [get_vertices(self.agent_poses[i], self.params["length"], self.params["width"]) for i in range(N)]
        col_flags, hit_idx = collision_multiple(np.stack(verts, axis=0))
        # Merge agent-agent collision flags with environment collision flags
        col_flags = col_flags.astype(np.int8, copy=False)
        for i in range(N):
            if self.collisions[i] == 1 or col_flags[i] == 1:
                self.collisions[i] = 1
            # collision_idx logic: if agent-agent collision, set; if env collision, maybe set to some special id
            # TODO: distinguish environment collisions in `collision_idx` instead of leaving them at -1.
        self.collision_idx[:] = hit_idx.astype(np.int32, copy=False)

        # --- 3) opponent occlusions via LiDAR footprints, etc.
        for i, agent in enumerate(self.agents):
            # update opponents poses etc
            if hasattr(agent, "update_opp_poses"):
                if i == 0:
                    opp_poses = self.agent_poses[1:, :]
                elif i == N - 1:
                    opp_poses = self.agent_poses[:N-1, :]
                else:
                    opp_poses = np.concatenate((self.agent_poses[:i, :], self.agent_poses[i+1:, :]), axis=0)
                agent.update_opp_poses(opp_poses)

            if hasattr(agent, "ray_cast_agents"):
                opp_verts = np.stack([verts[j] for j in range(N) if j != i], axis=0)
                agent.ray_cast_agents(opp_verts)
                scans_list[i] = agent.scan  # refresh occluded scan

            # Already accounted for environment collision via agent.in_collision
            if getattr(agent, "in_collision", False):
                self.collisions[i] = 1

        # --- 4) prepare observation dict
        scans = np.stack(scans_list, axis=0).astype(np.float32, copy=False)
        poses_x = self.agent_poses[:, 0].astype(np.float32, copy=False)
        poses_y = self.agent_poses[:, 1].astype(np.float32, copy=False)
        poses_theta = self.agent_poses[:, 2].astype(np.float32, copy=False)
        linear_vels_x = np.array([a.state[3] for a in self.agents], dtype=np.float32)
        linear_vels_y = np.zeros((N,), dtype=np.float32)  # TODO: expose real lateral velocity instead of zeros.
        ang_vels_z = np.array([a.state[5] for a in self.agents], dtype=np.float32)

        obs_dict = {
            "scans": scans,
            "poses_x": poses_x,
            "poses_y": poses_y,
            "poses_theta": poses_theta,
            "linear_vels_x": linear_vels_x,
            "linear_vels_y": linear_vels_y,
            "ang_vels_z": ang_vels_z,
            "collisions": self.collisions.copy(),  # per-step 0/1
        }
        return obs_dict


    def reset(self, poses: np.ndarray) -> dict:
        """
        Reset the simulator to the given poses and return initial vectorized observations.

        Args:
            poses: np.ndarray of shape (N, 3) with (x, y, theta) per agent.

        Returns:
            obs_dict with scans, poses, velocities, collisions.
        """
        if not isinstance(poses, np.ndarray) or poses.ndim != 2 or poses.shape[1] != 3:
            raise ValueError(f"poses must be an array with shape (N,3); got {getattr(poses, 'shape', None)}")
        if poses.shape[0] != self.num_agents:
            raise ValueError("Number of poses for reset does not match number of agents.")

        N = self.num_agents

        # Reset agent states
        for i in range(N):
            self.agents[i].reset(poses[i])
        self.agent_poses[:, :] = poses.astype(np.float32, copy=False)

        # Clear collisions
        self.collisions[:] = 0
        self.collision_idx[:] = -1

        # Compute scans
        scans_list = []
        for i in range(N):
            self.agents[i].update_scan(scans_list, i)
            scans_list.append(self.agents[i].scan)

        # Opponent occlusion
        verts = [get_vertices(self.agent_poses[i], self.params["length"], self.params["width"]) for i in range(N)]
        for i in range(N):
            opp_verts = np.stack([verts[j] for j in range(N) if j != i], axis=0)
            self.agents[i].opp_poses = opp_verts   # <-- ensure not None
            if hasattr(self.agents[i], "ray_cast_agents"):
                self.agents[i].ray_cast_agents(opp_verts)
                scans_list[i] = self.agents[i].scan

        # Collisions
        col_flags, hit_idx = collision_multiple(np.stack(verts, axis=0))
        self.collisions[:] = col_flags.astype(np.int8, copy=False)
        self.collision_idx[:] = hit_idx.astype(np.int32, copy=False)

        # Vectorize outputs
        poses_x       = poses[:, 0].astype(np.float32, copy=False)
        poses_y       = poses[:, 1].astype(np.float32, copy=False)
        poses_theta   = poses[:, 2].astype(np.float32, copy=False)
        linear_vels_x = np.array([getattr(a, "v_long", 0.0) for a in self.agents], dtype=np.float32)
        linear_vels_y = np.array([getattr(a, "v_lat",  0.0) for a in self.agents], dtype=np.float32)
        ang_vels_z    = np.array([getattr(a, "yaw_rate", 0.0) for a in self.agents], dtype=np.float32)

        scans = np.stack(scans_list, axis=0).astype(np.float32, copy=False)

        obs_dict = {
            "scans":         scans,
            "poses_x":       poses_x,
            "poses_y":       poses_y,
            "poses_theta":   poses_theta,
            "linear_vels_x": linear_vels_x,
            "linear_vels_y": linear_vels_y,
            "ang_vels_z":    ang_vels_z,
            "collisions":    self.collisions.copy(),
        }
        return obs_dict
