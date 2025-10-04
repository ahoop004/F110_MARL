
import warnings

import numpy as np
from numba import njit

from f110x.physics.integration import Integrator
from f110x.physics.vehicle import RaceCar
from f110x.physics.collision_models import get_vertices, collision_multiple


ENV_COLLISION_IDX = -2


@njit(cache=True)
def _merge_collision_results(
    agent_flags: np.ndarray,
    agent_indices: np.ndarray,
    env_mask: np.ndarray,
    out_collisions: np.ndarray,
    out_indices: np.ndarray,
    env_only_idx: int,
):
    count = out_collisions.shape[0]
    for i in range(count):
        env_hit = env_mask[i]
        agent_hit = agent_flags[i] != 0.0
        if agent_hit:
            out_collisions[i] = 1.0
            out_indices[i] = int(agent_indices[i])
        elif env_hit:
            out_collisions[i] = 1.0
            out_indices[i] = env_only_idx
        else:
            out_collisions[i] = 0.0
            out_indices[i] = -1


class Simulator(object):


    def __init__(
        self,
        params,
        num_agents,
        seed,
        time_step=0.01,
        *,
        integrator="RK4",
        lidar_dist=0.0,
        num_beams=1080,
    ):
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
        self.num_beams = int(num_beams)
        if isinstance(integrator, Integrator):
            integrator = integrator.value
        self.integrator = str(integrator)

        self.params = params
        self.agent_poses = np.empty((self.num_agents, 3), dtype=np.float32)
        self.agents = []
        self.collisions = np.zeros((self.num_agents,), dtype=np.float32)
        self.collision_idx = -1 * np.ones((self.num_agents,), dtype=np.int32)
        

        # initializing agents
        for i in range(self.num_agents):
            agent = RaceCar(
                params,
                self.seed,
                time_step=self.time_step,
                num_beams=self.num_beams,
                integrator=self.integrator,
                lidar_dist=lidar_dist,
            )
            self.agents.append(agent)

        self._state_dim = self.agents[0].state.shape[0] if self.agents else 0
        self._state_buffer = np.zeros((self.num_agents, self._state_dim), dtype=np.float64)
        self._pose_buffer = np.zeros((self.num_agents, 3), dtype=np.float64)
        self._verts_buffer_f64 = np.zeros((self.num_agents, 4, 2), dtype=np.float64)
        self._verts_buffer_f32 = np.zeros((self.num_agents, 4, 2), dtype=np.float32)
        self._scan_buffer = np.zeros((self.num_agents, self.num_beams), dtype=np.float32)
        self._env_collision_mask = np.zeros((self.num_agents,), dtype=np.bool_)
        opp_count = max(self.num_agents - 1, 1)
        self._opp_pose_buffer = np.zeros((opp_count, 3), dtype=np.float32)
        self._linear_vels_x = np.zeros((self.num_agents,), dtype=np.float32)
        self._linear_vels_y = np.zeros((self.num_agents,), dtype=np.float32)
        self._ang_vels_z = np.zeros((self.num_agents,), dtype=np.float32)



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

    def _ensure_scan_capacity(self, beam_count: int) -> None:
        if beam_count <= 0:
            beam_count = 1
        if self._scan_buffer.shape[1] == beam_count:
            return
        self._scan_buffer = np.zeros((self.num_agents, beam_count), dtype=np.float32)


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
        for i, agent in enumerate(self.agents):
            self._pose_buffer[i, 0] = agent.state[0]
            self._pose_buffer[i, 1] = agent.state[1]
            self._pose_buffer[i, 2] = agent.state[4]
            verts = get_vertices(
                self._pose_buffer[i],
                self.params['length'],
                self.params['width'],
            )
            self._verts_buffer_f64[i, :, :] = verts
            self._verts_buffer_f32[i, :, :] = verts

        col_flags, hit_idx = collision_multiple(self._verts_buffer_f64)
        np.copyto(self.collisions, np.asarray(col_flags, dtype=np.float32))
        np.copyto(self.collision_idx, np.asarray(hit_idx, dtype=np.int32))


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

        self.collisions.fill(0.0)
        self.collision_idx.fill(-1)
        env_collision_mask = self._env_collision_mask
        env_collision_mask.fill(False)

        self._ensure_scan_capacity(self.num_beams)

        state_buffer = self._state_buffer
        pose_buffer = self._pose_buffer

        for i, agent in enumerate(self.agents):
            np.copyto(state_buffer[i], agent.state)
            agent.update_pose(float(control_inputs[i, 0]), float(control_inputs[i, 1]))

            if getattr(agent, "in_collision", False):
                np.copyto(agent.state, state_buffer[i])
                agent.state[3] = 0.0
                agent.state[5] = 0.0
                agent.compute_scan()
                env_collision_mask[i] = True

            pose_buffer[i, 0] = agent.state[0]
            pose_buffer[i, 1] = agent.state[1]
            pose_buffer[i, 2] = agent.state[4]

            self.agent_poses[i, 0] = np.float32(agent.state[0])
            self.agent_poses[i, 1] = np.float32(agent.state[1])
            self.agent_poses[i, 2] = np.float32(agent.state[4])

            self._linear_vels_x[i] = np.float32(agent.state[3])
            self._linear_vels_y[i] = np.float32(agent.state[3] * np.sin(agent.state[6]))
            self._ang_vels_z[i] = np.float32(agent.state[5])

            scan_row = np.asarray(agent.scan, dtype=np.float32)
            if scan_row.shape[0] != self._scan_buffer.shape[1]:
                self._ensure_scan_capacity(scan_row.shape[0])
                for j in range(i):
                    prev = np.asarray(self.agents[j].scan, dtype=np.float32)
                    np.copyto(self._scan_buffer[j], prev)
                self.num_beams = self._scan_buffer.shape[1]
            np.copyto(self._scan_buffer[i], scan_row)

        # --- 2) agent-agent collisions (GJK)
        for i in range(N):
            verts = get_vertices(
                pose_buffer[i],
                self.params["length"],
                self.params["width"],
            )
            self._verts_buffer_f64[i, :, :] = verts
            self._verts_buffer_f32[i, :, :] = verts
        col_flags, hit_idx = collision_multiple(self._verts_buffer_f64)

        # --- 3) opponent occlusions via LiDAR footprints, etc.
        if N > 1:
            for i, agent in enumerate(self.agents):
                if hasattr(agent, "update_opp_poses"):
                    count = 0
                    for j in range(N):
                        if j == i:
                            continue
                        self._opp_pose_buffer[count, 0] = self.agent_poses[j, 0]
                        self._opp_pose_buffer[count, 1] = self.agent_poses[j, 1]
                        self._opp_pose_buffer[count, 2] = self.agent_poses[j, 2]
                        count += 1
                    agent.update_opp_poses(self._opp_pose_buffer[:count])

                if hasattr(agent, "ray_cast_agents"):
                    agent.ray_cast_agents(self._verts_buffer_f32, i)
                    scan_row = np.asarray(agent.scan, dtype=np.float32)
                    if scan_row.shape[0] != self._scan_buffer.shape[1]:
                        self._ensure_scan_capacity(scan_row.shape[0])
                        for j in range(N):
                            prev = np.asarray(self.agents[j].scan, dtype=np.float32)
                            np.copyto(self._scan_buffer[j], prev)
                        self.num_beams = self._scan_buffer.shape[1]
                    np.copyto(self._scan_buffer[i], scan_row)

                if getattr(agent, "in_collision", False):
                    env_collision_mask[i] = True

        _merge_collision_results(
            np.asarray(col_flags, dtype=np.float64),
            np.asarray(hit_idx, dtype=np.float64),
            env_collision_mask,
            self.collisions,
            self.collision_idx,
            ENV_COLLISION_IDX,
        )

        obs_dict = {
            "scans": self._scan_buffer.copy(),
            "poses_x": self.agent_poses[:, 0].copy(),
            "poses_y": self.agent_poses[:, 1].copy(),
            "poses_theta": self.agent_poses[:, 2].copy(),
            "linear_vels_x": self._linear_vels_x.copy(),
            "linear_vels_y": self._linear_vels_y.copy(),
            "ang_vels_z": self._ang_vels_z.copy(),
            "collisions": self.collisions.copy(),
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

        self.collisions.fill(0.0)
        self.collision_idx.fill(-1)
        env_collision_mask = self._env_collision_mask
        env_collision_mask.fill(False)

        self._ensure_scan_capacity(self.num_beams)

        pose_buffer = self._pose_buffer

        for i, agent in enumerate(self.agents):
            agent.reset(poses[i])

            pose_buffer[i, 0] = agent.state[0]
            pose_buffer[i, 1] = agent.state[1]
            pose_buffer[i, 2] = agent.state[4]

            self.agent_poses[i, 0] = np.float32(agent.state[0])
            self.agent_poses[i, 1] = np.float32(agent.state[1])
            self.agent_poses[i, 2] = np.float32(agent.state[4])

            v_long = float(getattr(agent, "v_long", agent.state[3]))
            v_lat = float(getattr(agent, "v_lat", agent.state[3] * np.sin(agent.state[6])))
            yaw_rate = float(getattr(agent, "yaw_rate", agent.state[5]))

            self._linear_vels_x[i] = np.float32(v_long)
            self._linear_vels_y[i] = np.float32(v_lat)
            self._ang_vels_z[i] = np.float32(yaw_rate)

            scan_row = np.asarray(agent.compute_scan(), dtype=np.float32)
            if scan_row.shape[0] != self._scan_buffer.shape[1]:
                self._ensure_scan_capacity(scan_row.shape[0])
                self.num_beams = self._scan_buffer.shape[1]
            np.copyto(self._scan_buffer[i], scan_row)

        for i in range(N):
            verts = get_vertices(
                pose_buffer[i],
                self.params["length"],
                self.params["width"],
            )
            self._verts_buffer_f64[i, :, :] = verts
            self._verts_buffer_f32[i, :, :] = verts
        col_flags, hit_idx = collision_multiple(self._verts_buffer_f64)

        if N > 1:
            for i, agent in enumerate(self.agents):
                if hasattr(agent, "update_opp_poses"):
                    count = 0
                    for j in range(N):
                        if j == i:
                            continue
                        self._opp_pose_buffer[count, 0] = self.agent_poses[j, 0]
                        self._opp_pose_buffer[count, 1] = self.agent_poses[j, 1]
                        self._opp_pose_buffer[count, 2] = self.agent_poses[j, 2]
                        count += 1
                    agent.update_opp_poses(self._opp_pose_buffer[:count])

                if hasattr(agent, "ray_cast_agents"):
                    agent.ray_cast_agents(self._verts_buffer_f32, i)
                    scan_row = np.asarray(agent.scan, dtype=np.float32)
                    if scan_row.shape[0] != self._scan_buffer.shape[1]:
                        self._ensure_scan_capacity(scan_row.shape[0])
                        self.num_beams = self._scan_buffer.shape[1]
                    np.copyto(self._scan_buffer[i], scan_row)

        _merge_collision_results(
            np.asarray(col_flags, dtype=np.float64),
            np.asarray(hit_idx, dtype=np.float64),
            env_collision_mask,
            self.collisions,
            self.collision_idx,
            ENV_COLLISION_IDX,
        )

        obs_dict = {
            "scans": self._scan_buffer.copy(),
            "poses_x": self.agent_poses[:, 0].copy(),
            "poses_y": self.agent_poses[:, 1].copy(),
            "poses_theta": self.agent_poses[:, 2].copy(),
            "linear_vels_x": self._linear_vels_x.copy(),
            "linear_vels_y": self._linear_vels_y.copy(),
            "ang_vels_z": self._ang_vels_z.copy(),
            "collisions": self.collisions.copy(),
        }
        return obs_dict
