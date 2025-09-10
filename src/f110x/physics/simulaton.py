
from enum import Enum
import warnings

import numpy as np
from numba import njit

from .vehicle import RaceCar
from .collision_models import get_vertices, collision_multiple


class Integrator(Enum):
    RK4 = 1
    Euler = 2


class Simulator(object):
    """
    Simulator class, handles the interaction and update of all vehicles in the environment

    Data Members:
        num_agents (int): number of agents in the environment
        time_step (float): physics time step
        agent_poses (np.ndarray(num_agents, 3)): all poses of all agents
        agents (list[RaceCar]): container for RaceCar objects
        collisions (np.ndarray(num_agents, )): array of collision indicator for each agent
        collision_idx (np.ndarray(num_agents, )): which agent is each agent in collision with

    """

    def __init__(self, params, num_agents, seed, time_step=0.01, ego_idx=0, integrator=Integrator.RK4, lidar_dist=0.0):
        """
        Init function

        Args:
            params (dict): vehicle parameter dictionary, includes {'mu', 'C_Sf', 'C_Sr', 'lf', 'lr', 'h', 'm', 'I', 's_min', 's_max', 'sv_min', 'sv_max', 'v_switch', 'a_max', 'v_min', 'v_max', 'length', 'width'}
            num_agents (int): number of agents in the environment
            seed (int): seed of the rng in scan simulation
            time_step (float, default=0.01): physics time step
            ego_idx (int, default=0): ego vehicle's index in list of agents
            lidar_dist (float, default=0): vertical distance between LiDAR and backshaft

        Returns:
            None
        """
        self.num_agents = num_agents
        self.seed = seed
        self.time_step = time_step
        self.ego_idx = ego_idx
        self.params = params
        self.agent_poses = np.empty((self.num_agents, 3))
        self.agents = []
        self.collisions = np.zeros((self.num_agents, ))
        self.collision_idx = -1 * np.ones((self.num_agents, ))

        # initializing agents
        for i in range(self.num_agents):
            if i == ego_idx:
                ego_car = RaceCar(params, self.seed, is_ego=True, time_step=self.time_step, integrator=integrator, lidar_dist=lidar_dist)
                self.agents.append(ego_car)
            else:
                agent = RaceCar(params, self.seed, is_ego=False, time_step=self.time_step, integrator=integrator, lidar_dist=lidar_dist)
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


    def step(self, control_inputs):
        """
        Steps the simulation environment

        Args:
            control_inputs (np.ndarray (num_agents, 2)): control inputs of all agents, first column is desired steering angle, second column is desired velocity
        
        Returns:
            observations (dict): dictionary for observations: poses of agents, current laser scan of each agent, collision indicators, etc.
        """


        agent_scans = []

        # looping over agents
        for i, agent in enumerate(self.agents):
            # update each agent's pose
            current_scan = agent.update_pose(control_inputs[i, 0], control_inputs[i, 1])
            agent_scans.append(current_scan)

            # update sim's information of agent poses
            self.agent_poses[i, :] = np.append(agent.state[0:2], agent.state[4])

        # check collisions between all agents
        self.check_collision()

        for i, agent in enumerate(self.agents):
            # update agent's information on other agents
            opp_poses = np.concatenate((self.agent_poses[0:i, :], self.agent_poses[i+1:, :]), axis=0)
            agent.update_opp_poses(opp_poses)

            # update each agent's current scan based on other agents
            agent.update_scan(agent_scans, i)

            # update agent collision with environment
            if agent.in_collision:
                self.collisions[i] = 1.

        # fill in observations
        # state is [x, y, steer_angle, vel, yaw_angle, yaw_rate, slip_angle]
        # collision_angles is removed from observations
        observations = {'ego_idx': self.ego_idx,
            'scans': [],
            'poses_x': [],
            'poses_y': [],
            'poses_theta': [],
            'linear_vels_x': [],
            'linear_vels_y': [],
            'ang_vels_z': [],
            'collisions': self.collisions}
        for i, agent in enumerate(self.agents):
            observations['scans'].append(agent_scans[i])
            observations['poses_x'].append(agent.state[0])
            observations['poses_y'].append(agent.state[1])
            observations['poses_theta'].append(agent.state[4])
            observations['linear_vels_x'].append(agent.state[3])
            observations['linear_vels_y'].append(0.)
            observations['ang_vels_z'].append(agent.state[5])

        return observations

    def reset(self, poses):
        """
        Resets the simulation environment by given poses

        Arges:
            poses (np.ndarray (num_agents, 3)): poses to reset agents to

        Returns:
            None
        """
        
        if poses.shape[0] != self.num_agents:
            raise ValueError('Number of poses for reset does not match number of agents.')

        # loop over poses to reset
        for i in range(self.num_agents):
            self.agents[i].reset(poses[i, :])
