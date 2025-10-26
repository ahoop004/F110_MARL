import warnings

import numpy as np
from numba import njit

from f110x.physics.dynamic_models import vehicle_dynamics_st, pid
from f110x.physics.integration import Integrator
from f110x.physics.laser_models import ScanSimulator2D, check_ttc_jit, ray_cast
from f110x.physics.collision_models import get_vertices


class RaceCar(object):
    """
    Base level race car class, handles the physics and laser scan of a single vehicle

    Data Members:
        params (dict): vehicle parameters dictionary

        time_step (float): physics timestep
        num_beams (int): number of beams in laser
        fov (float): field of view of laser
        state (np.ndarray (7, )): state vector [x, y, theta, vel, steer_angle, ang_vel, slip_angle]
        odom (np.ndarray(13, )): odometry vector [x, y, z, qx, qy, qz, qw, linear_x, linear_y, linear_z, angular_x, angular_y, angular_z]
        accel (float): current acceleration input
        steer_angle_vel (float): current steering velocity input
        in_collision (bool): collision indicator

    """

    # static objects that don't need to be stored in class instances
    scan_simulator = None
    cosines = None
    scan_angles = None
    side_distances = None

    def __init__(self, params, seed, time_step=0.01, num_beams=1080, fov=4.7, integrator=Integrator.RK4, lidar_dist=0.0):
        """
        Init function

        Args:
            params (dict): vehicle parameter dictionary, includes {'mu', 'C_Sf', 'C_Sr', 'lf', 'lr', 'h', 'm', 'I', 's_min', 's_max', 'sv_min', 'sv_max', 'v_switch', 'a_max': 9.51, 'v_min', 'v_max', 'length', 'width'}

            time_step (float, default=0.01): physics sim time step
            num_beams (int, default=1080): number of beams in the laser scan
            fov (float, default=4.7): field of view of the laser
            lidar_dist (float, default=0): vertical distance between LiDAR and backshaft

        Returns:
            None
        """

        # initialization
        self.params = params
        self.seed = seed
        self.opp_poses = []
        self.lidar_range= 30.0

        self.time_step = time_step
        self.num_beams = int(num_beams)
        self.fov = fov
        self.integrator = integrator
        self.lidar_dist = lidar_dist
        if self.integrator is Integrator.RK4:
            warnings.warn(f"Chosen integrator is RK4. This is different from previous versions of the gym.")

        # state is [x, y, steer_angle, vel, yaw_angle, yaw_rate, slip_angle]
        self.state = np.zeros((7, ))
        self._state_backup = np.zeros_like(self.state)

        # pose of opponents in the world
        self.opp_poses = None

        # control inputs
        self.accel = 0.0
        self.steer_angle_vel = 0.0

        # steering delay buffer
        self.steer_buffer_size = 2
        self._steer_buf = np.zeros(self.steer_buffer_size, dtype=np.float32)
        self._sb_head = 0  # next write index

        # reusable workspaces to reduce per-step allocations
        self._control_vec = np.zeros((2,), dtype=np.float64)
        self._rk_state1 = np.zeros_like(self.state, dtype=np.float64)
        self._rk_state2 = np.zeros_like(self.state, dtype=np.float64)
        self._rk_state3 = np.zeros_like(self.state, dtype=np.float64)
        self._rk_delta = np.zeros_like(self.state, dtype=np.float64)
        self._rk_accum = np.zeros_like(self.state, dtype=np.float64)
        self._scan_pose = np.zeros((3,), dtype=np.float64)

        # collision identifier
        self.in_collision = False

        # collision threshold for iTTC to environment
        self.ttc_thresh = 0.005

        self._refresh_param_cache()

        regenerate_scan = (
            RaceCar.scan_simulator is None
            or getattr(RaceCar, "_scan_beam_count", None) != self.num_beams
            or getattr(RaceCar, "_scan_fov", None) != fov
        )

        # initialize scan sim
        if regenerate_scan:
            self.scan_rng = np.random.default_rng(seed=self.seed)
            RaceCar.scan_simulator = ScanSimulator2D(self.num_beams, fov)

            scan_ang_incr = RaceCar.scan_simulator.get_increment()

            # angles of each scan beam, distance from lidar to edge of car at each beam, and precomputed cosines of each angle
            RaceCar.cosines = np.zeros((self.num_beams, ))
            RaceCar.scan_angles = np.zeros((self.num_beams, ))
            RaceCar.side_distances = np.zeros((self.num_beams, ))

            dist_sides = params['width']/2.
            dist_fr = (params['lf']+params['lr'])/2.

            for i in range(self.num_beams):
                angle = -fov/2. + i*scan_ang_incr
                RaceCar.scan_angles[i] = angle
                RaceCar.cosines[i] = np.cos(angle)

                if angle > 0:
                    if angle < np.pi/2:
                        # between 0 and pi/2
                        to_side = dist_sides / np.sin(angle)
                        to_fr = dist_fr / np.cos(angle)
                        RaceCar.side_distances[i] = min(to_side, to_fr)
                    else:
                        # between pi/2 and pi
                        to_side = dist_sides / np.cos(angle - np.pi/2.)
                        to_fr = dist_fr / np.sin(angle - np.pi/2.)
                        RaceCar.side_distances[i] = min(to_side, to_fr)
                else:
                    if angle > -np.pi/2:
                        # between 0 and -pi/2
                        to_side = dist_sides / np.sin(-angle)
                        to_fr = dist_fr / np.cos(-angle)
                        RaceCar.side_distances[i] = min(to_side, to_fr)
                    else:
                        # between -pi/2 and -pi
                        to_side = dist_sides / np.cos(-angle - np.pi/2)
                        to_fr = dist_fr / np.sin(-angle - np.pi/2)
                        RaceCar.side_distances[i] = min(to_side, to_fr)

            RaceCar._scan_beam_count = self.num_beams
            RaceCar._scan_fov = fov
        else:
            self.scan_rng = np.random.default_rng(seed=self.seed)

    def _refresh_param_cache(self) -> None:
        params = self.params
        self._mu = float(params.get('mu', 1.0489))
        self._C_Sf = float(params.get('C_Sf', 4.718))
        self._C_Sr = float(params.get('C_Sr', 5.4562))
        self._lf = float(params.get('lf', 0.15875))
        self._lr = float(params.get('lr', 0.17145))
        self._h = float(params.get('h', 0.074))
        self._m = float(params.get('m', 3.74))
        self._I = float(params.get('I', 0.04712))
        self._s_min = float(params.get('s_min', -0.4189))
        self._s_max = float(params.get('s_max', 0.4189))
        self._sv_min = float(params.get('sv_min', -3.2))
        self._sv_max = float(params.get('sv_max', 3.2))
        self._v_switch = float(params.get('v_switch', 7.319))
        self._a_max = float(params.get('a_max', 9.51))
        self._v_min = float(params.get('v_min', -5.0))
        self._v_max = float(params.get('v_max', 20.0))

        self._dyn_params = (
            self._mu,
            self._C_Sf,
            self._C_Sr,
            self._lf,
            self._lr,
            self._h,
            self._m,
            self._I,
            self._s_min,
            self._s_max,
            self._sv_min,
            self._sv_max,
            self._v_switch,
            self._a_max,
            self._v_min,
            self._v_max,
        )

        self._steer_min = self._s_min
        self._steer_max = self._s_max

    def update_params(self, params):
        """
        Updates the physical parameters of the vehicle
        Note that does not need to be called at initialization of class anymore

        Args:
            params (dict): new parameters for the vehicle

        Returns:
            None
        """
        self.params = params
        self._refresh_param_cache()

    def set_seed(self, seed):
        """
        Update the RNG seed used for LiDAR noise and reset behaviour.

        Args:
            seed (int): New seed value.
        """
        self.seed = int(seed)
        self.scan_rng = np.random.default_rng(seed=self.seed)
    
    def set_map(self, map_path: str, map_ext: str):
        """
        Configure the shared scan simulator only if the map asset changed.

        Args:
            map_path (str): absolute path to the map yaml file
            map_ext (str): extension of the map image file
        """

        cache_key = (str(map_path), str(map_ext))
        cached_key = getattr(RaceCar, "_map_cache_key", None)
        if cached_key == cache_key:
            return

        RaceCar.scan_simulator.set_map(map_path, map_ext)
        RaceCar._map_cache_key = cache_key

    def reset(self, pose):
        """
        Resets the vehicle to a pose
        
        Args:
            pose (np.ndarray (3, )): pose to reset the vehicle to

        Returns:
            None
        """
        # clear control inputs
        self.accel = 0.0
        self.steer_angle_vel = 0.0
        # clear collision indicator
        self.in_collision = False
        # clear state
        self.state = np.zeros((7, ))
        self.state[0:2] = pose[0:2]
        self.state[4] = pose[2]
        self.steer_buffer = np.empty((0, ))
        self._steer_buf.fill(0.0)
        self._sb_head = 0
        # reset scan random generator
        self.scan_rng = np.random.default_rng(seed=self.seed)

    def set_longitudinal_speed(self, speed: float) -> None:
        """Directly assign the vehicle's forward speed without touching pose."""
        try:
            value = float(speed)
        except (TypeError, ValueError):
            value = 0.0
        self.state[3] = value
        # Synchronise auxiliary attributes if the model exposes them
        setattr(self, "v_long", value)

    def ray_cast_agents(self, all_verts: np.ndarray, agent_index: int):
        """Modify scan by accounting for other agents' hulls.

        Args:
            all_verts: (N, 4, 2) contiguous vertex buffer for every agent.
            agent_index: Index of the current agent inside ``all_verts``.
        """
        if all_verts is None or all_verts.shape[0] <= 1:
            return

        scan_pose = np.array([
            self.state[0] + self.lidar_dist * np.cos(self.state[4]),
            self.state[1] + self.lidar_dist * np.sin(self.state[4]),
            self.state[4]
        ], dtype=np.float64)

        scan_view = np.asarray(self.scan, dtype=np.float32)

        total = all_verts.shape[0]
        for idx in range(total):
            if idx == agent_index:
                continue
            verts = np.asarray(all_verts[idx], dtype=np.float32)
            ray_cast(scan_pose, scan_view, RaceCar.scan_angles, verts)

        self.scan = scan_view

    def compute_scan(self) -> np.ndarray:
        """Recompute the LiDAR scan for the vehicle's current pose."""

        scan_pose = self._scan_pose
        scan_pose[0] = self.state[0] + self.lidar_dist * np.cos(self.state[4])
        scan_pose[1] = self.state[1] + self.lidar_dist * np.sin(self.state[4])
        scan_pose[2] = self.state[4]

        current_scan = RaceCar.scan_simulator.scan(scan_pose, self.scan_rng)
        if current_scan is None:
            self.scan = np.zeros((self.num_beams,), dtype=np.float32)
            self.in_collision = False
            return self.scan

        self.check_ttc(current_scan)
        self.scan = np.asarray(current_scan, dtype=np.float32)
        return self.scan

    def check_ttc(self, current_scan):
        if current_scan is None or not isinstance(current_scan, np.ndarray) or current_scan.size == 0:
            self.in_collision = False
            return
        try:
            in_collision = check_ttc_jit(
                current_scan, self.state[3],
                self.scan_angles, self.cosines,
                self.side_distances, self.ttc_thresh
            )
            self.in_collision = bool(in_collision)
        except Exception as e:
            print(f"[WARN] TTC check failed: {e}")
            self.in_collision = False

    def update_pose(self, raw_steer, vel):
        """
        Steps the vehicle's physical simulation

        Args:
            steer (float): desired steering angle
            vel (float): desired longitudinal velocity

        Returns:
            current_scan
        """

        # state is [x, y, steer_angle, vel, yaw_angle, yaw_rate, slip_angle]

        # steering delay
     
        self._steer_buf[self._sb_head] = float(raw_steer)      # write newest
        self._sb_head = (self._sb_head + 1) % self._steer_buf.size
        steer = float(self._steer_buf[self._sb_head])  


        # steering angle velocity input to steering velocity acceleration input
        accl, sv = pid(
            vel,
            steer,
            self.state[3],
            self.state[2],
            self._sv_max,
            self._a_max,
            self._v_max,
            self._v_min,
        )

        control_vec = self._control_vec
        control_vec[0] = sv
        control_vec[1] = accl
        dyn_params = self._dyn_params

        prev_state = self._state_backup
        np.copyto(prev_state, self.state)

        mode = self.integrator
        if isinstance(mode, Integrator):
            mode = mode.value
        mode = str(mode).upper()
        if mode == 'RK4':
            # RK4 integration
            k1 = vehicle_dynamics_st(self.state, control_vec, *dyn_params)

            rk_state1 = self._rk_state1
            rk_state2 = self._rk_state2
            rk_state3 = self._rk_state3
            rk_delta = self._rk_delta
            rk_accum = self._rk_accum

            np.multiply(k1, 0.5 * self.time_step, out=rk_delta)
            np.add(self.state, rk_delta, out=rk_state1)

            k2 = vehicle_dynamics_st(rk_state1, control_vec, *dyn_params)

            np.multiply(k2, 0.5 * self.time_step, out=rk_delta)
            np.add(self.state, rk_delta, out=rk_state2)

            k3 = vehicle_dynamics_st(rk_state2, control_vec, *dyn_params)

            np.multiply(k3, self.time_step, out=rk_delta)
            np.add(self.state, rk_delta, out=rk_state3)

            k4 = vehicle_dynamics_st(rk_state3, control_vec, *dyn_params)

            rk_accum[:] = k1
            rk_accum += k2
            rk_accum += k2
            rk_accum += k3
            rk_accum += k3
            rk_accum += k4

            np.multiply(rk_accum, self.time_step / 6.0, out=rk_delta)
            np.add(self.state, rk_delta, out=self.state)

        elif mode == 'EULER':
            f = vehicle_dynamics_st(self.state, control_vec, *dyn_params)
            np.multiply(f, self.time_step, out=self._rk_delta)
            np.add(self.state, self._rk_delta, out=self.state)

        else:
            raise SyntaxError(f"Invalid Integrator Specified. Provided {self.integrator}. Please choose RK4 or Euler")

        # # bound yaw angle
        # if self.state[4] > 2*np.pi:
        #     self.state[4] = self.state[4] - 2*np.pi
        # elif self.state[4] < 0:
        #     self.state[4] = self.state[4] + 2*np.pi
        if not np.all(np.isfinite(self.state)):
            # numerical blow-up; revert to previous stable state
            np.copyto(self.state, prev_state)
        else:
            # clamp state components to physically reasonable ranges
            steer_min = self._steer_min
            steer_max = self._steer_max
            v_min = self._v_min
            v_max = self._v_max
            yaw_rate_cap = 100.0
            slip_cap = np.pi / 2.0

            self.state[2] = float(np.clip(self.state[2], steer_min, steer_max))
            self.state[3] = float(np.clip(self.state[3], v_min, v_max))
            self.state[5] = float(np.clip(self.state[5], -yaw_rate_cap, yaw_rate_cap))
            self.state[6] = float(np.clip(self.state[6], -slip_cap, slip_cap))

            # keep orientation bounded for downstream trig
            self.state[4] = (self.state[4] + np.pi) % (2 * np.pi) - np.pi

            # ensure any remaining NaN/inf entries are neutralised
            np.nan_to_num(self.state, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

        # update scan
        scan_pose = self._scan_pose
        scan_pose[0] = self.state[0] + self.lidar_dist * np.cos(self.state[4])
        scan_pose[1] = self.state[1] + self.lidar_dist * np.sin(self.state[4])
        scan_pose[2] = self.state[4]
        current_scan = RaceCar.scan_simulator.scan(scan_pose, self.scan_rng)
        # current_scan = RaceCar.scan_simulator.scan(np.append(self.state[0:2],  self.state[4]), self.scan_rng)
        self.check_ttc(current_scan)

        self.scan = current_scan.astype(np.float32, copy=False)
        return self.scan



    def update_opp_poses(self, opp_poses):
        """
        Updates the vehicle's information on other vehicles

        Args:
            opp_poses (np.ndarray(num_other_agents, 3)): updated poses of other agents

        Returns:
            None
        """
        self.opp_poses = opp_poses


    def update_scan(self, agent_scans, agent_index):
        """
        Steps the vehicle's laser scan simulation
        Separated from update_pose because needs to update scan based on NEW poses of agents in the environment

        Args:
            agent scans list (modified in-place),
            agent index (int)

        Returns:
            None
        """

        if agent_scans is not None and len(agent_scans) > agent_index:
            current_scan = np.asarray(agent_scans[agent_index], dtype=np.float32)
            self.scan = current_scan
            if current_scan.size > 0:
                try:
                    self.check_ttc(current_scan)
                except Exception as e:
                    # safety fallback: don’t crash environment on TTC errors
                    print(f"[WARN] TTC check failed: {e}")
                    self.in_collision = False
            else:
                self.in_collision = False
            return

        current_scan = self.compute_scan()
        if current_scan.size == 0:
            self.in_collision = False
