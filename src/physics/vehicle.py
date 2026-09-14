import warnings
from collections.abc import Mapping
from types import MappingProxyType

import numpy as np
from numba import njit

from physics.dynamic_models import (
    vehicle_dynamics_st, pid, validate_vehicle_params, first_order_actuator_step,
    combined_slip_dynamics,
)
from physics.integration import Integrator
from physics.laser_models import ScanSimulator2D, check_ttc_jit, ray_cast
from physics.collision_models import get_vertices


def _calibration_metadata(value: Mapping, name: str):
    if (not isinstance(value, Mapping) or set(value) != {"id", "status", "source"}
            or any(not isinstance(item, str) or not item.strip() for item in value.values())
            or value["status"] not in {"uncalibrated", "measured"}):
        raise ValueError(f"{name}.calibration needs id, status (uncalibrated/measured), and source")
    return MappingProxyType(dict(value))


class WheelActuators:
    """Independent steering/wheel state for development of the nonlinear model.

    State and references are [steering angle (rad), wheel speed (rad/s)]. This
    component does not exert chassis forces and is not a selectable RaceCar
    model. No implicit transport delay or motor-torque equation is added.
    """

    _FIELDS = frozenset({
        "wheel_radius", "steering_time_constant", "wheel_speed_time_constant",
        "steering_min", "steering_max", "steering_rate_min", "steering_rate_max",
        "wheel_speed_min", "wheel_speed_max", "wheel_rate_min", "wheel_rate_max",
    })

    def __init__(self, config: Mapping):
        if not isinstance(config, Mapping):
            raise ValueError("wheel_actuators must be a mapping")
        expected = self._FIELDS | {"version", "calibration"}
        if set(config) != expected:
            raise ValueError(f"wheel_actuators fields must be exactly {sorted(expected)}")
        if type(config["version"]) is not int or config["version"] != 1:
            raise ValueError("wheel_actuators.version must be integer 1")
        calibration = _calibration_metadata(config["calibration"], "wheel_actuators")
        values = {}
        for name in self._FIELDS:
            value = config[name]
            if isinstance(value, (bool, np.bool_, str)) or not np.isscalar(value):
                raise ValueError(f"wheel_actuators.{name} must be a finite number")
            try:
                value = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"wheel_actuators.{name} must be a finite number") from exc
            if not np.isfinite(value):
                raise ValueError(f"wheel_actuators.{name} must be finite")
            values[name] = value
        for name in ("wheel_radius", "steering_time_constant", "wheel_speed_time_constant"):
            if values[name] <= 0:
                raise ValueError(f"wheel_actuators.{name} must be positive")
        for prefix in ("steering", "steering_rate", "wheel_speed", "wheel_rate"):
            lo, hi = values[f"{prefix}_min"], values[f"{prefix}_max"]
            if not lo <= 0 <= hi or lo >= hi or ("rate" in prefix and not lo < 0 < hi):
                raise ValueError(f"wheel_actuators.{prefix} limits must contain zero; rate limits must straddle it")
        if max(abs(values["steering_min"]), abs(values["steering_max"])) >= np.pi / 2:
            raise ValueError("wheel_actuators steering limits must be inside (-pi/2, pi/2)")
        self.params = MappingProxyType({**values, "version": 1,
                                       "calibration": MappingProxyType(dict(calibration))})
        self._bounds_low = np.array([values["steering_min"], values["wheel_speed_min"]])
        self._bounds_high = np.array([values["steering_max"], values["wheel_speed_max"]])
        self._time_constants = (values["steering_time_constant"], values["wheel_speed_time_constant"])
        self._rate_limits = ((values["steering_rate_min"], values["steering_rate_max"]),
                             (values["wheel_rate_min"], values["wheel_rate_max"]))
        self._state = np.zeros(2, dtype=np.float64)
        self._reference = self._state.copy()

    @property
    def state(self) -> np.ndarray:
        return self._state.copy()

    @property
    def reference(self) -> np.ndarray:
        """Applied reference after explicit command saturation."""
        return self._reference.copy()

    def reset(self, *, steering_angle=0.0, forward_speed=0.0, wheel_speed=None) -> None:
        """Clear command history; default to rolling speed only at initialization.

        An explicit wheel_speed initializes slip independently of forward_speed.
        Public spawn/reset integration belongs to the complete vehicle model.
        """
        if not np.isfinite(forward_speed):
            raise ValueError("forward_speed must be finite")
        if wheel_speed is None:
            wheel_speed = forward_speed / self.params["wheel_radius"]
        state = np.array([steering_angle, wheel_speed], dtype=np.float64)
        if (not np.all(np.isfinite(state)) or np.any(state < self._bounds_low)
                or np.any(state > self._bounds_high)):
            raise ValueError("Initial actuator state must be finite and within bounds")
        self._state[:] = state
        self._reference[:] = state

    def command(self, steering_angle: float, wheel_speed: float) -> None:
        reference = np.array([steering_angle, wheel_speed], dtype=np.float64)
        if not np.all(np.isfinite(reference)):
            raise ValueError("Actuator references must be finite")
        self._reference[:] = np.clip(reference, self._bounds_low, self._bounds_high)

    def sample(self, elapsed: float) -> np.ndarray:
        """Sample a held command at an integration stage without advancing state."""
        if not np.isfinite(elapsed) or elapsed < 0:
            raise ValueError("elapsed must be finite and nonnegative")
        return np.array([
            first_order_actuator_step(value, reference, elapsed, tau, *rates)
            for value, reference, tau, rates in zip(
                self._state, self._reference, self._time_constants, self._rate_limits)
        ])

    def advance(self, timestep: float) -> np.ndarray:
        if not np.isfinite(timestep) or timestep <= 0:
            raise ValueError("timestep must be finite and positive")
        self._state[:] = self.sample(timestep)
        return self.state


class CombinedSlipVehicle:
    """Coupled planar chassis/tire/actuator model for physics development.

    State is [x, y, psi, vx, vy, yaw_rate, delta, omega] in SI units. This class
    owns no map, sensors, trainer, or reward logic. RaceCar adapts it into the
    environment with explicit wheel-reference action and observation contracts.
    """

    _NUMERIC_FIELDS = (
        "m", "I", "lf", "lr", "h", "mu",
        "front_longitudinal_stiffness", "front_cornering_stiffness",
        "rear_longitudinal_stiffness", "rear_cornering_stiffness",
        "slip_speed_floor", "max_integration_step",
    )

    def __init__(self, config: Mapping, actuator_config: Mapping):
        expected = set(self._NUMERIC_FIELDS) | {
            "model", "model_version", "tire_model", "tire_id", "drivetrain", "calibration"}
        if not isinstance(config, Mapping) or set(config) != expected:
            raise ValueError(f"combined_slip_vehicle fields must be exactly {sorted(expected)}")
        for name, expected_value in (("model", "combined_slip_st"),
                                      ("tire_model", "smooth_friction_circle"),
                                      ("drivetrain", "shared_speed_awd")):
            if config[name] != expected_value:
                raise ValueError(f"combined_slip_vehicle.{name} must be {expected_value!r}")
        if type(config["model_version"]) is not int or config["model_version"] != 1:
            raise ValueError("combined_slip_vehicle.model_version must be integer 1")
        if not isinstance(config["tire_id"], str) or not config["tire_id"].strip():
            raise ValueError("combined_slip_vehicle.tire_id must be a nonempty string")
        params = dict(config)
        params["calibration"] = _calibration_metadata(config["calibration"], "combined_slip_vehicle")
        for name in self._NUMERIC_FIELDS:
            value = config[name]
            if isinstance(value, (bool, np.bool_, str)) or not np.isscalar(value):
                raise ValueError(f"combined_slip_vehicle.{name} must be a finite number")
            try:
                value = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"combined_slip_vehicle.{name} must be a finite number") from exc
            if not np.isfinite(value) or value < 0 or (value == 0 and name not in {"h", "mu"}):
                raise ValueError(f"combined_slip_vehicle.{name} must be finite and positive (h/mu may be zero)")
            params[name] = value
        self.params = MappingProxyType(params)
        self._actuators = WheelActuators(actuator_config)
        self._chassis = np.zeros(6, dtype=np.float64)
        self._dynamics_params = tuple(params[name] for name in self._NUMERIC_FIELDS[:-1]) + (
            self._actuators.params["wheel_radius"],)

    @property
    def state(self) -> np.ndarray:
        return np.concatenate((self._chassis, self._actuators.state))

    @property
    def reference(self) -> np.ndarray:
        return self._actuators.reference

    def reset(self, *, pose=(0.0, 0.0, 0.0), velocity=(0.0, 0.0), yaw_rate=0.0,
              steering_angle=0.0, wheel_speed=None) -> None:
        pose = np.asarray(pose, dtype=np.float64)
        velocity = np.asarray(velocity, dtype=np.float64)
        if pose.shape != (3,) or velocity.shape != (2,):
            raise ValueError("pose/velocity must have shapes (3,)/(2,)")
        chassis = np.concatenate((pose, velocity, [yaw_rate]))
        if not np.all(np.isfinite(chassis)):
            raise ValueError("Initial chassis state must be finite")
        # Validate the complete initial condition before changing either subsystem.
        actuators = WheelActuators(self._actuators.params)
        actuators.reset(steering_angle=steering_angle, forward_speed=velocity[0], wheel_speed=wheel_speed)
        combined_slip_dynamics(chassis, actuators.state, self._dynamics_params)
        self._chassis[:] = chassis
        self._actuators = actuators

    def command(self, steering_angle: float, wheel_speed: float) -> None:
        """Set steering radians and wheel rad/s, not chassis-speed commands."""
        self._actuators.command(steering_angle, wheel_speed)

    def diagnostics(self) -> dict:
        rhs, axles = combined_slip_dynamics(self._chassis, self._actuators.state, self._dynamics_params)
        yaw_rate = self._chassis[5]
        return {
            "contact_velocity": axles[:, :2].copy(),
            "slip_ratio": axles[:, 2].copy(),
            "slip_angle": axles[:, 3].copy(),
            "tire_forces": axles[:, 4:7].copy(),
            "body_acceleration": np.array([rhs[3] - yaw_rate * self._chassis[4],
                                           rhs[4] + yaw_rate * self._chassis[3]]),
            "yaw_acceleration": float(rhs[5]),
        }

    def advance(self, timestep: float) -> np.ndarray:
        """RK4 chassis integration with exact actuator samples at every stage.

        Explicit max_integration_step limits stiffness near zero contact speed.
        Failed steps raise without clipping, freezing, or committing partial state.
        """
        if not np.isfinite(timestep) or timestep <= 0:
            raise ValueError("timestep must be finite and positive")
        steps = int(np.ceil(timestep / self.params["max_integration_step"]))
        dt = timestep / steps
        state = self._chassis.copy()
        for index in range(steps):
            start = index * dt
            at_start = self._actuators.sample(start)
            at_middle = self._actuators.sample(start + dt / 2)
            at_end = self._actuators.sample((index + 1) * dt)
            k1, _ = combined_slip_dynamics(state, at_start, self._dynamics_params)
            k2, _ = combined_slip_dynamics(state + dt / 2 * k1, at_middle, self._dynamics_params)
            k3, _ = combined_slip_dynamics(state + dt / 2 * k2, at_middle, self._dynamics_params)
            k4, _ = combined_slip_dynamics(state + dt * k3, at_end, self._dynamics_params)
            state += dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        final_actuators = self._actuators.sample(timestep)
        combined_slip_dynamics(state, final_actuators, self._dynamics_params)
        if not np.all(np.isfinite(state)):
            raise ValueError("Nonfinite chassis state; reduce integration step or check parameters")
        self._actuators.advance(timestep)
        self._chassis[:] = state
        return self.state


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

    def __init__(
        self,
        params,
        seed,
        time_step=0.01,
        num_beams=1080,
        fov=4.7,
        integrator=Integrator.RK4,
        lidar_dist=0.0,
        scan_simulator=None,
    ):
        """
        Init function

        Args:
            params (dict): vehicle parameter dictionary, includes {'mu', 'C_Sf', 'C_Sr', 'lf', 'lr', 'h', 'm', 'I', 's_min', 's_max', 'sv_min', 'sv_max', 'v_switch', 'a_max': 9.51, 'v_min', 'v_max', 'length', 'width'}

            time_step (float, default=0.01): physics sim time step
            num_beams (int, default=1080): number of beams in the laser scan
            fov (float, default=4.7): field of view of the laser
            lidar_dist (float, default=0): vertical distance between LiDAR and backshaft
            scan_simulator: optional scanner shared by cars in the same environment

        Returns:
            None
        """

        # initialization
        self.params = validate_vehicle_params(params)
        self.nonlinear = self.params.get("model") == "combined_slip_st"
        self._physics = None
        if self.nonlinear:
            mode = integrator.value if isinstance(integrator, Integrator) else integrator
            if str(mode).upper() != "RK4":
                raise ValueError("combined_slip_st requires RK4 integration")
            self._physics = CombinedSlipVehicle(
                {k: v for k, v in self.params.items() if k not in {"length", "width", "wheel_actuators"}},
                self.params["wheel_actuators"])
        self._motion_frozen = False
        self.seed = seed
        self.opp_poses = []
        self.lidar_range= 30.0

        if not np.isfinite(time_step) or time_step <= 0:
            raise ValueError("physics time_step must be positive and finite")
        self.time_step = time_step
        self.num_beams = int(num_beams)
        self.fov = fov
        self.integrator = integrator
        self.lidar_dist = lidar_dist
        # Share the expensive distance transform between cars in one
        # Simulator, but never between concurrently active environments.
        self.scan_simulator = scan_simulator or ScanSimulator2D(self.num_beams, fov)
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

        self.scan_rng = np.random.default_rng(seed=self.seed)
        scan_ang_incr = self.scan_simulator.get_increment()

        # Beam geometry is environment-specific too: another environment may
        # use a different beam count or field of view.
        self.cosines = np.zeros((self.num_beams, ))
        self.scan_angles = np.zeros((self.num_beams, ))
        self.side_distances = np.zeros((self.num_beams, ))

        dist_sides = params['width']/2.
        dist_fr = (params['lf']+params['lr'])/2.

        for i in range(self.num_beams):
            angle = -fov/2. + i*scan_ang_incr
            self.scan_angles[i] = angle
            self.cosines[i] = np.cos(angle)

            if angle > 0:
                if angle < np.pi/2:
                    # between 0 and pi/2
                    to_side = dist_sides / np.sin(angle)
                    to_fr = dist_fr / np.cos(angle)
                    self.side_distances[i] = min(to_side, to_fr)
                else:
                    # between pi/2 and pi
                    to_side = dist_sides / np.cos(angle - np.pi/2.)
                    to_fr = dist_fr / np.sin(angle - np.pi/2.)
                    self.side_distances[i] = min(to_side, to_fr)
            else:
                if angle > -np.pi/2:
                    # between 0 and -pi/2
                    to_side = dist_sides / np.sin(-angle)
                    to_fr = dist_fr / np.cos(-angle)
                    self.side_distances[i] = min(to_side, to_fr)
                else:
                    # between -pi/2 and -pi
                    to_side = dist_sides / np.cos(-angle - np.pi/2)
                    to_fr = dist_fr / np.sin(-angle - np.pi/2)
                    self.side_distances[i] = min(to_side, to_fr)

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
        validated = validate_vehicle_params(params)
        if self.nonlinear or validated.get("model") == "combined_slip_st":
            raise ValueError("Recreate the environment to change nonlinear physics parameters or model")
        self.params = validated
        self._refresh_param_cache()

    @property
    def physics_state(self) -> np.ndarray:
        """Complete model state snapshot; nonlinear ordering is explicitly eight-state."""
        return self._physics.state if self.nonlinear else self.state.copy()

    @property
    def control_reference(self) -> np.ndarray:
        if not self.nonlinear:
            raise ValueError("Independent actuator references are available only for combined_slip_st")
        return self._physics.reference

    def physics_diagnostics(self) -> dict:
        if not self.nonlinear:
            raise ValueError("Tire-force diagnostics are available only for combined_slip_st")
        return self._physics.diagnostics()

    @property
    def body_velocity(self) -> tuple[float, float]:
        if self.nonlinear:
            state = self._physics.state
            return float(state[3]), float(state[4])
        speed = float(self.state[3])
        if abs(speed) < 0.5:
            return speed, 0.0
        return speed * float(np.cos(self.state[6])), speed * float(np.sin(self.state[6]))

    def _sync_physics_view(self) -> None:
        """Project into historical indices for geometry; never integrate this view."""
        x, y, psi, vx, vy, rate, delta, omega = self._physics.state
        sign = -1.0 if vx < 0 else 1.0
        speed = sign * np.hypot(vx, vy)
        beta = np.arctan2(sign * vy, sign * vx) if speed else 0.0
        self.state[:] = (x, y, delta, speed, (psi + np.pi) % (2*np.pi) - np.pi, rate, beta)

    def freeze_motion(self, previous_state=None, *, hold: bool = True) -> None:
        """Stop chassis and wheel motion; retain collision pose and steering."""
        if self.nonlinear:
            state = self.physics_state if previous_state is None else previous_state
            self._physics.reset(pose=state[:3], steering_angle=state[6], wheel_speed=0.0)
            self._motion_frozen = hold
            self._sync_physics_view()
        else:
            self.state[[3, 5, 6]] = 0.0

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
        Configure this environment's scan simulator if the map asset changed.

        Args:
            map_path (str): absolute path to the map yaml file
            map_ext (str): extension of the map image file
        """

        cache_key = (str(map_path), str(map_ext))
        cached_key = getattr(self.scan_simulator, "_map_cache_key", None)
        if cached_key == cache_key:
            return

        self.scan_simulator.set_map(map_path, map_ext)
        self.scan_simulator._map_cache_key = cache_key

    def reset(self, pose, *, friction_mu=None):
        """
        Resets the vehicle to a pose
        
        Args:
            pose (np.ndarray (3, )): pose to reset the vehicle to

        Returns:
            None
        """
        if friction_mu is not None and not self.nonlinear:
            raise ValueError("Episode friction requires combined_slip_st")
        if self.nonlinear:
            physics = self._physics
            if friction_mu is not None:
                config = dict(self._physics.params)
                config["mu"] = friction_mu
                physics = CombinedSlipVehicle(config, self.params["wheel_actuators"])
            physics.reset(pose=pose)
            self._physics = physics
            self._motion_frozen = False
        # clear control inputs
        self.accel = 0.0
        self.steer_angle_vel = 0.0
        # clear collision indicator
        self.in_collision = False
        # clear state
        self.state = np.zeros((7, ))
        self.state[0:2] = pose[0:2]
        self.state[4] = pose[2]
        if self.nonlinear:
            self._sync_physics_view()
        self.steer_buffer = np.empty((0, ))
        self._steer_buf.fill(0.0)
        self._sb_head = 0
        # reset scan random generator
        self.scan_rng = np.random.default_rng(seed=self.seed)

    def set_longitudinal_speed(self, speed: float) -> None:
        """Directly assign the vehicle's forward speed without touching pose."""
        if self.nonlinear:
            state = self._physics.state
            self._physics.reset(pose=state[:3], velocity=(float(speed), state[4]),
                                yaw_rate=state[5], steering_angle=state[6])
            self._sync_physics_view()
            return
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
            ray_cast(scan_pose, scan_view, self.scan_angles, verts)

        self.scan = scan_view

    def compute_scan(self) -> np.ndarray:
        """Recompute the LiDAR scan for the vehicle's current pose."""

        scan_pose = self._scan_pose
        scan_pose[0] = self.state[0] + self.lidar_dist * np.cos(self.state[4])
        scan_pose[1] = self.state[1] + self.lidar_dist * np.sin(self.state[4])
        scan_pose[2] = self.state[4]

        current_scan = self.scan_simulator.scan(scan_pose, self.scan_rng)
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
        if self.nonlinear:
            vx, vy = self.body_velocity
            # Project both body velocity components into each LiDAR direction.
            projected = vx * self.cosines + vy * np.sin(self.scan_angles)
            self.in_collision = bool(check_ttc_jit(
                current_scan, 1.0, self.scan_angles, projected,
                self.side_distances, self.ttc_thresh))
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
        vel (float): desired longitudinal velocity (legacy), or wheel rad/s
            reference (combined_slip_st)

        Returns:
            current_scan
        """

        if self.nonlinear:
            if not self._motion_frozen:
                self._physics.command(raw_steer, vel)  # rad, rad/s references
                self._physics.advance(self.time_step)
                self._sync_physics_view()
            return self.compute_scan()

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
        current_scan = self.scan_simulator.scan(scan_pose, self.scan_rng)
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
