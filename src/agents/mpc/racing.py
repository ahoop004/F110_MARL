"""Actuator-aware shooting MPC for fixed racing opponents.

The prediction model retains planar velocity and yaw momentum, with MF6.1
tire forces at static axle loads. It omits the plant's dynamic load transfer.
Track clearance uses the occupancy map and the entire rectangular footprint.
Traffic uses range-limited, perfect simulator states with constant world velocity
(including stationary wrecks). This privileged sensing contract is intentional.
"""
from __future__ import annotations

import numpy as np
from numba import njit
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.optimize import minimize
from utils.track_preview import _resample_uniform
from physics.dynamic_models import first_order_actuator_step
from physics.tire_models import MF61_KEYS, mf61_tire_force


@njit(cache=True)
def _clip(value, low, high):
    return min(max(value, low), high)


@njit(cache=True)
def _distance(field, x, y, origin, resolution):
    dx, dy = x - origin[0], y - origin[1]
    co, si = np.cos(origin[2]), np.sin(origin[2])
    u, v = (co * dx + si * dy) / resolution, (-si * dx + co * dy) / resolution
    ix, iy = int(np.floor(u)), int(np.floor(v))
    if ix < 0 or iy < 0 or ix + 1 >= field.shape[1] or iy + 1 >= field.shape[0]:
        return -1.0
    a, b = u - ix, v - iy
    return ((1-a)*(1-b)*field[iy, ix] + a*(1-b)*field[iy, ix+1]
            + (1-a)*b*field[iy+1, ix] + a*b*field[iy+1, ix+1])


@njit(cache=True)
def _chassis_rhs(state, delta, wheel, p):
    """MF6.1 forces with static axle loads; no algebraic yaw-rate clipping."""
    x, y, yaw, vx, _, _, vy, rate = state
    length, lr, mass, inertia = p[0], p[1], p[10], p[11]
    lf = length-lr
    co, si = np.cos(delta), np.sin(delta)
    front_v = vy+lf*rate
    front_load = mass*9.81*lr/length
    rear_load = mass*9.81*lf/length
    n = len(MF61_KEYS)
    fx_f, fy_f, _, _ = mf61_tire_force(
        vx*co+front_v*si, -vx*si+front_v*co, wheel,
        front_load, p[7], p[13:13+n], p[12])
    fx_r, fy_r, _, _ = mf61_tire_force(
        vx, vy-lr*rate, wheel, rear_load, p[7], p[13+n:], p[12])
    front_y = fx_f*si+fy_f*co
    return np.array([vx*np.cos(yaw)-vy*np.sin(yaw),
                     vx*np.sin(yaw)+vy*np.cos(yaw), rate,
                     (fx_f*co-fy_f*si+fx_r)/mass+rate*vy,
                     0., 0., (front_y+fy_r)/mass-rate*vx,
                     (lf*front_y-lr*fy_r)/inertia])


@njit(cache=True)
def predict_step(state, control, p, dt):
    """State x,y,yaw,vx,delta,rolling-wheel-speed,vy,yaw-rate; rad,m/s commands.

    Midpoint integration uses smaller steps for stiff low-speed tire response.
    Actuators use the same exact held-reference solution as the simulator.
    """
    state = state.copy()
    front_speed = state[3]*np.cos(state[4])+(state[6]+(p[0]-p[1])*state[7])*np.sin(state[4])
    contact_speed = min(abs(state[3]), abs(front_speed))
    # Reserve the possible speed drop over this decision when choosing the
    # stiffness bound. Cap at 20 ms even at high speed; use 4 ms near rest.
    contact_speed -= dt*(p[7]*9.81+abs(state[7]*state[6]))
    max_step = min(.02, .004*max(.5, contact_speed)/.5)
    steps = max(1, int(np.ceil(dt/max_step)))
    h = dt/steps
    for _ in range(steps):
        delta, wheel = state[4], state[5]
        mid = state + .5*h*_chassis_rhs(state, delta, wheel, p)
        mid_delta = first_order_actuator_step(delta, control[0], h/2, p[2], -p[3], p[3])
        mid_wheel = first_order_actuator_step(wheel, control[1], h/2, p[4], -p[5], p[5])
        state += h*_chassis_rhs(mid, mid_delta, mid_wheel, p)
        state[4] = first_order_actuator_step(delta, control[0], h, p[2], -p[3], p[3])
        state[5] = first_order_actuator_step(wheel, control[1], h, p[4], -p[5], p[5])
    return state


def _speed_profile(points, max_speed, mu, utilization, deceleration):
    """Curvature limits with a backward braking pass, in rolling-speed units."""
    indices = np.arange(len(points))
    before = points-points[np.maximum(indices-3, 0)]
    after = points[np.minimum(indices+3, len(points)-1)]-points
    lengths = .5*(np.linalg.norm(before, axis=1)+np.linalg.norm(after, axis=1))
    turns = np.arctan2(before[:, 0]*after[:, 1]-before[:, 1]*after[:, 0],
                      np.sum(before*after, axis=1))
    curvature = np.abs(turns)/np.maximum(lengths, 1e-6)
    grip = utilization*mu*9.81
    speeds = np.minimum(max_speed, np.sqrt(grip/np.maximum(curvature, 1e-6)))
    braking = min(deceleration, .5*grip)
    for i in range(len(points)-2, -1, -1):
        distance = np.linalg.norm(points[i+1]-points[i])
        speeds[i] = min(speeds[i], np.sqrt(speeds[i+1]**2+2*braking*distance))
    return speeds


@njit(cache=True)
def _limit_command(command, previous, dt, speed_rate, steering_rate):
    return np.array([_clip(command[0], previous[0]-steering_rate*dt,
                           previous[0]+steering_rate*dt),
                     _clip(command[1], previous[1]-speed_rate*dt,
                           previous[1]+speed_rate*dt)])


@njit(cache=True)
def _shoot(controls, initial, path, field, origin, resolution, footprint,
           traffic, p, dt, repeat, speed_limit, margin, reference_rate,
           steering_rate=1.5, grip_utilization=.8):
    state = initial[:8].copy()
    previous = np.array([initial[9], initial[8]])
    cost, min_clearance = 0.0, 1e6
    trajectory = np.empty((len(controls)*repeat, 8))
    last_index = 0
    previous_s = 0.0
    for k in range(len(controls)*repeat):
        command = controls[k // repeat].copy()
        command = _limit_command(command, previous, dt, reference_rate, steering_rate)
        # Penalize every executed change, including the first command and ramps
        # inside a knot. Compare against the applied reference, not raw knots.
        cost += 1.0*(command[0]-previous[0])**2/dt
        cost += .005*(command[1]-previous[1])**2
        previous = command
        state = predict_step(state, command, p, dt)
        if not np.isfinite(state).all():
            return np.inf, -np.inf, trajectory
        trajectory[k] = state
        # Local ordered centerline, extended across the finish seam by the caller.
        best, idx = 1e20, last_index
        for j in range(max(0, last_index-5), min(len(path)-1, last_index+30)):
            dx, dy = state[0]-path[j, 0], state[1]-path[j, 1]
            d = dx*dx+dy*dy
            if d < best:
                best, idx = d, j
        last_index = idx
        dx, dy = path[idx+1, 0]-path[idx, 0], path[idx+1, 1]-path[idx, 1]
        segment_length = max(np.sqrt(dx*dx+dy*dy), 1e-9)
        tx, ty = dx/segment_length, dy/segment_length
        ex, ey = state[0]-path[idx, 0], state[1]-path[idx, 1]
        contour = -ty*ex+tx*ey
        s = path[idx, 2] + _clip(tx*ex+ty*ey, 0., segment_length)
        if k == 0:
            previous_s = path[5, 2]  # caller leaves five points behind ego
        cost += 3.0*contour*contour*dt - 4.0*(s-previous_s)
        previous_s = s
        heading_error = np.arctan2(np.sin(state[2]-np.arctan2(ty, tx)), np.cos(state[2]-np.arctan2(ty, tx)))
        target_speed = min(speed_limit, path[idx, 3])
        cost += .3*heading_error**2*dt + .1*(state[3]-target_speed)**2*dt
        cost += 8.*max(0., state[3]-target_speed)**2*dt
        slip = np.arctan2(state[6], max(abs(state[3]), .5))
        cost += 30.*max(0., abs(slip)-.12)**2*dt
        lateral_demand = abs(state[3]*state[7])
        cost += 2.*max(0., lateral_demand-grip_utilization*p[7]*9.81)**2*dt
        co, si = np.cos(state[2]), np.sin(state[2])
        for point in footprint:
            x = state[0]+co*point[0]-si*point[1]
            y = state[1]+si*point[0]+co*point[1]
            clearance = _distance(field, x, y, origin, resolution)
            # Slack is relative to the required wall margin. Traffic below
            # already includes margin in the enclosing ellipse dimensions.
            min_clearance = min(min_clearance, clearance-margin)
            cost += 2000.*max(0., margin+.05-clearance)**2*dt
        # Oriented ellipse enclosing both vehicle rectangles: conservative during passes.
        t = (k+1)*dt
        for other in traffic:
            dx = state[0] - (other[0]+other[3]*t)
            dy = state[1] - (other[1]+other[4]*t)
            c, sn = np.cos(other[2]), np.sin(other[2])
            along, across = c*dx+sn*dy, -sn*dx+c*dy
            # Rotated ego support added to the other vehicle's half dimensions.
            angle = state[2]-other[2]
            a = p[8]/2*(1+abs(np.cos(angle)))+p[9]/2*abs(np.sin(angle))+margin+.03*t
            b = p[9]/2*(1+abs(np.cos(angle)))+p[8]/2*abs(np.sin(angle))+margin+.03*t
            separation = np.sqrt((along/a)**2+(across/b)**2)
            cost += 3000.*max(0., np.sqrt(2.)-separation)**2*dt
            min_clearance = min(min_clearance, (separation/np.sqrt(2.)-1)*min(a, b))
    return cost, min_clearance, trajectory


class RacingMPCAgent:
    """Bounded nonlinear shooting optimization; explicit brake candidate/fallback."""

    def __init__(self, config):
        cfg = config.get('params', config)
        self.agent_id = cfg.get('agent_id')
        self.max_speed = float(cfg.get('max_speed', 5.0))
        self.horizon = int(cfg.get('horizon', 30))
        self.knots = int(cfg.get('knots', 6))
        self.iterations = int(cfg.get('iterations', 18))
        self.max_evaluations = int(cfg.get('max_evaluations', 260))
        self.margin = float(cfg.get('margin', .10))
        self.sensing_range = float(cfg.get('sensing_range', 10.))
        self.acceleration = float(cfg.get('max_acceleration', 5.))
        self.steering_reference_rate = float(cfg.get('max_steering_reference_rate', 1.5))
        self.grip_utilization = float(cfg.get('grip_utilization', .8))
        if (self.horizon < 2 or self.knots < 2 or self.horizon % self.knots or
                self.iterations < 1 or self.max_evaluations < 1 or not np.isfinite([self.max_speed, self.margin,
                self.sensing_range, self.acceleration, self.steering_reference_rate,
                self.grip_utilization]).all() or self.max_speed <= 0
                or self.margin < 0 or self.sensing_range <= 0 or self.acceleration <= 0
                or self.steering_reference_rate <= 0 or not 0 < self.grip_utilization <= 1):
            raise ValueError('Invalid racing_mpc horizon, optimization, or physical limits')
        self.env = None
        self.reset()

    def set_env(self, env):
        if env.params.get('model') != 'combined_slip_st':
            raise ValueError('racing_mpc requires the current combined_slip_st vehicle')
        self.env = env
        v, a = env.params, env.params['wheel_actuators']
        self.dt = float(env.timestep)
        self.radius = a['wheel_radius']
        self.steer_min, self.steer_max = a['steering_min'], a['steering_max']
        self.max_speed = min(self.max_speed, a['wheel_speed_max']*self.radius)
        self.p = np.array([v['lf']+v['lr'], v['lr'], a['steering_time_constant'],
                          min(-a['steering_rate_min'], a['steering_rate_max']),
                          a['wheel_speed_time_constant'],
                          min(-a['wheel_rate_min'], a['wheel_rate_max'])*self.radius,
                          self.acceleration, v['mu'], v['length'], v['width'],
                          v['m'], v['I'], v['slip_speed_floor'],
                          *[v['front_tire'][key] for key in MF61_KEYS],
                          *[v['rear_tire'][key] for key in MF61_KEYS]])
        self._map_key = None
        self.reset()

    def reset(self):
        self._warm = None
        self._decisions = 0
        self.last_plan = {}

    def _map(self):
        key = (id(self.env.centerline_points), str(self.env.map_image_path))
        if key == self._map_key:
            return
        path, closed = _resample_uniform(np.asarray(self.env.centerline_points)[:, :2], .10)
        if not closed:
            raise ValueError('racing_mpc currently requires a closed centerline')
        self.path = np.asarray(path, dtype=np.float64)
        meta = self.env.map_meta
        self.origin = np.array(meta['origin'], dtype=np.float64)
        self.resolution = float(meta['resolution'])
        with Image.open(self.env.map_image_path) as image:
            pixels = np.flipud(np.asarray(image.convert('L'), dtype=np.float64)) / 255.
        occupancy = pixels if meta.get('negate', 0) else 1-pixels
        free = occupancy < float(meta.get('free_thresh', .196))
        self.field = (distance_transform_edt(free)-distance_transform_edt(~free))*self.resolution
        # Sample the entire rectangle, not only corners (thin walls can cross edges).
        xs = np.linspace(-self.p[8]/2, self.p[8]/2, int(np.ceil(self.p[8]/self.resolution))+1)
        ys = np.linspace(-self.p[9]/2, self.p[9]/2, int(np.ceil(self.p[9]/self.resolution))+1)
        self.footprint = np.array([(x, y) for x in xs for y in ys])
        # Retain the source array so an id reused after a map reload cannot hit
        # the cache for different geometry.
        self._map_points = self.env.centerline_points
        self._map_key = key
        self._warm = None

    def _traffic(self, pose, aid):
        if aid not in self.env.possible_agents:
            raise ValueError('racing_mpc requires its agent id to exclude itself from traffic')
        rows = []
        for other_id in self.env.possible_agents:
            if other_id == aid:
                continue
            other = self.env.get_agent_state(other_id)
            if np.linalg.norm(other.pose[:2]-pose[:2]) > self.sensing_range:
                continue
            co, si = np.cos(other.pose[2]), np.sin(other.pose[2])
            vx, vy = other.velocity
            rows.append([*other.pose, co*vx-si*vy, si*vx+co*vy])
        return np.array(rows, dtype=np.float64).reshape(-1, 5)

    def act(self, obs, deterministic=False, aid=None):
        if self.env is None:
            raise ValueError('Call set_env before using racing_mpc')
        self._map()
        agent_id = aid if aid is not None else self.agent_id
        ego = self.env.get_agent_state(agent_id)
        physics = self.env.get_global_state().metadata.get('physics', {})
        self.p[7] = float(physics.get('mu', self.env.params['mu']))
        if not np.isfinite(self.p[7]) or self.p[7] < 0:
            raise ValueError('Invalid racing_mpc episode friction')
        pose = np.asarray(obs['pose'], dtype=np.float64)
        initial = np.array([*pose, float(obs['velocity'][0]), float(obs['steering_angle']),
                            float(obs['wheel_speed'])*self.radius,
                            float(ego.velocity[1]), float(ego.angular_velocity),
                            float(obs['wheel_speed_reference'])*self.radius,
                            float(obs['steering_reference'])])
        if not np.isfinite(initial).all():
            raise ValueError('Nonfinite racing_mpc observation')
        nearest = int(np.argmin(np.sum((self.path-pose[:2])**2, axis=1)))
        count = int(np.ceil((self.max_speed*self.horizon*self.dt+4.)/.10))+10
        points = self.path[(np.arange(-5, count)+nearest) % len(self.path)]
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        speeds = _speed_profile(points, self.max_speed, self.p[7],
                                self.grip_utilization, self.acceleration)
        local = np.column_stack((points, arc, speeds))
        traffic = self._traffic(pose, aid if aid is not None else self.agent_id)
        repeat = self.horizon//self.knots
        args = (initial, local, self.field, self.origin, self.resolution,
                self.footprint, traffic, self.p, self.dt, repeat, self.max_speed,
                self.margin, self.acceleration, self.steering_reference_rate,
                self.grip_utilization)
        def objective(flat):
            return _shoot(flat.reshape(self.knots, 2), *args)[0]
        # Curvature-based seed gives the optimizer useful steering from rest.
        seed = np.zeros((self.knots, 2))
        seed[:, 1] = self.max_speed
        for k in range(self.knots):
            idx = min(len(points)-3, 5+int((k+.5)*repeat*self.dt*max(1., initial[3])/.10))
            u, v = points[idx]-points[idx-1], points[idx+1]-points[idx]
            turn = np.arctan2(u[0]*v[1]-u[1]*v[0], np.dot(u, v))/.10
            seed[k, 1] = speeds[idx]
            seed[k, 0] = np.clip(np.arctan(self.p[0]*turn), self.steer_min, self.steer_max)
        if self._warm is not None:
            seed = self._warm.copy()
        candidates = []
        starts = [seed]
        # Alternate pass-side starts avoid a symmetric stationary-obstacle minimum.
        # Re-solving all three starts at every decision wastes most traffic CPU
        # time after a safe passing trajectory has already been found. Retry
        # when the warm plan becomes unsafe, or periodically when following slowly.
        warm_clearance = _shoot(seed, *args)[1]
        retry_pass = (self._decisions % 10 == 0 and initial[3] < .6*self.max_speed)
        if len(traffic) and (warm_clearance < 0. or retry_pass):
            for bias in (-.12, .12):
                variant = seed.copy()
                variant[:2, 0] = np.clip(variant[:2, 0]+bias, self.steer_min, self.steer_max)
                starts.append(variant)
        bounds = [(self.steer_min, self.steer_max), (0., self.max_speed)]*self.knots
        for start in starts:
            result = minimize(objective, start.ravel(), method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': self.iterations, 'maxfun': self.max_evaluations,
                                       'ftol': 1e-5, 'eps': 1e-4, 'maxls': 6})
            # The best iterate may be useful even when the iteration budget expires.
            for flat in (start.ravel(), result.x):
                if np.isfinite(flat).all():
                    controls = flat.reshape(self.knots, 2)
                    cost, clearance, traj = _shoot(controls, *args)
                    if np.isfinite(cost):
                        candidates.append((cost, clearance, controls.copy(), traj))
        # Evaluate braking while following the plan, holding steering, and
        # straightening. Select the safest braking trajectory if none is feasible.
        braking_candidates = []
        for steering in (seed[:, 0], np.full(self.knots, initial[9]), np.zeros(self.knots)):
            brake = np.column_stack((steering, np.zeros(self.knots)))
            cost, clearance, traj = _shoot(brake, *args)
            braking_candidates.append((cost, clearance, brake, traj))
        candidates.extend(braking_candidates)
        feasible = [c for c in candidates if c[1] >= 0.]
        selected = (min(feasible, key=lambda c: c[0]) if feasible else
                    max(braking_candidates, key=lambda c: (c[1], -c[0])))
        self._warm = selected[2].copy()
        # Shift the piecewise controls by one decision, interpolating knot values.
        self._warm[:-1] += (self._warm[1:]-self._warm[:-1])/repeat
        action = selected[2][0].copy()
        fallback = not feasible
        if fallback:
            action[1] = 0.
            self._warm = None
        action = _limit_command(action, np.array([initial[9], initial[8]]), self.dt,
                                self.acceleration, self.steering_reference_rate)
        self.last_plan = {'predicted_safety_slack_m': float(selected[1]),
                          'friction_mu': float(self.p[7]),
                          'target_speed_mps': float(speeds[5]),
                          'brake_fallback': fallback, 'traffic_count': len(traffic),
                          'optimization_starts': len(starts),
                          'trajectory': selected[3]}
        self._decisions += 1
        return action.astype(np.float32)
