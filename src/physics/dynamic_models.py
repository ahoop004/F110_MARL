# Copyright 2020 Technical University of Munich, Professorship of Cyber-Physical Systems, Matthew O'Kelly, Aman Sinha, Hongrui Zheng

# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



"""
Prototype of vehicle dynamics functions and classes for simulating 2D Single
Track dynamic model
Following the implementation of commanroad's Single Track Dynamics model
Original implementation: https://gitlab.lrz.de/tum-cps/commonroad-vehicle-models/
Author: Hongrui Zheng
"""

import numpy as np
from numba import njit
from collections.abc import Mapping
from physics.tire_models import smooth_tire_force


LEGACY_MODEL = "legacy_st"
LEGACY_MODEL_VERSION = 1
GRAVITY = 9.81
_POSITIVE_PARAMS = frozenset({"lf", "lr", "m", "I", "length", "width",
                              "v_switch", "a_max", "ang_vel_max"})
_NONNEGATIVE_PARAMS = frozenset({"mu", "C_Sf", "C_Sr", "h"})
_LIMIT_PARAMS = frozenset({"s_min", "s_max", "sv_min", "sv_max", "v_min", "v_max"})


def validate_vehicle_params(params: Mapping) -> dict:
    """Validate partial legacy parameters or a complete nonlinear configuration.

    Model identity is optional for historical scenarios. Nonlinear configurations
    must supply their own complete parameter set without legacy default merging.
    The legacy speed controller divides by both v_max and -v_min, so physical
    limits must straddle zero even for a forward-only action wrapper.
    """
    if not isinstance(params, Mapping):
        raise ValueError("vehicle_params must be a mapping")
    if params.get("model") == "combined_slip_st":
        # Local import avoids a module cycle with the numerical kernels above.
        from physics.vehicle import CombinedSlipVehicle
        required = {"wheel_actuators", "length", "width"}
        if not required <= params.keys():
            raise ValueError("combined_slip_st requires wheel_actuators, length, and width")
        component = CombinedSlipVehicle(
            {k: v for k, v in params.items() if k not in required}, params["wheel_actuators"])
        dimensions = validate_vehicle_params({k: params[k] for k in ("length", "width")})
        return {**dict(component.params), **dimensions,
                "calibration": dict(component.params["calibration"]),
                "wheel_actuators": {**dict(component._actuators.params),
                    "calibration": dict(component._actuators.params["calibration"])}}
    if params.get("model", LEGACY_MODEL) != LEGACY_MODEL:
        raise ValueError("vehicle_params.model must be 'legacy_st' or 'combined_slip_st'")
    version = params.get("model_version", LEGACY_MODEL_VERSION)
    if type(version) is not int or version != LEGACY_MODEL_VERSION:
        raise ValueError("vehicle_params.model_version must be integer 1 for legacy_st")
    allowed = _POSITIVE_PARAMS | _NONNEGATIVE_PARAMS | _LIMIT_PARAMS | {"model", "model_version"}
    unknown = set(params) - allowed
    if unknown:
        raise ValueError(f"Unsupported legacy_st vehicle parameter(s): {sorted(unknown, key=str)}")
    result = dict(params)
    for name in set(params) - {"model", "model_version"}:
        value = params[name]
        if isinstance(value, (bool, str)) or not np.isscalar(value):
            raise ValueError(f"vehicle_params.{name} must be a finite number")
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"vehicle_params.{name} must be a finite number") from exc
        if not np.isfinite(value):
            raise ValueError(f"vehicle_params.{name} must be finite")
        if name in _POSITIVE_PARAMS and value <= 0:
            raise ValueError(f"vehicle_params.{name} must be positive")
        if name in _NONNEGATIVE_PARAMS and value < 0:
            raise ValueError(f"vehicle_params.{name} must be nonnegative")
        if name in {"s_min", "sv_min", "v_min"} and value >= 0:
            raise ValueError(f"vehicle_params.{name} must be negative for legacy_st")
        if name in {"s_max", "sv_max", "v_max"} and value <= 0:
            raise ValueError(f"vehicle_params.{name} must be positive for legacy_st")
        if name in {"s_min", "s_max"} and abs(value) >= np.pi / 2:
            raise ValueError(f"vehicle_params.{name} must lie strictly inside (-pi/2, pi/2)")
        result[name] = value
    return result


@njit(cache=True)
def first_order_actuator_step(value, reference, dt, time_constant, rate_min, rate_max):
    """Exact held-reference solution of a rate-limited first-order actuator.

    Parameters are validated by WheelActuators. Outside the exponential region,
    integrate the constant rate until abs(error) == rate_limit * time_constant;
    then integrate the exponential tail. This avoids timestep-dependent clipping
    and lets a coupled vehicle integrator sample exact actuator states at its
    intermediate stages. The reference must already satisfy position/speed bounds.
    """
    error = reference - value
    rate = rate_max if error >= 0.0 else -rate_min
    direction = 1.0 if error >= 0.0 else -1.0
    linear_time = max(0.0, abs(error) / rate - time_constant)
    if dt <= linear_time:
        return value + direction * rate * dt
    boundary = value + direction * rate * linear_time
    remaining = dt - linear_time
    return boundary + (reference - boundary) * (-np.expm1(-remaining / time_constant))


@njit(cache=True)
def combined_slip_dynamics(state, actuators, params):
    """Six chassis derivatives and axle diagnostics for smooth-circle ST v1.

    state = [x, y, psi, vx, vy, r]; actuators = [delta, omega]. Each diagnostic
    row is [u_tire, v_tire, kappa, alpha, Fx_tire, Fy_tire, Fz] (front, rear).
    Coefficients are normalized by axle load, allowing simultaneous force/load
    transfer to be solved algebraically rather than using stale acceleration.
    """
    m, inertia, lf, lr, h, mu, cxf, cyf, cxr, cyr, floor, radius = params
    psi, vx, vy, yaw_rate = state[2], state[3], state[4], state[5]
    delta, omega = actuators[0], actuators[1]
    cos_delta, sin_delta = np.cos(delta), np.sin(delta)
    # Contact velocities include yaw-induced motion about the center of mass.
    vf = vy + lf * yaw_rate
    uf = vx * cos_delta + vf * sin_delta
    vf = -vx * sin_delta + vf * cos_delta
    ur, vr = vx, vy - lr * yaw_rate
    fx_unit_f, fy_unit_f, kappa_f, alpha_f = smooth_tire_force(
        uf, vf, radius * omega, mu, cxf, cyf, floor)
    fx_unit_r, fy_unit_r, kappa_r, alpha_r = smooth_tire_force(
        ur, vr, radius * omega, mu, cxr, cyr, floor)
    ax_unit_f = fx_unit_f * cos_delta - fy_unit_f * sin_delta
    ax_unit_r = fx_unit_r
    wheelbase = lf + lr
    # Fzf=m*(g*lr-h*ax)/L, Fzr=m*(g*lf+h*ax)/L, ax=sum(Fx_body)/m.
    # ax is inertial acceleration resolved in body x, not dvx/dt=ax+r*vy.
    denominator = wheelbase + h * (ax_unit_f - ax_unit_r)
    if denominator <= 0.0:
        raise ValueError("Nonpositive load-transfer denominator; outside model validity")
    ax = GRAVITY * (lr * ax_unit_f + lf * ax_unit_r) / denominator
    fzf = m * (GRAVITY * lr - h * ax) / wheelbase
    fzr = m * (GRAVITY * lf + h * ax) / wheelbase
    if fzf < 0.0 or fzr < 0.0:
        raise ValueError("Negative axle normal load; wheel lift is outside model validity")
    fxf, fyf = fx_unit_f * fzf, fy_unit_f * fzf
    fxr, fyr = fx_unit_r * fzr, fy_unit_r * fzr
    fy_front_body = fxf * sin_delta + fyf * cos_delta
    ay = (fy_front_body + fyr) / m
    rhs = np.array([
        vx * np.cos(psi) - vy * np.sin(psi),
        vx * np.sin(psi) + vy * np.cos(psi),
        yaw_rate,
        ax + yaw_rate * vy,
        ay - yaw_rate * vx,
        (lf * fy_front_body - lr * fyr) / inertia,
    ])
    axles = np.array([
        [uf, vf, kappa_f, alpha_f, fxf, fyf, fzf],
        [ur, vr, kappa_r, alpha_r, fxr, fyr, fzr],
    ])
    if not np.all(np.isfinite(rhs)) or not np.all(np.isfinite(axles)):
        raise ValueError("Nonfinite combined-slip dynamics; outside model validity")
    return rhs, axles


@njit(cache=True)
def accl_constraints(vel, accl, v_switch, a_max, v_min, v_max):
    """
    Acceleration constraints, adjusts the acceleration based on constraints

        Args:
            vel (float): current velocity of the vehicle
            accl (float): unconstraint desired acceleration
            v_switch (float): switching velocity (velocity at which the acceleration is no longer able to create wheel spin)
            a_max (float): maximum allowed acceleration
            v_min (float): minimum allowed velocity
            v_max (float): maximum allowed velocity

        Returns:
            accl (float): adjusted acceleration
    """

    # positive accl limit
    if vel > v_switch:
        pos_limit = a_max*v_switch/vel
    else:
        pos_limit = a_max

    # accl limit reached?
    if (vel <= v_min and accl <= 0) or (vel >= v_max and accl >= 0):
        accl = 0.
    elif accl <= -a_max:
        accl = -a_max
    elif accl >= pos_limit:
        accl = pos_limit

    return accl

@njit(cache=True)
def steering_constraint(steering_angle, steering_velocity, s_min, s_max, sv_min, sv_max):
    """
    Steering constraints, adjusts the steering velocity based on constraints

        Args:
            steering_angle (float): current steering_angle of the vehicle
            steering_velocity (float): unconstraint desired steering_velocity
            s_min (float): minimum steering angle
            s_max (float): maximum steering angle
            sv_min (float): minimum steering velocity
            sv_max (float): maximum steering velocity

        Returns:
            steering_velocity (float): adjusted steering velocity
    """

    # constraint steering velocity
    if (steering_angle <= s_min and steering_velocity <= 0) or (steering_angle >= s_max and steering_velocity >= 0):
        steering_velocity = 0.
    elif steering_velocity <= sv_min:
        steering_velocity = sv_min
    elif steering_velocity >= sv_max:
        steering_velocity = sv_max

    return steering_velocity


@njit(cache=True)
def vehicle_dynamics_ks(x, u_init, mu, C_Sf, C_Sr, lf, lr, h, m, I, s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max):
    """
    Single Track Kinematic Vehicle Dynamics.

        Args:
            x (numpy.ndarray (3, )): vehicle state vector (x1, x2, x3, x4, x5)
                x1: x position in global coordinates
                x2: y position in global coordinates
                x3: steering angle of front wheels
                x4: velocity in x direction
                x5: yaw angle
            u (numpy.ndarray (2, )): control input vector (u1, u2)
                u1: steering angle velocity of front wheels
                u2: longitudinal acceleration

        Returns:
            f (numpy.ndarray): right hand side of differential equations
    """
    # wheelbase
    lwb = lf + lr

    # constraints
    u = np.array([steering_constraint(x[2], u_init[0], s_min, s_max, sv_min, sv_max), accl_constraints(x[3], u_init[1], v_switch, a_max, v_min, v_max)])

    # system dynamics
    f = np.array([x[3]*np.cos(x[4]),
         x[3]*np.sin(x[4]),
         u[0],
         u[1],
         x[3]/lwb*np.tan(x[2])])
    return f

@njit(cache=True)
def vehicle_dynamics_st(x, u_init, mu, C_Sf, C_Sr, lf, lr, h, m, I, s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max):
    """
    Single Track Dynamic Vehicle Dynamics.

        Args:
            x (numpy.ndarray (3, )): vehicle state vector (x1, x2, x3, x4, x5, x6, x7)
                x1: x position in global coordinates
                x2: y position in global coordinates
                x3: steering angle of front wheels
                x4: velocity in x direction
                x5: yaw angle
                x6: yaw rate
                x7: slip angle at vehicle center
            u (numpy.ndarray (2, )): control input vector (u1, u2)
                u1: steering angle velocity of front wheels
                u2: longitudinal acceleration

        Returns:
            f (numpy.ndarray): right hand side of differential equations
    """

    # gravity constant m/s^2
    g = 9.81

    # constraints
    u = np.array([steering_constraint(x[2], u_init[0], s_min, s_max, sv_min, sv_max), accl_constraints(x[3], u_init[1], v_switch, a_max, v_min, v_max)])

    # switch to kinematic model for small velocities
    if abs(x[3]) < 0.5:
        # wheelbase
        lwb = lf + lr

        # system dynamics
        x_ks = x[0:5]
        f_ks = vehicle_dynamics_ks(x_ks, u, mu, C_Sf, C_Sr, lf, lr, h, m, I, s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max)
        f = np.hstack((f_ks, np.array([u[1]/lwb*np.tan(x[2])+x[3]/(lwb*np.cos(x[2])**2)*u[0],
        0])))

    else:
        # system dynamics
        f = np.array([x[3]*np.cos(x[6] + x[4]),
            x[3]*np.sin(x[6] + x[4]),
            u[0],
            u[1],
            x[5],
            -mu*m/(x[3]*I*(lr+lf))*(lf**2*C_Sf*(g*lr-u[1]*h) + lr**2*C_Sr*(g*lf + u[1]*h))*x[5] \
                +mu*m/(I*(lr+lf))*(lr*C_Sr*(g*lf + u[1]*h) - lf*C_Sf*(g*lr - u[1]*h))*x[6] \
                +mu*m/(I*(lr+lf))*lf*C_Sf*(g*lr - u[1]*h)*x[2],
            (mu/(x[3]**2*(lr+lf))*(C_Sr*(g*lf + u[1]*h)*lr - C_Sf*(g*lr - u[1]*h)*lf)-1)*x[5] \
                -mu/(x[3]*(lr+lf))*(C_Sr*(g*lf + u[1]*h) + C_Sf*(g*lr-u[1]*h))*x[6] \
                +mu/(x[3]*(lr+lf))*(C_Sf*(g*lr-u[1]*h))*x[2]])

    return f

@njit(cache=True)
def pid(speed, steer, current_speed, current_steer, max_sv, max_a, max_v, min_v):
    """
    Basic controller for speed/steer -> accl./steer vel.

        Args:
            speed (float): desired input speed
            steer (float): desired input steering angle

        Returns:
            accl (float): desired input acceleration
            sv (float): desired input steering velocity
    """
    # steering
    steer_diff = steer - current_steer
    if np.fabs(steer_diff) > 1e-4:
        # scale steering velocity instead of pegging at the hard limit each step
        sv = steer_diff * 2.0
        if sv > max_sv:
            sv = max_sv
        elif sv < -max_sv:
            sv = -max_sv
    else:
        sv = 0.0

    # accl
    vel_diff = speed - current_speed
    # currently forward
    if current_speed > 0.:
        if vel_diff > 0:
            # accelerate
            kp = 7.5 * max_a / max_v
            accl = kp * vel_diff
        else:
            # braking
            kp = 6.0 * max_a / (-min_v)
            accl = kp * vel_diff
    # currently backwards
    else:
        if vel_diff > 0:
            # braking
            kp = 2.5 * max_a / max_v
            accl = kp * vel_diff
        else:
            # accelerating
            kp = 2.5 * max_a / (-min_v)
            accl = kp * vel_diff

    return accl, sv


@njit(cache=True)
def sample_wheel_actuators(state, reference, elapsed, constants, rates):
    """Sample both held actuator references without changing their initial state."""
    return np.array([first_order_actuator_step(state[i], reference[i], elapsed,
                    constants[i], rates[i][0], rates[i][1]) for i in range(2)])


@njit(cache=True)
def integrate_combined_slip(
    chassis, actuator_state, reference, constants, rates, params, timestep, max_step,
):
    """Integrate and validate a proposed state; the caller commits it atomically."""
    # Keep stage arithmetic and held-reference sampling identical to scalar RK4.
    # No fastmath: tire/load validation and reproducible rounding are intentional.
    steps = int(np.ceil(timestep / max_step))
    dt = timestep / steps
    state = chassis.copy()
    for index in range(steps):
        start = index * dt
        at_start = sample_wheel_actuators(actuator_state, reference, start, constants, rates)
        at_middle = sample_wheel_actuators(actuator_state, reference, start + dt / 2, constants, rates)
        at_end = sample_wheel_actuators(actuator_state, reference, (index + 1) * dt, constants, rates)
        k1, _ = combined_slip_dynamics(state, at_start, params)
        k2, _ = combined_slip_dynamics(state + dt / 2 * k1, at_middle, params)
        k3, _ = combined_slip_dynamics(state + dt / 2 * k2, at_middle, params)
        k4, _ = combined_slip_dynamics(state + dt * k3, at_end, params)
        state += dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    final_actuators = sample_wheel_actuators(actuator_state, reference, timestep, constants, rates)
    combined_slip_dynamics(state, final_actuators, params)
    if not np.all(np.isfinite(state)):
        raise ValueError("Nonfinite chassis state; reduce integration step or check parameters")
    return state, final_actuators
