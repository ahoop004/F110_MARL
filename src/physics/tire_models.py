"""Reduced combined-slip tire law; not an MF6.1/Pacejka implementation.

Version 1 uses normalized linear stiffness with smooth radial saturation inside
an isotropic friction circle. See docs/PHYSICS_MODEL.md for the derivation,
regularization, parameter units, and limitations. Inputs are validated by the
owning vehicle component before these numerical kernels are called.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def smooth_tire_force(u, v, rolling_speed, mu, c_long, c_lat, speed_floor):
    """Return (Fx/Fz, Fy/Fz, kappa, alpha) in the tire frame.

    Forward x, left y, positive wheel speed for forward rotation. Slip signs
    oppose contact-patch relative motion in forward and reverse travel. The
    symmetric longitudinal denominator stays finite at rest and wheel lock.
    Lateral slip uses the longitudinal contact speed, not chassis speed.
    """
    longitudinal_scale = max(abs(u), abs(rolling_speed), speed_floor)
    lateral_scale = max(abs(u), speed_floor)
    kappa = (rolling_speed - u) / longitudinal_scale
    alpha = np.arctan2(v, lateral_scale)
    qx = c_long * kappa
    qy = -c_lat * v / lateral_scale  # -C_alpha*tan(alpha), without trig roundoff
    demand = np.hypot(qx, qy)
    if mu == 0.0 or demand == 0.0:
        return 0.0, 0.0, kappa, alpha
    # Near zero tanh(z)/z -> 1, retaining the configured linear stiffness.
    # At high slip the vector magnitude approaches mu, sharing grip across axes.
    scale = mu * np.tanh(demand / mu) / demand
    return qx * scale, qy * scale, kappa, alpha
