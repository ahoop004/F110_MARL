"""Steady-state planar MF6.1 forces at nominal pressure and zero camber.

Pacejka, Tire and Vehicle Dynamics, 3rd ed., equations 4.E9--4.E29 and
4.E50--4.E67 (turn-slip factors and unlisted scale factors equal one).
Only Fx/Fy are modeled: no aligning moment, relaxation, camber, or pressure
state. Coefficients describe an effective axle, with FNOMIN its nominal load.
The low-speed/reverse continuation is explicit and is not a hardware fit.
See docs/PHYSICS_MODEL.md for sources, supported coefficients, and assumptions.
"""
from collections.abc import Mapping
import numpy as np
from numba import njit

MF61_KEYS = (
    "FNOMIN", "PCX1", "PDX1", "PDX2", "PEX1", "PEX2", "PEX3", "PEX4",
    "PKX1", "PKX2", "PKX3", "PHX1", "PHX2", "PVX1", "PVX2",
    "PCY1", "PDY1", "PDY2", "PEY1", "PEY2", "PEY3", "PKY1", "PKY2",
    "PKY4", "PHY1", "PHY2", "PVY1", "PVY2",
    "RBX1", "RBX2", "RCX1", "REX1", "REX2", "RHX1",
    "RBY1", "RBY2", "RBY3", "RCY1", "REY1", "REY2", "RHY1", "RHY2",
    "RVY1", "RVY2", "RVY4", "RVY5", "RVY6",
)


def validate_mf61_coefficients(config):
    """Require explicit coefficients; never interpret old stiffnesses as MF data."""
    if not isinstance(config, Mapping) or set(config) != set(MF61_KEYS):
        raise ValueError("MF6.1 coefficients must contain exactly: " + ", ".join(MF61_KEYS))
    values = {}
    for key in MF61_KEYS:
        value = config[key]
        if isinstance(value, (bool, np.bool_, str)) or not np.isscalar(value):
            raise ValueError(f"MF6.1 {key} must be a finite number")
        try:
            value = float(value)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(f"MF6.1 {key} must be a finite number") from exc
        if not np.isfinite(value):
            raise ValueError(f"MF6.1 {key} must be a finite number")
        values[key] = value
    for key in ("FNOMIN", "PCX1", "PDX1", "PKX1", "PCY1", "PDY1", "PKY2", "PKY4", "RBX1", "RBY1"):
        if values[key] <= 0:
            raise ValueError(f"MF6.1 {key} must be positive")
    if values["PKY1"] >= 0:
        raise ValueError("MF6.1 PKY1 must be negative for the left-positive slip-angle convention")
    for key in ("RCX1", "RCY1"):
        if not 0 < values[key] <= 1:
            raise ValueError(f"MF6.1 {key} must lie in (0, 1] for nonnegative combined-slip reduction")
    return values


@njit(cache=True)
def _mf_shape(value, stiffness, shape, curvature, cosine=False):
    z = stiffness * value
    phase = shape * np.arctan(z - curvature * (z - np.arctan(z)))
    return np.cos(phase) if cosine else np.sin(phase)


@njit(cache=True)
def mf61_tire_force(u, v, rolling_speed, normal_load, mu, p, speed_floor):
    """Return Fx, Fy [N], kappa, alpha [rad]; x forward, y left.

    Kappa uses ground speed (not wheel speed) as its denominator. The finite
    floor and abs(u) extend the forward-running equations through rest/reverse.
    Alpha is the signed contact-velocity angle; PKY1 < 0 restores lateral slip.
    mu acts as LMUX=LMUY; all other MF scaling factors are unity.
    """
    scale = max(abs(u), speed_floor)
    kappa = (rolling_speed - u) / scale
    alpha = np.arctan2(v, scale)
    if normal_load <= 0.0 or mu == 0.0:
        return 0.0, 0.0, kappa, alpha
    (fz0, cx, dx1, dx2, ex1, ex2, ex3, ex4, kx1, kx2, kx3,
     hx1, hx2, vx1, vx2, cy, dy1, dy2, ey1, ey2, ey3, ky1, ky2,
     ky4, hy1, hy2, vy1, vy2, bx1, bx2, cxa, exa1, exa2, hxa,
     by1, by2, by3, cyk, eyk1, eyk2, hyk1, hyk2,
     vyk1, vyk2, vyk4, vyk5, vyk6) = p
    dfz = (normal_load - fz0) / fz0
    mux = mu * (dx1 + dx2 * dfz)
    muy = mu * (dy1 + dy2 * dfz)
    kx = normal_load * (kx1 + kx2 * dfz) * np.exp(kx3 * dfz)
    ky = ky1 * fz0 * np.sin(ky4 * np.arctan(normal_load / (ky2 * fz0)))
    if mux <= 0.0 or muy <= 0.0 or kx <= 0.0 or ky >= 0.0:
        raise ValueError("MF6.1 coefficients outside valid normal-load range")
    dx, dy = mux * normal_load, muy * normal_load
    sx, sy = kappa + hx1 + hx2 * dfz, alpha + hy1 + hy2 * dfz
    ex = min(1.0, (ex1 + ex2 * dfz + ex3 * dfz * dfz) * (1.0 - ex4 * np.sign(sx)))
    ey = min(1.0, (ey1 + ey2 * dfz) * (1.0 - ey3 * np.sign(sy)))
    # Prime friction scaling in the longitudinal vertical shift (4.E8/E18).
    mu_prime = 10.0 * mu / (1.0 + 9.0 * mu)
    fx0 = dx * _mf_shape(sx, kx / (cx * dx), cx, ex) + normal_load * (vx1 + vx2 * dfz) * mu_prime
    fy0 = dy * _mf_shape(sy, ky / (cy * dy), cy, ey) + normal_load * (vy1 + vy2 * dfz) * mu
    # Combined-slip weighting functions, normalized to one at pure slip.
    bxa = bx1 / np.sqrt(1.0 + (bx2 * kappa) ** 2)
    exa = min(1.0, exa1 + exa2 * dfz)
    byk = by1 / np.sqrt(1.0 + (by2 * (alpha - by3)) ** 2)
    eyk = min(1.0, eyk1 + eyk2 * dfz)
    hyk = hyk1 + hyk2 * dfz
    gx0 = _mf_shape(hxa, bxa, cxa, exa, True)
    gy0 = _mf_shape(hyk, byk, cyk, eyk, True)
    if gx0 <= 1e-10 or gy0 <= 1e-10:
        raise ValueError("MF6.1 combined-slip normalization is singular")
    gx = _mf_shape(alpha + hxa, bxa, cxa, exa, True) / gx0
    gy = _mf_shape(kappa + hyk, byk, cyk, eyk, True) / gy0
    shift = (muy * normal_load * (vyk1 + vyk2 * dfz)
             / np.sqrt(1.0 + (vyk4 * alpha) ** 2)
             * np.sin(vyk5 * np.arctan(vyk6 * kappa)))
    return gx * fx0, gy * fy0 + shift, kappa, alpha
