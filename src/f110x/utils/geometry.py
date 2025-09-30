"""Lightweight geometry helpers for reward shaping and control logic."""

from __future__ import annotations

import math
from typing import Iterable, Tuple


def wrap_to_pi(angle: float) -> float:
    """Wrap an angle in radians to the interval [-pi, pi]."""

    wrapped = (angle + math.pi) % (2.0 * math.pi) - math.pi
    # Guard against tiny numerical drift at the boundary.
    if wrapped == -math.pi:
        return math.pi
    return wrapped


def resolve_pose(pose: Iterable[float]) -> Tuple[float, float, float]:
    """Coerce an arbitrary pose-like iterable to (x, y, theta)."""

    try:
        x, y, *rest = pose  # type: ignore[misc]
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError("pose must provide at least x and y") from exc

    theta = float(rest[0]) if rest else 0.0
    return float(x), float(y), theta


def compute_position2(
    target_pose: Iterable[float],
    d_back: float,
    lat_offset: float,
) -> Tuple[float, float]:
    """Return the Position-2 waypoint relative to the given target pose."""

    tx, ty, theta = resolve_pose(target_pose)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    xp2 = tx - d_back * cos_t - lat_offset * sin_t
    yp2 = ty - d_back * sin_t + lat_offset * cos_t
    return xp2, yp2


def relative_bearing(
    target_pose: Iterable[float],
    attacker_xy: Tuple[float, float],
) -> float:
    """Bearing from the target frame to the attacker (wrapped to [-pi, pi])."""

    tx, ty, theta = resolve_pose(target_pose)
    ax, ay = attacker_xy
    raw = math.atan2(ay - ty, ax - tx)
    return wrap_to_pi(raw - theta)


def heading_alignment(
    attacker_pose: Iterable[float],
    point_xy: Tuple[float, float],
) -> float:
    """Return |angle| between the attacker's heading and the vector to point."""

    ax, ay, atheta = resolve_pose(attacker_pose)
    px, py = point_xy
    direction = math.atan2(py - ay, px - ax)
    return abs(wrap_to_pi(direction - atheta))
